from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle, json

from .scoring import pick_tau_by_target, ridge_fit, ridge_apply, isotonic_fit, isotonic_apply
from .utils import drawdown_R
from .regimes import load_model as load_regime_model, assign_regime


def simulate_fold(symbol: str, bars: pd.DataFrame, features: pd.DataFrame, candidates: pd.DataFrame, events: pd.DataFrame, banks: Dict[str, Any], cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    # Per-regime training on TRAIN only; score on TEST only
    lookback = cfg["scoring"]["gate"]["lookback_hits_bars"]
    # Gates
    gate_cfg = cfg.get("scoring", {}).get("gate", {})
    min_good_hits = int(gate_cfg.get("min_good_hits", cfg.get("scoring", {}).get("min_good_hits", 0)))
    bad_to_good_max = float(gate_cfg.get("bad_to_good_max_ratio", cfg.get("scoring", {}).get("bad_to_good_max_ratio", 10.0)))
    kofn_cfg = cfg.get("scoring", {}).get("k_of_n", {"enabled": False, "required": 1})
    kofn_enabled = bool(kofn_cfg.get("enabled", False))
    kofn_required = int(kofn_cfg.get("required", 2))

    # Merge labels into candidates (supports new fields: ts, R, side)
    cols_ev = [c for c in ["bar_idx","label","R","ts","side"] if c in events.columns]
    df = candidates.merge(events[cols_ev], on="bar_idx", how="left")
    df["label"] = df["label"].fillna(0).astype(int)
    if "R" in df:
        df["realized_R"] = df["R"].fillna(0.0).astype(float)
    else:
        df["realized_R"] = 0.0
    if "timestamp" not in df and "ts" in df:
        df["timestamp"] = df["ts"]
    print(f"[simulate] {symbol} candidates={len(candidates)} events={len(events)} merged={len(df)}")

    if len(df) == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trades.csv").write_text("R\n", encoding="utf-8")
        stats = {"symbol": symbol, "trades": 0, "win_rate": 0.0, "sum_R": 0.0, "avg_R": 0.0, "median_R": 0.0, "max_dd_R": 0.0}
        with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats

    # Channels matrix
    channels = cfg["features"]["channels"]
    arrs = [features[ch].values for ch in channels if ch in features.columns]
    X = np.vstack(arrs)  # C x N

    # Load regime fingerprints if present
    art_dir = out_dir / "artifacts"
    regime_model_path = art_dir / "regime_fingerprints.pkl"
    regime_model = load_regime_model(regime_model_path) if regime_model_path.exists() else None
    regime_features = regime_model["features"] if regime_model else []

    # Helpers to load side-specific banks
    def load_side_banks(rid: int, side: str):
        root = art_dir / "banks" / f"regime_{rid}" / side
        if not root.exists():
            return None
        try:
            with open(root / "good.pkl", "rb") as f:
                good_map = pickle.load(f)
            with open(root / "bad.pkl", "rb") as f:
                bad_map = pickle.load(f)
        except Exception:
            return None
        out = {}
        for k in sorted(set(list(good_map.keys()) + list(bad_map.keys()))):
            out[k] = {"GOOD": good_map.get(k), "BAD": bad_map.get(k)}
        return out

    def load_global_side_banks(side: str):
        root = art_dir / "banks" / "global" / side
        if not root.exists():
            return None
        try:
            with open(root / "good.pkl", "rb") as f:
                good_map = pickle.load(f)
            with open(root / "bad.pkl", "rb") as f:
                bad_map = pickle.load(f)
        except Exception:
            return None
        out = {}
        for k in sorted(set(list(good_map.keys()) + list(bad_map.keys()))):
            out[k] = {"GOOD": good_map.get(k), "BAD": bad_map.get(k)}
        return out

    banks_cache: Dict[tuple, Any] = {}

    # Compute hits across lookback and min distances
    def compute_hits_and_dists(i0: int, bundle_any: Dict[str, Any]):
        good_hits = 0; bad_hits = 0; disc_hits = 0
        min_dg = float("inf"); min_db = float("inf")
        lengths_with_good = set()
        for i in range(max(0, i0 - lookback), i0 + 1):
            for banks_key, bundle in bundle_any.items():
                # pre/post from any shapelet meta
                pre = 20; post = 40
                for kind, bank in bundle.items():
                    if hasattr(bank, "shapelets") and len(bank.shapelets) > 0:
                        meta = bank.shapelets[0].meta
                        pre = int(meta.get("pre", pre)); post = int(meta.get("post", post))
                        break
                start = i - pre; end = i + post
                if start < 0 or end >= X.shape[1]:
                    continue
                seq = X[:, start:end + 1]
                seq_hits = {"GOOD": 0, "BAD": 0, "DISCORD": 0}
                for kind2, bank2 in bundle.items():
                    if not hasattr(bank2, "shapelets"):
                        continue
                    for sh in bank2.shapelets:
                        s = sh.series
                        mu_a = seq.mean(axis=1, keepdims=True); sd_a = seq.std(axis=1, keepdims=True) + 1e-12
                        mu_b = s.mean(axis=1, keepdims=True); sd_b = s.std(axis=1, keepdims=True) + 1e-12
                        A = (seq - mu_a) / sd_a; B = (s - mu_b) / sd_b
                        d = float(np.sqrt(((A - B) ** 2).sum()))
                        if d <= sh.epsilon:
                            seq_hits[kind2] += 1
                        if kind2 == "GOOD":
                            min_dg = min(min_dg, d)
                        elif kind2 == "BAD":
                            min_db = min(min_db, d)
                if seq_hits["GOOD"] > 0:
                    lengths_with_good.add(banks_key)
                good_hits += seq_hits["GOOD"]
                bad_hits += seq_hits["BAD"]
                disc_hits += seq_hits["DISCORD"]
        if not np.isfinite(min_dg):
            min_dg = 1e9
        if not np.isfinite(min_db):
            min_db = 1e9
        kofn_flag = 1.0 if (len(lengths_with_good) >= kofn_required) else 0.0
        return good_hits, bad_hits, disc_hits, float(min_dg), float(min_db), kofn_flag

    # Build per-row features
    rows = []
    for _, r in df.iterrows():
        i0 = int(r["bar_idx"])
        # assign regime
        if regime_model is not None:
            vreg = features.loc[i0, regime_features].to_numpy(dtype=float)
            rid = assign_regime(regime_model, vreg)
        else:
            rid = -1
        side = str(r.get("side", "long"))
        key = (rid, side)
        if key not in banks_cache:
            banks_cache[key] = load_side_banks(rid, side) if rid >= 0 else None
            if banks_cache[key] is None:
                banks_cache[key] = load_global_side_banks(side)
        bundle_reg = banks_cache[key] if banks_cache[key] is not None else {}
        good_hits, bad_hits, disc_hits, min_dg, min_db, kofn_flag = compute_hits_and_dists(i0, bundle_reg)
        ts = int(bars["timestamp"].iloc[i0])
        rows.append({
            "bar_idx": i0,
            "timestamp": ts,
            "regime_id": int(rid),
            "side": side,
            # ridge feature vector components
            "kappa_long": float(features["kappa_long"].iloc[i0]) if i0 < len(features) else 0.0,
            "accel": float(features["accel"].iloc[i0]) if i0 < len(features) else 0.0,
            "donch_pos": float(features["donch_pos"].iloc[i0]) if i0 < len(features) else 0.0,
            "slope": float(features["slope"].iloc[i0]) if i0 < len(features) else 0.0,
            "slope_r2": float(features["slope_r2"].iloc[i0]) if i0 < len(features) else 0.0,
            "bbw_p": float(features["bbw_p"].iloc[i0]) if i0 < len(features) else 0.0,
            "atr": float(features["atr"].iloc[i0]) if i0 < len(features) else 0.0,
            "body_frac": float(features["body_frac"].iloc[i0]) if i0 < len(features) else 0.0,
            "re_burst": float(features["re_burst"].iloc[i0]) if i0 < len(features) else 0.0,
            "good_hits": int(good_hits),
            "bad_hits": int(bad_hits),
            "min_dist_good": float(min_dg),
            "min_dist_bad": float(min_db),
            "k_of_n_flag": float(kofn_flag),
            # outcomes
            "realized_R": float(r.get("realized_R", 0.0)),
            "label": int(r.get("label", 0)),
        })
    feat_df = pd.DataFrame(rows)

    # Normalize timestamps into month labels
    _lo = int(min(feat_df["timestamp"])) if len(feat_df) else 0
    _hi = int(max(feat_df["timestamp"])) if len(feat_df) else 0
    if not (_lo > 10**9 and _hi > _lo):
        print("[simulate] WARN: unexpected timestamp range; continuing.")
    feat_df["month"] = pd.to_datetime(feat_df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_dir / "train_features.csv", index=False)

    # TRAIN/TEST split
    train_months = set(cfg.get("__current_train_months__", []))
    learn_src = feat_df[feat_df["month"].isin(train_months)].copy()
    test_frame = feat_df[~feat_df["month"].isin(train_months)].copy()
    learn_src = learn_src[learn_src["label"] != 0].copy()
    if len(learn_src) < 5:
        learn_src = feat_df[feat_df["label"] != 0].copy()

    FEATURE_ORDER = [
        "kappa_long", "accel", "donch_pos", "slope", "slope_r2", "bbw_p", "atr", "body_frac", "re_burst",
        "good_hits", "bad_hits", "min_dist_good", "min_dist_bad", "k_of_n_flag"
    ]

    lam = float(cfg["scoring"].get("ridge_lambda", 1.0))
    target_tpm = float(cfg["scoring"].get("target_trades_per_month", 0))
    min_prec = float(cfg["scoring"].get("min_precision_train", 0.0))
    min_iso = int(cfg["scoring"].get("min_events_for_isotonic", 500))
    models_dir = art_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    regimes = sorted([int(x) for x in feat_df["regime_id"].dropna().unique()])
    regime_models: Dict[tuple, Any] = {}
    train_stats = {}

    for rid in regimes:
        for side in ["long", "short"]:
            dfk = learn_src[(learn_src["regime_id"] == rid) & (learn_src["side"] == side)].copy()
            if len(dfk) < 5:
                continue
            Xk = dfk[FEATURE_ORDER].fillna(0).to_numpy(dtype=float)
            yk = dfk["realized_R"].clip(-2, 5).to_numpy(dtype=float)
            mdl = ridge_fit(Xk, yk, lam=lam)
            scores_k = ridge_apply(mdl, Xk)
            tau_k = pick_tau_by_target(scores_k, yk, dfk["month"].to_numpy(), target_tpm, min_prec, min_trades=10)
            iso = isotonic_fit(scores_k, yk) if len(dfk) >= min_iso else None
            # quick train diagnostics
            prec = float((yk[scores_k >= tau_k] > 0).mean()) if np.any(scores_k >= tau_k) else 0.0
            tpm = float(dfk[scores_k >= tau_k]["month"].value_counts().mean()) if np.any(scores_k >= tau_k) else 0.0
            print(f"[train] regime {rid} side {side}: n={len(dfk)} tau={tau_k:.3f} prec={prec:.2f} tpm={tpm:.1f}")
            rid_dir = models_dir / f"regime_{rid}" / side
            rid_dir.mkdir(parents=True, exist_ok=True)
            with open(rid_dir / "ridge.pkl", "wb") as f:
                pickle.dump({**mdl, "feature_order": FEATURE_ORDER}, f)
            if iso is not None:
                with open(rid_dir / "isotonic.pkl", "wb") as f:
                    pickle.dump(iso, f)
            with open(rid_dir / "tau.json", "w", encoding="utf-8") as f:
                json.dump({"tau": float(tau_k)}, f, indent=2)
            regime_models[(rid, side)] = {"ridge": mdl, "tau": float(tau_k), "iso": iso, "features": FEATURE_ORDER}
            pr = (yk > 0).mean() if len(yk) else 0.0
            train_stats[f"{rid}_{side}"] = {"n": int(len(dfk)), "win_rate": float(pr), "tau": float(tau_k)}

    with open(out_dir / "fold_summary_train.json", "w", encoding="utf-8") as f:
        json.dump({"per_regime": train_stats}, f, indent=2)

    # Simulate trades on TEST using per-regime routing and gates
    trades_R: List[float] = []
    per_bar_log = []
    for _, r in test_frame.iterrows():
        rid = int(r.get("regime_id", -1))
        side = str(r.get("side", "long"))
        mdl = regime_models.get((rid, side))
        if mdl is None:
            continue
        x = r[FEATURE_ORDER].to_numpy(dtype=float)[None, :]
        s_raw = float(ridge_apply(mdl["ridge"], x)[0])
        s_cal = float(isotonic_apply(mdl["iso"], np.array([s_raw]))[0]) if mdl.get("iso") is not None else s_raw
        good = float(r.get("good_hits", 0.0))
        bad = float(r.get("bad_hits", 0.0))
        # hard gates
        if good < min_good_hits:
            per_bar_log.append({"bar_idx": int(r["bar_idx"]), "regime_id": rid, "side": side, "pass": False, "reason": "min_good", "score": s_raw}); continue
        if good <= 0 and bad > 0:
            per_bar_log.append({"bar_idx": int(r["bar_idx"]), "regime_id": rid, "side": side, "pass": False, "reason": "bad_without_good", "score": s_raw}); continue
        if bad > bad_to_good_max * max(1.0, good):
            per_bar_log.append({"bar_idx": int(r["bar_idx"]), "regime_id": rid, "side": side, "pass": False, "reason": "bad_ratio", "score": s_raw}); continue
        if kofn_enabled and float(r.get("k_of_n_flag", 0.0)) < 1.0:
            per_bar_log.append({"bar_idx": int(r["bar_idx"]), "regime_id": rid, "side": side, "pass": False, "reason": "kofn", "score": s_raw}); continue
        tau = float(mdl["tau"])
        if s_raw >= tau:
            trades_R.append(float(r.get("realized_R", 0.0)))
            per_bar_log.append({"bar_idx": int(r["bar_idx"]), "regime_id": rid, "side": side, "pass": True, "score": s_raw, "calibrated": s_cal, "tau": tau})
        else:
            per_bar_log.append({"bar_idx": int(r["bar_idx"]), "regime_id": rid, "side": side, "pass": False, "score": s_raw, "calibrated": s_cal, "tau": tau})

    sum_R = float(np.sum(trades_R)) if trades_R else 0.0
    avg_R = float(np.mean(trades_R)) if trades_R else 0.0
    win_rate = float(np.mean([1 if x > 0 else 0 for x in trades_R])) if trades_R else 0.0
    median_R = float(np.median(trades_R)) if trades_R else 0.0
    max_dd = float(drawdown_R(trades_R)) if trades_R else 0.0

    stats = {"symbol": symbol, "trades": len(trades_R), "win_rate": win_rate, "sum_R": sum_R, "avg_R": avg_R, "median_R": median_R, "max_dd_R": max_dd}
    (out_dir / "trades.csv").write_text("R\n" + "\n".join([str(x) for x in trades_R]), encoding="utf-8")
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(out_dir / "per_bar_log.json", "w", encoding="utf-8") as f:
        json.dump(per_bar_log, f, indent=2)
    # per-regime metrics
    reg_test = {}
    # Determine if a bar became a trade
    decided = { (d.get("bar_idx"), d.get("regime_id"), d.get("side")) : d for d in per_bar_log }
    for rid in sorted(test_frame["regime_id"].dropna().unique()):
        rid = int(rid)
        for side in ["long","short"]:
            dfk = test_frame[(test_frame["regime_id"] == rid) & (test_frame["side"] == side)]
            Rs = []
            for _, rr in dfk.iterrows():
                key = (int(rr["bar_idx"]), rid, side)
                d = decided.get(key)
                if d and d.get("pass"):
                    Rs.append(float(rr.get("realized_R", 0.0)))
            reg_test[f"{rid}_{side}"] = {
                "trades": int(len(Rs)),
                "win_rate": float(np.mean([1 if x > 0 else 0 for x in Rs])) if Rs else 0.0,
                "avg_R": float(np.mean(Rs)) if Rs else 0.0,
                "sum_R": float(np.sum(Rs)) if Rs else 0.0,
                "max_dd_R": float(drawdown_R(Rs)) if Rs else 0.0,
            }
    with open(out_dir / "fold_summary.json", "w", encoding="utf-8") as f:
        json.dump({"per_regime_side_test": reg_test}, f, indent=2)

    return stats
