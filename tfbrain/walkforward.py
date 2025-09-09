from pathlib import Path
from typing import Dict, Any, List, Tuple
import json as jsonlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from .config import load_config
from .data import load_bars_1m, load_ticks
from .features import compute_features
from .cusum import cusum_events
from .candidates import build_candidates
from .label_ticks import first_touch_labels
from .mining import build_banks
from .regimes import fit_fingerprints, assign_regime, save_model
from .simulate import simulate_fold
from .utils import ensure_dir, json_dump

def month_list_to_folds(months: List[str], k_train:int, k_test:int, step:int):
    folds = []
    n = len(months)
    for start in range(0, n - (k_train + k_test) + 1, step):
        train = months[start:start+k_train]
        test  = months[start+k_train:start+k_train+k_test]
        folds.append((train, test))
    return folds

def run_walkforward(cfg: Dict[str, Any], root: Path):
    inputs = root / cfg["paths"]["inputs_dir"]
    outputs = root / cfg["paths"]["outputs_dir"]
    symbol = cfg["symbols"][0]
    months = cfg["data"]["months"]
    folds = month_list_to_folds(
        months,
        cfg["walkforward"]["train_months"],
        cfg["walkforward"]["test_months"],
        cfg["walkforward"]["step_months"],
    )

    all_stats = []
    pbar_folds = tqdm(range(len(folds)), desc=f"Walk-forward folds ({symbol})", unit="fold")
    for fi in pbar_folds:
        train_m, test_m = folds[fi]
        fold_dir = outputs / symbol / f"fold_{fi}"
        ensure_dir(fold_dir)

        tqdm.write(f"\nFold {fi} | Train: {train_m} -> Test: {test_m}")

        # STEP 1: Load bars & compute features
        step = tqdm(total=7, desc=f"Fold {fi}", unit="step")
        step.set_postfix_str("load bars")
        bars = load_bars_1m(inputs, symbol, train_m + test_m)
        step.update(1)

        step.set_postfix_str("features")
        feats = compute_features(
            bars,
            macro=cfg["features"]["windows"]["macro"],
            micro=cfg["features"]["windows"]["micro"],
            slope_M=cfg["features"]["windows"]["slope_M"],
        )
        step.update(1)

        # STEP 2: Fit regime fingerprints on TRAIN bars
        step.set_postfix_str("regimes")
        reg_cfg = cfg.get("regime", {"enabled": False})
        art_dir = fold_dir / "artifacts"
        ensure_dir(art_dir)
        regime_model = None
        if bool(reg_cfg.get("enabled", False)):
            regime_features = list(reg_cfg.get("features", []))
            if all(f in feats.columns for f in regime_features):
                # mark months for feats using bars timestamps
                months_all = pd.to_datetime(bars["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m")
                train_mask = months_all.isin(set(train_m))
                Xreg = feats.loc[train_mask, regime_features].reset_index(drop=True)
                regime_model = fit_fingerprints(Xreg, regime_features, int(reg_cfg.get("K", 8)), seed=int(reg_cfg.get("seed", 42)))
                save_model(regime_model, art_dir / "regime_fingerprints.pkl")
            else:
                tqdm.write("[regime] WARN: missing regime features; skipping fit.")
        step.update(1)

        # STEP 3: CUSUM calibration
        step.set_postfix_str("CUSUM")
        rr = np.log(bars["close"]).diff().fillna(0.0).values
        absr = np.abs(rr)
        drift_candidates = cfg["cusum"]["drift_k"]
        h_qs = cfg["cusum"]["h_quantiles"]
        k_val = 0.0
        if isinstance(drift_candidates, list) and "q50_abs_r" in drift_candidates:
            k_val = float(np.quantile(absr, 0.5))
        h_val = float(np.quantile(absr, h_qs[0])) if isinstance(h_qs, list) and len(h_qs) > 0 else float(np.quantile(absr, 0.85))
        cusum_df = cusum_events(bars, drift_k=k_val, h=h_val, sigma_window=cfg["cusum"].get("sigma_window"))
        step.update(1)

        # STEP 4: Candidate detection
        step.set_postfix_str("candidates")
        from copy import deepcopy
        cands = build_candidates(bars, feats, cusum_df, cfg, out_dir=fold_dir)

                # --- supply floor: if test-month candidates are too few, relax and retry ---
        min_test_cands = int(cfg.get("candidates", {}).get("min_test_candidates", 15))
        if min_test_cands > 0 and len(cands) > 0:
            test_set = set(test_m)
            cands["month"] = pd.to_datetime(cands["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m")
            n_test = int((cands["month"].isin(test_set)).sum())
            if n_test < min_test_cands:
                print(f"[cands] too few for test ({n_test}<{min_test_cands}); relaxing and retrying...")
                from copy import deepcopy
                cfg_relaxed = deepcopy(cfg)
                cand = cfg_relaxed.setdefault("candidates", {})
                comp = cand.setdefault("compression", {})
                cand["donch_n"] = max(15, int(cand.get("donch_n", cfg["features"]["windows"]["micro"])) - 10)
                cand["breakout_atr_buffer"] = max(0.0, float(cand.get("breakout_atr_buffer", 0.05)) - 0.03)
                cand["min_spacing_bars"] = max(1, int(cand.get("min_spacing_bars", 5)) - 2)
                comp["bbw_p_max"] = min(0.90, float(comp.get("bbw_p_max", 0.70)) + 0.10)
                comp["donch_w_p_max"] = min(0.90, float(comp.get("donch_w_p_max", 0.70)) + 0.10)
                sides = cfg_relaxed.setdefault("sides", {})
                sides["kappa_sign_thr"] = max(0.0, float(sides.get("kappa_sign_thr", 0.0)) - 0.05)
                sides["regime_gate"] = "both"

                cands = build_candidates(bars, feats, cusum_df, cfg_relaxed, out_dir=fold_dir)
                n_test_after = int((pd.to_datetime(cands["timestamp"], unit="ms", utc=True)
                                    .dt.strftime('%Y-%m').isin(test_set)).sum()) if len(cands) else 0
                (fold_dir/"relax_note.json").write_text(jsonlib.dumps({
                    "n_test_before": n_test,
                    "n_test_after": n_test_after,
                    "min_test_cands": min_test_cands,
                    "relaxed": True
                }, indent=2), encoding="utf-8")
        # --- end supply floor ---

        cands.to_parquet(fold_dir / "candidates.parquet", index=False)
        step.update(1)

        # STEP 5: Tick labeling (first-touch) with sides + counts
        step.set_postfix_str("label ticks")
        ticks = load_ticks(inputs, symbol, train_m + test_m)
        events = first_touch_labels(
            cands, ticks,
            sl_mult=float(cfg["risk"]["sl_mult"]),
            tp_mult=float(cfg["risk"]["tp_mult"]),
            timeout_bars=int(cfg["risk"]["timeout_bars"]),
            bar_seconds=60,
            slippage_bps=int(cfg.get("label", {}).get("slippage_bps", 3)),
        )
        # assign regime_id and symbol
        if regime_model is not None:
            rf = regime_model["features"]
            def _assign_row(bi: int) -> int:
                if bi < 0 or bi >= len(feats):
                    return -1
                v = feats.loc[int(bi), rf].to_numpy(dtype=float)
                try:
                    return assign_regime(regime_model, v)
                except Exception:
                    return -1
            events["regime_id"] = events["bar_idx"].astype(int).map(_assign_row)
        else:
            events["regime_id"] = -1
        events["sym"] = symbol
        events.to_parquet(fold_dir / "events.parquet", index=False)
        # counts by side and overall
        def _counts(df):
            return {"+1": int((df["label"]==1).sum()), "0": int((df["label"]==0).sum()), "-1": int((df["label"]==-1).sum())}
        counts = {
            "overall": _counts(events),
            "long": _counts(events[events["side"]=="long"]) if "side" in events else {"+1":0,"0":0,"-1":0},
            "short": _counts(events[events["side"]=="short"]) if "side" in events else {"+1":0,"0":0,"-1":0},
        }
        print(f"[labels] counts overall={counts['overall']} long={counts['long']} short={counts['short']}")
        (fold_dir / "label_counts.json").write_text(jsonlib.dumps(counts, indent=2), encoding="utf-8")
        overall_counts = counts.get("overall", {})
        print(f"[walkforward] fold={fi} label counts → overall={{'+1': {overall_counts.get('+1', 0)}, '0': {overall_counts.get('0', 0)}, '-1': {overall_counts.get('-1', 0)}}}")
        if int(overall_counts.get('+1', 0)) == 0 and int(overall_counts.get('0', 0)) == 0 and int(overall_counts.get('-1', 0)) == 0:
            print(f"[walkforward] WARNING: zero labeled events in fold {fi}. Proceeding.")
        step.update(1)

        # STEP 6: Mining on train only (with optional regimes + sides)
        step.set_postfix_str("mine motifs")
        events["month"] = events["ts"].map(lambda x: pd.to_datetime(x, unit="ms", utc=True).strftime("%Y-%m"))
        train_events = events[events["month"].isin(train_m)].copy()
        channels = cfg["features"]["channels"]
        import pickle
        # Build global side-specific banks as fallback
        banks_root = art_dir / "banks"
        ensure_dir(banks_root)
        for side in ["long","short"]:
            ev_s = train_events[train_events["side"] == side]
            if len(ev_s) == 0:
                continue
            b_s = build_banks(
                feats, ev_s,
                channels,
                cfg["mining"]["lengths"],
                cfg["mining"]["topk_per_class"],
                cfg["mining"]["eps_percentile"],
                meta={"symbol": symbol, "train_months": train_m, "side": side},
            )
            gdir = banks_root / "global" / side
            ensure_dir(gdir)
            with open(gdir / "good.pkl", "wb") as f:
                pickle.dump({k: v.get("GOOD") for k, v in b_s.items()}, f)
            with open(gdir / "bad.pkl", "wb") as f:
                pickle.dump({k: v.get("BAD") for k, v in b_s.items()}, f)
            with open(gdir / "meta.json", "w", encoding="utf-8") as f:
                jsonlib.dump({"side": side, "symbol": symbol, "train_months": train_m}, f, indent=2)

        # Per-regime + side banks
        reg_cfg = cfg.get("regime", {"enabled": False})
        per_regime_banks_meta = {}
        if regime_model is not None and bool(reg_cfg.get("enabled", False)):
            K = int(reg_cfg.get("K", 8))
            min_ev = int(reg_cfg.get("min_events_per_regime", 200))
            for k in range(K):
                ev_k = train_events[train_events["regime_id"] == k]
                per_regime_banks_meta[k] = {"events": int(len(ev_k))}
                if len(ev_k) < min_ev:
                    per_regime_banks_meta[k]["thin"] = True
                    continue
                for side in ["long","short"]:
                    ev_k_s = ev_k[ev_k["side"] == side]
                    if len(ev_k_s) == 0:
                        continue
                    b_k = build_banks(
                        feats, ev_k_s,
                        channels,
                        cfg["mining"]["lengths"],
                        cfg["mining"]["topk_per_class"],
                        cfg["mining"]["eps_percentile"],
                        meta={"symbol": symbol, "train_months": train_m, "regime": k, "side": side},
                    )
                    reg_dir = banks_root / f"regime_{k}" / side
                    ensure_dir(reg_dir)
                    with open(reg_dir / "good.pkl", "wb") as f:
                        pickle.dump({sk: sv.get("GOOD") for sk, sv in b_k.items()}, f)
                    with open(reg_dir / "bad.pkl", "wb") as f:
                        pickle.dump({sk: sv.get("BAD") for sk, sv in b_k.items()}, f)
                    with open(reg_dir / "meta.json", "w", encoding="utf-8") as f:
                        jsonlib.dump({"side": side, "symbol": symbol, "train_months": train_m, "regime": k}, f, indent=2)
        if per_regime_banks_meta:
            with open(art_dir/"regime_banks_meta.json","w",encoding="utf-8") as f:
                jsonlib.dump(per_regime_banks_meta, f, indent=2)
        step.update(1)

        # STEP 7: Simulate with frozen banks
        step.set_postfix_str("simulate")
        cfg_fold = dict(cfg)
        cfg_fold["__current_train_months__"] = train_m
        stats = simulate_fold(symbol, bars, feats, cands, events, {}, cfg_fold, fold_dir)
        json_dump(stats, fold_dir / "stats.json")
        step.update(1)

        # STEP 8: Finalize fold
        step.set_postfix_str("finalize")
        all_stats.append({"fold": fi, **stats, "train_months": train_m, "test_months": test_m})
        step.update(1)
        step.close()

    json_dump(all_stats, outputs / symbol / "walkforward_stats.json")
    tqdm.write("\nWalk-forward complete.")
    return all_stats

