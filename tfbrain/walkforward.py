from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import json
from .config import load_config
from .data import load_bars_1m, load_ticks
from .features import compute_features, assert_channels_exist
from .cusum import cusum_events
from .regime import fingerprint, kmeans_fit, assign_kmeans
from .candidates import detect_candidates
from .label_ticks import first_touch_labels
from .mining import build_banks_per_regime
from .gating import prepare_training_table, train_gates_on_fold
from .simulate import simulate_with_gates
from .utils import ensure_dir, json_dump, zscore_fit, zscore_apply

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
    folds = month_list_to_folds(months, cfg["walkforward"]["train_months"], cfg["walkforward"]["test_months"], cfg["walkforward"]["step_months"])

    all_stats = []
    pbar_folds = tqdm(range(len(folds)), desc=f"Walk-forward folds ({symbol})", unit="fold")
    for fi in pbar_folds:
        train_m, test_m = folds[fi]
        fold_dir = outputs / symbol / f"fold_{fi}"
        ensure_dir(fold_dir)
        tqdm.write(f"\n📦 Fold {fi} | Train: {train_m} → Test: {test_m}")

        step = tqdm(total=9, desc=f"Fold {fi}", unit="step")
        step.set_postfix_str("load bars")
        bars = load_bars_1m(inputs, symbol, train_m + test_m); step.update(1)

        step.set_postfix_str("features")
        feats = compute_features(bars, macro=cfg["features"]["windows"]["macro"], micro=cfg["features"]["windows"]["micro"], slope_M=cfg["features"]["windows"]["slope_M"])
        # Validate required feature channels early (fail fast)
        assert_channels_exist(feats, cfg["features"]["channels"]) 
        step.update(1)

        step.set_postfix_str("CUSUM")
        rr = np.log(bars["close"]).diff().fillna(0.0).values
        absr = np.abs(rr)
        drift_candidates = cfg["cusum"]["drift_k"]; h_qs = cfg["cusum"]["h_quantiles"]
        k_val = 0.0
        if isinstance(drift_candidates, list) and "q50_abs_r" in drift_candidates:
            k_val = float(np.quantile(absr, 0.5))
        h_val = float(np.quantile(absr, h_qs[0])) if isinstance(h_qs, list) and len(h_qs)>0 else float(np.quantile(absr, 0.85))
        cusum_df = cusum_events(bars, drift_k=k_val, h=h_val, sigma_window=cfg["cusum"].get("sigma_window")); step.update(1)

        step.set_postfix_str("regime fit")
        fp = fingerprint(feats, cusum_df, cfg)
        bars["month"] = pd.to_datetime(bars["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m")
        train_mask = bars["month"].isin(train_m)
        # K-means safety: guard k against small training set
        fp_train = fp[train_mask].values
        k_cfg = int(cfg.get("regime", {}).get("k", 6))
        n_train = len(fp_train)
        if n_train < 1:
            raise RuntimeError("Regime clustering: no training fingerprints available; check candidate/feature generation.")
        if n_train < k_cfg:
            tqdm.write(f"[regime] Reducing k from {k_cfg} -> {n_train} (train fingerprints too few).")
            k_cfg = n_train
        centers, _labels = kmeans_fit(fp_train, k=k_cfg, iters=50, seed=42)
        reg_ids = assign_kmeans(fp.values, centers)
        regimes = pd.DataFrame({"regime_id": reg_ids}, index=feats.index)
        regimes.to_csv(fold_dir/"regimes.csv", index=False)
        step.update(1)

        # z-normalize fingerprint and compute regime context
        fp_values = fp.values.astype(float)
        # train_mask already computed above from bars
        mu, sd = zscore_fit(fp_values[train_mask.values if hasattr(train_mask, 'values') else train_mask])
        fp_z = zscore_apply(fp_values, mu, sd)
        centers_z = zscore_apply(centers, mu, sd)

        regime_ctx = {
            "columns": list(fp.columns),
            "mu": mu.tolist(),
            "sd": sd.tolist(),
            "centers_z": centers_z.tolist(),
            "sim_type": cfg.get("regime", {}).get("ctx_sim", {}).get("type", "cosine"),
            "gamma": float(cfg.get("regime", {}).get("ctx_sim", {}).get("gamma", 1.0)),
        }
        art_dir = fold_dir / "artifacts"
        ensure_dir(art_dir)
        (art_dir / "regime_ctx.json").write_text(json.dumps(regime_ctx, indent=2), encoding="utf-8")

        step.set_postfix_str("candidates")
        cands = detect_candidates(bars, feats, cusum_df, cfg)
        cands.to_parquet(fold_dir/"candidates.parquet", index=False); step.update(1)

        step.set_postfix_str("label ticks")
        # Compute the minimal tick window we need for labeling
        if len(cands) > 0:
            t0 = int(cands["timestamp"].min())
            t1 = int(cands["timestamp"].max()) + int(cfg["label"]["time_limit_bars"]) * 60 * 1000
        else:
            t0 = None; t1 = None
        ticks = load_ticks(inputs, symbol, train_m + test_m, min_ts=t0, max_ts=t1, chunksize=2_000_000)
        events = first_touch_labels(cands, ticks, sl_mult=cfg["label"]["sl_mult"], tp_mult=cfg["label"]["tp_mult"], time_limit_bars=cfg["label"]["time_limit_bars"], bar_seconds=60, slippage_bps=cfg["label"]["slippage_bps"])
        events.to_parquet(fold_dir/"events.parquet", index=False); step.update(1)

        step.set_postfix_str("mine motifs")
        events["month"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m")
        ev_train = events[events["month"].isin(train_m)].copy()
        ev_train["regime_id"] = regimes["regime_id"].reindex(ev_train["bar_idx"].values).values
        banks = build_banks_per_regime(
            feats,
            ev_train,
            cfg["features"]["channels"],
            cfg["mining"]["lengths"],
            cfg["mining"]["topk_per_class"],
            cfg["mining"]["eps_percentile"],
            ev_train["regime_id"].values,
            cfg.get("mining", {}).get("ppv_prune", cfg.get("gating_search", {}).get("ppv_prune", 0.5)),
            meta={"symbol": symbol, "train_months": train_m},
            max_events_per_fold=int(cfg.get("mining", {}).get("max_events_per_fold", 25000))
        )
        import pickle
        art_dir = fold_dir/"artifacts"; ensure_dir(art_dir)
        with open(art_dir/"banks.pkl","wb") as f: pickle.dump(banks, f)
        step.update(1)

        step.set_postfix_str("calibrate gates")
        train_table = prepare_training_table(
            features=feats,
            candidates=cands,
            events=ev_train,
            regimes=regimes,
            banks=banks,
            fp_z=fp_z,
            regime_ctx=regime_ctx,
            channels=cfg["features"]["channels"],
            cfg=cfg
        )
        train_table.to_csv(fold_dir/"train_table.csv", index=False)
        gating = train_gates_on_fold(
            features=feats,
            train_df=train_table,
            banks=banks,
            regimes=regimes,
            channels=cfg["features"]["channels"],
            cfg=cfg
        )
        json_dump(gating, art_dir/"gating.json")
        step.update(1)

        step.set_postfix_str("simulate")
        # Restrict to test-only events for honest OOS evaluation
        if "month" not in events.columns:
            # derive if needed (YYYY-MM from timestamp)
            events["month"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m")
        events_test = events[events["month"].isin(test_m)].copy()

        stats = simulate_with_gates(symbol, feats, cands, events_test, banks, regimes, gating, cfg["features"]["channels"], fold_dir, fp_z=fp_z, regime_ctx=regime_ctx)
        json_dump(stats, fold_dir/"stats.json"); step.update(1); step.close()

        # Friendly summary per fold
        trades_n = stats.get("n_trades", stats.get("trades"))
        print(f"[fold {fi}] trades={trades_n} avg_R={stats.get('avg_R')} win_rate={stats.get('win_rate')} file={fold_dir/'stats.json'}")
        if not trades_n:
            print("Tip: 0 trades in TEST. Consider loosening gating (eps_multiplier ↑, K_hits ↓, bad_margin ↓, breakout_buffer_atr ↓) or loosening candidate compression/spacing. (Simulation uses test-only events.)")

        all_stats.append({"fold": fi, **stats, "train_months": train_m, "test_months": test_m})

    json_dump(all_stats, outputs / symbol / "walkforward_stats.json")
    tqdm.write("\n✅ Walk-forward complete.")
    return all_stats
