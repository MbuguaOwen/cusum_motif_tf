from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from .config import load_config
from .data import load_bars_1m, load_ticks
from .features import compute_features
from .cusum import cusum_events
from .candidates import detect_candidates
from .label_ticks import first_touch_labels
from .mining import build_banks
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
    folds = month_list_to_folds(months, cfg["walkforward"]["train_months"], cfg["walkforward"]["test_months"], cfg["walkforward"]["step_months"])
    all_stats = []
    for fi, (train_m, test_m) in enumerate(folds):
        fold_dir = outputs / symbol / f"fold_{fi}"
        ensure_dir(fold_dir)
        # Load bars for train+test union (features need continuity)
        bars = load_bars_1m(inputs, symbol, train_m + test_m)
        feats = compute_features(bars, macro=cfg["features"]["windows"]["macro"], micro=cfg["features"]["windows"]["micro"], slope_M=cfg["features"]["windows"]["slope_M"])
        # CUSUM calibration: choose drift/h by simple heuristic on train subset
        train_mask = feats["timestamp"].between(bars["timestamp"].iloc[0], bars["timestamp"].iloc[len(load_bars_1m(inputs, symbol, train_m)) -1] if len(train_m)>0 else feats["timestamp"].iloc[-1])
        # We'll pick first valid combo
        drift_candidates = cfg["cusum"]["drift_k"]
        h_qs = cfg["cusum"]["h_quantiles"]
        # compute rr std percentile on train
        import numpy as np
        rr = np.log(bars["close"]).diff().fillna(0).values
        absr = np.abs(rr)
        k_val = 0.0 if (isinstance(drift_candidates, list) and 0.0 in drift_candidates) else 0.0
        if isinstance(drift_candidates, list) and "q50_abs_r" in drift_candidates:
            k_val = float(np.quantile(absr, 0.5))
        h_val = float(np.quantile(absr, h_qs[0])) if isinstance(h_qs, list) and len(h_qs)>0 else float(np.quantile(absr, 0.85))
        cusum_df = cusum_events(bars, drift_k=k_val, h=h_val, sigma_window=cfg["cusum"].get("sigma_window"))
        # candidates (train+test)
        cands = detect_candidates(bars, feats, cusum_df, cfg)
        cands.to_parquet(fold_dir/"candidates.parquet", index=False)
        # labels on ticks (train+test) — but we will restrict mining to train only
        ticks = load_ticks(inputs, symbol, train_m + test_m)
        events = first_touch_labels(cands, ticks, sl_mult=cfg["label"]["sl_mult"], tp_mult=cfg["label"]["tp_mult"], time_limit_bars=cfg["label"]["time_limit_bars"], bar_seconds=60, slippage_bps=cfg["label"]["slippage_bps"])
        events.to_parquet(fold_dir/"events.parquet", index=False)
        # mining on train: filter events within train months
        # Map bar_idx to timestamp and month
        cands_idx_ts = cands[["bar_idx","timestamp"]].set_index("bar_idx")["timestamp"].to_dict()
        events["month"] = events["timestamp"].map(lambda x: pd.to_datetime(x, unit="ms", utc=True).strftime("%Y-%m"))
        train_events = events[events["month"].isin(train_m)].copy()
        # Build banks per lengths
        channels = cfg["features"]["channels"]
        banks = build_banks(feats, train_events, channels, cfg["mining"]["lengths"], cfg["mining"]["topk_per_class"], cfg["mining"]["eps_percentile"], meta={"symbol":symbol, "train_months":train_m})
        # Persist banks
        import pickle, os
        art_dir = fold_dir/"artifacts"; ensure_dir(art_dir)
        with open(art_dir/"banks.pkl","wb") as f:
            pickle.dump(banks, f)
        # Simulate on test using frozen banks
        stats = simulate_fold(symbol, bars, feats, cands, events, banks, cfg, fold_dir)
        json_dump(stats, fold_dir/"stats.json")
        all_stats.append({"fold": fi, **stats, "train_months": train_m, "test_months": test_m})
    # Save summary
    json_dump(all_stats, outputs / symbol / "walkforward_stats.json")
    return all_stats
