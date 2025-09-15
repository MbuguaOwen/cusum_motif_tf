from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_triggers.py — Stage-A search over candidate "trigger" thresholds to find a high-signal, high-supply region.
Evaluates on TRAIN months only (fast). No re-mining yet.

Usage (repo root, venv on):
  python scripts/sweep_triggers.py --config configs/motifs.yaml --symbol BTCUSDT

Output:
  outputs/<SYMBOL>/sweeps/trigger_sweep_summary.csv  (ranked)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, itertools
from pathlib import Path
import yaml, numpy as np, pandas as pd
from tqdm import tqdm

# project imports
from tfbrain.data import load_bars_1m, load_ticks
from tfbrain.features import compute_features
from tfbrain.cusum import cusum_events
from tfbrain.candidates import detect_candidates
from tfbrain.label_ticks import first_touch_labels

def make_grid():
    grid = {
        # Compression (OR between bbw_p and donch_w_p)
        "bbw_p_max":        [0.90, 0.95, 0.98],
        "donch_w_p_max":    [0.95, 0.98],
        "atr_p_max":        [0.80, 0.90],
        # Trigger
        "body_frac_min":    [0.35, 0.45, 0.55],
        "accel_min":        [0.50, 0.80, 1.00],
        "donch_pos_min_long":  [0.58, 0.65, 0.70],
        "spacing_bars":     [10, 20, 30],
        "require_cusum":    [False],  # add True if you want
        # Labeling
        "tp_mult":          [1.5, 2.0, 2.5],
        "sl_mult":          [1.2, 1.5, 2.0],
        "time_limit_bars":  [180, 240],
        "slippage_bps":     [1, 3],
    }
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        p = dict(zip(keys, combo))
        p["donch_pos_max_short"] = 1.0 - p["donch_pos_min_long"]
        yield p

def score_events(ev: pd.DataFrame, n_months: int) -> dict:
    if ev.empty:
        return {"n":0,"n_nz":0,"wins":0,"losses":0,"wr":0.0,"avg_R":0.0,"med_R":0.0,"R_sum":0.0,"tpm":0.0,"score":-1e9}
    nz = ev[ev["label"] != 0]
    n, n_nz = len(ev), len(nz)
    wins = int((nz["label"]==+1).sum()); losses = int((nz["label"]==-1).sum())
    wr = wins/(wins+losses) if (wins+losses)>0 else 0.0
    avg_R = float(nz["realized_R"].mean()) if n_nz else 0.0
    med_R = float(nz["realized_R"].median()) if n_nz else 0.0
    R_sum = float(nz["realized_R"].sum())
    tpm = n_nz / max(1, n_months)
    balance = 1.0 - abs(wr - 0.5)     # 1 at 50/50, 0 near 0/100
    score = avg_R*1.0 + 0.15*wr + 0.05*tpm + 0.1*balance
    return {"n":n,"n_nz":n_nz,"wins":wins,"losses":losses,"wr":wr,"avg_R":avg_R,"med_R":med_R,"R_sum":R_sum,"tpm":tpm,"score":score}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_combos", type=int, default=120)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    inputs = Path(cfg["paths"]["inputs_dir"])
    outputs = Path(cfg["paths"]["outputs_dir"])
    outdir = Path(args.outdir) if args.outdir else (outputs / args.symbol / "sweeps")
    outdir.mkdir(parents=True, exist_ok=True)

    months = cfg["data"]["months"]
    k_train = int(cfg["walkforward"]["train_months"])
    train_m = months[:k_train]

    # Load TRAIN once
    bars = load_bars_1m(inputs, args.symbol, train_m)
    feats = compute_features(bars,
                             macro=int(cfg["features"]["windows"]["macro"]),
                             micro=int(cfg["features"]["windows"]["micro"]),
                             slope_M=int(cfg["features"]["windows"]["slope_M"]))
    rr = np.log(bars["close"]).diff().fillna(0.0).values
    absr = np.abs(rr)
    k_val = float(np.quantile(absr, 0.5)) if "q50_abs_r" in cfg["cusum"].get("drift_k", []) else 0.0
    h_qs = cfg["cusum"].get("h_quantiles", [0.85]); h_val = float(np.quantile(absr, h_qs[0] if isinstance(h_qs, list) else 0.85))
    cus = cusum_events(bars, drift_k=k_val, h=h_val, sigma_window=cfg["cusum"].get("sigma_window"))

    # Ticks for TRAIN range
    min_ts, max_ts = int(bars["timestamp"].min()), int(bars["timestamp"].max())
    ticks = load_ticks(inputs, args.symbol, train_m, min_ts=min_ts, max_ts=max_ts)

    rows=[]; tested=0
    for params in tqdm(make_grid(), desc="Sweep triggers"):
        tested += 1
        if tested > args.max_combos: break

        cfg_local = {
            "features": cfg["features"],
            "candidates": {
                "compression": {
                    "bbw_p_max": params["bbw_p_max"],
                    "donch_w_p_max": params["donch_w_p_max"],
                    "atr_p_max": params["atr_p_max"],
                },
                "trigger": {
                    "body_frac_min": params["body_frac_min"],
                    "accel_min": params["accel_min"],
                    "donch_pos_min_long": params["donch_pos_min_long"],
                    "donch_pos_max_short": params["donch_pos_max_short"],
                },
                "spacing_bars": int(params["spacing_bars"]),
                "require_cusum": bool(params["require_cusum"]),
            },
        }

        cands = detect_candidates(bars, feats, cus if cfg_local["candidates"]["require_cusum"] else pd.DataFrame(), cfg_local)
        if cands.empty:
            rows.append({**params, "n_cands":0, "n":0, "n_nz":0, "wins":0, "losses":0, "wr":0.0, "avg_R":0.0, "med_R":0.0, "R_sum":0.0, "tpm":0.0, "score":-1e9})
            continue

        ev = first_touch_labels(
            candidates=cands,
            ticks=ticks,
            tp_mult=float(params["tp_mult"]),
            sl_mult=float(params["sl_mult"]),
            time_limit_bars=int(params["time_limit_bars"]),
            bar_seconds=60,
            slippage_bps=int(params["slippage_bps"]),
        )
        m = score_events(ev, n_months=k_train)
        rows.append({**params, "n_cands": len(cands), **m})

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out_csv = outdir / "trigger_sweep_summary.csv"
    df.to_csv(out_csv, index=False)
    print(df.head(10).to_string(index=False))
    print("\nSaved:", out_csv)

if __name__ == "__main__":
    main()
