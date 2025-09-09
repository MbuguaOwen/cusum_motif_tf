from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from .scoring import hits_from_banks, build_train_weights, score_row
from .mining import z_norm

def simulate_fold(symbol: str, bars: pd.DataFrame, features: pd.DataFrame, candidates: pd.DataFrame, events: pd.DataFrame, banks: Dict[str, Any], cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    # Build training frame from events to learn weights (if enabled)
    # For per-candidate window, compute hits summarised in a lookback range
    lookback = cfg["scoring"]["gate"]["lookback_hits_bars"]
    # Merge events into candidates by bar_idx
    df = candidates.merge(events[["bar_idx","label","realized_R"]], on="bar_idx", how="left")
    df["label"] = df["label"].fillna(0).astype(int)
    df["realized_R"] = df["realized_R"].fillna(0.0).astype(float)
    # Precompute sequences for windows per candidate for motif matching: use channels from config
    channels = cfg["features"]["channels"]
    arrs = [features[ch].values for ch in channels if ch in features.columns]
    X = np.vstack(arrs)  # C x N
    # For each candidate, take a short window [t0 - pre, t0 + post] matching the banks' lengths iteratively and produce hits counts aggregated in lookback window.
    rows = []
    for _, r in df.iterrows():
        i0 = int(r["bar_idx"])
        # aggregate hits across all bank lengths within the last `lookback` bars
        good_hits = 0; bad_hits = 0; disc_hits = 0
        for i in range(max(0, i0 - lookback), i0+1):
            for banks_key, bundle in banks.items():
                # derive pre/post from stored meta inside first bank if present
                for kind, bank in bundle.items():
                    if len(bank.shapelets)==0: 
                        continue
                    meta = bank.shapelets[0].meta
                    pre = int(meta.get("pre", 20)); post = int(meta.get("post", 40))
                    start = i - pre; end = i + post
                    if start < 0 or end >= X.shape[1]:
                        continue
                    seq = X[:, start:end+1]
                    gh, bh, dh = 0,0,0
                    # compute vs all kinds for the current banks_key
                    seq_hits = {"GOOD":0,"BAD":0,"DISCORD":0}
                    for kind2, bank2 in bundle.items():
                        for sh in bank2.shapelets:
                            # mv distance
                            s = sh.series
                            # z-norm distance
                            mu_a = seq.mean(axis=1, keepdims=True); sd_a = seq.std(axis=1, keepdims=True) + 1e-12
                            mu_b = s.mean(axis=1, keepdims=True); sd_b = s.std(axis=1, keepdims=True) + 1e-12
                            A = (seq-mu_a)/sd_a; B = (s-mu_b)/sd_b
                            d = float(np.sqrt(((A-B)**2).sum()))
                            if d <= sh.epsilon:
                                seq_hits[kind2] += 1
                    good_hits += seq_hits["GOOD"]
                    bad_hits  += seq_hits["BAD"]
                    disc_hits += seq_hits["DISCORD"]
        row = {
            "bar_idx": i0,
            "good_hits": good_hits,
            "neg_bad_hits": -bad_hits,
            "neg_disc_hits": -disc_hits,
            "accel": float(features["accel"].iloc[i0]) if i0 < len(features) else 0.0,
            "kappa_long": float(features["kappa_long"].iloc[i0]) if i0 < len(features) else 0.0,
            "donch_pos": float(features["donch_pos"].iloc[i0]) if i0 < len(features) else 0.0,
            "realized_R": float(r["realized_R"]),
            "label": int(r["label"]),
        }
        rows.append(row)
    train_frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(out_dir/"train_features.csv", index=False)
    # learn weights
    if cfg["scoring"].get("learn_weights", True) and len(train_frame) > 5:
        w = build_train_weights(train_frame, lam=float(cfg["scoring"].get("ridge_lambda", 1.0)))
        weights = w
    else:
        # fallback fixed weights
        ww = np.array([
            cfg["scoring"]["good_weight"],
            -cfg["scoring"]["bad_weight"],
            -cfg["scoring"]["discord_weight"],
            cfg["scoring"]["accel_weight"],
            cfg["scoring"]["kappa_weight"],
            cfg["scoring"]["donchpos_weight"],
            0.0 # bias
        ], dtype=float)
        class W: pass
        weights = W(); weights.w = ww; weights.tau = 0.0
    # Simulate trades on test using weights and threshold
    trades = []
    for _, r in train_frame.iterrows():
        score = r["good_hits"] + r["neg_bad_hits"] + r["neg_disc_hits"]
        # recompute with learned weights
        x = np.array([
            r["good_hits"], r["neg_bad_hits"], r["neg_disc_hits"], r["accel"], r["kappa_long"], r["donch_pos"], 1.0
        ])
        s_final = float(x @ weights.w)
        fire = (s_final >= weights.tau) and (r["neg_bad_hits"] >= 0)  # no BAD dominance
        if fire:
            trades.append(r["realized_R"])
    # Stats
    trades_R = trades
    sum_R = float(np.sum(trades_R)) if trades_R else 0.0
    avg_R = float(np.mean(trades_R)) if trades_R else 0.0
    win_rate = float(np.mean([1 if x>0 else 0 for x in trades_R])) if trades_R else 0.0
    median_R = float(np.median(trades_R)) if trades_R else 0.0
    max_dd = float(__import__("tfbrain.utils", fromlist=[""]).utils.drawdown_R(trades_R)) if trades_R else 0.0  # small trick, but ok
    
    stats = {
        "symbol": symbol,
        "trades": len(trades_R),
        "win_rate": win_rate,
        "sum_R": sum_R,
        "avg_R": avg_R,
        "median_R": median_R,
        "max_dd_R": max_dd,
    }
    import json
    (out_dir/"trades.csv").write_text("R\n" + "\n".join([str(x) for x in trades_R]), encoding="utf-8")
    with open(out_dir/"stats.json","w",encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
