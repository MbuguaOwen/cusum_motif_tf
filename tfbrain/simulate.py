import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .mining import mv_distance

def _seq(features, channels, start, end):
    arrs = [features[ch].values for ch in channels]
    X = np.vstack(arrs)
    if start < 0 or end >= X.shape[1]: return None
    seg = X[:, start:end+1]
    if np.any(~np.isfinite(seg)): return None
    return seg

def simulate_with_gates(symbol: str, features: pd.DataFrame, candidates: pd.DataFrame, events: pd.DataFrame, banks: Dict[str,Any], regimes: pd.DataFrame, gating: Dict[str,Any], channels: list, out_dir: Path, fp_z, regime_ctx: Dict[str,Any]) -> Dict[str,Any]:
    epsm = gating["eps_mult"]; K = gating["K"]; dm = gating["bad_margin"]; bb = gating["breakout_buffer_atr"]; tau = gating["tau"]
    w = np.array(gating["w"], dtype=float)
    # Context similarity config
    from .utils import cosine_sim, rbf_sim
    sim_type = regime_ctx.get("sim_type", "cosine")
    gamma = float(regime_ctx.get("gamma", 1.0))
    centers_z = np.array(regime_ctx["centers_z"], dtype=float)
    ev = events.merge(candidates[["bar_idx","side","donch_high","donch_low","atr","close"]], on="bar_idx", how="left")
    ev["regime_id"] = regimes["regime_id"].reindex(ev["bar_idx"].values).values
    trades = []
    for _, r in ev.iterrows():
        i0 = int(r["bar_idx"]); side = int(r["side"]); reg_id = 0 if np.isnan(r["regime_id"]) else int(r["regime_id"])
        if side==+1:
            if not (float(r["close"]) >= float(r["donch_high"]) + bb*float(r["atr"])): 
                continue
        else:
            if not (float(r["close"]) <= float(r["donch_low"]) - bb*float(r["atr"])):
                continue
        good_min = np.inf; bad_min = np.inf; good_hits=0; bad_hits=0
        for reg_key, lengths in banks.items():
            if reg_key != f"reg{int(reg_id)}": continue
            for _, bundle in lengths.items():
                for kind, bank in bundle.items():
                    for sh in bank.shapelets:
                        pre = int(sh.meta["pre"]); post = int(sh.meta["post"])
                        seq = _seq(features, channels, i0 - pre, i0 + post)
                        if seq is None: continue
                        d = mv_distance(seq, sh.series)
                        if kind=="GOOD":
                            good_min = min(good_min, d)
                            if d <= sh.epsilon*epsm: good_hits += 1
                        else:
                            bad_min = min(bad_min, d)
                            if d <= sh.epsilon*epsm: bad_hits += 1
        if not (good_hits >= K and bad_hits == 0 and (bad_min - good_min) >= dm):
            continue
        # Score with the full ridge vector used in training
        llr = (bad_min - good_min)

        # Pull per-bar feature values (guard against index bounds)
        acc = float(features["accel"].iloc[i0]) if 0 <= i0 < len(features) else 0.0
        kap = float(features["kappa_long"].iloc[i0]) if 0 <= i0 < len(features) else 0.0
        don = float(features["donch_pos"].iloc[i0]) if 0 <= i0 < len(features) else 0.0

        # Real context similarity, z-normalized, same as training
        if 0 <= reg_id < len(centers_z):
            zrow = np.array(fp_z[i0], dtype=float) if hasattr(fp_z, "__getitem__") else fp_z[i0]
            center = centers_z[reg_id]
            ctx = cosine_sim(zrow, center) if sim_type == "cosine" else rbf_sim(zrow, center, gamma=gamma)
        else:
            ctx = 0.0

        # Training vector shape: [LLR_sum, accel, kappa_long, donch_pos, ctx_sim, 1]
        x = np.array([llr, acc, kap, don, ctx, 1.0], dtype=float)

        # If training produced a 6-length w, use it; otherwise fall back to LLR
        S = float(np.dot(x, np.array(w))) if isinstance(w, (list, tuple, np.ndarray)) and len(w) == 6 else float(llr)
        if S < tau:
            continue

        trades.append({
            "timestamp": int(r.get("timestamp", 0)),
            "bar_idx": int(r["bar_idx"]),
            "side": int(r["side"]),
            "S": float(S),
            "tau": float(tau),
            "LLR_sum": float(llr),
            "accel": float(acc),
            "kappa_long": float(kap),
            "donch_pos": float(don),
            "ctx_sim": float(ctx),
            "realized_R": float(r["realized_R"])
        })
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write rich trades file
    import csv, json
    trades_fp = out_dir / "trades.csv"
    fieldnames = ["timestamp","bar_idx","side","S","tau","LLR_sum","accel","kappa_long","donch_pos","ctx_sim","realized_R"]
    with open(trades_fp, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        for row in trades:
            wtr.writerow(row)

    # Compute stats from realized_R
    R_list = [t["realized_R"] for t in trades]
    stats = {
        "symbol": symbol,
        "trades": len(R_list),
        "win_rate": float(np.mean([1.0 if x > 0 else 0.0 for x in R_list])) if R_list else 0.0,
        "sum_R": float(np.sum(R_list)) if R_list else 0.0,
        "avg_R": float(np.mean(R_list)) if R_list else 0.0,
        "median_R": float(np.median(R_list)) if R_list else 0.0
    }
    (out_dir/"stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
