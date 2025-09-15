
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_triggers_fast.py — Big-move trigger discovery (fast + robust) with RR & volatility sweeps.

Stage A (FAST): bar-based first-touch over (family × params × RR pairs × vol filter),
                quality-first rank with dynamic viability gates (dependent on RR).
Stage B (REFINE): relabel top-K with tick-accurate first-touch and re-rank.
Optionally emit a YAML overlay for the winner.

Defaults:
  - RR pairs: 20x60 (3:1), 15x45 (3:1 but smaller), 12x36 (3:1), change via --pairs
  - time_limit: 7 days (change via --time-limit-days)
  - volatility prefilter: --atrp-min (e.g., 0.60 keeps top 40% ATR)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd, yaml
from tqdm import tqdm

# project imports
from tfbrain.data import load_bars_1m, load_ticks
from tfbrain.features import compute_features
from tfbrain.label_ticks import first_touch_labels

BAR_SEC = 60
SLIPPAGE_BPS = 1

# ---------- globals set from CLI ----------
TIME_LIMIT = 7 * 1440
SPACING = 30

# ---------- helpers ----------
def spacing_filter(idxs: np.ndarray, spacing: int = SPACING) -> np.ndarray:
    if len(idxs) == 0: return idxs
    keep=[]; last=-10**9
    for i in np.asarray(idxs, dtype=int):
        if i - last >= spacing:
            keep.append(i); last=i
    return np.array(keep, dtype=int)

def mk_candidates(bars: pd.DataFrame, feats: pd.DataFrame, idxs: np.ndarray, side: int) -> pd.DataFrame:
    if len(idxs)==0:
        return pd.DataFrame(columns=["bar_idx","timestamp","side","close","atr"])
    out = pd.DataFrame({
        "bar_idx": idxs,
        "timestamp": bars["timestamp"].iloc[idxs].astype(int).values,
        "side": np.full(len(idxs), side, dtype=int),
        "close": bars["close"].iloc[idxs].astype(float).values,
        "atr": feats["atr"].iloc[idxs].astype(float).values,
    })
    return out[out["atr"]>0].reset_index(drop=True)

def bar_first_touch(bars: pd.DataFrame, cands: pd.DataFrame, tp_atr: float, sl_atr: float, time_limit_bars: int) -> pd.DataFrame:
    """Approximate first-touch with OHLC; early-out on touch."""
    if cands.empty:
        return pd.DataFrame(columns=["label","realized_R"])
    hi = bars["high"].values; lo = bars["low"].values
    out = np.zeros((len(cands), 2), dtype=float)  # label, realized_R
    for idx, r in enumerate(cands.itertuples(index=False)):
        i = int(r.bar_idx); s = int(r.side); px = float(r.close); atr = float(r.atr)
        if not np.isfinite(atr) or atr <= 0:
            continue
        tp = px + s*tp_atr*atr
        sl = px - s*sl_atr*atr
        hit = 0; rr = 0.0
        j_end = min(i + time_limit_bars, len(bars)-1)
        for j in range(i+1, j_end+1):
            if s > 0:
                if lo[j] <= sl: hit=-1; rr=-sl_atr; break
                if hi[j] >= tp: hit=+1; rr=+tp_atr; break
            else:
                if hi[j] >= tp: hit=-1; rr=-sl_atr; break
                if lo[j] <= sl: hit=+1; rr=+tp_atr; break
        out[idx,0] = hit
        out[idx,1] = rr
    return pd.DataFrame(out, columns=["label","realized_R"])

def rr_viability(tp_atr: float, sl_atr: float, wr: float, avg_R: float, n_nz: int) -> bool:
    """
    Dynamic viability gates:
      - need some sample,
      - avg_R > 0,
      - WR above (break-even + margin)
    Break-even p* = sl / (sl + tp). We require wr >= p* + 0.02.
    """
    if n_nz < 200: return False
    if avg_R <= 0.0: return False
    p_star = sl_atr / (sl_atr + tp_atr)
    return wr >= (p_star + 0.02)

def score(ev: pd.DataFrame, n_months: int, tp_atr: float, sl_atr: float) -> dict:
    if ev.empty:
        return {"n":0,"n_nz":0,"wins":0,"losses":0,"wr":0.0,"avg_R":0.0,"med_R":0.0,"R_sum":0.0,"tpm":0.0,"balance":0.0,"score":-1e9}
    nz = ev[ev["label"]!=0]
    n, n_nz = len(ev), len(nz)
    wins = int((nz["label"]==+1).sum()); losses = int((nz["label"]==-1).sum())
    wr = wins/max(1,wins+losses)
    avg_R = float(nz["realized_R"].mean()) if n_nz else 0.0
    med_R = float(nz["realized_R"].median()) if n_nz else 0.0
    R_sum = float(nz["realized_R"].sum())
    tpm = n_nz / max(1, n_months)
    balance = 1.0 - abs(wr - 0.5)
    if not rr_viability(tp_atr, sl_atr, wr, avg_R, n_nz):
        sc = -1e9
    else:
        sc = 0.7*avg_R + 0.3*med_R + 0.25*math.log1p(tpm) - 0.2*abs(wr - (sl_atr/(sl_atr+tp_atr)))
    return {"n":n,"n_nz":n_nz,"wins":wins,"losses":losses,"wr":wr,"avg_R":avg_R,"med_R":med_R,"R_sum":R_sum,"tpm":tpm,"balance":balance,"score":sc}

# ---------- triggers ----------
def candidates_for_trigger(bars, feats, family: str, p: dict, spacing: int, atrp_min: float) -> pd.DataFrame:
    accel     = feats["accel"].values
    donch_pos = feats["donch_pos"].values
    body_frac = feats["body_frac"].values
    atr       = feats["atr"].values
    slope     = feats.get("slope", pd.Series(np.nan, index=feats.index)).values
    slope_r2  = feats.get("slope_r2", pd.Series(np.nan, index=feats.index)).values
    bbw_p     = feats.get("bbw_p", pd.Series(np.nan, index=feats.index)).values
    kappaL    = feats.get("kappa_long", pd.Series(np.nan, index=feats.index)).values
    trend_age = feats.get("trend_age", pd.Series(0, index=feats.index)).values
    # volatility percentile (derive if absent)
    if "atr_p" in feats.columns:
        atr_p = feats["atr_p"].values
    else:
        atr_p = pd.Series(atr).rank(pct=True).values
    vol_ok = (atr_p >= atrp_min)
    rng = (bars["high"].values - bars["low"].values) / np.maximum(1e-12, atr)

    long_idx = np.array([], dtype=int); short_idx = np.array([], dtype=int)

    if family == "accel_loc":
        a = p.get("accel_min", 1.2); b = p.get("body_min", 0.60); posL = p.get("pos_min_long", 0.70); posS = 1.0-posL
        L = np.where(vol_ok & (accel>=+a) & (donch_pos>=posL) & (body_frac>=b))[0]
        S = np.where(vol_ok & (accel<=-a) & (donch_pos<=posS) & (body_frac>=b))[0]
        long_idx, short_idx = spacing_filter(L, spacing), spacing_filter(S, spacing)

    elif family == "squeeze_rel":
        sL = int(p.get("squeeze_lookback", 40)); sT = float(p.get("squeeze_p_max", 0.12))
        b  = p.get("body_min", 0.60); a = p.get("accel_min", 1.0)
        min_bbw = pd.Series(bbw_p).rolling(sL, min_periods=sL).min().shift(1).fillna(np.inf).values
        L = np.where(vol_ok & (min_bbw<=sT) & (accel>=+a) & (body_frac>=b))[0]
        S = np.where(vol_ok & (min_bbw<=sT) & (accel<=-a) & (body_frac>=b))[0]
        long_idx, short_idx = spacing_filter(L, spacing), spacing_filter(S, spacing)

    elif family == "slope_body":
        s_min = float(p.get("slope_min", 0.0)); r2_min = float(p.get("slope_r2_min", 0.60))
        b = p.get("body_min", 0.60); posL = p.get("pos_min_long", 0.70); posS = 1.0-posL
        L = np.where(vol_ok & (slope>=+s_min)&(slope_r2>=r2_min)&(donch_pos>=posL)&(body_frac>=b))[0]
        S = np.where(vol_ok & (slope<=-s_min)&(slope_r2>=r2_min)&(donch_pos<=posS)&(body_frac>=b))[0]
        long_idx, short_idx = spacing_filter(L, spacing), spacing_filter(S, spacing)

    elif family == "dwell_trend":
        dwell = int(p.get("dwell_min", 8)); b = p.get("body_min", 0.60)
        L = np.where(vol_ok & (trend_age>=dwell)&(kappaL>0)&(body_frac>=b))[0]
        S = np.where(vol_ok & (trend_age>=dwell)&(kappaL<0)&(body_frac>=b))[0]
        long_idx, short_idx = spacing_filter(L, spacing), spacing_filter(S, spacing)

    elif family == "range_thrust":
        r_thr = float(p.get("range_atr_min", 2.5)); b = p.get("body_min", 0.60)
        L = np.where(vol_ok & (rng>=r_thr)&(kappaL>0)&(body_frac>=b))[0]
        S = np.where(vol_ok & (rng>=r_thr)&(kappaL<0)&(body_frac>=b))[0]
        long_idx, short_idx = spacing_filter(L, spacing), spacing_filter(S, spacing)

    return pd.concat([
        mk_candidates(bars, feats, long_idx, +1),
        mk_candidates(bars, feats, short_idx, -1)
    ], ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--max-per-trigger", type=int, default=4)
    ap.add_argument("--topk-refine", type=int, default=25)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--emit-config", action="store_true", help="Write configs/motifs_trigger_opt.yaml for best refined row.")
    ap.add_argument("--families", nargs="*", default=["accel_loc","squeeze_rel","slope_body","dwell_trend","range_thrust"])

    # RR & time-limit sweep
    ap.add_argument("--pairs", type=str, default="20x60,15x45,12x36",
                    help='Comma-separated RR pairs like "20x60,15x45,12x36" (SLxTP in ATR).')
    ap.add_argument("--time-limit-days", type=float, default=7.0)
    ap.add_argument("--spacing", type=int, default=30)
    ap.add_argument("--atrp-min", type=float, default=0.60, help="Keep bars with ATR percentile >= this (0..1).")

    args = ap.parse_args()

    global TIME_LIMIT, SPACING
    TIME_LIMIT = int(args.time_limit_days * 1440)
    SPACING = int(args.spacing)

    # parse RR pairs
    pairs=[]
    for tok in args.pairs.split(","):
        tok = tok.strip()
        if "x" not in tok: continue
        sl, tp = tok.split("x")
        pairs.append((float(sl), float(tp)))
    assert len(pairs) > 0, "No valid --pairs given."

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    inputs = Path(cfg["paths"]["inputs_dir"])
    outputs = Path(cfg["paths"]["outputs_dir"])
    sym = args.symbol
    outdir = outputs / sym / "sweeps"; outdir.mkdir(parents=True, exist_ok=True)

    months = cfg["data"]["months"]; k_train = int(cfg["walkforward"]["train_months"])
    train_m = months[:k_train]
    print(f"TRAIN months: {train_m[0]}..{train_m[-1]} ({k_train} months)")
    print(f"pairs={pairs}  time_limit={TIME_LIMIT} bars (~{TIME_LIMIT/1440:.2f} days)  spacing={SPACING}  atr_p_min={args.atrp_min}")

    # Load once
    bars = load_bars_1m(inputs, sym, train_m)
    feats = compute_features(bars,
                             macro=int(cfg["features"]["windows"]["macro"]),
                             micro=int(cfg["features"]["windows"]["micro"]),
                             slope_M=int(cfg["features"]["windows"]["slope_M"]))

    # Presets (tighter, big-move bias)
    presets = {
        "accel_loc": [
            {"accel_min":1.2, "body_min":0.60, "pos_min_long":0.70},
            {"accel_min":1.5, "body_min":0.60, "pos_min_long":0.75},
            {"accel_min":1.8, "body_min":0.65, "pos_min_long":0.75},
            {"accel_min":2.0, "body_min":0.65, "pos_min_long":0.80},
        ],
        "squeeze_rel": [
            {"squeeze_lookback":40, "squeeze_p_max":0.12, "accel_min":1.0, "body_min":0.60},
            {"squeeze_lookback":40, "squeeze_p_max":0.10, "accel_min":1.2, "body_min":0.60},
            {"squeeze_lookback":30, "squeeze_p_max":0.10, "accel_min":1.5, "body_min":0.65},
            {"squeeze_lookback":30, "squeeze_p_max":0.08, "accel_min":1.5, "body_min":0.65},
        ],
        "slope_body": [
            {"slope_min":0.0, "slope_r2_min":0.60, "body_min":0.60, "pos_min_long":0.70},
            {"slope_min":0.0, "slope_r2_min":0.70, "body_min":0.65, "pos_min_long":0.75},
            {"slope_min":0.0, "slope_r2_min":0.75, "body_min":0.65, "pos_min_long":0.80},
            {"slope_min":0.0, "slope_r2_min":0.80, "body_min":0.70, "pos_min_long":0.80},
        ],
        "dwell_trend": [
            {"dwell_min":8, "body_min":0.60},
            {"dwell_min":10, "body_min":0.60},
            {"dwell_min":12, "body_min":0.65},
            {"dwell_min":14, "body_min":0.65},
        ],
        "range_thrust": [
            {"range_atr_min":2.5, "body_min":0.60},
            {"range_atr_min":3.0, "body_min":0.60},
            {"range_atr_min":3.5, "body_min":0.65},
            {"range_atr_min":4.0, "body_min":0.65},
        ],
    }

    fams = [f for f in args.families if f in presets]

    # --------------- Stage A ---------------
    stageA_csv = outdir / "triggers_fast_stageA.csv"
    rowsA = []
    done = set()
    if args.resume and stageA_csv.exists():
        df_old = pd.read_csv(stageA_csv)
        rowsA.extend(df_old.to_dict("records"))
        for r in rowsA:
            k = json.dumps({"family": r["family"], "sl_atr": r.get("sl_atr"), "tp_atr": r.get("tp_atr"),
                            **{x:r.get(x) for x in r.keys() if x in ["accel_min","body_min","pos_min_long","squeeze_lookback","squeeze_p_max","slope_min","slope_r2_min","dwell_min","range_atr_min","atrp_min"]}},
                           sort_keys=True)
            done.add(k)

    combos = []
    for fam in fams:
        for p in presets[fam][:args.max_per_trigger]:
            for (sl_atr, tp_atr) in pairs:
                combos.append((fam, p, sl_atr, tp_atr))

    for i,(fam,p,sl_atr,tp_atr) in enumerate(tqdm(combos, desc="Stage A (bars)")):
        key = json.dumps({"family": fam, **p, "sl_atr": sl_atr, "tp_atr": tp_atr, "atrp_min": args.atrp_min}, sort_keys=True)
        if key in done: continue
        cands = candidates_for_trigger(bars, feats, fam, p, spacing=SPACING, atrp_min=args.atrp_min)
        if cands.empty:
            stat = {"n_cands":0,"n":0,"n_nz":0,"wins":0,"losses":0,"wr":0.0,"avg_R":0.0,"med_R":0.0,"R_sum":0.0,"tpm":0.0,"balance":0.0,"score":-1e9}
        else:
            ev = bar_first_touch(bars, cands, tp_atr=tp_atr, sl_atr=sl_atr, time_limit_bars=TIME_LIMIT)
            stat = score(ev, n_months=len(train_m), tp_atr=tp_atr, sl_atr=sl_atr); stat["n_cands"] = len(cands)
        rowsA.append({"family": fam, **p, "sl_atr": sl_atr, "tp_atr": tp_atr, "atrp_min": args.atrp_min, **stat})
        if (i+1) % max(1, args.every) == 0:
            pd.DataFrame(rowsA).sort_values(["score","avg_R","tpm"], ascending=[False,False,False]).to_csv(stageA_csv, index=False)

    dfA = pd.DataFrame(rowsA).sort_values(["score","avg_R","tpm"], ascending=[False,False,False])
    dfA.to_csv(stageA_csv, index=False)
    print("\nStage A saved:", stageA_csv)
    print(dfA.head(12).to_string(index=False))

    # --------------- Stage B ---------------
    topk = int(args.topk_refine)
    if topk <= 0 or dfA.empty:
        print("Skip Stage B (no rows or topk<=0)."); return

    # Load ticks once for the train window
    min_ts, max_ts = int(bars["timestamp"].min()), int(bars["timestamp"].max())
    ticks = load_ticks(inputs, sym, train_m, min_ts=min_ts, max_ts=max_ts)

    rowsB = []
    for _, r in tqdm(dfA.head(topk).iterrows(), total=min(topk, len(dfA)), desc="Stage B (ticks)"):
        fam = r["family"]
        sl_atr = float(r["sl_atr"]); tp_atr = float(r["tp_atr"])
        p = {k: r[k] for k in r.index if k in ["accel_min","body_min","pos_min_long","squeeze_lookback","squeeze_p_max","slope_min","slope_r2_min","dwell_min","range_atr_min"] and pd.notna(r[k])}
        cands = candidates_for_trigger(bars, feats, fam, p, spacing=SPACING, atrp_min=float(r.get("atrp_min", 0.0)))
        if cands.empty:
            stat = {"n_cands":0,"n":0,"n_nz":0,"wins":0,"losses":0,"wr":0.0,"avg_R":0.0,"med_R":0.0,"R_sum":0.0,"tpm":0.0,"balance":0.0,"score":-1e9}
        else:
            ev = first_touch_labels(
                candidates=cands, ticks=ticks,
                tp_mult=tp_atr, sl_mult=sl_atr,
                time_limit_bars=TIME_LIMIT, bar_seconds=BAR_SEC,
                slippage_bps=SLIPPAGE_BPS
            )
            stat = score(ev, n_months=len(train_m), tp_atr=tp_atr, sl_atr=sl_atr); stat["n_cands"] = len(cands)
        rowsB.append({"family": fam, **p, "sl_atr": sl_atr, "tp_atr": tp_atr, **stat})

    dfB = pd.DataFrame(rowsB).sort_values(["score","avg_R","tpm"], ascending=[False,False,False]).reset_index(drop=True)
    stageB_csv = outdir / "triggers_fast_stageB.csv"
    dfB.to_csv(stageB_csv, index=False)
    print("\nStage B saved:", stageB_csv)
    print(dfB.head(12).to_string(index=False))

    # Emit YAML for the top refined row
    if args.emit_config and not dfB.empty:
        best = dfB.iloc[0].to_dict()
        y = {
          "candidates": {"trigger": {}, "spacing_bars": SPACING, "require_cusum": False},
          "labeling": {
            "tp_mult": best["tp_atr"], "sl_mult": best["sl_atr"],
            "time_limit_bars": TIME_LIMIT, "slippage_bps": SLIPPAGE_BPS
          }
        }
        fam = best["family"]
        if fam == "accel_loc":
            y["candidates"]["trigger"].update({
              "accel_min": float(best.get("accel_min", 1.2)),
              "body_frac_min": float(best.get("body_min", 0.60)),
              "donch_pos_min_long": float(best.get("pos_min_long", 0.70)),
              "donch_pos_max_short": float(1.0 - best.get("pos_min_long", 0.70)),
            })
        elif fam == "squeeze_rel":
            y["candidates"]["trigger"].update({
              "squeeze_lookback": int(best.get("squeeze_lookback", 40)),
              "squeeze_p_max": float(best.get("squeeze_p_max", 0.12)),
              "accel_min": float(best.get("accel_min", 1.0)),
              "body_frac_min": float(best.get("body_min", 0.60)),
              "donch_pos_min_long": 0.70, "donch_pos_max_short": 0.30,
            })
        elif fam == "slope_body":
            y["candidates"]["trigger"].update({
              "slope_t_min": float(best.get("slope_min", 0.0)),
              "slope_r2_min": float(best.get("slope_r2_min", 0.60)),
              "body_frac_min": float(best.get("body_min", 0.60)),
              "donch_pos_min_long": float(best.get("pos_min_long", 0.70)),
              "donch_pos_max_short": float(1.0 - best.get("pos_min_long", 0.70)),
            })
        elif fam == "dwell_trend":
            y["candidates"]["trigger"].update({
              "cusum_dwell_min": int(best.get("dwell_min", 8)),
              "body_frac_min": float(best.get("body_min", 0.60)),
            })
        elif fam == "range_thrust":
            y["candidates"]["trigger"].update({
              "tr_atr_min": float(best.get("range_atr_min", 2.5)),
              "body_frac_min": float(best.get("body_min", 0.60)),
            })
        with open("configs/motifs_trigger_opt.yaml","w",encoding="utf-8") as fh:
            yaml.safe_dump(y, fh, sort_keys=False)
        print("\n[OK] Wrote configs/motifs_trigger_opt.yaml with best refined trigger + labeling.")

if __name__ == "__main__":
    main()
