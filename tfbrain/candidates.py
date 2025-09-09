from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

# Shared candidate schema (order enforced)
CAND_SCHEMA = [
    "bar_idx","timestamp","ts","side","entry_price","atr",
    "donch_pos","kappa_long","slope","slope_r2","bbw_p","atr_p"
]


def _make_side_candidates(df: pd.DataFrame, bars: pd.DataFrame, cfg: dict, side: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    side: "long" or "short"
    Uses Donchian breakout with ATR buffer and compression precondition.
    Directional guards:
      long  → kappa_long >= +thr, slope >= +thr
      short → kappa_long <= -thr, slope <= -thr
    Returns a DataFrame with at least:
      ["bar_idx","timestamp","ts","side","entry_price","atr","donch_pos","kappa_long","slope","slope_r2","bbw_p","atr_p"]
    """
    assert side in ("long", "short")
    n = len(df)
    # Params
    donch_n = int(cfg.get("candidates", {}).get("donch_n", int(cfg["features"]["windows"]["micro"])) )
    atr_w = int(cfg.get("risk", {}).get("atr_window", 50))
    buf = float(cfg.get("candidates", {}).get("breakout_atr_buffer", 0.0))
    spacing = int(cfg.get("candidates", {}).get("min_spacing_bars", 20))
    comp = cfg.get("candidates", {}).get("compression", {})
    bbw_max = float(comp.get("bbw_p_max", 0.35))
    donch_w_max = float(comp.get("donch_w_p_max", 0.35))
    # Core series
    highs = bars["high"].rolling(donch_n, min_periods=donch_n).max()
    lows  = bars["low"].rolling(donch_n, min_periods=donch_n).min()
    # ATR (classic)
    tr = np.maximum(bars["high"]-bars["low"], np.maximum((bars["high"]-bars["close"].shift()).abs(), (bars["low"]-bars["close"].shift()).abs()))
    atr = tr.rolling(atr_w, min_periods=atr_w).mean()

    # compression flags from features
    bbw_p = df.get("bbw_p", pd.Series([np.nan]*n))
    donch_w_p = df.get("donch_w_p", pd.Series([np.nan]*n))
    atr_p = df.get("atr_p", pd.Series([np.nan]*n))
    slope = df.get("slope", pd.Series([np.nan]*n))

    # directional thresholds for sign gating
    sign_thr = float(cfg.get("sides", {}).get("kappa_sign_thr", 0.0))

    blockers = {"considered": 0, "spacing": 0, "not_compressed": 0, "breakout": 0, "sign_gate": 0}
    out_rows = []
    last_idx = -10**9
    for i in range(n):
        blockers["considered"] += 1
        if i - last_idx < spacing:
            blockers["spacing"] += 1; continue
        # compression precondition (features' percentiles are in [0,1])
        compressed = (pd.notna(bbw_p.iloc[i]) and float(bbw_p.iloc[i]) <= bbw_max) or (pd.notna(donch_w_p.iloc[i]) and float(donch_w_p.iloc[i]) <= donch_w_max)
        if not compressed:
            blockers["not_compressed"] += 1; continue
        # sign gate by side
        kl = float(np.nan_to_num(df.get("kappa_long", pd.Series([np.nan]*n)).iloc[i]))
        slp = float(np.nan_to_num(slope.iloc[i]))
        if side == "long":
            if not (kl >= +sign_thr and slp >= +0.0):
                blockers["sign_gate"] += 1; continue
        else:
            if not (kl <= -sign_thr and slp <= +0.0):
                blockers["sign_gate"] += 1; continue
        # breakout rule with ATR buffer
        c = float(bars["close"].iloc[i])
        dH = float(highs.iloc[i]) if pd.notna(highs.iloc[i]) else np.nan
        dL = float(lows.iloc[i]) if pd.notna(lows.iloc[i]) else np.nan
        atr_i = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
        if not (np.isfinite(dH) and np.isfinite(dL) and np.isfinite(atr_i)):
            blockers["breakout"] += 1; continue
        ok = c > dH + buf * atr_i if side == "long" else c < dL - buf * atr_i
        if not ok:
            blockers["breakout"] += 1; continue
        # donch pos
        donch_pos = (c - dL) / (dH - dL) if (dH - dL) != 0 else np.nan
        out_rows.append({
            "bar_idx": int(i),
            "timestamp": int(bars["timestamp"].iloc[i]),
            "ts": int(bars["timestamp"].iloc[i]),
            "side": side,
            "entry_price": c,
            "atr": atr_i,
            "donch_pos": float(donch_pos) if np.isfinite(donch_pos) else np.nan,
            "kappa_long": kl,
            "slope": slp,
            "slope_r2": float(np.nan_to_num(df.get("slope_r2", pd.Series([np.nan]*n)).iloc[i])),
            "bbw_p": float(np.nan_to_num(df.get("bbw_p", pd.Series([np.nan]*n)).iloc[i])),
            "atr_p": float(np.nan_to_num(df.get("atr_p", pd.Series([np.nan]*n)).iloc[i])),
        })
        last_idx = i
    df_out = pd.DataFrame(out_rows, columns=CAND_SCHEMA) if len(out_rows) else pd.DataFrame(columns=CAND_SCHEMA)
    return df_out, blockers


def build_candidates(bars: pd.DataFrame, features: pd.DataFrame, cusum_df: Optional[pd.DataFrame], cfg: dict, out_dir: Optional[Path] = None) -> pd.DataFrame:
    out = []
    blocks: Dict[str, Any] = {}
    if cfg.get("sides", {}).get("allow_long", True):
        c_long, b_long = _make_side_candidates(features, bars, cfg, "long")
        # ensure schema even if empty
        if set(c_long.columns) != set(CAND_SCHEMA):
            c_long = c_long.reindex(columns=CAND_SCHEMA)
        out.append(c_long); blocks["long"] = b_long
    if cfg.get("sides", {}).get("allow_short", False):
        c_short, b_short = _make_side_candidates(features, bars, cfg, "short")
        if set(c_short.columns) != set(CAND_SCHEMA):
            c_short = c_short.reindex(columns=CAND_SCHEMA)
        out.append(c_short); blocks["short"] = b_short

    # If both sides produced zero rows, return an empty, schema-correct DF
    if not any(len(df_side) for df_side in out):
        cands = pd.DataFrame(columns=CAND_SCHEMA)
    else:
        cands = pd.concat(out, ignore_index=True)
        if "ts" in cands.columns and not cands.empty:
            cands = cands.sort_values("ts").reset_index(drop=True)

    # Optional regime sign gate (safe on empty)
    if cfg.get("sides", {}).get("regime_gate", "sign") == "sign" and not cands.empty:
        thr = float(cfg["sides"].get("kappa_sign_thr", 0.0))
        keep_long  = (cands["side"]=="long")  & (cands["kappa_long"] >= +thr)
        keep_short = (cands["side"]=="short") & (cands["kappa_long"] <= -thr)
        cands = cands[ keep_long | keep_short ].reset_index(drop=True)

    # blockers telemetry (robust)
    if out_dir is not None:
        try:
            (out_dir/"cand_blockers.json").write_text(pd.Series(blocks, dtype="object").to_json(indent=2), encoding="utf-8")
        except Exception:
            pass
    print(f"[cands] blockers: {blocks}")
    return cands
