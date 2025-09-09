from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def detect_candidates(bars: pd.DataFrame, features: pd.DataFrame, cusum_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return DataFrame of candidate entries with trigger confirmations and feature snapshot."""
    df = features.copy()
    df["t_index"] = np.arange(len(df))
    df["timestamp"] = bars["timestamp"].values
    # donchian window = micro
    micro = int(cfg["features"]["windows"]["micro"])
    dh = bars["high"].rolling(micro, min_periods=micro).max()
    dl = bars["low"].rolling(micro, min_periods=micro).min()
    atr = df["atr"]
    # thresholds
    bbw_max = cfg["entry"]["compression"]["bbw_p_max"]
    donch_w_max = cfg["entry"]["compression"]["donch_w_p_max"]
    atr_p_max = cfg["entry"]["compression"]["atr_p_max"]
    body_min = cfg["entry"]["trigger"]["body_frac_min"]
    accel_min = cfg["entry"]["trigger"]["accel_min"]
    donch_pos_min = cfg["entry"]["trigger"]["donch_pos_min"]
    spacing = cfg["entry"]["spacing_bars"]
    require_cusum = bool(cfg["entry"]["require_cusum"])
    # α from train fold will be set later; here we store buffer_atr placeholder (e.g., q75 to be injected upstream)
    alpha_buffer_atr = None  # will be added by caller if needed
    
    t_bursts = set(cusum_df["idx"].tolist()) if cusum_df is not None and len(cusum_df)>0 else set()
    candidates = []
    last_idx = -10**9
    for i in range(len(df)):
        # spacing
        if i - last_idx < spacing:
            continue
        # require CUSUM at i (or nearby) if enabled
        if require_cusum and (i not in t_bursts):
            continue
        # preconditions (compression)
        if not ( (df["bbw_p"].iloc[i] <= bbw_max/100.0) or (df["donch_w_p"].iloc[i] <= donch_w_max/100.0) ):
            continue
        if not (df["atr_p"].iloc[i] <= atr_p_max/100.0):
            continue
        # trigger (longs and shorts will be considered separately later; here we just record context)
        c = bars["close"].iloc[i]
        dH = dh.iloc[i]; dL = dl.iloc[i]
        donch_pos = (c - dL) / (dH - dL) if (dH - dL) != 0 else np.nan
        if not (df["body_frac"].iloc[i] >= body_min):
            continue
        if not (df["accel"].iloc[i] >= accel_min):
            continue
        if not (donch_pos >= donch_pos_min):
            continue
        # store
        candidates.append({
            "bar_idx": i,
            "timestamp": int(bars["timestamp"].iloc[i]),
            "close": float(c),
            "atr": float(atr.iloc[i]),
            "donch_high": float(dH) if not np.isnan(dH) else None,
            "donch_low": float(dL) if not np.isnan(dL) else None,
            "donch_pos": float(donch_pos) if not np.isnan(donch_pos) else None,
            "kappa_long": float(df["kappa_long"].iloc[i]) if not np.isnan(df["kappa_long"].iloc[i]) else None,
            "kappa_short": float(df["kappa_short"].iloc[i]) if not np.isnan(df["kappa_short"].iloc[i]) else None,
            "accel": float(df["accel"].iloc[i]) if not np.isnan(df["accel"].iloc[i]) else None,
            "body_frac": float(df["body_frac"].iloc[i]) if not np.isnan(df["body_frac"].iloc[i]) else None,
        })
        last_idx = i
    out = pd.DataFrame(candidates)
    if len(out)==0:
        out = pd.DataFrame(columns=["bar_idx","timestamp","close","atr","donch_high","donch_low","donch_pos","kappa_long","kappa_short","accel","body_frac"])
    return out
