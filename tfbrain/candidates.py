import numpy as np
import pandas as pd

def _frac(x):
    try:
        v = float(x); return v/100.0 if v>1.0 else v
    except Exception:
        return x

def detect_candidates(bars: pd.DataFrame, features: pd.DataFrame, cusum_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = features.copy()
    df["idx"] = np.arange(len(df))
    df["timestamp"] = bars["timestamp"].values
    micro = int(cfg["features"]["windows"]["micro"])
    dh = bars["high"].rolling(micro, min_periods=micro).max()
    dl = bars["low"].rolling(micro, min_periods=micro).min()
    atr = df["atr"]

    bbw_max = _frac(cfg["candidates"]["compression"]["bbw_p_max"])
    donch_w_max = _frac(cfg["candidates"]["compression"]["donch_w_p_max"])
    atr_p_max = _frac(cfg["candidates"]["compression"]["atr_p_max"])
    body_min = float(cfg["candidates"]["trigger"]["body_frac_min"])
    accel_min = float(cfg["candidates"]["trigger"]["accel_min"])
    pos_min_long = _frac(cfg["candidates"]["trigger"]["donch_pos_min_long"])
    pos_max_short = _frac(cfg["candidates"]["trigger"]["donch_pos_max_short"])
    spacing = int(cfg["candidates"]["spacing_bars"])
    require_cusum = bool(cfg["candidates"]["require_cusum"])

    burst_idx_up = set(); burst_idx_dn = set()
    if cusum_df is not None and len(cusum_df)>0:
        burst_idx_up = set(cusum_df[cusum_df["dir"]==+1]["idx"].tolist())
        burst_idx_dn = set(cusum_df[cusum_df["dir"]==-1]["idx"].tolist())

    candidates = []
    last_long = -10**9; last_short = -10**9

    for i in range(len(df)):
        c = bars["close"].iloc[i]
        dH = dh.iloc[i]; dL = dl.iloc[i]
        if np.isnan(dH) or np.isnan(dL): 
            continue
        donch_pos = (c - dL) / (dH - dL) if (dH - dL) != 0 else np.nan

        if not ((df["bbw_p"].iloc[i] <= bbw_max) or (df["donch_w_p"].iloc[i] <= donch_w_max)):
            continue
        if not (df["atr_p"].iloc[i] <= atr_p_max):
            continue
        if not (df["body_frac"].iloc[i] >= body_min):
            continue

        # LONG
        if i - last_long >= spacing:
            if (not require_cusum) or (i in burst_idx_up):
                if (df["accel"].iloc[i] >= accel_min) and (donch_pos >= pos_min_long):
                    candidates.append({
                        "bar_idx": i, "timestamp": int(df["timestamp"].iloc[i]), "side": +1,
                        "close": float(c), "atr": float(atr.iloc[i]),
                        "donch_high": float(dH), "donch_low": float(dL), "donch_pos": float(donch_pos)
                    })
                    last_long = i
        # SHORT
        if i - last_short >= spacing:
            if (not require_cusum) or (i in burst_idx_dn):
                if (df["accel"].iloc[i] <= -accel_min) and (donch_pos <= pos_max_short):
                    candidates.append({
                        "bar_idx": i, "timestamp": int(df["timestamp"].iloc[i]), "side": -1,
                        "close": float(c), "atr": float(atr.iloc[i]),
                        "donch_high": float(dH), "donch_low": float(dL), "donch_pos": float(donch_pos)
                    })
                    last_short = i
    return pd.DataFrame(candidates, columns=["bar_idx","timestamp","side","close","atr","donch_high","donch_low","donch_pos"]) 
