import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import linear_regression_slope_r2

def compute_features(bars: pd.DataFrame, macro: int=60, micro: int=15, slope_M:int=60) -> pd.DataFrame:
    df = bars.copy().sort_values("timestamp").reset_index(drop=True)
    df["log_close"] = np.log(df["close"].replace(0, np.nan))
    df["ret"] = df["log_close"].diff()

    def tstat(series, window):
        mu = series.rolling(window, min_periods=window).mean()
        sd = series.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
        return (mu / sd) * np.sqrt(window)

    df["kappa_long"]  = tstat(df["ret"], macro)
    df["kappa_short"] = tstat(df["ret"], micro)
    df["accel"] = df["kappa_short"] - df["kappa_long"]

    slopes, r2s = [], []
    closes = df["log_close"].values
    for i in tqdm(range(len(df)), desc="Features: slope/R²", leave=False):
        j = i - slope_M + 1
        if j < 0:
            slopes.append(np.nan); r2s.append(np.nan); continue
        y = closes[j:i+1]
        s, r = linear_regression_slope_r2(y)
        slopes.append(s); r2s.append(r)
    df["slope"] = slopes
    df["slope_r2"] = r2s

    tr = np.maximum(
        df["high"]-df["low"],
        np.maximum((df["high"]-df["close"].shift()).abs(), (df["low"]-df["close"].shift()).abs())
    )
    df["atr"] = tr.rolling(micro, min_periods=micro).mean()

    m = df["close"].rolling(micro, min_periods=micro).mean()
    sd = df["close"].rolling(micro, min_periods=micro).std(ddof=0)
    upper = m + 2*sd; lower = m - 2*sd
    bbw = (upper - lower).abs()

    dh = df["high"].rolling(micro, min_periods=micro).max()
    dl = df["low"].rolling(micro, min_periods=micro).min()
    donch_w = (dh - dl).abs()

    df["donch_pos"] = (df["close"] - dl) / (dh - dl).replace(0, np.nan)
    df["body_frac"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = (df["high"] - df["close"]).clip(lower=0)
    lower_wick = (df["close"] - df["low"]).clip(lower=0)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["wick_bias"] = (upper_wick - lower_wick) / rng

    bar_range = (df["high"] - df["low"]).abs()
    N = micro
    df["re_burst"] = (bar_range > bar_range.rolling(N, min_periods=N).quantile(0.9)).astype(float)

    def pct_last(x): return pd.Series(x).rank(pct=True).iloc[-1]
    df["atr_p"] = df["atr"].rolling(60, min_periods=60).apply(pct_last)
    df["bbw_p"] = bbw.rolling(60, min_periods=60).apply(pct_last)
    df["donch_w_p"] = donch_w.rolling(60, min_periods=60).apply(pct_last)

    sign = np.sign(df["kappa_long"].fillna(0))
    age, last_flip = [], 0
    for i, s in enumerate(sign):
        if i == 0:
            age.append(0); continue
        if s != sign[i-1]:
            last_flip = 0
        else:
            last_flip += 1
        age.append(last_flip)
    df["trend_age"] = age

    # Attach feature channel names for diagnostics (non-breaking; still return DataFrame)
    produced = [
        "kappa_long","kappa_short","accel","slope","slope_r2","atr",
        "atr_p","bbw_p","donch_w_p","donch_pos","body_frac","wick_bias",
        "re_burst","trend_age"
    ]
    df.attrs["feature_channels_lower"] = [c.lower() for c in produced if c in df.columns]
    return df

def assert_channels_exist(df: pd.DataFrame, channels: list[str]):
    cols = {c.lower() for c in df.columns}
    need = [c for c in channels if c.lower() not in cols]
    if need:
        raise KeyError(f"Missing feature channels: {need}. Present: {sorted(list(cols))}")
