from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import zscore, pct_rank, linear_regression_slope_r2

def compute_features(bars: pd.DataFrame, macro: int=60, micro: int=15, slope_M:int=60) -> pd.DataFrame:
    df = bars.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    # log returns
    df["log_close"] = np.log(df["close"].replace(0, np.nan))
    df["ret"] = df["log_close"].diff()
    # t-stat proxies: mean/std * sqrt(n) ~ (rolling mean / rolling std) * sqrt(window)
    def tstat(series, window):
        mu = series.rolling(window, min_periods=window).mean()
        sd = series.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
        return (mu / sd) * np.sqrt(window)
    df["kappa_long"]  = tstat(df["ret"], macro)
    df["kappa_short"] = tstat(df["ret"], micro)
    df["accel"] = df["kappa_short"] - df["kappa_long"]
    # slope & r2 over M bars
    slopes = []
    r2s = []
    closes = df["log_close"].values
    for i in range(len(df)):
        j = i - slope_M + 1
        if j < 0:
            slopes.append(np.nan); r2s.append(np.nan); continue
        y = closes[j:i+1]
        s, r = linear_regression_slope_r2(y)
        slopes.append(s); r2s.append(r)
    df["slope"] = slopes
    df["slope_r2"] = r2s
    # ATR (classic)
    tr = np.maximum(df["high"]-df["low"], np.maximum((df["high"]-df["close"].shift()).abs(), (df["low"]-df["close"].shift()).abs()))
    df["atr"] = tr.rolling(micro, min_periods=micro).mean()
    # Percentiles
    def pct(s, w):
        return s.rolling(w, min_periods=w).apply(lambda x: (x.rank(pct=True).iloc[-1]))
    # Bollinger width
    m = df["close"].rolling(micro, min_periods=micro).mean()
    sd = df["close"].rolling(micro, min_periods=micro).std(ddof=0)
    upper = m + 2*sd; lower = m - 2*sd
    bbw = (upper - lower).abs()
    # Donchian
    dh = df["high"].rolling(micro, min_periods=micro).max()
    dl = df["low"].rolling(micro, min_periods=micro).min()
    donch_w = (dh - dl).abs()
    # positions & structure
    df["donch_pos"] = (df["close"] - dl) / (dh - dl).replace(0, np.nan)
    df["body_frac"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = (df["high"] - df["close"]).clip(lower=0)
    lower_wick = (df["close"] - df["low"]).clip(lower=0)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["wick_bias"] = (upper_wick - lower_wick) / rng
    # Range expansion flag (WR7-like on 1m): today's bar range vs last N
    bar_range = (df["high"] - df["low"]).abs()
    N = micro
    re_flag = (bar_range > bar_range.rolling(N, min_periods=N).quantile(0.9)).astype(float)
    df["re_burst"] = re_flag
    # percentiles
    df["atr_p"] = df["atr"].rolling(macro, min_periods=macro).apply(lambda x: (pd.Series(x).rank(pct=True).iloc[-1]))
    df["bbw_p"] = bbw.rolling(macro, min_periods=macro).apply(lambda x: (pd.Series(x).rank(pct=True).iloc[-1]))
    df["donch_w_p"] = donch_w.rolling(macro, min_periods=macro).apply(lambda x: (pd.Series(x).rank(pct=True).iloc[-1]))
    # trend age (bars since sign flip in kappa_long)
    sign = np.sign(df["kappa_long"].fillna(0))
    age = []
    last_flip = 0
    for i, s in enumerate(sign):
        if i==0:
            age.append(0); continue
        if s != sign[i-1]:
            last_flip = 0
        else:
            last_flip += 1
        age.append(last_flip)
    df["trend_age"] = age
    return df
