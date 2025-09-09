import os, json, math, pickle, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_ms(ts):
    # Accept ms int or ISO string -> ms int
    if isinstance(ts, (int, np.integer, float, np.floating)):
        return int(ts)
    try:
        return int(pd.to_datetime(ts, utc=True).value // 10**6)
    except Exception:
        return int(pd.to_datetime(ts, utc=True).value // 10**6)

def rolling_percentile(s: pd.Series, window: int, q: float) -> pd.Series:
    return s.rolling(window, min_periods=window).quantile(q/100.0)

def pct_rank(s: pd.Series, window: int) -> pd.Series:
    # Percentile rank inside rolling window
    return s.rolling(window, min_periods=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - m) / (sd.replace(0, np.nan))

def linear_regression_slope_r2(y: np.ndarray):
    x = np.arange(len(y))
    x = (x - x.mean())/ (x.std() + 1e-12)
    X = np.c_[np.ones_like(x), x]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = ((y - y_hat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum() + 1e-12
    r2 = 1 - ss_res/ss_tot
    slope = beta[1]
    return slope, r2

def drawdown_R(rs: List[float]) -> float:
    # worst cumulative sum drawdown in R
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return abs(dd)

def month_from_ts_ms(ms: int, tz="UTC"):
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert(tz).strftime("%Y-%m")

def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def json_dump(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_csv_or_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def quantize_price(price: float, step: Optional[float]) -> float:
    if (step is None) or (step <= 0): return price
    return round(price / step) * step
