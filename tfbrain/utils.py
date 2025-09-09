import os, json, math, pickle, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_ms(ts):
    # Accept seconds/ms/μs ints or ISO strings -> ms int
    if isinstance(ts, (int, np.integer, float, np.floating)):
        n = int(ts)
        if n > 10**14:   # microseconds
            return n // 1000
        if n > 10**12:   # milliseconds
            return n
        if n > 10**9:    # seconds
            return n * 1000
        return n
    # strings / timestamps
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(dt):
        # last resort: treat as number
        try:
            return to_ms(float(ts))
        except Exception:
            raise ValueError(f"Cannot parse timestamp: {ts}")
    return int(dt.value // 10**6)

def rolling_percentile(s: pd.Series, window: int, q: float) -> pd.Series:
    return s.rolling(window, min_periods=window).quantile(q/100.0)

def pct_rank(s: pd.Series, window: int) -> pd.Series:
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
    cum = 0.0; peak = 0.0; dd = 0.0
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
