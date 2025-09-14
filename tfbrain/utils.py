import os, json, math, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_ms(ts):
    if isinstance(ts, (int, np.integer, float, np.floating)):
        n = int(ts)
        if n > 10**14:   # microseconds
            return n // 1000
        if n > 10**12:   # milliseconds
            return n
        if n > 10**9:    # seconds
            return n * 1000
        return n
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(dt):
        try:
            return to_ms(float(ts))
        except Exception:
            raise ValueError(f"Cannot parse timestamp: {ts}")
    return int(dt.value // 10**6)

def json_dump(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_csv_or_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

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

# Similarity and z-score helpers
def _safe_norm(x: np.ndarray) -> float:
    n = float(np.linalg.norm(x))
    return n if np.isfinite(n) and n > 1e-12 else 1.0

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = _safe_norm(a); bn = _safe_norm(b)
    return float(np.dot(a, b) / (an * bn))

def rbf_sim(a: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> float:
    diff = a - b
    d2 = float(np.dot(diff, diff))
    if not np.isfinite(d2): d2 = 1e6
    return float(np.exp(-gamma * d2))

def zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    return mu, sd

def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sd
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z
