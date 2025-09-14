import numpy as np
import pandas as pd

def kmeans_fit(X: np.ndarray, k:int=8, iters:int=50, seed:int=42):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    centers = X[idx].copy()
    for _ in range(iters):
        dists = ((X[:,None,:] - centers[None,:,:])**2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.vstack([X[labels==j].mean(axis=0) if (labels==j).any() else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers):
            centers = new_centers; break
        centers = new_centers
    return centers, labels

def assign_kmeans(X: np.ndarray, centers: np.ndarray):
    dists = ((X[:,None,:] - centers[None,:,:])**2).sum(axis=2)
    return dists.argmin(axis=1)

def fingerprint(features: pd.DataFrame, cusum_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rate = np.zeros(len(features), dtype=float)
    if cusum_df is not None and len(cusum_df)>0:
        burst_idx = cusum_df["idx"].values
        for i in burst_idx:
            if 0 <= i < len(rate):
                rate[i] = 1.0
    cusum_rate = pd.Series(rate).rolling(120, min_periods=1).sum() / 120.0
    cols = cfg["regime"]["fingerprint"]
    fp = pd.DataFrame(index=features.index)
    for c in cols:
        if c == "cusum_rate":
            fp[c] = cusum_rate.values
        else:
            fp[c] = features[c].astype(float)
    fp = fp.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return fp
