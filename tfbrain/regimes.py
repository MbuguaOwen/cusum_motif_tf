from __future__ import annotations
import numpy as np, pickle
from pathlib import Path
from typing import Dict, Any

def zscore_fit(X: np.ndarray):
    mu, sigma = X.mean(0), X.std(0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def zscore_apply(X, mu, sigma):
    return (X - mu) / sigma

def kmeans(X: np.ndarray, K: int, n_init=10, iters=100, seed=42):
    rng = np.random.default_rng(seed)
    best_inertia, best_C = np.inf, None
    for _ in range(n_init):
        idx = rng.choice(len(X), K, replace=False)
        C = X[idx].copy()
        for _ in range(iters):
            D = ((X[:, None, :] - C[None, :, :])**2).sum(-1)
            lbl = D.argmin(1)
            C_new = np.vstack([X[lbl==k].mean(0) if np.any(lbl==k) else C[k] for k in range(K)])
            if np.allclose(C_new, C): break
            C = C_new
        inertia = ((X - C[lbl])**2).sum()
        if inertia < best_inertia:
            best_inertia, best_C = inertia, C
    return best_C

def fit_fingerprints(df, regime_features, K, seed=42):
    X = df[regime_features].to_numpy(dtype=float)
    mu, sigma = zscore_fit(X)
    Z = zscore_apply(X, mu, sigma)
    C = kmeans(Z, K=K, seed=seed)
    return {"mu": mu, "sigma": sigma, "centroids": C, "features": regime_features}

def assign_regime(fprint_model: Dict[str, Any], row_vec: np.ndarray):
    mu, sigma, C = fprint_model["mu"], fprint_model["sigma"], fprint_model["centroids"]
    z = (row_vec - mu) / sigma
    D = ((C - z)**2).sum(1)
    return int(np.argmin(D))

def save_model(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f: pickle.dump(obj, f)

def load_model(path: Path):
    with open(path, "rb") as f: return pickle.load(f)

