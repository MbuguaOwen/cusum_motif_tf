import numpy as np
import pandas as pd

def cusum_events(bars: pd.DataFrame, drift_k: float, h: float, sigma_window: int=None) -> pd.DataFrame:
    """CUSUM on log returns. Returns a DataFrame with t_burst indexes and directions."""
    df = bars.sort_values("timestamp").reset_index(drop=True)
    r = np.log(df["close"]).diff().fillna(0.0).to_numpy()

    if sigma_window:
        sigma = (
            pd.Series(r)
            .rolling(sigma_window, min_periods=sigma_window)
            .std(ddof=0)
            .bfill()
            .to_numpy()
        )
        sigma[sigma <= 0] = 1e-12
        rr = r / sigma
    else:
        rr = r

    Gp = 0.0; Gn = 0.0
    bursts = []
    for i, x in enumerate(rr):
        Gp = max(0.0, Gp + x - drift_k)
        Gn = min(0.0, Gn + x + drift_k)
        if Gp >= h:
            bursts.append((i, +1)); Gp = 0.0; Gn = 0.0
        elif Gn <= -h:
            bursts.append((i, -1)); Gp = 0.0; Gn = 0.0

    return pd.DataFrame(bursts, columns=["idx", "dir"])
