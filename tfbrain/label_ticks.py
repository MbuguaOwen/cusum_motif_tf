import numpy as np
import pandas as pd
from tqdm import tqdm

def first_touch_labels(candidates: pd.DataFrame, ticks: pd.DataFrame, sl_mult: float, tp_mult: float, time_limit_bars:int, bar_seconds:int=60, slippage_bps:int=3, quant_step:float=None):
    """Label each candidate with tick-first-touch triple barrier. ticks: timestamp (ms), price."""
    events = []
    if len(candidates)==0:
        return pd.DataFrame(columns=["timestamp","bar_idx","entry","tp","sl","exit_ts","exit_price","label","realized_R","t_to_exit_s"])
    ticks = ticks.sort_values("timestamp").reset_index(drop=True)
    t_idx = 0
    t_times = ticks["timestamp"].values
    t_prices = ticks["price"].values.astype(float)
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Tick label"):
        t0 = int(row["timestamp"])
        entry = float(row["close"])
        atr0 = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
        if atr0 <= 0:  # skip if invalid
            continue
        tp = entry + tp_mult*atr0
        sl = entry - sl_mult*atr0
        # find starting tick index
        while t_idx < len(t_times) and t_times[t_idx] < t0:
            t_idx += 1
        label = 0; exit_price = entry; exit_ts = t0
        limit_ts = t0 + time_limit_bars*bar_seconds*1000
        j = t_idx
        while j < len(t_times) and t_times[j] <= limit_ts:
            p = t_prices[j]
            # check touches (for longs)
            if p >= tp:
                label = +1; exit_price = p; exit_ts = int(t_times[j]); break
            if p <= sl:
                label = -1; exit_price = p; exit_ts = int(t_times[j]); break
            j += 1
        # apply slippage (bps)
        slip = entry * (slippage_bps/10000.0)
        if label == +1:
            exit_price = exit_price - slip
        elif label == -1:
            exit_price = exit_price + slip
        R = (exit_price - entry) / (sl_mult*atr0)  # denom in SL ATRs -> R units
        events.append({
            "timestamp": t0,
            "bar_idx": int(row["bar_idx"]),
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "exit_ts": exit_ts,
            "exit_price": exit_price,
            "label": label,
            "realized_R": float(R),
        })
    return pd.DataFrame(events)
