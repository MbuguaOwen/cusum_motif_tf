import numpy as np
import pandas as pd
from tqdm import tqdm

def first_touch_labels(candidates: pd.DataFrame, ticks: pd.DataFrame, sl_mult: float, tp_mult: float, time_limit_bars:int, bar_seconds:int=60, slippage_bps:int=3):
    events = []
    if len(candidates)==0:
        return pd.DataFrame(columns=["timestamp","bar_idx","side","entry","tp","sl","exit_ts","exit_price","label","realized_R"]
                            )
    ticks = ticks.sort_values("timestamp").reset_index(drop=True)
    t_idx = 0
    t_times = ticks["timestamp"].values
    t_prices = ticks["price"].values.astype(float)
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Tick label", leave=False):
        t0 = int(row["timestamp"])
        entry = float(row["close"])
        atr0 = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
        side = int(row["side"])  # +1 long, -1 short
        if atr0 <= 0:
            continue
        if side == +1:
            tp = entry + tp_mult*atr0
            sl = entry - sl_mult*atr0
        else:
            tp = entry - tp_mult*atr0
            sl = entry + sl_mult*atr0
        while t_idx < len(t_times) and t_times[t_idx] < t0:
            t_idx += 1
        label = 0; exit_price = entry; exit_ts = t0
        limit_ts = t0 + time_limit_bars*bar_seconds*1000
        j = t_idx
        while j < len(t_times) and t_times[j] <= limit_ts:
            p = t_prices[j]
            if side == +1:
                if p >= tp: label=+1; exit_price=p; exit_ts=int(t_times[j]); break
                if p <= sl: label=-1; exit_price=p; exit_ts=int(t_times[j]); break
            else:
                if p <= tp: label=+1; exit_price=p; exit_ts=int(t_times[j]); break
                if p >= sl: label=-1; exit_price=p; exit_ts=int(t_times[j]); break
            j += 1
        slip = entry * (slippage_bps/10000.0)
        if label == +1:
            exit_price = exit_price - slip if side==+1 else exit_price + slip
        elif label == -1:
            exit_price = exit_price + slip if side==+1 else exit_price - slip
        R = side * (exit_price - entry) / (sl_mult*atr0)
        events.append({
            "timestamp": t0, "bar_idx": int(row["bar_idx"]), "side": side,
            "entry": entry, "tp": tp, "sl": sl,
            "exit_ts": exit_ts, "exit_price": exit_price,
            "label": label, "realized_R": float(R)
        })
    return pd.DataFrame(events)
