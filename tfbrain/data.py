from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from .utils import to_ms, read_csv_or_parquet

def load_bars_1m(inputs_dir: Path, symbol: str, months: List[str]) -> pd.DataFrame:
    rows = []
    for m in months:
        csv = inputs_dir / "bars_1m" / f"{symbol}_{m}.csv"
        pq  = inputs_dir / "bars_1m" / f"{symbol}_{m}.parquet"
        fp = csv if csv.exists() else pq
        if not fp.exists():
            raise FileNotFoundError(f"Bars missing for {symbol} {m}: {fp.name}")
        df = read_csv_or_parquet(fp)
        # expected columns: timestamp, open, high, low, close, volume
        if "timestamp" not in df.columns:
            # try to infer index
            if df.index.name:
                df = df.reset_index()
                df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df["timestamp"] = df["timestamp"].apply(to_ms)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out

def load_ticks(inputs_dir: Path, symbol: str, months: List[str]) -> pd.DataFrame:
    rows = []
    for m in months:
        fp = inputs_dir / "ticks" / f"{symbol}_{m}.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Ticks missing for {symbol} {m}: {fp.name}")
        df = pd.read_csv(fp)
        # expected: timestamp, price, qty, is_buyer_maker
        df["timestamp"] = df["timestamp"].astype("int64")
        rows.append(df)
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    return out
