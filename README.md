# CUSUM‑Hybrid Motif Trend‑Following Brain

**Tagline:** Bars for brains, ticks for receipts.  
- **Features & regimes:** 1‑minute bars (cleaner, more stable).  
- **Ignition:** event‑time **CUSUM** on bars + breakout confirmation.  
- **Labeling & fills:** **tick first‑touch** (realistic TP/SL sequencing, slippage).  
- **Memory:** class‑aware **motifs/shapelets** (GOOD/BAD/DISCORD) mined around event onsets.  
- **Scoring & gating:** weighted motif hits + statistical context; **weights learned** on train folds via ridge regression.  
- **Validation:** strict walk‑forward (k→1, step=1).

## Folder layout
```
cusum_motif_tf/
  configs/
    motifs.yaml           # main config
  inputs/
    bars_1m/              # your 1m OHLCV files per month (CSV or Parquet)
    ticks/                # your tick prints per month (CSV only for now)
  outputs/                # results per run/fold/symbol
  tfbrain/                # package code (run with: python -m tfbrain ...)
  README.md
  requirements.txt
```

### Expected input file naming
- **Bars (1m)**: `inputs/bars_1m/{SYMBOL}_{YYYY-MM}.csv` *or* `.parquet`  
  CSV columns: `timestamp, open, high, low, close, volume` (timestamp in ms or iso).  
- **Ticks**: `inputs/ticks/{SYMBOL}_{YYYY-MM}.csv`  
  Columns: `timestamp, price, qty, is_buyer_maker` (timestamp in ms).

> Run **one symbol at a time** (as configured). Months are the rolling train/test splits.

## Quickstart
1) Create & edit `configs/motifs.yaml` (see defaults).  
2) Place your data into `inputs/bars_1m` and `inputs/ticks`.  
3) Run the 7‑step pipeline via walk‑forward:

```bash
# Windows PowerShell
python -m tfbrain --config configs/motifs.yaml --mode walkforward

# Or step-by-step
python -m tfbrain --config configs/motifs.yaml --mode features
python -m tfbrain --config configs/motifs.yaml --mode candidates
python -m tfbrain --config configs/motifs.yaml --mode label
python -m tfbrain --config configs/motifs.yaml --mode mine
python -m tfbrain --config configs/motifs.yaml --mode simulate
```

## Outputs (per symbol & fold under `outputs/`)
- `candidates.parquet` — deterministic candidate entries & feature snapshots.
- `events.parquet` — tick first‑touch labels (+1/−1/0), realized R, times to PT/SL.
- `artifacts/*.pkl` — motif banks with ε calibration and metadata.
- `trades.csv` — simulated trades using frozen banks + risk.
- `stats.json` — fold metrics (sum_R, avg_R, win_rate, median_R, max_dd_R, trades, etc.).

## Dependencies (minimal)
```
numpy
pandas
pyyaml
tqdm
```
No scikit‑learn required; ridge weights are learned in‑house (NumPy).

## Notes
- All thresholds are statistical (quantiles, t‑stats) and calibrated **on train only**.
- Labeling is tick‑truth first‑touch (no bar fantasy). Slippage & quantized fills included.
- Gating is **BAD‑aware**. We prefer fewer A+ trades with higher realized R.
