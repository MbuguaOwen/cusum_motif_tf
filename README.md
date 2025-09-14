# CUSUM‑Hybrid Motif Trend‑Following — v2 (Regime‑Routed, LLR‑Gated)

> **Bars for brains, ticks for receipts.** This is a research‑grade toolkit to design, mine, calibrate, and walk‑forward a selective **trend‑following** entry engine using **regime fingerprinting** and **multivariate motif/shapelet** evidence.

---

## 0) TL;DR (Quickstart)

```bash
# 1) Create venv & install
python -m venv venv
# Windows PowerShell:
.env\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate
pip install -r requirements.txt

# 2) Drop your data under inputs/ (see §2 for formats)
#    - inputs/bars_1m/SYMBOL_YYYY-MM.csv|parquet
#    - inputs/ticks/SYMBOL_YYYY-MM.csv

# 3) Edit configs/motifs.yaml (symbol + months)

# 4) Run end-to-end walk-forward
python -m tfbrain --config configs/motifs.yaml --mode walkforward
```

Artifacts land under `outputs/{SYMBOL}/fold_{i}/…` with **candidates**, **events (labels)**, **banks (motifs)**, **gates**, **trades**, and **stats**.

---

## 1) Project Layout

```
cusum_motif_tf_v2/
  configs/
    motifs.yaml        # ← main configuration with full commentary
  inputs/
    bars_1m/           # ← 1-minute bars (CSV/Parquet)
    ticks/             # ← tick prints (CSV)
  outputs/             # ← all artifacts per fold
  tfbrain/             # ← pipeline code (module entry: python -m tfbrain)
  README.md
  requirements.txt
  .gitignore
  .gitattributes
```

---

## 2) Data: Formats & Examples

### 2.1 Bars (1‑minute, preferred for features)
**Path:** `inputs/bars_1m/SYMBOL_YYYY-MM.csv` (or `.parquet`)  
**Accepted columns:** one of:
- **Binance headerless 12‑col** CSV → auto‑mapped to:
  `open_time, open, high, low, close, volume, close_time, quote_volume, n_trades, taker_buy_base, taker_buy_quote, ignore`
- **Named columns:** must include `timestamp` or `open_time`, plus `open, high, low, close, volume`

**Timestamp units accepted:** seconds / milliseconds / microseconds; we normalize to **ms**.

**Example (headerless):**
```
1738368000000000,102429.56,102431.38,102333.00,102333.01,13.95242,1738368059999999,1428274.9594,3208,2.41966,247693.8046,0
```

### 2.2 Ticks (first‑touch labeling; “receipts”)
**Path:** `inputs/ticks/SYMBOL_YYYY-MM.csv`  
**Required columns:** `timestamp, price` (+ optional `qty, is_buyer_maker`)  
**Timestamp:** integer seconds/ms/µs accepted (auto‑normalized).

**Example:**
```csv
timestamp,price,qty,is_buyer_maker
1743465600108,124.53,21.248,1
1743465600614,124.53,0.058,0
```

> **Tip:** If your schema differs, adjust the column names in place, or extend `tfbrain/data.py` to map your variants.

---

## 3) Pipeline Overview (What happens under the hood)

**Bars → Features → CUSUM → Regimes → Candidates → Tick Labels → Mining → Gating Search → Simulation (test)**

1. **Features (bars, stat‑first)** — trend, vol, structure (“Core‑12”):
   - `kappa_long, kappa_short, accel = short−long`
   - `slope, slope_r2` (linear fit on log‑close)
   - `atr, bbw (Bollinger width), donch_w (Donchian width)` + **percentiles** (`atr_p, bbw_p, donch_w_p`)
   - `donch_pos, body_frac, wick_bias, re_burst, trend_age`

2. **CUSUM (event‑time ignition)** on log returns (optional sigma normalization). Emits burst indices (+1/−1).

3. **Regime Fingerprinting** (k‑means on compact state vector):  
   `[atr_p, bbw_p, donch_w_p, kappa_long, accel, slope_r2, donch_pos, body_frac, wick_bias, cusum_rate]`  
   → Every bar gets a **regime_id** for routing mining & gating.

4. **Entry Candidates (side‑aware)** — *compression + ignition*:
   - Compression (accepts **fractions** or **percent**): `bbw_p ≤ 0.70` **or** `donch_w_p ≤ 0.80`, **and** `atr_p ≤ 0.30`
   - Ignition: `body_frac ≥ 0.60`, `|accel| ≥ 1.5`, plus **donch_pos** bias (≥0.80 long / ≤0.20 short)
   - Optional: require CUSUM burst in the right direction
   - Spacing: `spacing_bars = 45` to prevent clustering

5. **Tick First‑Touch Triple‑Barrier Labels (truth)** (per candidate, **per side**):
   - Targets: `tp = entry ± tp_mult * ATR`, `sl = entry ∓ sl_mult * ATR`
   - First touch decides **+1 win / −1 loss**; else **0 timeout** by `time_limit_bars`
   - Applies simple slippage (bps) to exit
   - Computes realized **R** in SL units

6. **Motif/Shapelet Mining** *(per side × regime)*:
   - Extract multivariate windows around events (GOOD vs non‑GOOD) for lengths `{16|32 pre, 32|48 post}`
   - Distance = z‑normalized Euclidean across channels (fast & stable)
   - Greedy medoid selection → **top‑K** shapelets per class/length
   - Calibrate each shapelet’s **ε** at the q‑percentile of in‑class distances
   - **PPV pruning**: drop motifs with PPV < 0.6 (unique‑wins bias)

7. **Post‑Mining Gating Search** (train‑fold only; **selection before simulation**):
   - Grid knobs:
     - **ε‑multiplier α** ∈ `[0.8, 0.9, 1.0, 1.1]` (tighten/loosen)
     - **K‑of‑N GOOD hits** ∈ `[1,2,3]`
     - **BAD veto margin Δ** ∈ `[0.0,0.2,0.5]` (require min distance gap)
     - **Breakout buffer** ∈ `[0, 0.5, 1.0] * ATR` (price beyond Donchian)
   - Evidence score (LLR‑style **margin proxy**): `LLR ≈ bad_min − good_min`
   - Learn a small **ridge** weight vector on fired samples (R as target):  
     features = `[LLR, accel, kappa_long, donch_pos, ctx_sim, bias]`
   - Choose score threshold **τ** to **maximize train E[R]** under constraints:  
     *min trades/month*, *max drawdown in R*

8. **Simulation (test)** — freeze banks + selected gates/weights, apply to test candidates to get trades & KPIs.

---

## 4) Configuration (configs/motifs.yaml)

```yaml
paths:
  inputs_dir: "inputs"
  outputs_dir: "outputs"

symbols: ["BTCUSDT"]
data: { months: ["2025-01","2025-02","2025-03","2025-04"], timezone: "UTC" }

features:
  windows: { macro: 60, micro: 15, slope_M: 60 }
  channels: [kappa_long, kappa_short, accel, slope_r2, atr_p, bbw_p, donch_w_p, donch_pos, body_frac, wick_bias, re_burst, trend_age]

cusum: { drift_k: [0.0, "q50_abs_r"], h_quantiles: [0.80, 0.85, 0.90], sigma_window: 60, cooloff_bars: 45 }

regime:
  k: 8
  fingerprint: [atr_p, bbw_p, donch_w_p, kappa_long, accel, slope_r2, donch_pos, body_frac, wick_bias, cusum_rate]
  ctx_tau: 0.3

candidates:
  compression: { bbw_p_max: 0.70, donch_w_p_max: 0.80, atr_p_max: 0.30 }   # accepts 0..1 or %
  trigger: { body_frac_min: 0.60, accel_min: 1.5, donch_pos_min_long: 0.80, donch_pos_max_short: 0.20 }
  spacing_bars: 45
  require_cusum: true

label: { sl_mult: 8, tp_mult: 60, time_limit_bars: 1800, slippage_bps: 3 }

mining:
  lengths: [ {pre: 16, post: 32}, {pre: 32, post: 48} ]
  topk_per_class: 20
  eps_percentile: 15
  banks: [GOOD, BAD]

gating_search:
  eps_multiplier: [0.8, 0.9, 1.0, 1.1]
  K_hits: [1, 2, 3]
  lookback_L: [20, 30, 50]           # reserved for advanced scoring
  bad_margin: [0.0, 0.2, 0.5]
  breakout_buffer_atr: [0.0, 0.5, 1.0]
  ppv_prune: 0.6
  min_trades_per_month: 3
  max_dd_R: 8
  ridge_lambda: 1.0
```

Context similarity (ctx_sim)
- Compares each bar's regime fingerprint to its regime center using cosine or RBF on z-scored features (z-stats fit on TRAIN only).
- The same ctx_sim is computed in training and simulation, so learned ridge weights apply 1:1 at runtime.
- Configure via `regime.ctx_sim` in `configs/motifs.yaml`:
  - `type`: "cosine" | "rbf"
  - `gamma`: float (for "rbf")

**Key knobs you’ll tune most often:**
- `candidates.compression` thresholds (your given `0.70 / 0.80` are defaulted here)
- `label.sl_mult, label.tp_mult, time_limit_bars`
- `mining.lengths, topk_per_class`
- `gating_search.*` (coverage vs. precision trade‑off)

---

## 5) Walk‑Forward Protocol

- Rolling folds: **train 3 months → test 1 month** (defaults).  
- Per fold:
  1. Load bars/ticks → compute features → CUSUM → regimes
  2. Generate **candidates**
  3. Label with **tick first‑touch**
  4. **Mine motifs per (side × regime)** on **train only**
  5. **Calibrate gates** (train) → choose `(α, K, Δ, breakout buffer, τ, weights)`
  6. **Simulate** (test) with frozen banks & gates
- Aggregated metrics in `outputs/{SYMBOL}/walkforward_stats.json`

---

## 6) Outputs & What to Inspect

- `fold_i/candidates.parquet` — raw entry candidates (bar_idx, side, prices, ATR)
- `fold_i/events.parquet` — labeled outcomes (+1/0/−1) with realized **R**
- `fold_i/artifacts/banks.pkl` — dict of motif banks per regime/length/class
- `fold_i/train_table.csv` — training rows (one per candidate, with evidence features)
- `fold_i/artifacts/gating.json` — selected hyperparams & learned weights
- `fold_i/trades.csv` — test trades’ **R**
- `fold_i/stats.json` — test KPIs (trades, win_rate, sum_R, avg_R, median_R, max_dd_R)
- `walkforward_stats.json` — cross‑fold summary

---

## 7) Research Methodology (how to make this *profitable & robust*)

- **Optimize E[R] of fired trades**, not accuracy/recall. Precision is king.
- **PPV pruning** enforces *unique wins* by dropping motifs that hit losses.
- **ε multiplier (α)** controls **coverage**. Tight → fewer, cleaner trades.
- **BAD veto + margin (Δ)** kills ambiguous look‑alikes.
- **K‑of‑N** across lengths enforces structural coherence.
- **Regime routing** avoids context drift (squeeze motifs don’t fire in vol‑shock regimes).
- **Timeouts**: treated as **0R** by default (realistic).

> Re‑mine **only** when: you change lengths/channels, quarterly refresh, or drift triggers (distribution shift in features; portfolio E[R] erosion).

---

## 8) Troubleshooting

- **“Bars missing … timestamp/open_time column”**  
  Your CSV is headerless or different order. Use Binance 12‑col headerless or include `timestamp/open_time`. See §2.1.
- **“Ticks missing …”**  
  Ensure `timestamp,price` exist. Names can be `ts/time` and `p/last_price` (mapped in code).
- **Empty candidates / zero trades**  
  Loosen `candidates.compression` and/or `trigger` thresholds; reduce `spacing_bars`; relax gating (e.g., `K=1`, `Δ=0.0`, `α=1.1`).
- **Too many trades / low E[R]**  
  Tighten `α` to `0.9` or `0.8`, increase `Δ`, raise breakout buffer to `1.0` ATR, increase `K`.
- **Performance slow**  
  Start with fewer months or reduce `topk_per_class` and motif lengths. Parquet bars are faster than CSV.

---

## 9) Extending (roadmap hooks in code)

- **True LLR from distance histograms** per motif (KDE/hist) → replace margin proxy.
- **Isotonic calibration** of score→E[R].
- **Mahalanobis context similarity** in scoring (currently placeholder set to 1.0).
- **Per‑side thresholds** and dynamic SL/TP ladders.
- **Execution simulator** (queueing, partial fills) once you’re ready.

---

## 10) Reproducibility

- K‑means seeded (`seed=42`), greedy selection deterministic given data order.
- Artifacts versioned per fold; keep zips of `outputs/` with the exact `configs/motifs.yaml` used.
- For strict determinism, pin numpy/pandas versions (see `requirements.txt`).

---

## 11) Glossary

- **Motif/Shapelet:** short multivariate subsequence (across selected features) that characterizes wins or losses.
- **ε (epsilon):** max z‑norm distance for a motif to “hit” (calibrated by in‑class percentile).
- **PPV:** Precision of motif hits on GOOD vs BAD.
- **LLR (proxy):** Evidence margin `bad_min − good_min` (higher is better).
- **Regime:** Cluster label of market state derived from fingerprint features.

---

## 12) License & Safety

This is **research tooling**. No broker/exchange execution, no financial advice. Validate thoroughly before any live use.

---

## 13) Command Reference

- Full run:  
  `python -m tfbrain --config configs/motifs.yaml --mode walkforward`

- Data months & symbol live in `configs/motifs.yaml`.  
  Add files into `inputs/` and extend `data.months` accordingly.

---

**Build it like a pro:** start tight (few months), confirm the bank/gates make sense, then extend months/symbols. Keep your eye on **E[R]**, **PPV**, and **DD in R**. Memory is power; selection is edge.
