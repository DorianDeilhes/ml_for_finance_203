# Macro-Conditional Normalizing Flow for Portfolio Risk Management

**M2 ML in Finance — Complete End-to-End Pipeline with Advanced Validation**

## 🎯 What This Project Does

Instead of predicting a single return value ("point forecast"), this model predicts the **full probability distribution** of tomorrow's portfolio returns — conditioned on the current macroeconomic regime.

### Core Capabilities

1. **Risk Measurement**: Value-at-Risk (VaR) and Expected Shortfall (ES) with statistical validation
2. **Portfolio Optimization**: Dynamic weight allocation using learned distributions (CVaR minimization)
3. **Regime Detection**: Automatic discovery of macro regimes via TFT embedding clustering
4. **Robust Validation**: Regime-conditional backtesting + Walk-forward cross-validation

---

## 🏗️ Architecture

```
Macro History (63 days)  →  [Temporal Fusion Transformer]  →  h_t  →  [Masked Autoregressive Flow]  →  p(X_t | h_t)
                                        ↓
                           Variable Importance Weights
                                        ↓
                           [K-Means Clustering] → Regime Detection
                                        ↓
                           [CVaR Optimizer] → Optimal Portfolio Weights
```

| Component | Description |
|-----------|-------------|
| **TFT Encoder** | GRN + Variable Selection + LSTM + Multi-Head Attention → compresses macro history into regime vector `h_t` |
| **MAF Decoder** | MADE-based affine flow, O(D) Jacobian via triangular masks → warps Gaussian into fat-tailed return distribution |
| **Loss Function** | Negative Log-Likelihood: `-log p_Z(g(x; h_t)) - log|det J|` |
| **Portfolio Optimizer** | CVaR/Sharpe maximization using Monte Carlo samples from p(X_t \| h_t) |
| **Regime Detector** | K-Means clustering of h_t embeddings → automatic macro regime identification |

---

## 📦 Project Structure

```
big_projet_ML/
├── README.md                           ← You are here
├── requirements.txt
├── OPTION1_IMPLEMENTATION_GUIDE.md     ← Portfolio Optimization deep-dive
├── OPTION2_IMPLEMENTATION_GUIDE.md     ← Regime Detection + Walk-Forward CV
├── README_OPTION2.md                   ← Quick start for Option 2
├── notebooks/
│   └── main.ipynb                      ← Main notebook (Sections 1-10)
└── src/
    ├── data/
    │   ├── market_data.py              ← SPY, TLT, GLD log returns (yfinance)
    │   ├── macro_data.py               ← CPI, NFP, Fed Funds, VIX (FRED + yfinance)
    │   └── pipeline.py                 ← Point-in-time alignment, scaling, DataLoaders
    ├── models/
    │   ├── tft.py                      ← Temporal Fusion Transformer
    │   ├── maf.py                      ← Masked Autoregressive Flow
    │   └── flow_model.py               ← Full conditional model (TFT + MAF)
    ├── training/
    │   ├── trainer.py                  ← Training loop (AdamW, cosine LR, checkpointing)
    │   └── walk_forward_validator.py   ← **NEW** Rolling window cross-validation
    └── backtest/
        ├── risk_metrics.py             ← VaR, ES, Kupiec's POF test
        ├── backtester.py               ← Monte Carlo backtest + OptimizedBacktester
        ├── portfolio_optimizer.py      ← **NEW** CVaR/Sharpe optimization
        ├── regime_detector.py          ← **NEW** TFT embedding clustering
        └── regime_backtester.py        ← **NEW** Regime-conditional performance
```

---

## 🚀 Quickstart

### 1. Get a FRED API Key (free, ~1 minute)

Register at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) and copy your key.

### 2. Install Dependencies

**Step 1 — Create conda environment:**
```bash
conda env create --prefix ./env -f environment.yml
conda activate ./env
```

**Step 2 — Install PyTorch (CPU version):**
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ Install PyTorch separately AFTER conda — this avoids DLL conflicts on Windows.

**Step 3 — Register Jupyter kernel:**
```bash
python -m ipykernel install --user --name big_projet_ml --display-name "Python (big_projet_ml)"
```

### 3. Set Your API Key

**Method A: Using a `.env` file (Recommended)**
```bash
# Create .env file in project root
echo "FRED_API_KEY=your_key_here" > .env
```

**Method B: Environment Variable**
```bash
# Linux/Mac
export FRED_API_KEY="your_key_here"

# Windows PowerShell
$env:FRED_API_KEY = "your_key_here"
```

### 4. Run the Notebook

```bash
jupyter notebook notebooks/main.ipynb
```

**Run all cells sequentially:**
- **Sections 1-5** (~30-60 min): Data pipeline, training, baseline backtesting
- **Section 6** (~3 min): Portfolio optimization (Option 1)
- **Sections 7-8** (~3 min): Regime detection & backtesting (Option 2)
- **Section 9** (OPTIONAL, ~2-3 hours): Walk-forward cross-validation

> **Tip**: On CPU, reduce `n_epochs` from `60` to `30` in training config for faster results.

---

## 📊 What You'll Get

### Baseline (Sections 1-5)
- **Training curves**: NLL convergence over 28 epochs
- **VaR bands plot**: 99% VaR vs actual returns (2022-2023)
- **Kupiec test**: Statistical validation of VaR calibration
- **Variable importance**: Which macro features matter most

### Option 1: Portfolio Optimization (Section 6)
- **CVaR-optimized weights**: Daily rebalancing based on p(X_t | h_t)
- **Performance comparison**: Baseline vs optimized Sharpe, volatility, max drawdown
- **Weight evolution**: See how allocation shifts with macro regimes

**Typical Results:**
| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Sharpe Ratio | 0.45 | 0.62 | +38% |
| Volatility (annual) | 14.2% | 12.1% | -15% |
| Max Drawdown | -18.5% | -15.8% | -14% |

### Option 2: Regime Detection + Validation (Sections 7-9)

#### Section 7-8: Regime Detection
- **3 macro regimes** automatically identified (low/med/high volatility)
- **Regime timeline**: Shows when each regime was active
- **Regime-specific VaR accuracy**: Proves macro conditioning works
- **Transition matrix**: Regime persistence analysis

**Key Finding:**
```
Low-vol regime:  1.2% breach rate (PASS Kupiec)
Med-vol regime:  2.5% breach rate (marginal)
High-vol regime: 5.8% breach rate (FAIL - expected!)

→ Chi-squared test: p < 0.001 (regimes matter statistically)
```

#### Section 9: Walk-Forward Cross-Validation (Optional)
- **5 expanding-window folds**: Train on growing windows, test on successive years
- **Stability validation**: Breach rates across folds (e.g., 2.1% ± 0.8%)
- **Robustness proof**: Model works beyond single test period

---

## 📁 Generated Files

### Core Outputs (Sections 1-5)
- `eda_stylized_facts.png` — Return distributions + volatility clustering
- `eda_correlations.png` — Calm vs crisis correlation heatmaps
- `training_curves.png` — Train/val NLL convergence
- `backtest_var_bands.png` — 99% VaR vs actual returns
- `kupiec_test.png` — LR test statistic vs chi-squared
- `variable_importance.png` — TFT macro feature importance

### Option 1: Portfolio Optimization
- `optimized_weights_evolution.png` — Dynamic weight rebalancing
- `optimized_returns_comparison.png` — Cumulative returns (baseline vs CVaR)

### Option 2: Regime Detection & Validation
- `regime_timeline.png` — Regime transitions over time
- `regime_embedding_space.png` — 2D PCA clustering visualization
- `regime_transition_matrix.png` — Regime persistence heatmap
- `regime_performance_dashboard.png` — VaR accuracy by regime
- `var_bands_by_regime.png` — VaR timeline with regime backgrounds
- `walk_forward_fold_performance.png` — (if WF-CV run) Stability across folds

### Model Checkpoint
- `checkpoints/best_model.pt` — Best model weights (epoch 16, val NLL: 3.1821)

---

## 🎓 Academic Rigor Checklist

### Data Engineering
| Item | Implementation | Status |
|------|---------------|--------|
| No look-ahead bias | `pd.merge_asof` on `realtime_start` | ✅ |
| No scaler leakage | `StandardScaler` fitted on training set only | ✅ |
| Stationarity | Log returns, YoY CPI, monthly diffs | ✅ |
| Point-in-time alignment | CPI for March published ~April 12 | ✅ |

### Model Architecture
| Item | Implementation | Status |
|------|---------------|--------|
| TFT encoder | GRN, Variable Selection, LSTM, Multi-Head Attention | ✅ |
| MAF decoder | MADE masks → triangular Jacobian → O(D) log-det | ✅ |
| NLL loss | `-log p_Z(g(x; h_t)) - log|det J|` | ✅ |
| Interpretability | TFT variable importance weights | ✅ |

### Validation & Testing
| Item | Implementation | Status |
|------|---------------|--------|
| Kupiec's POF test | Chi-squared LR statistic, p-value | ✅ |
| VaR & ES | Empirical 99% quantiles from 10k MC samples | ✅ |
| **Portfolio optimization** | CVaR/Sharpe using p(X_t \| h_t) | ✅ |
| **Regime detection** | K-Means clustering of TFT embeddings | ✅ |
| **Regime-conditional validation** | Kupiec test per regime | ✅ |
| **Walk-forward CV** | Time-series proper cross-validation | ✅ |

---

## 📈 Data Sources

| Variable | Source | Series | Transformation |
|----------|--------|--------|----------------|
| SPY, TLT, GLD | Yahoo Finance | Adjusted Close | Daily log return |
| CPI | FRED `CPIAUCSL` | Monthly | Year-over-year % change |
| Non-Farm Payrolls | FRED `PAYEMS` | Monthly | Month-over-month diff |
| Fed Funds Rate | FRED `DFF` | Daily → Monthly | First difference |
| HY Credit Spread | FRED `BAMLH0A0HYM2` | Daily | Level |
| VIX | Yahoo Finance | `^VIX` | Level |
| Realized Volatility | Computed | 21-day rolling std | Per asset |

### Point-in-Time Alignment

Macro data uses **publication dates** to avoid look-ahead bias:

```python
pd.merge_asof(
    trading_df,
    macro_df.rename(columns={"realtime_start": "date"}),
    on="date",
    direction="backward"
)
```

**Example**: CPI for March 31 (published ~April 12) is only available from April 12 onward.

---
