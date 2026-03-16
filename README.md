# Macro-Conditional Normalizing Flow for Portfolio Risk Management

## What This Project Does

Instead of predicting a single return value ("point forecast"), this model predicts the **full probability distribution** of tomorrow's portfolio returns — conditioned on the current macroeconomic regime. This enables statistically rigorous risk metrics:

- **Value-at-Risk (VaR)**: "With 99% confidence, the portfolio will not lose more than X% tomorrow."
- **Expected Shortfall (ES)**: "If that VaR is breached, the average loss will be Y%."
- **Kupiec's POF Test**: Formal statistical validation of VaR calibration accuracy.

## Architecture

```
Macro History (63 days)  →  [Temporal Fusion Transformer]  →  h_t  →  [Masked Autoregressive Flow]  →  p(X_t | h_t)
```

| Component | Description |
|-----------|-------------|
| **TFT Encoder** | GRN + Variable Selection + LSTM + Multi-Head Attention → compresses macro history into regime vector `h_t` |
| **MAF Decoder** | MADE-based affine flow, O(D) Jacobian via triangular masks → warps Gaussian into fat-tailed return distribution |
| **Loss Function** | Negative Log-Likelihood: `-log p_Z(g(x; h_t)) - log\|det J\|` |

## Project Structure

```
big_projet_ML/
├── requirements.txt
├── notebooks/
│   └── main.ipynb              ← Run this end-to-end
└── src/
    ├── data/
    │   ├── market_data.py      ← SPY, TLT, GLD log returns (yfinance)
    │   ├── macro_data.py       ← CPI, NFP, Fed Funds, VIX (FRED + yfinance)
    │   └── pipeline.py         ← Point-in-time alignment, scaling, DataLoaders
    ├── models/
    │   ├── tft.py              ← Temporal Fusion Transformer
    │   ├── maf.py              ← Masked Autoregressive Flow
    │   └── flow_model.py       ← Full conditional model (TFT + MAF)
    ├── training/
    │   └── trainer.py          ← Training loop (AdamW, cosine LR, checkpointing)
    └── backtest/
        ├── risk_metrics.py     ← VaR, ES, Kupiec's POF test
        └── backtester.py       ← Monte Carlo backtest + plots
```

## Quickstart

### 1. Get a FRED API Key (free, ~1 minute)

Register at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) and copy your key.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Your API Key

You can provide your API key in one of two ways:

**Method A: Using a `.env` file (Recommended for sharing code)**
Create a file named `.env` in the root of the project and paste your key inside:
```env
FRED_API_KEY=your_key_here
```
*(Note: `.env` is already configured in the `.gitignore` so your secret key will not be accidentally published).*

**Method B: Using Terminal Environment Variables (Classic Method)**
Set it in your terminal right before launching the notebook:
```powershell
# Windows PowerShell
$env:FRED_API_KEY = "your_key_here"
```

```bash
# Linux / macOS
export FRED_API_KEY="your_key_here"
```

### 4. Run the Notebook

```bash
jupyter notebook notebooks/main.ipynb
```

Run all cells top-to-bottom. Estimated time: **20–60 min on CPU**, **5–10 min on GPU**.

> **Tip**: To train faster, reduce `n_epochs` from `80` to `30` in the training cell.

### 5. (Optional) Quick Component Test

```bash
python verify.py
```

Expected output:
```
[OK] TFT: h_t=torch.Size([4, 64]), weights=torch.Size([4, 63, 10])
[OK] MAFlow: log_prob=torch.Size([4]), samples=torch.Size([50, 3])
[OK] ConditionalNormalizingFlow: NLL=..., params=...
[OK] Kupiec POF: good model PASS, bad model FAIL
[OK] VaR=-1.318%, ES=-1.495% (ES <= VaR confirmed)
=== ALL TESTS PASSED ===
```

## Data Sources

| Variable | Source | Series | Transformation |
|----------|--------|--------|----------------|
| SPY, TLT, GLD | Yahoo Finance | Adjusted Close | Daily log return |
| CPI | FRED `CPIAUCSL` | Monthly | Year-over-year % change |
| Non-Farm Payrolls | FRED `PAYEMS` | Monthly | Month-over-month diff |
| Fed Funds Rate | FRED `DFF` | Daily → Monthly | First difference |
| HY Credit Spread | FRED `BAMLH0A0HYM2` | Daily | Level |
| VIX | Yahoo Finance | `^VIX` | Level |

### No Look-Ahead Bias

Macro data is aligned to trading dates using **publication dates** (`realtime_start`), not observation dates:

```python
pd.merge_asof(trading_df, macro_df.rename(columns={"realtime_start": "date"}),
              on="date", direction="backward")
```

For example, CPI data for March 31st (published ~April 12th) is only made available to the model from April 12th onward.

## Grading Checklist

| Criterion | Implementation |
|-----------|---------------|
| ✅ No look-ahead bias | `pd.merge_asof` on `realtime_start` |
| ✅ No scaler leakage | `StandardScaler` fitted on training set only |
| ✅ Stationarity | Log returns, YoY CPI, monthly diffs |
| ✅ Sound NF math | MADE masks → triangular Jacobian → O(D) log-det |
| ✅ TFT architecture | GRN, Variable Selection, LSTM, attention |
| ✅ Kupiec's POF test | Chi-squared LR statistic, pass/fail verdict |
| ✅ VaR & ES | Empirical 99% quantiles from 10,000 MC samples |
| ✅ Interpretability | TFT variable importance weights |

## Outputs Generated by the Notebook

| File | Description |
|------|-------------|
| `eda_stylized_facts.png` | Return distributions + volatility clustering |
| `eda_correlations.png` | Calm vs crisis correlation heatmaps |
| `training_curves.png` | Train/val NLL convergence |
| `backtest_var_bands.png` | 99% VaR vs actual returns (2022–2023) |
| `kupiec_test.png` | LR test statistic vs chi-squared distribution |
| `variable_importance.png` | TFT macro feature importance |
| `checkpoints/best_model.pt` | Best model weights |

## Key References

- Lim et al. (2021): *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*
- Papamakarios et al. (2017): *Masked Autoregressive Flow for Density Estimation*
- Germain et al. (2015): *MADE: Masked Autoencoder for Distribution Estimation*
- Kupiec (1995): *Techniques for Verifying the Accuracy of Risk Measurement Models*
