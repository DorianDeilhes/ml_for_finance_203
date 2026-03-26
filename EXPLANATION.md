# Main Notebook — Detailed Notes

These notes are meant to be *more detailed than the README* and to follow **exactly** the same structure as [notebooks/main.ipynb](notebooks/main.ipynb).


## Table of Contents (mirrors the notebook)

- [I) Problem and ML Formulation](#problem-formulation)
- [1) Data Pipeline](#data-pipeline)
- [1.2) Exploratory Data Analysis (EDA)](#eda)
- [2) Model Architecture](#model-architecture)
- [3) Training](#training)
- [4) Financial Backtesting](#backtesting)
- [5) Interpretability](#interpretability)
- [6) Summary](#summary)
- [7) Walk-Forward Cross-Validation](#walk-forward-cv)
- [Glossary](#glossary)

## Source-code map

- Data:
  - [src/data/market_data.py](src/data/market_data.py)
  - [src/data/macro_data.py](src/data/macro_data.py)
  - [src/data/pipeline.py](src/data/pipeline.py)
- Models:
  - [src/models/tft.py](src/models/tft.py)
  - [src/models/maf.py](src/models/maf.py)
  - [src/models/flow_model.py](src/models/flow_model.py)
- Training:
  - [src/training/trainer.py](src/training/trainer.py)
- Backtesting:
  - [src/backtest/backtester.py](src/backtest/backtester.py)
  - [src/backtest/risk_metrics.py](src/backtest/risk_metrics.py)
  - [src/backtest/benchmarks.py](src/backtest/benchmarks.py)
  - [src/backtest/benchmark_backtest.py](src/backtest/benchmark_backtest.py)

<a id="problem-formulation"></a>
# I) Problem and ML Formulation (Notebook Intro)

This chapter corresponds to the very first markdown cell in [notebooks/main.ipynb](notebooks/main.ipynb).

## I.1 What problem are we solving?

We are **not** trying to predict a single number like “tomorrow’s SPY return”.

We want a *risk model* that outputs a **full probability distribution** for tomorrow’s *joint* returns of a small portfolio:

- assets: SPY (equities), TLT (US Treasury bonds), GLD (gold)
- output: a distribution over $X_t \in \mathbb{R}^3$

Why? Because risk management is about tail events:
- **VaR** (Value-at-Risk) is a quantile of the loss/return distribution.
- **ES** (Expected Shortfall) is the average loss conditional on being in the tail.

Classical parametric VaR methods often assume:
- returns are Gaussian (thin tails)
- parameters are constant (stationary)
- correlations are stable

But in reality:
- tails are heavy (crashes happen more often than Gaussian)
- volatility clusters
- correlations increase in crises (“everything sells off together”)

The key modeling hypothesis in this project:

> The return distribution is **regime-dependent**, and macro/market indicators help describe the regime.

So we want a model that learns:

$$p(X_t \mid M_{<t})$$

where $M_{<t}$ is the macro/market feature history from the past 63 trading days.

## I.2 Conditional density estimation (what it means)

A *point forecast* model learns $\hat{x}_t = f(M_{<t})$.

A *density model* learns a full conditional distribution:

$$p_\theta(x \mid M_{<t})$$

so we can compute probabilities and quantiles.

In particular, for VaR at level $\alpha=0.01$ (99% VaR):

- Define portfolio return (equal weight by default):

$$r^{(p)}_t = w^\top X_t, \quad w = (1/3,1/3,1/3)$$

- VaR is the $\alpha$-quantile of the portfolio return:

$$\mathrm{VaR}_{\alpha,t} = q_{\alpha}(r^{(p)}_t \mid M_{<t})$$

Interpretation (returns): “With probability $1-\alpha$, tomorrow’s return will be above VaR.”

- ES is the mean conditional tail return:

$$\mathrm{ES}_{\alpha,t} = \mathbb{E}[r^{(p)}_t \mid r^{(p)}_t \le \mathrm{VaR}_{\alpha,t}]$$

## I.3 Training objective: Negative Log-Likelihood (NLL)

If the model defines a valid conditional density, we can train by maximum likelihood:

$$\theta^* = \arg\max_\theta \sum_{t} \log p_\theta(X_t \mid M_{<t})$$

Equivalently, minimize the negative log-likelihood (NLL):

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(X_t \mid M_{<t})$$

This differs from MSE:
- MSE forces a point prediction.
- NLL trains the *shape* of the distribution: mean, variance, skew, heavy tails, correlation.

## I.4 Architecture at a glance

The project uses a **conditional normalizing flow**:

$$[M_{<t}] \xrightarrow{\text{TFT}} h_t \xrightarrow{\text{MAF}} p(X_t \mid h_t)$$

- TFT (Temporal Fusion Transformer) compresses the macro sequence into a fixed vector $h_t$.
- MAF (Masked Autoregressive Flow) uses $h_t$ to condition a flexible invertible mapping that defines the density of $X_t$.

You can locate this in code here:
- encoder: [src/models/tft.py](src/models/tft.py)
- flow: [src/models/maf.py](src/models/maf.py)
- wrapper: [src/models/flow_model.py](src/models/flow_model.py)

## I.5 Design constraints (engineering integrity)

This project is carefully designed to avoid the most common “finance ML” pitfalls:

1. **No look-ahead bias**
   - macro data is aligned by *publication date* (vintages), not observation date.

2. **No leakage in scaling**
   - scalers are fit on the training period only.

3. **Stationarity transformations**
   - prices → log returns
   - CPI level → YoY % change
   - NFP level → monthly difference
   - Fed Funds daily level → monthly, first-difference

4. **Formal validation**
   - VaR calibration tested out-of-sample via Kupiec’s POF test.

<a id="data-pipeline"></a>
# 1) Data Pipeline (Download → Clean → Align → Scale → Sequences)

This chapter corresponds to notebook section:
- “1. Data Pipeline”
- “1.1 Point-in-Time Data Alignment”

Primary source file: [src/data/pipeline.py](src/data/pipeline.py)
Supporting files:
- [src/data/market_data.py](src/data/market_data.py)
- [src/data/macro_data.py](src/data/macro_data.py)

## 1.0 The goal of the data pipeline

We want to build supervised learning samples of the form:

- input $M_{<t}$: a sequence of **63 trading days** of features (macro + volatility)
- target $X_t$: the **asset log returns** (SPY, TLT, GLD) on day $t$

Formally:
- $M_{<t} \in \mathbb{R}^{63 \times F}$
- $X_t \in \mathbb{R}^{D}$ with $D=3$

where $F$ is the number of features.

In this repository, $F$ is **not abstract**: it is literally “all columns in the master dataset except the return columns”. Concretely (see [src/data/pipeline.py](src/data/pipeline.py)):

- **Targets (returns)** (excluded from $M_{<t}$, used for $X_t$): `SPY_ret`, `TLT_ret`, `GLD_ret`
- **Features** (used for $M_{<t}$, i.e. $F=8$):
  - Macro features: `CPI`, `NFP`, `FedFundsRate`, `HYSpread`, `VIX`
  - Market-regime features (realized vol): `SPY_RealVol21d`, `TLT_RealVol21d`, `GLD_RealVol21d`

So, in this project:

$$M_{<t} \in \mathbb{R}^{63 \times 8}, \quad X_t \in \mathbb{R}^{3}.$$

### Why is $X_t$ “3 numbers” if we predict a distribution?

Because a conditional density model is trained on **realized samples**.

- The model outputs a *distribution* $p_\theta(\cdot\mid M_{<t})$ over $\mathbb{R}^3$.
- On each day $t$, we observe exactly **one** realized return vector $x_t = (r^{SPY}_t, r^{TLT}_t, r^{GLD}_t)$.
- Maximum-likelihood training uses that realized vector as a sample and optimizes:

$$\max_\theta \sum_t \log p_\theta(x_t\mid M_{<t}).$$

So the target stored in the dataset is a 3-vector, but the *object we learn* is a full distribution over such 3-vectors.

The pipeline has 5 constraints:

1. **Point-in-time (PIT) correctness**: macro values used on day $t$ must have been published by day $t$.
2. **Stationarity**: transform raw macro/price levels to stable-ish differences/returns.
3. **No leakage**: scalers must be fit on training only.
4. **Time-respecting split**: train/val/test are chronological.
5. **Tensor-ready**: produce PyTorch `DataLoader`s.

---

## 1.1 market_data.py: returns and realized volatility

### What is downloaded?

`download_market_data()` downloads adjusted close prices for the tickers:
- SPY, TLT, GLD

Then it computes **daily log returns**:

$$r_t = \log\left(\frac{P_t}{P_{t-1}}\right)$$

Why log returns?
- time-additive: $\log(P_t/P_{t-k}) = \sum_{i=0}^{k-1} r_{t-i}$
- often more Gaussian-ish than simple returns

### Rolling realized volatility

`compute_rolling_realized_vol(returns, window=21)` computes rolling std:

$$\hat\sigma_t = \mathrm{Std}(r_{t-20:t})$$

Annualize (optional):

$$\hat\sigma^{(ann)}_t = \hat\sigma_t \sqrt{252}$$

These are not "macro" in the economic sense, but they are regime indicators.

---

## 1.2 macro_data.py: macro indicators with vintages

### Key concept: observation date vs publication date

For many macro series:
- the value refers to a month (observation date)
- but it becomes known later (publication date)
- and it may be revised later (multiple releases)

We store two date concepts:

- `observation_date`: “what period does this measurement belong to?”
- `realtime_start`: “when did the public first have access to this number?”

In your code, “download vintages” means: download data with `realtime_start` so later we can align strictly by publication date.

### Which series?

From FRED (via fredapi):
- CPIAUCSL → CPI
- PAYEMS → NFP
- DFF → FedFundsRate
- BAMLH0A0HYM2 → HYSpread

From Yahoo Finance:
- ^VIX → VIX

### Stationarity transforms

Raw macro series are usually nonstationary. So we transform:

**Definition (YoY).** “YoY” means **Year-over-Year** change: compare a value to the same month one year earlier.

- CPI level → YoY % change:
  $$\mathrm{CPI\_YoY}_t = 100 \cdot \left(\frac{\mathrm{CPI}_t}{\mathrm{CPI}_{t-12}} - 1\right)$$

  In the final dataset, the column is still named `CPI`, but it contains this YoY-transformed value.

- NFP level → monthly change:
  $$\Delta \mathrm{NFP}_t = \mathrm{NFP}_t - \mathrm{NFP}_{t-1}$$

  In the final dataset, the column is still named `NFP`, but it contains this month-over-month difference.

- Fed Funds daily level → monthly first-difference:
  - resample to month start (MS)
  - take last value in month
  - difference:

  $$\Delta r^{FF}_t = r^{FF}_t - r^{FF}_{t-1}$$

  In the final dataset, the column is still named `FedFundsRate`, but it contains this monthly first-difference.

HYSpread and VIX are used as levels (already closer to stationary / mean-reverting).

### Why are some transforms “yearly” and others “monthly”?

This is a pragmatic stationarity + interpretability choice, aligned with how each series behaves:

- **CPI** has strong trend and seasonality in levels; YoY removes most seasonality and turns “price level” into “inflation rate”.
- **NFP** is a level series (employment). A **monthly difference** turns it into “jobs added/lost”, which is closer to stationary and easier to interpret as a macro shock.
- **Fed Funds Rate (DFF)** is available daily, but markets often react to *changes* in the policy stance rather than the absolute level. Resampling to monthly reduces high-frequency noise, and differencing emphasizes “tightening vs easing”.
- **HYSpread** and **VIX** are already mean-reverting regime indicators in level form (stress gauges), so differencing is not strictly necessary.

### Important implementation detail: chunking FRED vintage requests

`fred.get_series_all_releases()` has API limits. In [src/data/macro_data.py](src/data/macro_data.py), the function `_download_fred_series_with_vintages` splits the date range into 5-year chunks to avoid hitting FRED’s “max vintage dates” errors.

---

## 1.3 pipeline.py: the orchestration

### Step A: Build the master daily dataset

Function: `build_master_dataset(fred_api_key, start, end)`.

Conceptually, it builds a daily table indexed by trading days, where each row contains:

- returns for SPY/TLT/GLD on day $t$
- macro features **as known on day $t$**
- rolling realized vol features on day $t$

So each row is a single day. This is still a *tabular time series*, not yet sequences.

### Step B: Point-in-time alignment (the core trick)

Function: `_pit_merge_macro(trading_index, macro_series, value_col)`.

We create a trading calendar `trading_index` from market returns (because markets define our day-to-day timeline).

Then, for each trading day $t$, we select the most recent macro value with:

$$\text{publication date} \le t$$

Implementation uses:

- `pd.merge_asof(..., direction='backward')`

Interpretation:
- “Take today’s date and look backward to the last published macro datapoint.”

This is the exact place where look-ahead bias is prevented.

### Step C: Forward-fill macro values

After merging, macro series are sparse (monthly). We forward-fill:

- if CPI was published on April 12, we assume traders use that CPI value on April 12, April 13, ..., until the next CPI release.

### Step D: Drop NaNs

The pipeline drops rows that still contain missing values after forward-fill.

This usually removes:
- early rows before enough macro data exists
- early rows before rolling vol is “warmed up”

### Step E: Train/Val/Test split (chronological)

In `build_pipeline()`:

- train: up to 2016-12-31
- val: 2017-01-01 to 2021-12-31
- test: from 2022-01-01 onward

### Step F: Scaling with no leakage

Two scalers:
- macro scaler: `RobustScaler` (less sensitive to outliers)
- return scaler: `StandardScaler`

More explicitly:

- `StandardScaler`: $z = \frac{x-\mu}{\sigma}$ (center by mean, scale by standard deviation)
- `RobustScaler`: $z = \frac{x-\mathrm{median}(x)}{\mathrm{IQR}(x)}$ where $\mathrm{IQR}=Q_{0.75}-Q_{0.25}$

Why this split?

- Macro and regime features can contain **outliers** (spikes in VIX/spreads), and we don’t want scaling to be dominated by a few crisis points → `RobustScaler`.
- Returns are typically already centered near 0 and are the *objects whose distribution we model* → using `StandardScaler` is a simple, common normalization.

Fit scalers on training only:

- `macro_scaler.fit(master_train[feature_cols])`
- `ret_scaler.fit(master_train[ret_cols])`

Then apply to train/val/test.

### Step G: Build sequences

Function: `build_sequences(master_scaled, seq_len=63)`.

For each day index $t \ge 63$:

- input: $X^{(in)}_t = \text{features}[t-63:t]$ (the last 63 rows, excluding current day)
- target: $y_t = \text{returns}[t]$ (returns on the current day)

This matches the notebook notation $M_{<t}$ (past) → $X_t$ (today).

Outputs:
- `macro_seqs`: shape `(N, 63, F)`
- `asset_returns`: shape `(N, 3)`
- `date_list`: length `N`

Here:

- $F=8$ is the number of feature columns listed in 1.0.
- $D=3$ is the number of assets (SPY/TLT/GLD).
- $N$ is the number of training examples after windowing:

$$ N = (\text{rows in the split}) - 63. $$

This is why each split (train/val/test) loses the first 63 rows: you can’t form a full history window before that.

### Step H: Wrap into PyTorch DataLoaders

The pipeline creates:

- `TensorDataset(torch.tensor(X), torch.tensor(y))`
- `DataLoader(dataset, batch_size=64, shuffle=...)`

So each batch is:
- macro batch: `(batch, 63, F)`
- returns batch: `(batch, 3)`

Terminology recap:

- `batch` (often written $B$) is the mini-batch size (here 64).
- $N$ is the total number of windows/examples in the dataset split.
- The DataLoader iterates over $N$ windows in chunks of size $B$.

This is what the model will consume.

#### What a batch really *means* (shapes + intuition)

When you iterate over `train_loader`, you get something like:

- `X_batch` (the inputs): shape `(64, 63, F)`
  - `64` = batch size, i.e. **64 different days / training examples** sampled from the training period
  - `63` = sequence length (the last 63 trading days of features leading up to each day)
  - `F` = number of features per day
    - in this project, $F$ is typically around 8 (macro features + realized vol features)
    - if you keep only macro indicators (CPI, NFP, VIX, …) then you might see something like $F=5$

  Mental model: “Here are 64 different scenarios; for each one, here are the 63 prior days of the $F$ forces that define the regime.”

- `y_batch` (the targets): shape `(64, 3)`
  - `64` = the same 64 days / examples as in `X_batch`
  - `3` = log-returns for **SPY, TLT, GLD** on the prediction day $t$ (i.e. the day immediately after each 63-day history window)

  Mental model: “For those 64 scenarios, here are the realized returns for the day after the window.”

This is *super important*: each sample is a **window** of past information, and the target is the return vector for the **day right after that window**. Shuffling does not break the internal chronological order *inside* each window.

#### Why does `train_loader` use `shuffle=True`?

In [src/data/pipeline.py](src/data/pipeline.py), the training loader is created with `shuffle=True`.

This does **not** mean we shuffle time *inside* a sequence window. Each individual example still contains the correct ordered history of 63 days.

What shuffling does is randomize the **order of the windows** that the optimizer sees. That matters because:

- If you train strictly chronologically (Monday → Tuesday → Wednesday → …), gradient descent can become biased toward the most recent regime it saw (e.g., late-sample bull market days dominate the recent updates).
- With shuffling, each mini-batch tends to mix regimes (calm years, crisis periods, recoveries), which forces the model to learn patterns that generalize across regimes.

That’s also why we keep `shuffle=False` for validation and test loaders: evaluation is time-ordered, and many backtesting routines want predictions aligned with their true dates.

---

## 1.4 What is the final data used by the model?

At the end, the model uses only the DataLoader batches:

- a tensor `macro_seq` (history features)
- a tensor `returns` (targets)

Everything else (pandas, numpy) is preparation.

The final features are exactly:

- Macro features: `CPI`, `NFP`, `FedFundsRate`, `HYSpread`, `VIX`
- Realized vols: `SPY_RealVol21d`, `TLT_RealVol21d`, `GLD_RealVol21d`

So $F \approx 8$ in the notebook.

### A small “why do we drop rows?” intuition

Even with forward-filling, there are unavoidable warm-up requirements:

- **CPI YoY** needs 12 months of history before the first YoY value exists.
- **Realized vol (21d)** needs a few weeks of returns to stabilize.
- **Sequence windows (63d)** require 63 trading days of history before the first training example exists.

Dropping the initial NaNs is therefore expected and is part of producing a clean, strictly point-in-time dataset.

Targets are:
- `SPY_ret`, `TLT_ret`, `GLD_ret`

<a id="eda"></a>
# 1.bis) Exploratory Data Analysis (EDA)

This chapter mirrors notebook section “1.bis Exploratory Data Analysis”.

The EDA in the notebook is not decorative: it is there to justify *why* we need a conditional heavy-tailed density model.

## 1.bis.1 Stylized facts of returns

The notebook plots, on the training period:

1) **Return distributions vs Normal fit**
- It shows histograms of daily log returns for SPY/TLT/GLD.
- Then it overlays a fitted Normal density.

Key concept: **excess kurtosis**.

- For a perfect Gaussian, excess kurtosis is 0.
- Financial returns often have excess kurtosis > 0.

Meaning: more probability mass in the tails (extreme events) than Gaussian.

This is a big reason to avoid Gaussian VaR.

2) **Volatility clustering**

They compute rolling volatility:

$$\hat\sigma_t = \sqrt{252}\,\mathrm{Std}(r_{t-20:t})$$

You will observe clusters where volatility stays high for weeks/months.

That means risk is time-varying.

## 1.bis.2 Correlation is regime-dependent

The notebook compares correlation matrices:

- “calm” years: 2012–2014
- “stress” years: 2008–2009, 2020

Key insight:

> Cross-asset correlation is not constant. In crises, correlations often increase.

This motivates modeling a **joint distribution** $p(X_t \mid M_{<t})$ rather than three independent marginals.

## 1.bis.3 What EDA implies for modeling choices

- Heavy tails → choose a flexible distributional family (normalizing flow).
- Time-varying volatility and correlations → condition the distribution on macro/market regime.
- Joint modeling → flow dimension is D=3, not D=1.

In this project, these requirements correspond to:
- TFT encoder for regime context: [src/models/tft.py](src/models/tft.py)
- MAF flow for joint density: [src/models/maf.py](src/models/maf.py)
<a id="model-architecture"></a>
# 2) Model Architecture (TFT encoder + MAF flow)

This chapter corresponds to notebook section “2. Model Architecture / 2.1 Instantiate the Conditional Normalizing Flow”.

Primary source files:
- [src/models/flow_model.py](src/models/flow_model.py)
- [src/models/tft.py](src/models/tft.py)
- [src/models/maf.py](src/models/maf.py)

## 2.1 What enters the model? (shapes)

From the data pipeline, each training example is:

- `macro_seq`: tensor of shape `(B, T, F)`
  - $B$ batch size (e.g., 64)
  - $T=63$ sequence length
  - $F$ number of features (≈ 8)

- `x` (returns): tensor of shape `(B, D)`
  - $D=3$ (SPY, TLT, GLD)

The model outputs during training:
- `nll`: scalar loss (mean NLL over batch)
- `var_weights`: tensor `(B, T, F)` interpretability weights

## 2.2 The big idea of a normalizing flow

A normalizing flow defines a complicated density by transforming a simple base random variable.

Let $z$ be a simple base variable with known density $p_Z(z)$ (often Normal). A flow uses an invertible mapping $f$ such that:

$$x = f(z; h), \qquad z = f^{-1}(x; h)$$

Here $h$ is a conditioning context (“macro regime embedding”).

Change of variables formula:

$$\log p_X(x \mid h) = \log p_Z(z) + \log\left|\det\left(\frac{\partial z}{\partial x}\right)\right|$$

Key requirement: we must compute the Jacobian determinant efficiently.

## 2.3 Why autoregressive flows? (MAF)

A Masked Autoregressive Flow (MAF) is designed so that the Jacobian is triangular.

A single MAF layer defines for each dimension $i$:

$$z_i = (x_i - \mu_i(x_{<i}, h))\exp(-\alpha_i(x_{<i}, h))$$

Then:

$$\log\left|\det\left(\frac{\partial z}{\partial x}\right)\right| = -\sum_{i=1}^{D} \alpha_i$$

That is **O(D)**, cheap.

This is implemented in [src/models/maf.py](src/models/maf.py) in `MAFLayer.forward()`.

## 2.4 MADE masks (how the triangular structure is enforced)

We need the network producing $(\alpha,\mu)$ to obey:

$$(\alpha_i,\mu_i) \text{ may depend on } x_1,\dots,x_{i-1} \text{ but not on } x_i, x_{i+1},\dots$$

MADE (Masked Autoencoder for Distribution Estimation) enforces this by multiplying weight matrices by binary masks.

In code:
- `MaskedLinear` holds a `mask` buffer and uses `weight * mask`.
- `_setup_masks()` assigns “degrees” to each unit and builds masks.

Context injection:
- context units are degree 0 → visible to all outputs.

## 2.5 Conditioning on macro regime: the TFT encoder

The flow needs a context vector $h_t$ summarizing the last 63 days of macro features.

TFT produces:

- $h_t \in \mathbb{R}^{d}$ (e.g., $d=128$)
- variable selection weights (interpretable)

Core TFT components in this simplified implementation:

1) **Per-feature embedding**: each scalar feature becomes a $d$-dim vector.

2) **Variable Selection Network (VSN)**
- uses GRNs to transform each variable
- uses a “selection GRN + softmax” to produce weights over variables

3) **LSTM encoder**
- captures sequential patterns and persistence

4) **Multihead self-attention**
- allows the model to focus on the most relevant time steps

5) **Final GRN + last time step extraction**
- takes the last time step as the summary context vector $h_t$.

All of this is in [src/models/tft.py](src/models/tft.py).

## 2.6 End-to-end wrapper: ConditionalNormalizingFlow

`ConditionalNormalizingFlow` in [src/models/flow_model.py](src/models/flow_model.py) simply composes:

- `self.tft(macro_seq) -> (h_t, weights)`
- `self.flow.log_prob(x, context=h_t) -> log p(x|h_t)`

Training loss:

$$\mathrm{NLL} = -\frac{1}{B}\sum_{b=1}^{B} \log p(x^{(b)} \mid h^{(b)})$$

## 2.7 Student-t base distribution (heavy tails)

In [src/models/maf.py](src/models/maf.py), `MAFlow` uses a **Student-t** base distribution rather than Normal:

- learnable degrees of freedom $\nu > 2$

Student-t has heavier tails than Normal. Smaller $\nu$ → heavier tails.

This is a modeling choice aligned with financial returns.

## 2.8 Sanity check in the notebook

The notebook does a forward pass and prints an “initial NLL”.

For a baseline intuition: if returns were standard Normal and $D=3$, expected negative log-likelihood per sample is roughly

$$\frac{D}{2}\log(2\pi) + \frac{1}{2}\mathbb{E}[\|x\|^2]$$

The notebook prints the term $\frac{D}{2}\log(2\pi)$ as a rough anchor.

---

## 2.9 What you should be able to explain

1) State the change-of-variables formula.
2) Explain why triangular Jacobian → logdet is sum of diagonal logs.
3) Explain why MAF sampling is slower than training (inverse is sequential).
4) Explain what the TFT variable weights mean.
<a id="training"></a>
# 3) Training (Objective, optimizer, stability)

This chapter mirrors notebook section “3. Training” including:
- “3.1 Naive Benchmarks — Fitting”
- “3.2 TFT + MAF — Training”

Primary source file: [src/training/trainer.py](src/training/trainer.py)
Supporting model: [src/models/flow_model.py](src/models/flow_model.py)

## 3.1 Benchmarks (why they exist)

The notebook compares 3 approaches:

1) Gaussian parametric VaR (constant)
- Assumes portfolio returns are iid Normal.
- Produces a constant VaR for every day.

2) Gradient boosting quantile regression
- Uses macro features to predict only the 1% quantile.
- Regime-aware (in a limited sense), but **not a full distribution**.

3) TFT + MAF conditional flow
- Learns full conditional density.
- Can produce VaR, ES, scenario generation.

Code references:
- Gaussian + GB: [src/backtest/benchmarks.py](src/backtest/benchmarks.py)

## 3.2 Training objective (NLL for conditional flow)

The model outputs a log-likelihood:

$$\log p_\theta(x \mid h)$$

Loss is the negative mean log-likelihood:

$$\mathcal{L}(\theta) = -\frac{1}{B}\sum_{b=1}^{B} \log p_\theta(x^{(b)} \mid h^{(b)})$$

Expanding flow log-probability:

$$\log p(x\mid h) = \log p_Z(g(x;h)) + \log |\det J_{g}(x;h)|$$

where $g$ is the data→noise mapping and $J_g$ is its Jacobian.

In MAF layers, $\log |\det J|$ is a sum of $-\alpha_i$ terms (triangular Jacobian).

## 3.3 What a training iteration does

One step on a batch $(M_{<t}, X_t)$:

1) TFT computes $h_t = \mathrm{TFT}(M_{<t})$.
2) Flow computes $\log p(X_t\mid h_t)$.
3) Loss is `nll = -log_prob.mean()`.
4) Backprop updates all parameters (TFT + flow).

This is implemented in `Trainer._run_epoch()`.

## 3.4 Optimization choices (why these)

In [src/training/trainer.py](src/training/trainer.py):

- Optimizer: AdamW
  - Adam with decoupled weight decay.
  - Often more stable for Transformers than vanilla SGD.

- Scheduler: CosineAnnealingLR + warmup
  - Warmup: linearly increases LR for the first few epochs.
  - Cosine decay: slowly reduces LR to help convergence.

- Gradient clipping
  - `clip_grad_norm_` prevents exploding gradients.

- Early stopping
  - If validation NLL does not improve for `patience` epochs, stop.

- Checkpointing
  - Saves the best validation NLL model to `checkpoints/best_model.pt`.

## 3.5 What is “validation NLL” telling you?

NLL is not accuracy, it is *calibration of the density*.

Lower NLL usually means:
- the model assigns higher probability to the realized returns
- the distribution better matches the data

But beware:
- lower NLL doesn’t automatically guarantee good VaR calibration.
- that’s why we still run **Kupiec POF** out-of-sample.

## 3.6 “Why training flows can be unstable” (important intuition)

Normalizing flows are powerful because they model densities exactly.
But that also means:
- they can overfit tails if not regularized
- they can produce numerical issues if scales (alphas) explode

Stability tools in this repo:
- bounded alpha in MADE: `alpha = 3*tanh(alpha/3)` in [src/models/maf.py](src/models/maf.py)
- optional batch norm layers between flow steps
- robust scaling of macro features (RobustScaler)

## 3.7 What you should be able to explain

1) Why NLL is the correct objective for distribution modeling.
2) Why early stopping is done on validation NLL.
3) Why we clip gradients.
4) Why AdamW + warmup is common for attention-based encoders.
<a id="backtesting"></a>
# 4) Financial Backtesting (VaR/ES + Kupiec test)

This chapter mirrors notebook section “4. Financial Backtesting” with subsections:
- 4.1 Gaussian VaR
- 4.2 GB Quantile VaR
- 4.3 TFT + MAF Monte Carlo backtest

Primary source files:
- [src/backtest/backtester.py](src/backtest/backtester.py)
- [src/backtest/risk_metrics.py](src/backtest/risk_metrics.py)
- [src/backtest/benchmarks.py](src/backtest/benchmarks.py)
- [src/backtest/benchmark_backtest.py](src/backtest/benchmark_backtest.py)

## 4.0 What is a VaR backtest?

Each day in the test period, we produce a predicted VaR (at level 99%):

$$\mathrm{VaR}_{0.01,t}$$

Then we compare it to the realized portfolio return $r^{(p)}_t$.

A **breach** occurs if:

$$r^{(p)}_t < \mathrm{VaR}_{0.01,t}$$

Over $n$ test days, a perfectly calibrated 99% VaR should breach about $\alpha n$ times with $\alpha=0.01$.

## 4.1 Benchmark 1: Gaussian VaR

File: [src/backtest/benchmarks.py](src/backtest/benchmarks.py)

Steps:
1) Compute equal-weighted portfolio returns on the training set.
2) Fit $\mu$ and $\sigma$.
3) VaR is constant:

$$\mathrm{VaR}_{\alpha} = \mu + \sigma\Phi^{-1}(\alpha)$$

This ignores regimes entirely.

Notebook alignment detail:
- notebook uses unscaled returns for this benchmark.

## 4.2 Benchmark 2: Gradient Boosting quantile VaR

File: [src/backtest/benchmarks.py](src/backtest/benchmarks.py)

This uses a regression model with quantile loss to predict the 1% quantile.

Important modeling limitation:
- it predicts only **one number** (VaR), not a full distribution.
- you don’t naturally get ES or scenarios.

The notebook uses as input features:
- the *last time step* in the macro window (not the whole sequence).

That is a strong simplifying assumption.

## 4.3 Main model backtest: TFT + MAF via Monte Carlo

File: [src/backtest/backtester.py](src/backtest/backtester.py)

### 4.3.1 What does the model produce each day?

For each day $t$ in the test set, we have a macro sequence $M_{<t}$.

The model defines a conditional density $p(X_t\mid M_{<t})$.

But VaR is a quantile, which is hard to compute analytically.

So we use **Monte Carlo**:

1) Compute context:

$$h_t = \mathrm{TFT}(M_{<t})$$

2) Draw $S$ samples:

$$X_t^{(s)} \sim p(X_t\mid h_t), \quad s=1..S$$

3) Convert to portfolio returns:

$$r_t^{(p,s)} = w^\top X_t^{(s)}$$

4) Estimate VaR as empirical quantile:

$$\widehat{\mathrm{VaR}}_{\alpha,t} = \mathrm{Quantile}_\alpha(\{r_t^{(p,s)}\}_{s=1}^S)$$

5) Estimate ES as mean below VaR:

$$\widehat{\mathrm{ES}}_{\alpha,t} = \frac{1}{|\mathcal{T}|}\sum_{s \in \mathcal{T}} r_t^{(p,s)}\quad\text{where }\mathcal{T}=\{s: r_t^{(p,s)} \le \widehat{\mathrm{VaR}}_{\alpha,t}\}$$

Functions:
- `compute_var`, `compute_es`: [src/backtest/risk_metrics.py](src/backtest/risk_metrics.py)

### 4.3.2 Why does the backtester precompute h_t?

In [src/backtest/backtester.py](src/backtest/backtester.py), `run()` does:

- loop over `test_loader` once to compute all contexts $h_t$
- then loop day-by-day to sample from the flow

This matters because:
- TFT forward pass is relatively expensive
- doing it inside the Monte Carlo loop would be extremely slow

### 4.3.3 Scaling: why inverse_transform matters

The model is trained on *scaled returns*.

But risk metrics must be computed in real return units.

So backtester does:
- draw samples in scaled space
- then unscale: `ret_scaler.inverse_transform(samples)`

Same for actual returns.

## 4.4 Kupiec’s Proportion of Failures (POF) test

File: [src/backtest/risk_metrics.py](src/backtest/risk_metrics.py)

Kupiec tests whether the breach frequency matches the intended breach probability $\alpha$.

- $n$ observations
- $x$ breaches
- observed breach rate: $\hat{p} = x/n$

Likelihood ratio statistic:

$$\mathrm{LR} = -2\left[ \log L(p=\alpha) - \log L(p=\hat{p}) \right]$$

where the Bernoulli likelihood is:

$$\log L(p) = x\log p + (n-x)\log(1-p)$$

So:

$$\mathrm{LR} = -2\Big[x\log\alpha + (n-x)\log(1-\alpha) - x\log\hat{p} - (n-x)\log(1-\hat{p})\Big]$$

Under $H_0$ (correct calibration), $\mathrm{LR} \sim \chi^2(1)$ approximately.

Decision rule:
- reject if p-value < 0.05 (or LR > 3.841)

Interpretation from the code:
- too many breaches → VaR too optimistic → underestimates tail risk
- too few breaches → VaR too conservative → overestimates tail risk

## 4.5 What the backtest produces

The backtester returns a DataFrame indexed by date with columns:
- `actual_port_return`
- `var_99`, `es_99`
- `breach` (0/1)
- percent versions `var_99_pct`, `es_99_pct`
- plus per-asset actual returns

Plots:
- VaR bands plot (with breach markers)
- histogram distributions for selected days (optional)

## 4.6 What you should be able to explain

1) Why VaR needs a distribution, not a point forecast.
2) Why Monte Carlo approximates quantiles.
3) Why Kupiec is a calibration test (frequency), not a “profit” metric.
<a id="interpretability"></a>
# 5) Interpretability: TFT Variable Importance

This chapter mirrors notebook section “5. Model Interpretability: Variable Importance (TFT)”.

Primary implementation:
- variable selection weights produced by the TFT: [src/models/tft.py](src/models/tft.py)
- helper method: `ConditionalNormalizingFlow.get_variable_importance`: [src/models/flow_model.py](src/models/flow_model.py)

## 5.1 What are we interpreting?

The model is:

$$M_{<t} \to h_t \to p(X_t\mid h_t)$$

The flow part (MAF) is powerful but not naturally interpretable.

The TFT part includes a Variable Selection Network (VSN) that outputs **weights** per feature per time step.

These weights are a form of learned “attention over variables”.

## 5.2 Where do weights come from?

In [src/models/tft.py](src/models/tft.py):

- each feature is embedded into a vector
- the VSN computes:
  - a transformed representation per variable
  - a selection score per variable
- softmax turns scores into weights that sum to 1 across variables.

So for each sample (batch element) and each time step:

$$w_{t,i} \ge 0, \quad \sum_{i=1}^{F} w_{t,i} = 1$$

Interpretation:
- if $w_{t,i}$ is large, feature $i$ is important at time $t$ for building the regime representation.

## 5.3 How the notebook computes importance

The notebook loops over `test_loader` and calls:

- `model.get_variable_importance(macro_seq)`

which returns an average over:
- batch dimension
- time dimension

So output is:
- a vector of length $F$ (one scalar per feature)

This is plotted as a horizontal bar chart.

## 5.4 Cautions (interpretability is not causality)

Important: these weights are **not causal effects**.

They tell you:
- which features the model used to compress information into $h_t$

They do not prove:
- “CPI causes returns”

To reason about causality you would need an entirely different setup.

## 5.5 What you should be able to explain

1) Why weights sum to 1 (softmax).
2) Why “importance” is model-dependent.
3) Why interpretability here is about representation, not causality.
<a id="summary"></a>
# 6) Summary (What to remember)

This chapter mirrors notebook section “6. Summary”.

## 6.1 What the project achieved

- Built a point-in-time (PIT) dataset combining:
  - daily asset returns (SPY, TLT, GLD)
  - macro and market regime indicators (CPI, NFP, Fed Funds diff, HY spread, VIX)
  - rolling realized volatilities

- Trained a conditional density model:
  - TFT encodes macro history to a regime vector $h_t$
  - MAF defines a flexible conditional distribution of returns

- Produced daily out-of-sample risk forecasts:
  - 99% VaR and 99% ES

- Evaluated calibration statistically:
  - Kupiec POF test on breach frequency

## 6.2 What is the “core intellectual contribution” here?

The *conceptual leap* is:

> Predicting *distributions conditioned on regimes*, rather than point forecasts.

The *engineering leap* is:

> Point-in-time alignment using publication dates (vintages) to prevent look-ahead bias.

The *quantitative validation* is:

> Kupiec’s POF test to test VaR calibration out-of-sample.

## 6.3 What to verify if results look suspicious

- Did you set `FRED_API_KEY` correctly?
- Did the pipeline use publication dates (`realtime_start`) and `merge_asof` backward?
- Are scalers fit on training only?
- Are your `test_loader` batches unshuffled?

## 6.4 What you should be able to reproduce offline

- Implement a PIT merge for one macro variable.
- Implement log returns and rolling volatility.
- Derive the flow change-of-variables log-likelihood.
- Implement VaR and ES from Monte Carlo samples.
- Implement Kupiec POF test.
<a id="walk-forward-cv"></a>
# 7) Walk-Forward Cross-Validation (Expanding window + warm start)

This chapter mirrors notebook section “7. Walk-Forward Cross-Validation”.

Primary implementation: `build_walk_forward_pipeline()` in [src/data/pipeline.py](src/data/pipeline.py).

## 7.1 Why walk-forward CV for time series?

Standard k-fold cross-validation randomly shuffles data. That breaks time order and leaks future information.

In time series, a valid evaluation must be chronological:
- train on the past
- validate/test on the future

Walk-forward does exactly that, repeatedly.

## 7.2 Expanding window vs rolling window

Two common schemes:

1) Rolling window
- Always train on last N years.
- Old data is dropped.

2) Expanding window (used here)
- Train set grows over time.
- You keep old regimes (e.g., 2008 crisis) in the training set forever.

Why choose expanding here?
- deep models and flows are data-hungry
- you want the model to see multiple regimes

## 7.3 The fold definition in this repo

In [src/data/pipeline.py](src/data/pipeline.py), `build_walk_forward_pipeline` generates folds like:

- initial train end year: 2016
- validation: next 1 year
- test: next 1 year
- then move forward by `test_years` (1 year)

So roughly:
- fold 1: train ≤ 2016, val 2017, test 2018
- fold 2: train ≤ 2017, val 2018, test 2019
- ...

(Exact depends on `start_year`, `end_year`, etc.)

## 7.4 Warm start vs cold start

Cold start:
- reinitialize model weights every fold.

Warm start (used in notebook):
- keep the model weights from previous fold.
- continue training (“fine-tune”) on expanded training data.

Why warm start helps:
- much faster
- preserves learned representations of macro regimes

But be careful:
- the optimizer state can cause weird dynamics.
- in the notebook, the trainer resets scheduler and LR (`reset_for_new_fold`).

## 7.5 Scaling within folds

Each fold fits scalers on the fold’s training set only.

That preserves “no leakage” property at every fold.

## 7.6 Output of the walk-forward generator

Each yield gives:

- train_loader
- val_loader
- test_loader
- ret_scaler (for inverse transforming returns)
- fold_info dict (dates, feature count, etc.)

The notebook then:
- trains (or warm-starts) a model
- runs a backtest on the fold’s test period
- concatenates all fold test results into one long out-of-sample record

## 7.7 What you should be able to explain

1) Why random splits are invalid for financial time series.
2) Why expanding windows are often preferred for data-hungry deep models.
3) What warm start changes (and why resetting LR scheduler matters).
<a id="glossary"></a>
# Glossary (terms you must own)

Use this as your offline dictionary.

## Autoregressive
A factorization where dimension $i$ depends only on previous dimensions. In flows, it creates a triangular Jacobian.

## Batch
A group of samples processed together in one forward/backward pass. In PyTorch, shape often begins with `(batch_size, ...)`.

## Change of variables (flows)
If $x=f(z)$ invertible, then:

$$\log p_X(x)=\log p_Z(f^{-1}(x)) + \log\left|\det \frac{\partial f^{-1}}{\partial x}\right|$$

## Conditional density estimation
Learning $p(x\mid c)$ for a context $c$. Here $c=M_{<t}$ or its embedding $h_t$.

## Data leakage
Using future information in training, even indirectly (e.g., fitting a scaler on train+test).

## Expected Shortfall (ES)
Mean of tail losses beyond VaR:

$$\mathrm{ES}_\alpha = \mathbb{E}[R \mid R \le \mathrm{VaR}_\alpha]$$

## Look-ahead bias
Using information that was not available at the decision time (e.g., using March CPI on April 1 when it’s published April 12).

## MAF (Masked Autoregressive Flow)
A normalizing flow where each layer uses an autoregressive network (MADE) to produce affine parameters.

## MADE
Masked Autoencoder for Distribution Estimation. Enforces autoregressive dependency by masking weights.

## PIT (Point-in-Time)
A dataset that reflects the information actually available on each historical date.

## RobustScaler
Scaling using median and IQR, less sensitive to outliers than StandardScaler.

## Stationarity
A process whose statistical properties (mean/variance/distribution) are stable over time. Many ML methods assume approximate stationarity.

## TFT (Temporal Fusion Transformer)
A sequence model combining variable selection networks, LSTM, attention, and gated residual networks. Here used as regime encoder.

## Value-at-Risk (VaR)
Quantile of return/loss distribution. For returns:

$$\mathrm{VaR}_{\alpha} = q_{\alpha}(R)$$

At 99% VaR, $\alpha=0.01$.

## Vintage (macro data)
A particular historical release of a macro datapoint. “Downloading vintages” means keeping publication date (`realtime_start`) and using the first release to avoid revision leakage.
