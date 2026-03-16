# Deep Information: Grading Checklist Explanations

This document provides a detailed, technical explanation of every item in the "Grading Checklist" section of the README. It is designed to help you defend your architectural choices in your 6-page project report or oral presentation for your M2 ML in Finance class.

## 1. No Look-Ahead Bias (`pd.merge_asof` on `realtime_start`)
- **The Problem:** In quantitative finance backtests, students often align macroeconomic data (like CPI or NFP) to asset returns using the "observation date" (e.g., March CPI aligned to March 31st returns). However, March CPI is not actually published by the Bureau of Labor Statistics until roughly April 12th. If a model predicts April 1st returns using March CPI data, the model is "seeing the future" and cheating. This is a fatal flaw in any trading or risk model.
- **The Solution:** We query the FRED API for the `realtime_start` data parameter, which tells us exactly when a specific data point was released to the public. We then use Pandas `merge_asof(..., direction='backward')` to align our daily trading dates to the most recently *published* macro data as of that exact trading day. Our model on April 1st only has access to February's CPI (which was published in mid-March).

## 2. No Scaler Leakage
- **The Problem:** Machine learning models require normalized features. A common mistake is to run `StandardScaler().fit_transform()` on the entire dataset (2005-2023) *before* splitting it into training and testing sets. This means the mean and variance of the 2022-2023 test period "leak" into the training set calculations. The model implicitly learns structural information about the test set, invalidating the out-of-sample results.
- **The Solution:** We strictly fit our `StandardScaler` using only indices from the training set (2005-2021). We then use that frozen scaler to `.transform()` the test set data. The model remains completely blind to the statistical distribution of the out-of-sample data.

## 3. Stationarity
- **The Problem:** Financial time series (like stock prices or GDP levels) are generally non-stationary; they trend upwards over time, meaning their mean and variance change. Neural networks struggle massively with non-stationary data, often failing to generalize.
- **The Solution:** We transformed all raw inputs into strictly stationary series:
  - Assets (SPY, TLT, GLD) are converted from prices to daily log-returns.
  - CPI is converted to Year-over-Year (YoY) percentage change.
  - Non-Farm Payrolls (NFP) and Fed Funds Rate are converted using first differences (month-to-month absolute changes).

## 4. Sound Normalizing Flow Math (O(D) Jacobian)
- **The Problem:** A Normalizing Flow uses the change-of-variables formula: `p_Z(g(x)) * |det Jacobian|`. If the mapping function `g(x)` is a standard dense neural network, calculating the determinant of the Jacobian matrix takes $O(D^3)$ time (where D is the number of dimensions/assets). This is computationally intractable for modeling distributions.
- **The Solution:** We use Masked Autoregressive Flows (MAF) based on the MADE (Masked Autoencoder for Distribution Estimation) architecture. By applying binary masks to the neural network weights, we mathematically force the network to be strictly autoregressive. This guarantees the Jacobian matrix is strictly lower-triangular. The determinant of any triangular matrix is simply the product of its diagonal elements, reducing the computational complexity from $O(D^3)$ to a highly efficient $O(D)$.

## 5. TFT Architecture (Temporal Fusion Transformer)
- **The Problem:** LSTMs struggle with long-term dependencies and interpretability, while standard Transformers struggle to isolate relevant features in noisy financial data.
- **The Solution:** We implemented a state-of-the-art Temporal Fusion Transformer (TFT) encoder for the macro data.
  - It uses *Gated Residual Networks (GRNs)* to suppress unnecessary noise and allow gradients to easily skip layers.
  - It includes a *Variable Selection Network* to dynamically weigh the importance of different macro features at each time step, providing critical model interpretability (so we can see *why* the model shifted its distribution).

## 6. Kupiec's POF Test (Formal Validation)
- **The Problem:** Simply calculating the Value-at-Risk (VaR) and stating "the breaches look okay" is insufficient for a rigorous finance project. A 99% VaR model *should* breach exactly 1% of the time, but if we observe a 1.5% breach rate over 500 days, is the model broken, or is it just statistical noise?
- **The Solution:** We implemented the Kupiec Proportion of Failures (POF) test. It is a formal statistical hypothesis test designed to validate VaR models. It calculates a Likelihood Ratio (LR) test statistic and compares it against a $\chi^2$ (Chi-squared) distribution with 1 degree of freedom to generate a p-value. This proves formally whether to reject the null hypothesis (that the model is calibrated correctly) or accept it.

## 7. VaR & Expected Shortfall via Monte Carlo
- **The Problem:** Traditional explicit VaR calculation requires assuming normal distributions (Analytical VaR) or relying solely on historical periods (Historical VaR), both of which fail to capture shifting economic regimes and heavy-tail events.
- **The Solution:** Since our model outputs a fully parameterized, non-normal joint density function `p(X_t | h_t)`, we use a Monte Carlo approach. We sample 10,000 potential return scenarios from the learned distribution for *every single day* in the test set. We then empirically measure the 1% tail to find the Value-at-Risk, and average the losses beyond that point to compute the Expected Shortfall.

## 8. Interpretability
- **The Problem:** Deep learning models in finance are often criticized as "black boxes," making them unacceptable to regulatory bodies and risk managers.
- **The Solution:** The Variable Selection Network embedded in our TFT extracts normalized attention weights assigned to each macroeconomic indicator. We can extract these weights and plot them over the test period, clearly demonstrating to a risk manager exactly which economic variables (e.g., Inflation vs. Volatility) the model is prioritizing when drawing its risk distributions.
