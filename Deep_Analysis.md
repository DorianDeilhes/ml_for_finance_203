# Project Deep Analysis: Macro-Conditional Risk Forecasting

## 1. Modeling Methodologies

In this project, we transition from naive, static risk estimation techniques to highly dynamic, macroeconomic-conditional deep learning architectures. The overarching goal is to accurately forecast the 99% **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** of a multi-asset portfolio.

For students encountering these concepts for the first time:
* **Value at Risk (VaR)** answers the question: *"Under normal market conditions, what is the maximum amount of money we expect to lose on our worst 1% of trading days?"*
* **Expected Shortfall (ES)** answers: *"If we do actually hit that catastrophic 1% day, what is the average amount of money we will lose?"*

To solve this, we tracked the evolution of mathematical finance by building three progressively advanced models.

---

### 1.1 Baseline 1: Parametric Gaussian VaR (The Traditional Standard)

**Intuition & Concept:**
The Gaussian VaR approach is the most widespread baseline used in traditional banking and quantitative finance (Jorion, 2006). The core intuition is aggressively simple: let's assume the daily returns of the stock market behave perfectly like a standard "bell curve" (a Normal Gaussian distribution). 

To calculate risk, the model finds the historical average return of the portfolio (the center of the bell) and the standard deviation (how wide the bell is). By consulting standard statistical tables, the model simply draws a line at the exact 1% tail of that bell curve and declares that as the ultimate limits of our risk.

**Mathematical Formulation:**
Given historical portfolio returns $R_t$, we estimate the sample mean $\hat{\mu}$ and the sample standard deviation $\hat{\sigma}$. The out-of-sample $1-\alpha$ VaR (where $\alpha = 0.01$ for 99% confidence) is computed using the inverse cumulative distribution function (CDF), $\Phi^{-1}$, of the standard normal distribution:

$$ \text{VaR}_\alpha = \hat{\mu} + \Phi^{-1}(\alpha) \hat{\sigma} $$

**Rationale, Limitations & Academic Context:**
* **Why use it?** It is exceptionally fast to compute, requires almost zero computing power, and is incredibly easy to explain to stakeholders.
* **Why does it fail?** It suffers from two fatal flaws that empirically contradict how financial markets work (Cont, 2001):
  1. **Blindness (Constancy):** The model assigns a single, constant risk number to the future. It outputs the exact same risk estimate whether the economy is in a peaceful 10-year bull run or amidst a global pandemic. It completely ignores macroeconomic context.
  2. **Fat Tails (Leptokurtosis):** The real stock market crashes much harder, and far more frequently, than a clean Gaussian bell curve predicts. If stock returns truly followed a Gaussian distribution, a crash like "Black Monday" or the "2008 Financial Crisis" would be mathematically impossible (a 10-to-20 sigma event that should occur once in the lifespan of the universe). Consequently, this model aggressively *underestimates* true market risk.

---

### 1.2 Baseline 2: Gradient Boosting Quantile Regression (GBQuantile)

**Intuition & Concept:**
To fix the "blindness" of the Gaussian model, we transition into Machine Learning. Instead of forcing reality into a rigid bell curve, what if we simply fed the current macroeconomic weather (e.g., today's US Inflation rate, today's Unemployment rate) into an algorithm, and asked it to directly predict the 1% boundary?

We utilize Gradient Boosting models (ensembles of decision trees that iteratively correct each other's mistakes). As the economy shifts, the trees navigate the economic indicators and output a highly flexible, dynamic VaR limit that rises during turmoil and shrinks during peace.

**Mathematical Formulation:**
In standard machine learning, algorithms usually use Mean Squared Error (MSE) to predict the *average* outcome. However, we don't care about the average day; we only care about the worst 1% of days. 

To achieve this, we use **Quantile Regression** via the *Pinball Loss* function (Koenker & Bassett, 1978). The Pinball Loss penalizes the model asymmetrically. If predicting the 1% quantile ($\alpha = 0.01$), the model is penalized 99 times harder for overestimating the return compared to underestimating it. For a prediction $\hat{q}$ and actual portfolio return $y$:

$$ \mathcal{L}_\alpha(y, \hat{q}) = \begin{cases} \alpha(y - \hat{q}) & \text{if } y \ge \hat{q} \\ (1 - \alpha)(\hat{q} - y) & \text{if } y < \hat{q} \end{cases} $$

The trees $F(x)$ are fitted such that $F(x_{t}) \approx \text{VaR}_{\alpha, t}$.

**Rationale, Limitations & Academic Context:**
* **Why use it?** It is *distribution-free*. It makes zero assumptions about whether the market is a bell curve or not. Furthermore, it successfully reacts to non-linear macroeconomic inputs like sudden dips in the Fed Funds rate.
* **Why does it fail?** It operates in a vacuum. It looks only at today's raw numbers and ignores the crucial *momentum* of the economy (e.g., it sees inflation is 5%, but misses the context that inflation rose from 2% to 5% over 6 months). More importantly, it is merely a "point estimator"—it yields a single number (the VaR line). Because the GBQuantile model doesn't understand the full landscape of the tail beyond that line, it is mathematically impossible for it to calculate Expected Shortfall (ES).

---

### 1.3 The Core Architecture: Temporal Fusion Transformer + Masked Autoregressive Flow (TFT-MAF)

**Intuition & Concept:**
To achieve a state-of-the-art solution, we must solve two problems simultaneously: we need to understand the *trajectory* of the macroeconomy, and we need to model the *entire shape* of the market's losses. We do this by combining two massive neural network frameworks:

1. **The Brain (The TFT):** We deploy a Temporal Fusion Transformer (Lim et al., 2021). It ingests the past 63 days of the entire economy. It acts like an advanced reader, utilizing "Attention mechanisms" to ignore daily noise and focus strictly on the historical moments that matter. It compresses this history into a highly dense "context vector" (a mathematical summary of the current economic regime).
2. **The Mathematical Engine (The MAF):** We pass this regime summary to a Masked Autoregressive Flow (Papamakarios et al., 2017; Germain et al., 2015). A Normalizing Flow is a generational leap in statistics. Imagine taking a simple, easy-to-understand shape (like a standard blob of mathematical clay) and applying a series of complex Neural Network twists, stretches, and bends to it until the clay completely reshapes itself to perfectly match the chaotic, multi-dimensional reality of the stock market. 

**Mathematical Formulation:**
1. **Sequence Representation (TFT):** The TFT reads the sequence of macro features $\mathbf{M}_{t-63:t}$ and emits a context conditioning vector $h_t$:
   $$ h_t = \text{TFT}(\mathbf{M}_{t-63:t}) $$
   
2. **Density Estimation (MAF):** The joint distribution of our assets $R_t \in \mathbb{R}^D$ is modeled autoregressively. The MAF applies a sequence of invertible affine transformations $f$ to a base density $p_Z(z)$. 
   
   To decisively defeat the exact flaw that brought down the Gaussian Baseline (fat tails), we **do not** use a Gaussian base. Instead, we use a heavy-tailed multivariate **Student-T distribution**, and we force the neural network to mathematically learn the exact Degrees of Freedom ($\nu$). 
   $$ R_t = f^{-1}(z; h_t) \quad \text{where} \quad z \sim \text{Student-T}(\nu) $$
   
   The model is trained by maximizing the exact log-likelihood via the Change-of-Variables theorem:
   $$ \log p(R_t | h_t) = \log p_Z(f(R_t; h_t)) + \log \left| \det \frac{\partial f}{\partial R_t} \right| $$

**Rationale, Limitations & Academic Context:**
* **Why use it?** It represents the pinnacle of modern risk modeling. The TFT perfectly registers macroeconomic momentum. The MAF, conditioned on this specific economic regime, learns the entire continuous probability landscape of the portfolio. Because we possess the full parametric map, we can run Monte Carlo simulations (simulating 10,000 alternate realities of tomorrow) to effortlessly, seamlessly, and highly accurately extract both VaR and Expected Shortfall.
* **Why is it hard?** Normalizing flows are notoriously data-hungry and slow to train. The autoregressive sampling phase is sequential $(O(D))$ across the assets, requiring substantially longer computational inference times than standard machine learning trees.
