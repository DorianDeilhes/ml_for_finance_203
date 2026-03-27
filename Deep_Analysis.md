# Project Deep Analysis: Macro-Conditional Risk Forecasting

## 0. Glossary of Key Terms (The Jargon Buster)

Before diving into the complex models, it's crucial to understand the fundamental concepts we are trying to predict and the phenomena we must navigate.

*   **Value-at-Risk (VaR):** The threshold of pain. It answers: *"Under normal market conditions, what is the maximum amount of money we expect to lose on our worst 1% of trading days?"* If our 99% VaR is -3%, we are 99% confident our portfolio won't lose more than 3% in a single day.
*   **Expected Shortfall (ES) / Conditional VaR:** The reality of the catastrophe. It answers: *"If we do actually cross the VaR threshold and experience a top 1% worst day, what will the average loss be?"* It measures the "tail risk" that VaR ignores.
*   **Fat Tails (Leptokurtosis):** In a standard "Gaussian" bell curve, extreme events (like a market crash) represent a minuscule fraction of a percent of probability (essentially impossible). In the real stock market, extreme events happen much more frequently. The "tails" of the distribution hold more probability mass; they are "fat".
*   **Volatility Clustering:** The financial phenomenon where calm markets tend to stay calm, and turbulent markets tend to stay turbulent. It proves that market risk is dynamic, not a static, eternal constant.
*   **Normalizing Flow:** An advanced mathematical framework in Deep Learning. It learns to transform a simple, easy-to-understand probability distribution (like a basic bell curve) into a highly complex, multi-dimensional distribution that perfectly matches chaotic real-world data (like the stock market).
*   **Autoregressive Model:** A model that predicts a value based on the previous values in a sequence. In the context of predicting multiple assets, evaluating the risk of Asset B is conditioned on what happened to Asset A. 

---

## 1. Modeling Methodologies

In this project, we transition from naive, static risk estimation techniques to highly dynamic, macroeconomic-conditional deep learning architectures. The overarching goal is to accurately forecast the 99% VaR and Expected Shortfall of a multi-asset portfolio.

We solved this by tracking the evolution of quantitative finance through three progressively advanced models.

---

### 1.1 Baseline 1: Parametric Gaussian VaR (The Traditional Standard)

**Intuition & Concept:**
The Gaussian VaR approach is the most widespread baseline used in traditional banking. The core intuition is aggressively simple: let's assume the daily returns of the stock market behave perfectly like a standard "bell curve" (a Normal Gaussian distribution).

To calculate risk, the model finds the historical average return of the portfolio (the center of the bell) and the standard deviation (how wide the bell is). By consulting standard statistical tables, the model simply draws a line at the exact 1% tail of that bell curve and declares that as the ultimate limits of our risk.

**Mathematical Formulation:**
Given historical portfolio returns $R_t$, we estimate the sample mean $\hat{\mu}$ and the sample standard deviation $\hat{\sigma}$. The out-of-sample $1-\alpha$ VaR (where $\alpha = 0.01$ for 99% confidence) is computed using the inverse cumulative distribution function (CDF), $\Phi^{-1}$, of the standard normal distribution:

$$ \text{VaR}_\alpha = \hat{\mu} + \Phi^{-1}(\alpha) \hat{\sigma} $$

**Rationale, Limitations & Academic Context:**
*   **Why use it?** It is exceptionally fast to compute, requires almost zero computing power, and is incredibly easy to explain to stakeholders.
*   **Why does it fail?** It suffers from two fatal flaws that empirically contradict how financial markets work:
    1.  **Blindness (Constancy):** The model assigns a single, constant risk number to the future. It outputs the exact same risk estimate whether the economy is in a peaceful 10-year bull run or amidst a global pandemic. It completely ignores macroeconomic context and *volatility clustering*.
    2.  **Thin Tails:** If stock returns truly followed a Gaussian distribution, a crash like the "2008 Financial Crisis" would be mathematically almost impossible. Consequently, this model aggressively *underestimates* true market risk.

---

### 1.2 Baseline 2: Gradient Boosting Quantile Regression (GBQuantile)

**Intuition & Concept:**
To fix the "blindness" of the Gaussian model, we transition into Machine Learning. Instead of forcing reality into a rigid bell curve, what if we simply fed the current macroeconomic weather (e.g., today's US Inflation rate, today's Unemployment rate, the VIX) into an algorithm, and asked it to directly predict where the 1% worst-case line sits?

We utilize **Gradient Boosting**, which is based on **Decision Trees**. A decision tree is like a flowchart that splits data based on yes/no questions (e.g., "Is VIX > 20? If yes, go left..."). The tree ends in a "leaf" that provides a numerical prediction. An **ensemble** simply means we combine hundreds of these fragile, individual trees together to make a single, highly robust prediction.

The "Gradient Boosting" part works sequentially: Tree 1 makes a crude, weak prediction. We then calculate its "mistakes" by subtracting Tree 1's prediction from the actual real-world market result (these mistakes are mathematically termed *residuals*). Tree 2 is then trained *exclusively* to predict those residuals. If Tree 1 under-predicted a loss by 2%, Tree 2 tries to output exactly 2%. By simply adding their outputs together (Tree 1 + Tree 2), the final prediction mathematically zeros out the error. Tree 3 then targets the tiny leftover error from Tree 2, and so on.

Concretely, as the economy shifts, the trees learn precise IF/THEN rules (e.g., "IF CPI is rising AND VIX > 30, then drop the VaR limit to -6%"). This outputs a highly flexible, dynamic VaR limit that expands risk boundaries during turmoil and shrinks them during peace.

**Mathematical Formulation:**
In standard machine learning, algorithms usually use Mean Squared Error (MSE) to predict the *average* outcome. However, we don't care about the average day; we only care about the worst 1% of days. 

To achieve this, we use **Quantile Regression** via the *Pinball Loss* function. The Pinball Loss penalizes the model asymmetrically. If predicting the 1% quantile ($\alpha = 0.01$), the model is penalized 99 times harder for overestimating the return (acting too optimistic) compared to underestimating it. For a prediction $\hat{q}$ and actual portfolio return $y$:

$$ \mathcal{L}_\alpha(y, \hat{q}) = \begin{cases} \alpha(y - \hat{q}) & \text{if } y \ge \hat{q} \\ (1 - \alpha)(\hat{q} - y) & \text{if } y < \hat{q} \end{cases} $$

The ensemble of trees, denoted mathematically as the function $F$, takes today's macroeconomic data array $x_t$ and directly outputs a single number. They are fitted such that this output exactly approximates the Value-at-Risk boundary: $F(x_{t}) \approx \text{VaR}_{\alpha, t}$.

**Rationale, Limitations & Academic Context:**
*   **Why use it?** It is *distribution-free*. It makes zero assumptions about whether the market is a bell curve or not. Furthermore, it successfully reacts to **non-linear macroeconomic inputs**. (A linear model assumes a 1-point rise in VIX always equals a 0.5% drop in returns. A non-linear tree can learn that VIX rising from 10 to 15 means almost nothing, but crossing 25 triggers a massive crash limit calculation).
*   **Why does it fail?** It operates in a vacuum. It looks only at today's raw numbers and ignores the crucial *momentum* of the economy. More importantly, it is merely a "point estimator"—it yields a single number (the VaR line). Because the GBQuantile model doesn't understand the full landscape of the tail beyond that line, **it is mathematically impossible for it to calculate Expected Shortfall (ES)**.

---

### 1.3 The Core Architecture: Temporal Fusion Transformer + Masked Autoregressive Flow (TFT-MAF)

**Intuition & Concept:**
To achieve a state-of-the-art solution, we must solve two problems simultaneously: we need to understand the *trajectory* of the macroeconomy (momentum), and we need to model the *entire shape* of the market's losses (distribution). We do this by combining two massive neural network frameworks:

1.  **The Brain (The Temporal Fusion Transformer - TFT):** The TFT ingests the past 63 days of the entire economy and processes the information via specific internal mechanisms:
    *   *Variable Selection Network:* It dynamically assigns weights (from 0 to 1) to each incoming macroeconomic factor. If inflation is driving the market this month but unemployment is noise, it mathematically mutes unemployment and amplifies inflation.
    *   *Sequential Momentum (LSTM):* An LSTM (Long Short-Term Memory) is a neural network chunk with a "memory cell" that tracks trends. It understands that inflation rising consistently over 3 months is far more dangerous than a random 1-day spike.
    *   *Attention Mechanisms:* Standard networks forget older data. Attention calculates similarity scores between past historical days and today. If today looks exactly like the start of a crash 40 days ago, Attention allows the model to "jump back" and look heavily at that specific sequence, completely ignoring the boring 39 days in between.
    *   *Context Vector:* The raw input is massive ($63 \text{ days} \times 8 \text{ variables} = 504 \text{ numbers}$). The TFT uses the above mechanisms to compress this giant matrix into a single dense 1D vector (e.g., just 64 numbers) that deeply and mathematically summarizes the *exact* state of the current economic regime. 

2.  **The Mathematical Engine (The Masked Autoregressive Flow - MAF):** We pass this regime summary to a Normalizing Flow. Imagine taking a simple block of mathematical "clay" (a standard base distribution $z$, like a basic bell curve). The MAF morphs this clay into our highly complex market returns $X$ utilizing a sequence of **Affine Transformations** (the "twists, stretches, and shifts").
    Concretely, for each asset dimension $i$, the neural network outputs two parameters: a translation parameter $\mu_i$ (the shift) and a scaling parameter $\alpha_i$ (the stretch). The simple variable $z_i$ is mapped into reality using the exact formula: $x_i = z_i \times \exp(\alpha_i) + \mu_i$. 
    By applying this simple algebraic equation stacked through multiple layers (like kneading and folding dough repeatedly), the final shape morphs into a highly irregular, multidimensional probability distribution perfectly boundary-fitted for today's market regime. It does this *Autoregressively* (calculating the equation for Asset A, then calculating Asset B conditioned tightly on Asset A’s output).
    *   *Is it the best approach?* Advanced **Diffusion Models** (the same generative technology behind AI image creators like Midjourney) represent a formidable competing approach. They learn distributions by systematically injecting mathematical noise into data until it collapses into pure Gaussian fuzz, and then train a neural network to systematically "reverse-denoise" it back to reality. While incredibly powerful at generating data, running a 1,000-step reverse-diffusion process to sample a real-time portfolio risk limit every single day is computationally far too heavy for practical quant finance. On the other end, simpler models like GARCH are lightning-fast but structurally rigid, consistently failing to grasp extreme non-linearities. Therefore, the TFT-MAF system is firmly considered the state-of-the-art optimal balance between extreme data-driven flexibility and realistic computational feasibility.

**Mathematical Formulation:**
1.  **Sequence Representation (TFT):** The TFT reads the sequence of macro features $\mathbf{M}_{<t}$ and emits a context conditioning vector $h_t$:
    $$ h_t = \text{TFT}(\mathbf{M}_{<t}) $$
    
2.  **Density Estimation (MAF):** The joint distribution of our assets $X_t \in \mathbb{R}^D$ is modeled autoregressively. The MAF applies a sequence of invertible transformations $f$ to a base variable $z$. 
    
    The genius of the *Autoregressive* mechanism is that it makes the Jacobian matrix (the mathematical derivative of the transformation) strictly triangular. This means calculating the determinant is simply adding up the diagonals, a fast $O(D)$ operation, allowing us to train the model efficiently via the Change-of-Variables theorem:
    $$ \log p_X(X_t|h_t) = \log p_Z(z) + \log \left| \det \left(\frac{\partial z}{\partial X_t}\right) \right| $$

3.  **The Heavy Tail Engine (Student-T Base):** To decisively defeat the exact flaw that brought down the Gaussian Baseline (thin tails), we **do not** start with a Gaussian block of clay. Instead, we use a heavy-tailed multivariate **Student-T distribution**, and we force the neural network to mathematically learn the exact Degrees of Freedom ($\nu$). 
    $$ z \sim \text{Student-T}(\nu) $$
    *Why learn $\nu$?* Instead of a human guessing a static risk number, we use gradient descent. The neural network sweeps through the historical data, observes the worst crashes, and mathematically solves for exactly how "fat" the distribution's tails need to be to account for those catastrophic anomalies explicitly.

**Rationale, Limitations & Academic Context:**
*   **Why use it?** It represents the pinnacle of modern risk modeling. The TFT perfectly registers macroeconomic momentum. The MAF, conditioned on this specific economic regime, learns the entire continuous probability landscape of the portfolio while accurately modeling catastrophic crashes via the Student-t distribution. Because we possess the full parametric map, we can run Monte Carlo simulations (simulating 10,000 alternate realities of tomorrow) to effortlessly and highly accurately calculate *both* VaR and Expected Shortfall.
*   **Why is it hard?** Normalizing flows are notoriously data-hungry and slow to train. The autoregressive sampling phase is sequential $(O(D))$ across the assets, requiring substantially longer computational inference times than standard machine learning trees.

---

## 2. Validation Methodology: Walk-Forward Cross-Validation

**Intuition & Concept:**
In standard machine learning, researchers typically shuffle all their data and take a random 80% to train and 20% to test. In financial time-series analysis, this is dangerous due to **look-ahead bias** (e.g., the model accidentally memorizing the 2020 pandemic crash to falsely predict data in 2018) and **non-stationarity** (the economy of 2015 behaves entirely differently than 2023).

To solve this, our codebase formally implements an expanding **Walk-Forward Cross-Validation** approach (executed via `run_walkforward.py`).

**How it works structurally:**
1.  **Initial Training:** We train the full model on a strictly historical block (e.g., 2005 to 2016).
2.  **Strictly Future Testing:** We test the model’s VaR accuracy out-of-sample purely on the next unseen block (e.g., the year 2017).
3.  **Window Expansion:** We then advance time. Our new training block *expands* to include 2017 (so, 2005 to 2017).
4.  **Continuous Testing:** We test the updated model on the next year, 2018. We repeat this expanding process fold-by-fold until the present day.

**Code Implementation & The "Warm Start" Advantage:**
Re-training an entire deep learning TFT-MAF architecture from absolute scratch for every single shifting year is extremely computationally expensive. 
In our `run_walkforward.py` pipeline, we circumvent this by implementing a **"Warm Start"** architecture. When shifting from Fold 1 (training up to 2016) to Fold 2 (training up to 2017), the model does *not* format and initialize with random weights. Instead, it resets the learning rate but initializes with the fundamentally trained weights of Fold 1 (`trainer.reset_for_new_fold(...)`).

**Advantages of this approach:**
*   **Honest Out-of-Sample Testing:** The model absolutely never sees future data during any training block, providing a deeply realistic and robust benchmark of how the algorithm would actually perform in live trading.
*   **Regime Adaptability:** By perpetually adding the most recent year to its training memory during the test, the model naturally updates its understanding of prevailing macroeconomic conditions.
*   **Computational Efficiency:** Warm starts drastically cut down calculation times across multiple moving folds of data, enabling highly advanced deep learning networks to be efficiently backtested.

---

## Appendix: Oral Presentation Q&A Preparation

To ensure you are fully prepared to defend this mathematical architecture during a presentation, here are critical questions a professor or senior quant might ask, along with the intuitive answers:

**1. "Why did you use an Autoregressive Normalizing Flow instead of just predicting a single VaR number using a standard Neural Network?"**
*   **Answer:** If we only predict a single number (a point estimate), we become blind to the "shape" of the extreme risk tail. It makes calculating Expected Shortfall (ES) impossible because ES requires integrating the *entire* mass past the VaR cutoff. A Normalizing Flow generates the full continuous probability density function, letting us seamlessly calculate both VaR, ES, and run Monte Carlo simulations.

**2. "Explain exactly why the Jacobian determinant in your MAF model is $O(D)$ and why that matters computationally."**
*   **Answer:** In the Change of Variables theorem, calculating the determinant of a large matrix usually scales cubically, which would paralyze a neural network. Because our flow is *Autoregressive* (Asset 2 strictly depends on Asset 1), the resulting Jacobian matrix is perfectly triangular. The mathematics of linear algebra dictate that the determinant of a triangular matrix is just the sum of its diagonals. This reduces the heavy mathematical burden down to linear time, $O(D)$, allowing us to train large datasets swiftly.

**3. "Why use a Student-t base distribution instead of a standard Gaussian base distribution for your Normalizing Flow?"**
*   **Answer:** Financial markets famously possess "fat tails" (excess kurtosis); extreme events like the 2008 crash or 2020 pandemic occur far too frequently to fit in a Gaussian normal bell curve. If we start our Normalizing Flow with Gaussian "clay," the neural network struggles immensely to stretch that thin tail out far enough. By explicitly starting with a heavy-tailed Student-t base and allowing gradient descent to learn the exact *Degrees of Freedom* ($\nu$), we give the model massive mathematical leverage to account for catastrophic crashes easily.

**4. "In Gradient Boosting, exactly how does the model 'learn from its mistakes'?"**
*   **Answer:** It functions additively. Tree 1 attempts a prediction but is highly inaccurate. We calculate the difference between the prediction and reality (the *residual* error). Tree 2 is then trained *not* to predict the market, but strictly to predict Tree 1's error. When we add the output of Tree 1 and Tree 2 together, the error is canceled out. The ensemble builds up by repeating this residual-targeting process hundreds of times.

**5. "How does the Walk-Forward 'Warm Start' logic prevent look-ahead bias while speeding up the code?"**
*   **Answer:** Walk-Forward definitively stops look-ahead bias by only testing on chronological windows that exist strictly *after* the training window cuts off. To prevent the massive cost of retraining the deep neural network from randomized scratch every time we advance a year, the 'Warm Start' initializes Fold 2 with the solved, trained weights from Fold 1. It only needs a few epochs to quickly calibrate exactly what changed in that newly added year of data, massively reducing training time while preserving the strict chronological purity of the test.
