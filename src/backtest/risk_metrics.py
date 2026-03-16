"""
risk_metrics.py
---------------
Financial Risk Metrics for Portfolio Backtesting.

Computes key institutional risk metrics from Monte Carlo samples generated
by the conditional normalizing flow:

    1. Value-at-Risk (VaR): The maximum expected loss at a given confidence level.
       "With 99% confidence, the portfolio will not lose more than VaR tomorrow."

    2. Expected Shortfall (ES / CVaR): The average loss when VaR is breached.
       "If the 99% VaR is breached, the average loss will be ES."

    3. Kupiec's Proportion of Failures (POF) Test: A formal statistical test
       checking whether the observed breach rate equals the theoretical rate.
       H0: p_breach = alpha (model is correctly calibrated)
       H1: p_breach != alpha (model is mis-calibrated)

References:
    - Kupiec (1995): "Techniques for verifying the accuracy of risk measurement models"
    - McNeil, Frey, Embrechts (2015): "Quantitative Risk Management"
"""

from typing import Dict, NamedTuple, Optional

import numpy as np
from scipy import stats


class KupiecResult(NamedTuple):
    """Result of Kupiec's POF test."""
    n_obs: int                  # Total number of observations
    n_breaches: int             # Observed number of VaR breaches
    expected_breaches: float    # Expected number of breaches under H0
    breach_rate: float          # Actual breach rate = n_breaches / n_obs
    expected_rate: float        # Theoretical breach rate = alpha
    lr_statistic: float         # Likelihood Ratio test statistic (chi-squared)
    p_value: float              # p-value (< 0.05 → reject H0 at 5% level)
    reject_h0: bool             # True if model is mis-calibrated at 5% level
    interpretation: str         # Human-readable verdict


def compute_var(
    samples: np.ndarray,
    alpha: float = 0.01,
    portfolio_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the Value-at-Risk (VaR) from Monte Carlo samples.

    For an equal-weight (or user-specified-weight) portfolio, aggregates
    the individual asset returns into a single portfolio return, then
    computes the alpha-quantile (left tail).

    Parameters
    ----------
    samples : np.ndarray of shape (n_samples, D)
        Monte Carlo samples of asset returns from the flow.
    alpha : float
        VaR confidence level expressed as the tail probability.
        alpha=0.01 corresponds to 99% VaR.
    portfolio_weights : np.ndarray of shape (D,), optional
        Portfolio weights (must sum to 1). Defaults to equal weighting.

    Returns
    -------
    float
        The VaR value. A negative number means a loss.
        E.g., VaR = -0.03 means "99% confidence the portfolio loses ≤ 3%".
    """
    D = samples.shape[1]
    if portfolio_weights is None:
        portfolio_weights = np.ones(D) / D

    # Compute portfolio returns for each sample
    portfolio_returns = samples @ portfolio_weights  # (n_samples,)

    # VaR is the alpha-quantile of the portfolio return distribution
    var = np.quantile(portfolio_returns, alpha)
    return float(var)


def compute_es(
    samples: np.ndarray,
    alpha: float = 0.01,
    portfolio_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the Expected Shortfall (ES / CVaR) from Monte Carlo samples.

    ES is the average return conditional on the return being below the VaR:
        ES = E[R | R ≤ VaR_alpha]

    ES is a coherent risk measure (unlike VaR) and is required by Basel III.

    Parameters
    ----------
    samples : np.ndarray of shape (n_samples, D)
    alpha : float
        Tail probability (default: 0.01 for 99% ES).
    portfolio_weights : np.ndarray of shape (D,), optional

    Returns
    -------
    float
        The ES value (a negative number representing average tail loss).
    """
    D = samples.shape[1]
    if portfolio_weights is None:
        portfolio_weights = np.ones(D) / D

    portfolio_returns = samples @ portfolio_weights
    var = np.quantile(portfolio_returns, alpha)

    # ES = mean of returns in the left tail (those below VaR)
    tail_losses = portfolio_returns[portfolio_returns <= var]
    if len(tail_losses) == 0:
        return float(var)
    return float(tail_losses.mean())


def kupiec_pof_test(
    breaches: int,
    n: int,
    alpha: float = 0.01,
) -> KupiecResult:
    """
    Kupiec's Proportion of Failures (POF) Likelihood Ratio Test.

    Tests whether the observed VaR breach rate equals the theoretical rate
    (alpha). Under H0: the model is correctly calibrated.

    The LR statistic is:
        LR = -2 * [log L(H0) - log L(H1)]
        LR = -2 * [n_breaches * log(alpha) + (n-n_breaches) * log(1-alpha)
                  - n_breaches * log(p_hat) - (n-n_breaches) * log(1-p_hat)]

    where p_hat = n_breaches / n is the observed breach rate.

    Under H0, LR ~ chi-squared(1). Reject H0 if LR > 3.841 (5% critical value).

    Parameters
    ----------
    breaches : int
        Number of days where actual loss exceeded predicted VaR.
    n : int
        Total number of backtesting days.
    alpha : float
        The theoretical tail probability (e.g., 0.01 for 99% VaR).

    Returns
    -------
    KupiecResult
        Detailed test result including LR statistic, p-value, and verdict.
    """
    p_hat = breaches / n  # Maximum likelihood estimate of true breach probability

    # Edge cases: perfect or zero breaches
    if p_hat == 0.0:
        # No breaches: very unlikely if alpha > 0, but handle gracefully
        lr_stat = -2 * (breaches * np.log(alpha + 1e-10)
                        + (n - breaches) * np.log(1 - alpha))
    elif p_hat == 1.0:
        lr_stat = -2 * (breaches * np.log(1.0 / n)
                        + (n - breaches) * np.log(1 - 1.0 / n))
    else:
        # Full LR statistic
        log_l0 = (breaches * np.log(alpha)
                  + (n - breaches) * np.log(1 - alpha))
        log_l1 = (breaches * np.log(p_hat)
                  + (n - breaches) * np.log(1 - p_hat))
        lr_stat = -2 * (log_l0 - log_l1)

    # p-value from chi-squared distribution with 1 degree of freedom
    p_value = float(1 - stats.chi2.cdf(lr_stat, df=1))
    reject_h0 = p_value < 0.05

    expected_breaches = n * alpha

    if reject_h0:
        if p_hat > alpha:
            verdict = (f"FAIL: Model UNDER-estimates tail risk. "
                       f"Observed {p_hat:.2%} breaches vs expected {alpha:.2%}. "
                       f"VaR is too optimistic.")
        else:
            verdict = (f"FAIL: Model OVER-estimates tail risk. "
                       f"Observed {p_hat:.2%} breaches vs expected {alpha:.2%}. "
                       f"VaR is too conservative.")
    else:
        verdict = (f"PASS: Model is well-calibrated at the {(1-alpha)*100:.0f}% level. "
                   f"Observed {breaches} breaches vs expected {expected_breaches:.1f}.")

    return KupiecResult(
        n_obs=n,
        n_breaches=breaches,
        expected_breaches=expected_breaches,
        breach_rate=p_hat,
        expected_rate=alpha,
        lr_statistic=float(lr_stat),
        p_value=p_value,
        reject_h0=reject_h0,
        interpretation=verdict,
    )


def compute_portfolio_stats(
    actual_returns: np.ndarray,
    var_series: np.ndarray,
    es_series: np.ndarray,
    portfolio_weights: Optional[np.ndarray] = None,
    alpha: float = 0.01,
) -> Dict:
    """
    Compute comprehensive backtest statistics for a portfolio.

    Parameters
    ----------
    actual_returns : np.ndarray of shape (T, D)
        Actual daily log returns for each asset.
    var_series : np.ndarray of shape (T,)
        Predicted daily VaR values.
    es_series : np.ndarray of shape (T,)
        Predicted daily ES values.
    portfolio_weights : np.ndarray, optional
        Weights per asset (default: equal weight).
    alpha : float
        VaR significance level.

    Returns
    -------
    dict
        Summary statistics including Kupiec test, mean VaR, mean ES, etc.
    """
    D = actual_returns.shape[1] if actual_returns.ndim == 2 else 1
    if portfolio_weights is None:
        portfolio_weights = np.ones(D) / D

    # Compute actual portfolio returns
    if actual_returns.ndim == 2:
        port_returns = actual_returns @ portfolio_weights
    else:
        port_returns = actual_returns

    # VaR breaches: actual loss worse than predicted VaR
    breaches_mask = port_returns < var_series
    n_breaches = int(breaches_mask.sum())
    n_total = len(port_returns)

    # Kupiec's test
    kupiec = kupiec_pof_test(n_breaches, n_total, alpha)

    # Additional statistics
    avg_breach_severity = float(
        (port_returns[breaches_mask] - var_series[breaches_mask]).mean()
    ) if n_breaches > 0 else 0.0

    return {
        "n_days": n_total,
        "n_breaches": n_breaches,
        "breach_rate": n_breaches / n_total,
        "expected_breach_rate": alpha,
        "mean_var": float(var_series.mean()),
        "mean_es": float(es_series.mean()),
        "mean_port_return": float(port_returns.mean()),
        "std_port_return": float(port_returns.std()),
        "kupiec_test": kupiec,
        "avg_breach_severity": avg_breach_severity,
        "max_drawdown_day": float(port_returns.min()),
    }


if __name__ == "__main__":
    # Unit test: verify Kupiec's test
    print("=== Kupiec's POF Test Verification ===\n")

    # Good model: 99% VaR with ~1% breaches
    print("Perfect model (5 breaches / 500 days ≈ 1%):")
    result = kupiec_pof_test(breaches=5, n=500, alpha=0.01)
    print(f"  LR Stat: {result.lr_statistic:.4f}, p-value: {result.p_value:.4f}")
    print(f"  {result.interpretation}\n")

    # Bad model: too many breaches
    print("Bad model (25 breaches / 500 days = 5%):")
    result = kupiec_pof_test(breaches=25, n=500, alpha=0.01)
    print(f"  LR Stat: {result.lr_statistic:.4f}, p-value: {result.p_value:.4f}")
    print(f"  {result.interpretation}\n")

    # VaR / ES computation
    np.random.seed(0)
    samples = np.random.randn(10_000, 3) * 0.01  # Simulate daily returns
    var = compute_var(samples, alpha=0.01)
    es  = compute_es(samples, alpha=0.01)
    print(f"VaR (99%): {var*100:.3f}%")
    print(f"ES  (99%): {es*100:.3f}%")
