"""
risk_metrics.py
---------------
Financial risk metrics for portfolio backtesting.

Computes Value-at-Risk (VaR), Expected Shortfall (ES), and Kupiec's
Proportion of Failures (POF) test from Monte Carlo flow samples.

References:
    - Kupiec (1995): "Techniques for verifying the accuracy of risk measurement models"
    - McNeil, Frey, Embrechts (2015): "Quantitative Risk Management"
"""

from typing import Dict, NamedTuple, Optional

import numpy as np
from scipy import stats


class KupiecResult(NamedTuple):
    """Result of Kupiec's POF test."""
    n_obs: int
    n_breaches: int
    expected_breaches: float
    breach_rate: float
    expected_rate: float
    lr_statistic: float
    p_value: float
    reject_h0: bool
    interpretation: str


def compute_var(
    samples: np.ndarray,
    alpha: float = 0.01,
    portfolio_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Value-at-Risk (VaR) from Monte Carlo samples.

    Parameters
    ----------
    samples : np.ndarray of shape (n_samples, D)
        Monte Carlo samples of asset returns from the flow.
    alpha : float
        Tail probability. alpha=0.01 corresponds to 99% VaR.
    portfolio_weights : np.ndarray of shape (D,), optional
        Portfolio weights summing to 1. Defaults to equal weighting.

    Returns
    -------
    float
        VaR value (negative = loss). E.g., -0.03 means losses exceed 3%
        with probability alpha.
    """
    D = samples.shape[1]
    if portfolio_weights is None:
        portfolio_weights = np.ones(D) / D

    portfolio_returns = samples @ portfolio_weights
    return float(np.quantile(portfolio_returns, alpha))


def compute_es(
    samples: np.ndarray,
    alpha: float = 0.01,
    portfolio_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Expected Shortfall (ES / CVaR) from Monte Carlo samples.

    ES is the mean return conditional on falling below VaR:
        ES = E[R | R <= VaR_alpha]

    ES is a coherent risk measure required by Basel III.

    Parameters
    ----------
    samples : np.ndarray of shape (n_samples, D)
    alpha : float
        Tail probability (default: 0.01 for 99% ES).
    portfolio_weights : np.ndarray of shape (D,), optional

    Returns
    -------
    float
        ES value (negative = average tail loss).
    """
    D = samples.shape[1]
    if portfolio_weights is None:
        portfolio_weights = np.ones(D) / D

    portfolio_returns = samples @ portfolio_weights
    var = np.quantile(portfolio_returns, alpha)
    tail_losses = portfolio_returns[portfolio_returns <= var]
    return float(tail_losses.mean()) if len(tail_losses) > 0 else float(var)


def kupiec_pof_test(
    breaches: int,
    n: int,
    alpha: float = 0.01,
) -> KupiecResult:
    """
    Kupiec's Proportion of Failures (POF) Likelihood Ratio Test.

    Tests whether the observed VaR breach rate equals the theoretical rate alpha.
    H0: p_breach = alpha (model is correctly calibrated).

    LR statistic:
        LR = -2 * [n_breaches * log(alpha) + (n - n_breaches) * log(1 - alpha)
                   - n_breaches * log(p_hat) - (n - n_breaches) * log(1 - p_hat)]

    Under H0, LR ~ chi-squared(1). Reject H0 if LR > 3.841 (5% critical value).

    Parameters
    ----------
    breaches : int
        Number of days where actual loss exceeded predicted VaR.
    n : int
        Total number of backtesting days.
    alpha : float
        Theoretical tail probability (e.g., 0.01 for 99% VaR).

    Returns
    -------
    KupiecResult
        LR statistic, p-value, and calibration verdict.
    """
    p_hat = breaches / n

    if p_hat == 0.0:
        # No breaches: log(0) avoided by adding small epsilon
        lr_stat = -2 * (breaches * np.log(alpha + 1e-10)
                        + (n - breaches) * np.log(1 - alpha))
    elif p_hat == 1.0:
        lr_stat = -2 * (breaches * np.log(1.0 / n)
                        + (n - breaches) * np.log(1 - 1.0 / n))
    else:
        log_l0 = breaches * np.log(alpha) + (n - breaches) * np.log(1 - alpha)
        log_l1 = breaches * np.log(p_hat) + (n - breaches) * np.log(1 - p_hat)
        lr_stat = -2 * (log_l0 - log_l1)

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
        verdict = (f"PASS: Model is well-calibrated at the {(1 - alpha) * 100:.0f}% level. "
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
    portfolio_weights : np.ndarray of shape (D,), optional
        Weights per asset. Defaults to equal weight.
    alpha : float
        VaR significance level.

    Returns
    -------
    dict
        Summary statistics including Kupiec test result, mean VaR/ES,
        portfolio return stats, and breach severity.
    """
    D = actual_returns.shape[1] if actual_returns.ndim == 2 else 1
    if portfolio_weights is None:
        portfolio_weights = np.ones(D) / D

    port_returns = actual_returns @ portfolio_weights if actual_returns.ndim == 2 else actual_returns

    breaches_mask = port_returns < var_series
    n_breaches = int(breaches_mask.sum())
    n_total = len(port_returns)

    kupiec = kupiec_pof_test(n_breaches, n_total, alpha)

    avg_breach_severity = (
        float((port_returns[breaches_mask] - var_series[breaches_mask]).mean())
        if n_breaches > 0 else 0.0
    )

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
    print("=== Kupiec's POF Test Verification ===\n")

    print("Perfect model (5 breaches / 500 days ≈ 1%):")
    result = kupiec_pof_test(breaches=5, n=500, alpha=0.01)
    print(f"  LR Stat: {result.lr_statistic:.4f}, p-value: {result.p_value:.4f}")
    print(f"  {result.interpretation}\n")

    print("Bad model (25 breaches / 500 days = 5%):")
    result = kupiec_pof_test(breaches=25, n=500, alpha=0.01)
    print(f"  LR Stat: {result.lr_statistic:.4f}, p-value: {result.p_value:.4f}")
    print(f"  {result.interpretation}\n")

    np.random.seed(0)
    samples = np.random.randn(10_000, 3) * 0.01
    var = compute_var(samples, alpha=0.01)
    es = compute_es(samples, alpha=0.01)
    print(f"VaR (99%): {var * 100:.3f}%")
    print(f"ES  (99%): {es * 100:.3f}%")