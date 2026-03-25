"""
benchmarks.py
-------------
Naive VaR benchmark: parametric Gaussian model.

Assumes portfolio returns follow a stationary Normal distribution
fitted on the training set. VaR is constant across time — no regime
conditioning. Serves as a baseline to compare against the
macro-conditional normalizing flow.
"""

import numpy as np
from scipy import stats


class GaussianVaR:
    """Parametric Gaussian VaR benchmark.

    Fits the mean and standard deviation of equal-weighted portfolio
    returns on the training set, then estimates VaR as the alpha-quantile
    of the fitted Normal distribution. The VaR is constant across all
    test days — it ignores macro regime and volatility clustering.

    Parameters
    ----------
    alpha : float
        Tail probability (default: 0.01 for 99% VaR).
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self.mu: float = 0.0
        self.sigma: float = 1.0
        self.var: float = 0.0

    def fit(self, train_returns: np.ndarray) -> "GaussianVaR":
        """Estimate mean and std from equal-weighted training returns.

        Parameters
        ----------
        train_returns : np.ndarray of shape (T, D)
            Daily log returns for D assets over T training days.

        Returns
        -------
        self
        """
        portfolio_returns = train_returns.mean(axis=1)
        self.mu = float(portfolio_returns.mean())
        self.sigma = float(portfolio_returns.std())
        self.var = float(self.mu + stats.norm.ppf(self.alpha) * self.sigma)
        return self

    def predict_var(self, n_days: int) -> np.ndarray:
        """Return constant VaR estimates for each test day.

        Parameters
        ----------
        n_days : int
            Number of test days.

        Returns
        -------
        np.ndarray of shape (n_days,)
        """
        return np.full(n_days, self.var)
