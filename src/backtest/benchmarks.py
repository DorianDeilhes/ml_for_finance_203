"""
benchmarks.py
-------------
Naive VaR benchmarks: parametric Gaussian and Gradient Boosting quantile models.

Both serve as baselines to compare against the macro-conditional normalizing flow.
- GaussianVaR: constant VaR, ignores regime and volatility clustering.
- GBQuantileVaR: predicts the alpha-quantile directly from macro features,
  but produces a point estimate only (no full distribution, no ES).
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor


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


class GBQuantileVaR:
    """Gradient Boosting quantile regression VaR benchmark.

    Trains a GradientBoostingRegressor with quantile loss on the
    last observation of each training macro sequence. Predicts the
    alpha-quantile of the portfolio return directly from macro features —
    no distributional assumption, but no full distribution either.

    Parameters
    ----------
    alpha : float
        Tail probability (default: 0.01 for 99% VaR).
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum tree depth.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        n_estimators: int = 200,
        max_depth: int = 3,
    ) -> None:
        self.alpha = alpha
        self.model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

    def fit(
        self,
        train_macro_seqs: np.ndarray,
        train_returns: np.ndarray,
    ) -> "GBQuantileVaR":
        """Fit the quantile regression model on training data.

        Uses the last time step of each macro sequence as features,
        and the equal-weighted portfolio return as the target.

        Parameters
        ----------
        train_macro_seqs : np.ndarray of shape (T, seq_len, F)
            Macro feature sequences from the training set.
        train_returns : np.ndarray of shape (T, D)
            Daily asset log returns for T training days.

        Returns
        -------
        self
        """
        # Use last macro observation of each window as feature vector
        macro_last = train_macro_seqs[:, -1, :]
        portfolio_returns = train_returns.mean(axis=1)
        self.model.fit(macro_last, portfolio_returns)
        return self

    def predict_var(self, test_macro_seqs: np.ndarray) -> np.ndarray:
        """Predict daily VaR from macro features on the test set.

        Parameters
        ----------
        test_macro_seqs : np.ndarray of shape (T, seq_len, F)

        Returns
        -------
        np.ndarray of shape (T,)
        """
        macro_last = test_macro_seqs[:, -1, :]
        return self.model.predict(macro_last)
