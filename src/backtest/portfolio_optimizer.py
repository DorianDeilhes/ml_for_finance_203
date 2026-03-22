"""
portfolio_optimizer.py
----------------------
Dynamic Portfolio Optimization using Conditional Distributions.

For each trading day, optimizes portfolio weights based on the learned
conditional distribution p(X_t | h_t), enabling:

1. CVaR Minimization: Find weights that minimize tail risk
   min_w CVaR(w @ X | h_t)  subject to: sum(w)=1, w_i >= 0

2. Sharpe Ratio Maximization: Find weights that maximize risk-adjusted returns
   max_w (E[w @ X] - r_f) / sqrt(Var[w @ X])  subject to: sum(w)=1, w_i >= 0

The optimized weights can then be backtested to show:
- Sharpe ratio improvement vs equal-weight baseline
- VaR/CVaR reduction
- In-sample vs out-of-sample stability

References:
  - Rockafellar & Uryasev (2000): "Optimization of Conditional Value-at-Risk"
  - Markowitz (1952): "Portfolio Selection"
"""

from typing import Dict, Tuple, Optional, Callable
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Dynamic portfolio optimizer using Monte Carlo samples from conditional distributions.

    Methods
    -------
    optimize_cvar(...) -> Tuple[np.ndarray, float]
        Minimize CVaR (Conditional Value-at-Risk / Expected Shortfall).
        Returns optimal weights and minimum CVaR value.

    optimize_sharpe(...) -> Tuple[np.ndarray, float]
        Maximize Sharpe ratio. Returns optimal weights and maximum Sharpe ratio.

    optimize_equal_weight(...) -> np.ndarray
        Compute weights for equal-weighted (1/N) portfolio.

    compute_portfolio_cvar(...) -> float
        Compute CVaR of a portfolio given its weights and return samples.

    compute_portfolio_sharpe(...) -> float
        Compute Sharpe ratio of a portfolio.

    compute_portfolio_returns(...) -> np.ndarray
        Aggregate individual asset returns into portfolio returns.
    """

    def __init__(
        self,
        n_assets: int,
        alpha: float = 0.01,
        risk_free_rate: float = 0.00,
        allow_short_selling: bool = False,
        max_weight: float = 1.0,
    ):
        """
        Parameters
        ----------
        n_assets : int
            Number of assets in the portfolio.
        alpha : float
            Tail probability for CVaR computation (default: 0.01 for 99% VaR).
        risk_free_rate : float
            Daily risk-free rate for Sharpe computation (default: 0.00).
        allow_short_selling : bool
            If False, weights are constrained to [0, 1] (long-only).
        max_weight : float
            Maximum allowed weight per asset (default: 1.0, i.e., no upper bound).
        """
        self.n_assets = n_assets
        self.alpha = alpha
        self.risk_free_rate = risk_free_rate
        self.allow_short_selling = allow_short_selling
        self.max_weight = max_weight

    @staticmethod
    def compute_portfolio_returns(
        samples: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute portfolio returns from asset return samples and weights.

        Parameters
        ----------
        samples : np.ndarray of shape (n_samples, n_assets)
            Monte Carlo samples of asset returns.
        weights : np.ndarray of shape (n_assets,)
            Portfolio weights (must sum to 1).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Portfolio returns for each sample.
        """
        return samples @ weights

    def compute_portfolio_var(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute Value-at-Risk (alpha-quantile of portfolio returns).

        Parameters
        ----------
        samples : np.ndarray
        weights : np.ndarray

        Returns
        -------
        float
            VaR value (negative for losses).
        """
        port_returns = self.compute_portfolio_returns(samples, weights)
        return np.quantile(port_returns, self.alpha)

    def compute_portfolio_cvar(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute Conditional Value-at-Risk (CVaR / Expected Shortfall).

        CVaR = E[R | R <= VaR]. This is the average loss in the tail.

        Parameters
        ----------
        samples : np.ndarray of shape (n_samples, n_assets)
        weights : np.ndarray of shape (n_assets,)

        Returns
        -------
        float
            CVaR value (negative for losses).
        """
        port_returns = self.compute_portfolio_returns(samples, weights)
        var = np.quantile(port_returns, self.alpha)
        tail_returns = port_returns[port_returns <= var]
        if len(tail_returns) == 0:
            return float(var)
        return float(tail_returns.mean())

    def compute_portfolio_mean_return(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute mean portfolio return (expected value)."""
        port_returns = self.compute_portfolio_returns(samples, weights)
        return float(port_returns.mean())

    def compute_portfolio_std_return(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute portfolio return volatility (std dev)."""
        port_returns = self.compute_portfolio_returns(samples, weights)
        return float(port_returns.std())

    def compute_portfolio_sharpe(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute Sharpe ratio.

        Sharpe Ratio = (E[R] - r_f) / std(R)

        Parameters
        ----------
        samples : np.ndarray
        weights : np.ndarray

        Returns
        -------
        float
            Sharpe ratio (higher is better).
        """
        mean_ret = self.compute_portfolio_mean_return(samples, weights)
        std_ret = self.compute_portfolio_std_return(samples, weights)
        if std_ret < 1e-6:
            return 0.0
        return float((mean_ret - self.risk_free_rate) / std_ret)

    def optimize_cvar(
        self,
        samples: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize portfolio weights to minimize CVaR (tail risk).

        Solves:
            min_w CVaR(w @ X | h_t)
            subject to: sum(w) = 1, w_i >= 0 (long-only)

        Uses SLSQP (Sequential Least Squares Programming) for constrained optimization.

        Parameters
        ----------
        samples : np.ndarray of shape (n_samples, n_assets)
            Monte Carlo samples from the conditional distribution.
        initial_weights : np.ndarray, optional
            Starting point for optimization (default: equal-weight 1/N).
        verbose : bool
            If True, print optimization details.

        Returns
        -------
        Tuple[np.ndarray, float]
            (optimal_weights, minimum_cvar_value)
        """
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets

        # Objective: minimize tail risk (maximize CVaR since it's a negative value)
        def objective(w):
            return -self.compute_portfolio_cvar(samples, w)

        # Constraint 1: sum(w) = 1
        def constraint_sum(w):
            return np.sum(w) - 1.0

        constraints = {"type": "eq", "fun": constraint_sum}

        # Bounds: long-only if not allow_short_selling
        if self.allow_short_selling:
            bounds = [(None, None)] * self.n_assets
        else:
            bounds = [(0.0, self.max_weight)] * self.n_assets

        options = {"maxiter": 500, "ftol": 1e-8}

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if verbose:
            logger.info(f"CVaR optimization: success={result.success}, "
                       f"n_iter={result.nit}, final CVaR={result.fun:.6f}")

        optimal_weights = result.x
        min_cvar = float(result.fun)

        return optimal_weights, min_cvar

    def optimize_sharpe(
        self,
        samples: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize portfolio weights to maximize Sharpe ratio.

        Solves:
            max_w Sharpe(w) = (E[w@X] - r_f) / std(w@X)
            subject to: sum(w) = 1, w_i >= 0

        Since we want to maximize Sharpe, we minimize -Sharpe.

        Parameters
        ----------
        samples : np.ndarray
        initial_weights : np.ndarray, optional
        verbose : bool

        Returns
        -------
        Tuple[np.ndarray, float]
            (optimal_weights, maximum_sharpe_ratio)
        """
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets

        # Objective: minimize -Sharpe (to maximize Sharpe)
        def objective(w):
            return -self.compute_portfolio_sharpe(samples, w)

        def constraint_sum(w):
            return np.sum(w) - 1.0

        constraints = {"type": "eq", "fun": constraint_sum}

        if self.allow_short_selling:
            bounds = [(None, None)] * self.n_assets
        else:
            bounds = [(0.0, self.max_weight)] * self.n_assets

        options = {"maxiter": 500, "ftol": 1e-8}

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if verbose:
            max_sharpe = -result.fun
            logger.info(f"Sharpe optimization: success={result.success}, "
                       f"n_iter={result.nit}, max Sharpe={max_sharpe:.6f}")

        optimal_weights = result.x
        max_sharpe = float(-result.fun)

        return optimal_weights, max_sharpe

    def optimize_equal_weight(self) -> np.ndarray:
        """Compute equal-weight (1/N) portfolio."""
        return np.ones(self.n_assets) / self.n_assets

    def get_asset_labels(self, tickers: list) -> Dict[str, float]:
        """Format optimized weights with asset names for display."""
        # Placeholder; typically called with tickers=['SPY', 'TLT', 'GLD']
        return {ticker: 0.0 for ticker in tickers}


class DynamicPortfolioMetrics:
    """
    Compute across-day metrics for dynamic portfolios.

    Useful for comparing constant-weight vs optimized allocation performance
    over the full backtest period.
    """

    @staticmethod
    def compute_cumulative_returns(
        daily_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cumulative returns (geometric compound).

        Parameters
        ----------
        daily_returns : np.ndarray of shape (n_days,)
            Daily portfolio returns.

        Returns
        -------
        np.ndarray of shape (n_days,)
            Cumulative returns = prod(1 + r_t) - 1
        """
        return np.cumprod(1 + daily_returns) - 1

    @staticmethod
    def compute_daily_sharpe(
        daily_returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Compute annualized Sharpe ratio from daily returns."""
        excess_returns = daily_returns - risk_free_rate
        if daily_returns.std() < 1e-6:
            return 0.0
        return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())

    @staticmethod
    def compute_max_drawdown(
        cumulative_returns: np.ndarray,
    ) -> float:
        """
        Compute maximum drawdown.

        Drawdown = (peak_value - current_value) / peak_value
        Max Drawdown = min(drawdown over time)
        """
        running_max = np.maximum.accumulate(1 + cumulative_returns)
        drawdown = (running_max - (1 + cumulative_returns)) / running_max
        return float(np.max(drawdown))

    @staticmethod
    def compute_calmar_ratio(
        daily_returns: np.ndarray,
    ) -> float:
        """
        Compute Calmar Ratio = Annual Return / Max Drawdown.

        Useful for evaluating risk-adjusted performance.
        """
        annual_return = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
        cumret = DynamicPortfolioMetrics.compute_cumulative_returns(daily_returns)
        max_dd = DynamicPortfolioMetrics.compute_max_drawdown(cumret)
        if max_dd < 1e-6:
            return 0.0
        return float(annual_return / max_dd)

    @staticmethod
    def compare_portfolios(
        baseline_returns: np.ndarray,
        optimized_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compare baseline (equal-weight) vs optimized portfolio performance.

        Parameters
        ----------
        baseline_returns : np.ndarray of shape (n_days,)
            Daily returns of the baseline (equal-weight) portfolio.
        optimized_returns : np.ndarray of shape (n_days,)
            Daily returns of the optimized portfolio.

        Returns
        -------
        Dict[str, float]
            Comparison metrics:
            - baseline_sharpe, optimized_sharpe
            - baseline_volatility, optimized_volatility
            - baseline_max_dd, optimized_max_dd
            - return_improvement, sharpe_improvement, volatility_reduction
        """
        baseline_sharpe = DynamicPortfolioMetrics.compute_daily_sharpe(baseline_returns)
        optimized_sharpe = DynamicPortfolioMetrics.compute_daily_sharpe(optimized_returns)

        baseline_vol = baseline_returns.std() * np.sqrt(252)
        optimized_vol = optimized_returns.std() * np.sqrt(252)

        baseline_cumret = DynamicPortfolioMetrics.compute_cumulative_returns(baseline_returns)
        optimized_cumret = DynamicPortfolioMetrics.compute_cumulative_returns(optimized_returns)

        baseline_dd = DynamicPortfolioMetrics.compute_max_drawdown(baseline_cumret)
        optimized_dd = DynamicPortfolioMetrics.compute_max_drawdown(optimized_cumret)

        return {
            "baseline_sharpe": baseline_sharpe,
            "optimized_sharpe": optimized_sharpe,
            "sharpe_improvement": optimized_sharpe - baseline_sharpe,
            "baseline_volatility": baseline_vol,
            "optimized_volatility": optimized_vol,
            "volatility_reduction": baseline_vol - optimized_vol,
            "baseline_max_dd": baseline_dd,
            "optimized_max_dd": optimized_dd,
            "dd_reduction": baseline_dd - optimized_dd,
            "baseline_cumulative_return": baseline_cumret[-1],
            "optimized_cumulative_return": optimized_cumret[-1],
            "return_improvement": optimized_cumret[-1] - baseline_cumret[-1],
        }
