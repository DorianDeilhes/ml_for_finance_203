"""

backtester.py

-------------

Out-of-Sample Financial Backtester for Portfolio Risk Management.



For each trading day in the test set, uses the trained ConditionalNormalizingFlow

to:

1. Encode the macro context h_t via the TFT.

2. Draw 10,000 Monte Carlo samples from the conditional distribution p(X_t | h_t).

3. Compute portfolio VaR (99%) and Expected Shortfall from those samples.

4. Record whether the actual portfolio return breached the VaR (a "breach").

5. Run Kupiec's POF test on the full breach series.

6. Generate visualization plots (VaR bands vs actual returns, breach markers).

"""



import logging

import os

from typing import Dict, List, Optional, Tuple



import matplotlib.dates as mdates

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from tqdm import tqdm



from src.models.flow_model import ConditionalNormalizingFlow

from src.backtest.risk_metrics import (

    compute_var,

    compute_es,

    kupiec_pof_test,

    compute_portfolio_stats,

    KupiecResult,

)



logger = logging.getLogger(__name__)





class Backtester:

    """

    Out-of-Sample Backtester for the Conditional Normalizing Flow.



    Parameters

    ----------

    model : ConditionalNormalizingFlow

        Trained model (best checkpoint should be loaded before calling run()).

    test_loader : DataLoader

        Test set DataLoader (macro_seq, returns) -- must NOT be shuffled.

    test_dates : np.ndarray

        Array of dates corresponding to each test observation.

    ret_scaler : StandardScaler

        The scaler used on returns (needed to invert scaling for actual $ metrics).

    tickers : list of str

        Asset names for labeling.

    n_mc_samples : int

        Number of Monte Carlo samples per day.

    alpha : float

        VaR confidence level tail probability (default: 0.01 for 99% VaR).

    portfolio_weights : np.ndarray, optional

        Asset weights (default: equal).

    device : torch.device, optional

        Device to run inference on.

    """



    def __init__(

        self,

        model: ConditionalNormalizingFlow,

        test_loader,

        test_dates: np.ndarray,

        ret_scaler,

        tickers: List[str],

        n_mc_samples: int = 10_000,

        alpha: float = 0.01,

        portfolio_weights: Optional[np.ndarray] = None,

        device: Optional[torch.device] = None,

    ) -> None:

        self.model = model

        self.test_loader = test_loader

        self.test_dates  = pd.to_datetime(test_dates)

        self.ret_scaler  = ret_scaler

        self.tickers = tickers

        self.n_mc_samples = n_mc_samples

        self.alpha = alpha



        D = len(tickers)

        self.portfolio_weights = (

            portfolio_weights if portfolio_weights is not None else np.ones(D) / D

        )



        if device is None:

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device



        # Results storage

        self.results: Optional[pd.DataFrame] = None

        # MC portfolio-return samples stored for a handful of representative days

        # (used by plot_return_distributions)

        self._stored_port_samples: dict = {}



    def run(self) -> pd.DataFrame:

        """

        Execute the full out-of-sample backtest.



        Optimisation: the TFT encoder is run in batches over the entire test set

        first to pre-compute all context vectors h_t, then per-day MAF sampling

        is performed using those cached vectors.  This eliminates the overhead of

        running the (relatively heavy) TFT once per day inside the inner loop.



        Returns

        -------

        pd.DataFrame

            Backtest results indexed by date. Columns:

            - actual_port_return: true portfolio return that day

            - var_99: predicted 99% VaR

            - es_99: predicted 99% ES

            - breach: 1 if actual < var_99, else 0

            - var_99_pct, es_99_pct: in percentage form

            - individual asset returns: SPY_actual, TLT_actual, GLD_actual

        """

        self.model.eval()

        self.model.to(self.device)



        # -- Step 1: batch-encode all macro sequences -> collect h_t and returns --

        logger.info("Pre-computing context vectors h_t for all %d test days...", len(self.test_dates))

        all_h_t: List[torch.Tensor] = []

        all_returns: List[np.ndarray] = []



        with torch.no_grad():

            for macro_seq, returns in self.test_loader:

                macro_seq = macro_seq.to(self.device)

                h_t, _ = self.model.tft(macro_seq)   # (batch, d_model)

                all_h_t.append(h_t.cpu())

                all_returns.append(returns.numpy())



        all_h_t_tensor = torch.cat(all_h_t, dim=0)          # (N, d_model)

        all_returns_arr = np.concatenate(all_returns, axis=0) # (N, D)



        # -- Step 2: per-day MAF sampling using cached h_t --------------------

        logger.info(

            "Sampling %d MC draws per day for %d days...",

            self.n_mc_samples, len(all_h_t_tensor),

        )

        var_list: List[float] = []

        es_list:  List[float] = []



        # Choose up to 3 breach days + 3 calm days to store full samples for plots

        # (decided after run() completes, so we do a two-pass store below)

        _port_sample_store: dict = {}  # date_idx -> 1-D portfolio return samples



        with torch.no_grad():

            for i in tqdm(range(len(all_h_t_tensor)), desc="Backtesting"):

                h_t_i = all_h_t_tensor[i:i+1].to(self.device)   # (1, d_model)

                samples = self.model.flow.sample(

                    self.n_mc_samples, context=h_t_i

                ).cpu().numpy()  # (n_samples, D)



                samples_unscaled = self.ret_scaler.inverse_transform(samples)



                var = compute_var(

                    samples_unscaled, alpha=self.alpha,

                    portfolio_weights=self.portfolio_weights,

                )

                es = compute_es(

                    samples_unscaled, alpha=self.alpha,

                    portfolio_weights=self.portfolio_weights,

                )

                var_list.append(var)

                es_list.append(es)



                # Store 1-D portfolio-return samples for representative days

                # (always stored; trimmed to a handful after breach detection below)

                port_samples = samples_unscaled @ self.portfolio_weights

                _port_sample_store[i] = port_samples



        # -- Step 3: assemble results DataFrame -------------------------------

        actual_unscaled = self.ret_scaler.inverse_transform(all_returns_arr)

        port_returns = actual_unscaled @ self.portfolio_weights



        n_dates = min(len(self.test_dates), len(var_list))

        dates    = self.test_dates[:n_dates]

        var_arr  = np.array(var_list[:n_dates])

        es_arr   = np.array(es_list[:n_dates])

        port_arr = port_returns[:n_dates]

        actual_arr = actual_unscaled[:n_dates]

        breaches = (port_arr < var_arr).astype(int)



        results = pd.DataFrame({

            "actual_port_return": port_arr,

            "var_99":  var_arr,

            "es_99":   es_arr,

            "breach":  breaches,

            "var_99_pct": var_arr * 100,

            "es_99_pct":  es_arr  * 100,

        }, index=dates)



        for j, ticker in enumerate(self.tickers):

            results[f"{ticker}_actual"] = actual_arr[:, j]



        self.results = results



        # -- Step 4: keep stored samples only for representative days ----------

        breach_idx = np.where(breaches)[0]

        calm_idx   = np.where(~breaches.astype(bool))[0]

        keep_idx = set()

        if len(breach_idx) > 0:

            keep_idx.update(breach_idx[:3].tolist())

        step = max(1, len(calm_idx) // 3)

        keep_idx.update(calm_idx[::step][:3].tolist())

        self._stored_port_samples = {

            dates[i]: _port_sample_store[i] for i in keep_idx if i < n_dates

        }



        # -- Step 5: Kupiec's test ---------------------------------------------

        n_breaches = int(breaches.sum())

        n_total    = len(breaches)

        kupiec_result = kupiec_pof_test(n_breaches, n_total, self.alpha)



        logger.info("=== Backtest Results ===")

        logger.info("Period: %s to %s (%d days)", dates[0].date(), dates[-1].date(), n_total)

        logger.info("VaR breaches: %d / %d (%.2f%% observed vs %.2f%% expected)",

                    n_breaches, n_total, n_breaches / n_total * 100, self.alpha * 100)

        logger.info("Kupiec LR: %.4f, p-value: %.4f", kupiec_result.lr_statistic, kupiec_result.p_value)

        logger.info(kupiec_result.interpretation)



        self.kupiec_result = kupiec_result

        return results



    def plot_var_bands(

        self,

        output_path: Optional[str] = None,

        title: str = "Out-of-Sample 99% VaR vs Actual Portfolio Returns",

        figsize: Tuple[int, int] = (14, 7),

    ) -> plt.Figure:

        """

        Plot predicted 99% VaR bands overlaid on actual portfolio returns,

        with breach events highlighted.



        Parameters

        ----------

        output_path : str, optional

            If provided, save the figure to this path.

        title : str

            Plot title.

        figsize : tuple

            Figure dimensions.



        Returns

        -------

        matplotlib.figure.Figure

        """

        if self.results is None:

            raise RuntimeError("Must call run() before plotting.")



        df = self.results

        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})



        ax1 = axes[0]



        # Actual portfolio returns

        ax1.plot(

            df.index, df["actual_port_return"] * 100,

            color="#3a86ff", linewidth=0.8, alpha=0.9, label="Actual Portfolio Return",

        )



        # VaR band (99% lower bound)

        ax1.fill_between(

            df.index, df["var_99_pct"], df["var_99_pct"] - 1.0,

            alpha=0.3, color="#ff006e", label="99% VaR",

        )

        ax1.plot(

            df.index, df["var_99_pct"],

            color="#ff006e", linewidth=1.2, linestyle="--",

        )



        # ES line

        ax1.plot(

            df.index, df["es_99_pct"],

            color="#fb8500", linewidth=1.0, linestyle=":",

            alpha=0.8, label="99% Expected Shortfall",

        )



        # Mark breach events

        breach_dates = df.index[df["breach"] == 1]

        breach_returns = df.loc[df["breach"] == 1, "actual_port_return"] * 100

        ax1.scatter(

            breach_dates, breach_returns,

            color="#ff006e", s=30, zorder=5, label=f"VaR Breaches ({len(breach_dates)})",

            marker="v",

        )



        ax1.axhline(0, color="white", linewidth=0.5, alpha=0.5)

        ax1.set_ylabel("Daily Portfolio Return (%)", fontsize=11)

        ax1.set_title(title, fontsize=13, fontweight="bold")

        ax1.legend(loc="upper right", fontsize=9)

        ax1.grid(True, alpha=0.3)



        # Bottom panel: VaR breach indicators

        ax2 = axes[1]

        ax2.bar(

            df.index, df["breach"],

            color="#ff006e", alpha=0.8, width=1.0,

            label=f"VaR Breach (n={int(df['breach'].sum())})",

        )

        ax2.set_ylabel("Breach", fontsize=10)

        ax2.set_yticks([0, 1])

        ax2.set_xlabel("Date", fontsize=10)

        ax2.legend(loc="upper right", fontsize=9)

        ax2.grid(True, alpha=0.2)



        # Format x-axis dates

        for ax in axes:

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")



        # Add Kupiec test annotation

        kupiec = self.kupiec_result

        annotation = (

            f"Kupiec's POF: LR={kupiec.lr_statistic:.3f}, "

            f"p={kupiec.p_value:.3f} "

            f"({'PASS PASS' if not kupiec.reject_h0 else 'FAIL FAIL'})"

        )

        ax1.annotate(

            annotation,

            xy=(0.02, 0.04),

            xycoords="axes fraction",

            fontsize=9,

            color="#8ecae6" if not kupiec.reject_h0 else "#ff006e",

            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),

        )



        plt.tight_layout()



        if output_path:

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

            fig.savefig(output_path, dpi=150, bbox_inches="tight")

            logger.info("VaR plot saved to %s", output_path)



        return fig



    def plot_return_distributions(

        self,

        n_days: int = 5,

        output_path: Optional[str] = None,

    ) -> plt.Figure:

        """

        Plot the predicted portfolio return distribution (histogram) for selected

        test days alongside the VaR and actual return, showing how the model

        adapts to different macro regimes.



        Uses MC samples stored during run() for representative days (breach days

        and calm days).



        Parameters

        ----------

        n_days : int

            Maximum number of days to plot (limited by stored samples).

        output_path : str, optional

            File path to save the figure.



        Returns

        -------

        matplotlib.figure.Figure

        """

        if self.results is None:

            raise RuntimeError("Must call run() before plotting.")

        if not self._stored_port_samples:

            raise RuntimeError("No stored samples found. Was run() called successfully?")



        df = self.results

        selected_dates = sorted(self._stored_port_samples.keys())[:n_days]



        fig, axes = plt.subplots(1, len(selected_dates), figsize=(5 * len(selected_dates), 4))

        if len(selected_dates) == 1:

            axes = [axes]



        for ax, date in zip(axes, selected_dates):

            row = df.loc[date]

            port_samples = self._stored_port_samples[date]



            # Histogram of MC portfolio return samples

            ax.hist(

                port_samples * 100, bins=80, density=True,

                color="#3a86ff", alpha=0.6, label="MC distribution",

            )

            ax.axvline(

                row["var_99"] * 100, color="#ff006e", linestyle="--",

                linewidth=1.8, label=f"VaR: {row['var_99']*100:.2f}%",

            )

            ax.axvline(

                row["actual_port_return"] * 100, color="#fb8500", linestyle="-",

                linewidth=2, label=f"Actual: {row['actual_port_return']*100:.2f}%",

            )

            is_breach = bool(row["breach"])

            ax.set_title(

                f"{date.date()}"

                + (" <- BREACH" if is_breach else ""),

                fontsize=10,

                color="#ff006e" if is_breach else "white",

            )

            ax.set_xlabel("Portfolio Return (%)")

            ax.set_ylabel("Density")

            ax.legend(fontsize=8)

            ax.grid(True, alpha=0.3)



        plt.suptitle("Predicted Portfolio Return Distributions (Selected Days)", fontsize=12)

        plt.tight_layout()



        if output_path:

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

            fig.savefig(output_path, dpi=150, bbox_inches="tight")



        return fig



    def summary(self) -> pd.DataFrame:

        """

        Return a summary table of key backtest metrics.



        Returns

        -------

        pd.DataFrame with columns: Metric, Value.

        """

        if self.results is None:

            raise RuntimeError("Must call run() before summary().")



        df = self.results

        kupiec = self.kupiec_result



        rows = [

            ("Backtest Period", f"{df.index[0].date()} to {df.index[-1].date()}"),

            ("Total Days", len(df)),

            ("VaR Confidence Level", f"{(1-self.alpha)*100:.0f}%"),

            ("Expected Breaches", f"{kupiec.expected_breaches:.1f}"),

            ("Observed Breaches", kupiec.n_breaches),

            ("Observed Breach Rate", f"{kupiec.breach_rate:.2%}"),

            ("Kupiec LR Statistic", f"{kupiec.lr_statistic:.4f}"),

            ("Kupiec p-value", f"{kupiec.p_value:.4f}"),

            ("Kupiec Result", "PASS PASS" if not kupiec.reject_h0 else "FAIL FAIL"),

            ("Mean Daily VaR", f"{df['var_99'].mean()*100:.3f}%"),

            ("Mean Daily ES", f"{df['es_99'].mean()*100:.3f}%"),

            ("Mean Actual Daily Return", f"{df['actual_port_return'].mean()*100:.3f}%"),

            ("Worst Day (actual)", f"{df['actual_port_return'].min()*100:.3f}%"),

        ]



        return pd.DataFrame(rows, columns=["Metric", "Value"])


# ============================================================================
# OPTIMIZED BACKTESTER: Dynamic Portfolio Optimization
# ============================================================================


class OptimizedBacktester:
    """
    Extended backtester that computes optimal portfolio weights at each time step
    using the learned conditional distribution p(X_t | h_t).

    Compares:
    1. Baseline (equal-weight) portfolio
    2. CVaR-minimized portfolio (least tail risk)
    3. Sharpe-maximized portfolio (best risk-adjusted returns)

    Metrics tracked:
    - Daily portfolio returns for each strategy
    - Rolling Sharpe ratio, volatility, max drawdown
    - Comparison of VaR/CVaR accuracy across strategies
    - Weight stability over time

    Parameters
    ----------
    baseline_backtester : Backtester
        A completed baseline backtest with results.
    optimization_method : {'cvar', 'sharpe'}
        Which optimization objective to use (default: 'cvar').
    allow_short_selling : bool
        If False, weights are constrained to [0, 1] (long-only).
    """

    def __init__(
        self,
        baseline_backtester: Backtester,
        optimization_method: str = "cvar",
        allow_short_selling: bool = False,
    ):
        self.baseline_backtester = baseline_backtester
        self.optimization_method = optimization_method
        self.allow_short_selling = allow_short_selling

        # Import optimizer here to avoid circular imports
        from src.backtest.portfolio_optimizer import (
            PortfolioOptimizer,
            DynamicPortfolioMetrics,
        )

        self.PortfolioOptimizer = PortfolioOptimizer
        self.DynamicPortfolioMetrics = DynamicPortfolioMetrics

        self.device = baseline_backtester.device
        self.model = baseline_backtester.model
        self.test_loader = baseline_backtester.test_loader
        self.test_dates = baseline_backtester.test_dates
        self.ret_scaler = baseline_backtester.ret_scaler
        self.tickers = baseline_backtester.tickers
        self.n_mc_samples = baseline_backtester.n_mc_samples
        self.alpha = baseline_backtester.alpha

        self.results: Optional[pd.DataFrame] = None
        self.comparison_metrics: Optional[Dict] = None

    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Run optimized backtester alongside baseline.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (detailed_results_df, comparison_metrics_dict)

        detailed_results_df columns:
            - date (index)
            - baseline_return, optimized_return, optimized_weights_*
            - baseline_var, optimized_var
            - baseline_es, optimized_es
            - actual_return (ground truth), breach_baseline, breach_optimized

        comparison_metrics_dict:
            - baseline_sharpe, optimized_sharpe, sharpe_improvement
            - baseline_volatility, optimized_volatility, volatility_reduction
            - baseline_max_dd, optimized_max_dd, dd_reduction
            - baseline_cumulative_return, optimized_cumulative_return
        """
        logger.info("=" * 70)
        logger.info("OPTIMIZED BACKTESTER: Computing optimal weights...")
        logger.info("=" * 70)

        self.model.eval()
        self.model.to(self.device)

        # Pre-compute h_t for all test days
        logger.info("Pre-computing context vectors h_t for all %d test days...",
                   len(self.test_dates))
        all_h_t: List[torch.Tensor] = []
        all_returns: List[np.ndarray] = []

        with torch.no_grad():
            for macro_seq, returns in self.test_loader:
                macro_seq = macro_seq.to(self.device)
                h_t, _ = self.model.tft(macro_seq)
                all_h_t.append(h_t.cpu())
                all_returns.append(returns.numpy())

        all_h_t_tensor = torch.cat(all_h_t, dim=0)
        all_returns_arr = np.concatenate(all_returns, axis=0)

        # Unscale actual returns
        actual_unscaled = self.ret_scaler.inverse_transform(all_returns_arr)

        # Initialize optimizer
        optimizer = self.PortfolioOptimizer(
            n_assets=len(self.tickers),
            alpha=self.alpha,
            allow_short_selling=self.allow_short_selling,
        )

        # Storage for results
        n_days = len(all_h_t_tensor)
        baseline_returns: List[float] = []
        optimized_returns: List[float] = []
        optimized_weights_list: List[np.ndarray] = []
        baseline_var_list: List[float] = []
        baseline_es_list: List[float] = []
        optimized_var_list: List[float] = []
        optimized_es_list: List[float] = []

        # Get baseline weights
        baseline_weights = self.baseline_backtester.portfolio_weights

        # Per-day optimization
        logger.info("Optimizing weights for %d days (method='%s')...",
                   n_days, self.optimization_method)
        with torch.no_grad():
            for i in tqdm(range(n_days), desc="Optimizing"):
                h_t_i = all_h_t_tensor[i:i + 1].to(self.device)

                # Draw MC samples
                samples = self.model.flow.sample(
                    self.n_mc_samples, context=h_t_i
                ).cpu().numpy()
                samples_unscaled = self.ret_scaler.inverse_transform(samples)

                # Baseline metrics
                baseline_var = compute_var(
                    samples_unscaled, alpha=self.alpha,
                    portfolio_weights=baseline_weights,
                )
                baseline_es = compute_es(
                    samples_unscaled, alpha=self.alpha,
                    portfolio_weights=baseline_weights,
                )
                baseline_var_list.append(baseline_var)
                baseline_es_list.append(baseline_es)

                # Optimize weights
                if self.optimization_method == "cvar":
                    opt_weights, _ = optimizer.optimize_cvar(
                        samples_unscaled,
                        initial_weights=baseline_weights,
                    )
                elif self.optimization_method == "sharpe":
                    opt_weights, _ = optimizer.optimize_sharpe(
                        samples_unscaled,
                        initial_weights=baseline_weights,
                    )
                else:
                    raise ValueError(f"Unknown optimization method: {self.optimization_method}")

                optimized_weights_list.append(opt_weights)

                # Optimized metrics
                opt_var = compute_var(
                    samples_unscaled, alpha=self.alpha,
                    portfolio_weights=opt_weights,
                )
                opt_es = compute_es(
                    samples_unscaled, alpha=self.alpha,
                    portfolio_weights=opt_weights,
                )
                optimized_var_list.append(opt_var)
                optimized_es_list.append(opt_es)

                # Actual returns
                actual_returns_day = actual_unscaled[i]
                baseline_ret = float(actual_returns_day @ baseline_weights)
                optimized_ret = float(actual_returns_day @ opt_weights)
                baseline_returns.append(baseline_ret)
                optimized_returns.append(optimized_ret)

        # Assemble results DataFrame
        baseline_returns = np.array(baseline_returns)
        optimized_returns = np.array(optimized_returns)
        baseline_var_list = np.array(baseline_var_list)
        optimized_var_list = np.array(optimized_var_list)
        baseline_es_list = np.array(baseline_es_list)
        optimized_es_list = np.array(optimized_es_list)

        breaches_baseline = (baseline_returns < baseline_var_list).astype(int)
        breaches_optimized = (optimized_returns < optimized_var_list).astype(int)

        results = pd.DataFrame({
            "actual_port_return": baseline_returns,  # same for both strategies
            "baseline_return": baseline_returns,
            "optimized_return": optimized_returns,
            "baseline_var": baseline_var_list,
            "optimized_var": optimized_var_list,
            "baseline_es": baseline_es_list,
            "optimized_es": optimized_es_list,
            "baseline_breach": breaches_baseline,
            "optimized_breach": breaches_optimized,
        }, index=self.test_dates[:n_days])

        # Add individual asset returns
        for j, ticker in enumerate(self.tickers):
            results[f"{ticker}_actual"] = actual_unscaled[:n_days, j]

        # Add optimized weights columns
        weights_array = np.array(optimized_weights_list)
        for j, ticker in enumerate(self.tickers):
            results[f"weight_{ticker}"] = weights_array[:, j]

        self.results = results
        logger.info("Optimization complete. Generating comparison metrics...")

        # Compare performance
        comparison = self.DynamicPortfolioMetrics.compare_portfolios(
            baseline_returns, optimized_returns
        )
        self.comparison_metrics = comparison

        logger.info("=" * 70)
        logger.info("OPTIMIZATION RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info("Baseline (Equal-Weight) vs Optimized (%s)", self.optimization_method.upper())
        logger.info("-" * 70)
        logger.info("Sharpe Ratio:        %.4f  →  %.4f  (Δ = %.4f)",
                   comparison["baseline_sharpe"], comparison["optimized_sharpe"],
                   comparison["sharpe_improvement"])
        logger.info("Volatility (annual): %.2f%%  →  %.2f%%  (Δ = %.2f%%)",
                   comparison["baseline_volatility"] * 100,
                   comparison["optimized_volatility"] * 100,
                   comparison["volatility_reduction"] * 100)
        logger.info("Max Drawdown:        %.2f%%  →  %.2f%%  (Δ = %.2f%%)",
                   comparison["baseline_max_dd"] * 100,
                   comparison["optimized_max_dd"] * 100,
                   comparison["dd_reduction"] * 100)
        logger.info("Cumulative Return:   %.2f%%  →  %.2f%%  (Δ = %.2f%%)",
                   comparison["baseline_cumulative_return"] * 100,
                   comparison["optimized_cumulative_return"] * 100,
                   comparison["return_improvement"] * 100)
        logger.info("=" * 70)

        return results, comparison

    def plot_weight_evolution(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        Plot how optimal portfolio weights evolve over time.

        Shows whether the model converges to a stable allocation or
        actively rebalances in response to changing macro regimes.

        Parameters
        ----------
        output_path : str, optional
            File path to save the figure.
        figsize : tuple
            Figure dimensions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise RuntimeError("Must call run() before plot_weight_evolution()")

        df = self.results
        fig, axes = plt.subplots(len(self.tickers), 1, figsize=figsize, sharex=True)
        if len(self.tickers) == 1:
            axes = [axes]

        for j, ticker in enumerate(self.tickers):
            weight_col = f"weight_{ticker}"
            ax = axes[j]
            ax.plot(df.index, df[weight_col] * 100, linewidth=1.5, label=ticker)
            ax.fill_between(df.index, 0, df[weight_col] * 100, alpha=0.3)
            ax.set_ylabel(f"{ticker} Weight (%)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Date")
        fig.suptitle(
            f"Optimal Portfolio Weight Evolution ({self.optimization_method.upper()} Optimization)",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
            logger.info(f"Saved weight evolution plot to {output_path}")

        return fig

    def plot_return_comparison(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot cumulative returns: baseline vs optimized.

        Parameters
        ----------
        output_path : str, optional
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise RuntimeError("Must call run() before plot_return_comparison()")

        df = self.results
        baseline_cumret = self.DynamicPortfolioMetrics.compute_cumulative_returns(
            df["baseline_return"].values
        )
        optimized_cumret = self.DynamicPortfolioMetrics.compute_cumulative_returns(
            df["optimized_return"].values
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df.index, baseline_cumret * 100, linewidth=2, label="Baseline (Equal-Weight)",
               color="#3a86ff")
        ax.plot(df.index, optimized_cumret * 100, linewidth=2,
               label=f"Optimized ({self.optimization_method.upper()})",
               color="#fb8500")
        ax.fill_between(df.index, baseline_cumret * 100, optimized_cumret * 100,
                       alpha=0.2, color="gray")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_title("Baseline vs Optimized Portfolio: Cumulative Returns", fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
            logger.info(f"Saved return comparison plot to {output_path}")

        return fig

    def summary(self) -> pd.DataFrame:
        """
        Generate a summary comparison table.

        Returns
        -------
        pd.DataFrame
            Comparison metrics (Baseline vs Optimized)
        """
        if self.comparison_metrics is None:
            raise RuntimeError("Must call run() before summary()")

        metrics = self.comparison_metrics
        rows = [
            ("Strategy", "Baseline (Equal-Weight)  |  Optimized"),
            ("", ""),
            ("Sharpe Ratio (Annual)", f"{metrics['baseline_sharpe']:.4f}  →  {metrics['optimized_sharpe']:.4f}"),
            ("Sharpe Improvement", f"{metrics['sharpe_improvement']:+.4f}"),
            ("", ""),
            ("Volatility (Annual)", f"{metrics['baseline_volatility']*100:.2f}%  →  {metrics['optimized_volatility']*100:.2f}%"),
            ("Volatility Reduction", f"{metrics['volatility_reduction']*100:+.2f}%"),
            ("", ""),
            ("Max Drawdown", f"{metrics['baseline_max_dd']*100:.2f}%  →  {metrics['optimized_max_dd']*100:.2f}%"),
            ("Drawdown Reduction", f"{metrics['dd_reduction']*100:+.2f}%"),
            ("", ""),
            ("Cumulative Return", f"{metrics['baseline_cumulative_return']*100:.2f}%  →  {metrics['optimized_cumulative_return']*100:.2f}%"),
            ("Return Improvement", f"{metrics['return_improvement']*100:+.2f}%"),
        ]

        return pd.DataFrame(rows, columns=["Metric", "Value"])

