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
        Test set DataLoader (macro_seq, returns) — must NOT be shuffled.
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

    def run(self) -> pd.DataFrame:
        """
        Execute the full out-of-sample backtest.

        For each test day t:
        - Encode macro context h_t
        - Draw n_mc_samples from p(X_t | h_t)
        - Unscale samples to true return scale
        - Compute VaR and ES for the portfolio
        - Record actual portfolio return and breach indicator

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

        var_list: List[float] = []
        es_list:  List[float] = []
        actual_returns_list: List[np.ndarray] = []
        date_idx = 0

        logger.info(
            "Running backtest: %d days, %d MC samples each...",
            len(self.test_dates), self.n_mc_samples,
        )

        with torch.no_grad():
            for macro_seq, returns in tqdm(self.test_loader, desc="Backtesting"):
                batch_size = macro_seq.shape[0]
                macro_seq = macro_seq.to(self.device)
                returns   = returns.to(self.device)

                # Process each day in the batch individually (each needs own samples)
                for i in range(batch_size):
                    # Single macro sequence (1, seq_len, F)
                    seq_i = macro_seq[i:i+1]
                    ret_i = returns[i].cpu().numpy()

                    # Draw Monte Carlo samples from conditional flow
                    samples = self.model.sample(seq_i, n_samples=self.n_mc_samples)
                    samples_np = samples.cpu().numpy()  # (n_samples, D)

                    # Invert scaling: bring samples back to true log-return scale
                    samples_unscaled = self.ret_scaler.inverse_transform(samples_np)
                    ret_i_unscaled   = self.ret_scaler.inverse_transform(ret_i.reshape(1, -1)).flatten()

                    # Compute portfolio risk metrics
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
                    actual_returns_list.append(ret_i_unscaled)

        # Build results DataFrame
        actual_returns_arr = np.array(actual_returns_list)  # (T, D)
        port_returns = actual_returns_arr @ self.portfolio_weights

        n_dates = min(len(self.test_dates), len(var_list))
        dates = self.test_dates[:n_dates]
        var_arr = np.array(var_list[:n_dates])
        es_arr  = np.array(es_list[:n_dates])
        port_arr = port_returns[:n_dates]
        actual_arr = actual_returns_arr[:n_dates]

        breaches = (port_arr < var_arr).astype(int)

        results = pd.DataFrame({
            "actual_port_return": port_arr,
            "var_99":  var_arr,
            "es_99":   es_arr,
            "breach":  breaches,
            "var_99_pct": var_arr * 100,
            "es_99_pct":  es_arr  * 100,
        }, index=dates)

        # Add individual asset returns
        for j, ticker in enumerate(self.tickers):
            results[f"{ticker}_actual"] = actual_arr[:, j]

        self.results = results

        # Run Kupiec's test
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
            f"({'PASS ✓' if not kupiec.reject_h0 else 'FAIL ✗'})"
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
        Plot the predicted return distribution for selected test days,
        showing how the model adapts to different macro regimes.

        Parameters
        ----------
        n_days : int
            Number of representative days to plot.
        output_path : str, optional
            File path to save the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise RuntimeError("Must call run() before plotting.")

        df = self.results

        # Select representative days: some breach, some calm
        breach_days = df[df["breach"] == 1].index
        calm_days   = df[df["breach"] == 0].index

        selected = []
        if len(breach_days) > 0:
            selected.append(breach_days[len(breach_days) // 2])
        selected += list(calm_days[::len(calm_days) // (n_days - len(selected))])[:n_days - len(selected)]

        fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4))
        if len(selected) == 1:
            axes = [axes]

        for ax, date in zip(axes, selected):
            row = df.loc[date]
            ax.axvline(
                row["var_99"] * 100, color="#ff006e", linestyle="--",
                linewidth=1.5, label=f"VaR: {row['var_99']*100:.2f}%",
            )
            ax.axvline(
                row["actual_port_return"] * 100, color="#3a86ff", linestyle="-",
                linewidth=2, label=f"Actual: {row['actual_port_return']*100:.2f}%",
            )
            ax.set_title(str(date.date()), fontsize=10)
            ax.set_xlabel("Portfolio Return (%)")
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
            ("Kupiec Result", "PASS ✓" if not kupiec.reject_h0 else "FAIL ✗"),
            ("Mean Daily VaR", f"{df['var_99'].mean()*100:.3f}%"),
            ("Mean Daily ES", f"{df['es_99'].mean()*100:.3f}%"),
            ("Mean Actual Daily Return", f"{df['actual_port_return'].mean()*100:.3f}%"),
            ("Worst Day (actual)", f"{df['actual_port_return'].min()*100:.3f}%"),
        ]

        return pd.DataFrame(rows, columns=["Metric", "Value"])
