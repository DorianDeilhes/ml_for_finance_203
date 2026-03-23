"""
backtester.py
-------------
Out-of-sample backtester for the macro-conditional normalizing flow.

Runs the trained model on the test set to produce daily VaR and ES estimates,
detects breaches, and validates calibration via Kupiec's POF test.
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

from src.backtest.risk_metrics import (
    KupiecResult,
    compute_es,
    compute_var,
    compute_portfolio_stats,
    kupiec_pof_test,
)
from src.models.flow_model import ConditionalNormalizingFlow

logger = logging.getLogger(__name__)


class Backtester:
    """
    Out-of-sample backtester for the ConditionalNormalizingFlow.

    For each test day, encodes the macro context via the TFT, draws Monte Carlo
    samples from the conditional distribution, computes VaR and ES, and records
    whether the actual return breached the VaR. Runs Kupiec's POF test on the
    full breach series.

    Parameters
    ----------
    model : ConditionalNormalizingFlow
        Trained model (best checkpoint should be loaded before calling run()).
    test_loader : DataLoader
        Test set DataLoader yielding (macro_seq, returns). Must not be shuffled.
    test_dates : np.ndarray
        Dates corresponding to each test observation.
    ret_scaler : StandardScaler
        Scaler used on returns, applied in inverse to recover original scale.
    tickers : list of str
        Asset names for labeling.
    n_mc_samples : int
        Number of Monte Carlo samples per day.
    alpha : float
        VaR tail probability (default: 0.01 for 99% VaR).
    portfolio_weights : np.ndarray, optional
        Asset weights. Defaults to equal weighting.
    device : torch.device, optional
        Inference device. Defaults to CUDA if available, else CPU.
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
        self.test_dates = pd.to_datetime(test_dates)
        self.ret_scaler = ret_scaler
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

        self.results: Optional[pd.DataFrame] = None
        self._stored_port_samples: dict = {}

    def run(self) -> pd.DataFrame:
        """
        Execute the full out-of-sample backtest.

        TFT encoding is batched over the entire test set first to pre-compute
        all context vectors h_t. Per-day MAF sampling then uses these cached
        vectors, avoiding redundant TFT passes in the inner loop.

        Returns
        -------
        pd.DataFrame
            Backtest results indexed by date. Columns include actual portfolio
            return, var_99, es_99, breach flag, percentage forms, and per-asset
            actual returns.
        """
        self.model.eval()
        self.model.to(self.device)

        logger.info(
            "Pre-computing context vectors h_t for all %d test days...",
            len(self.test_dates),
        )
        all_h_t: List[torch.Tensor] = []
        all_returns: List[np.ndarray] = []

        with torch.no_grad():
            for macro_seq, returns in self.test_loader:
                h_t, _ = self.model.tft(macro_seq.to(self.device))
                all_h_t.append(h_t.cpu())
                all_returns.append(returns.numpy())

        all_h_t_tensor = torch.cat(all_h_t, dim=0)
        all_returns_arr = np.concatenate(all_returns, axis=0)

        logger.info(
            "Sampling %d MC draws per day for %d days...",
            self.n_mc_samples, len(all_h_t_tensor),
        )

        var_list: List[float] = []
        es_list: List[float] = []
        _port_sample_store: dict = {}

        with torch.no_grad():
            for i in tqdm(range(len(all_h_t_tensor)), desc="Backtesting"):
                h_t_i = all_h_t_tensor[i:i + 1].to(self.device)
                samples = self.model.flow.sample(
                    self.n_mc_samples, context=h_t_i
                ).cpu().numpy()
                samples_unscaled = self.ret_scaler.inverse_transform(samples)

                var_list.append(compute_var(
                    samples_unscaled, alpha=self.alpha,
                    portfolio_weights=self.portfolio_weights,
                ))
                es_list.append(compute_es(
                    samples_unscaled, alpha=self.alpha,
                    portfolio_weights=self.portfolio_weights,
                ))
                _port_sample_store[i] = samples_unscaled @ self.portfolio_weights

        actual_unscaled = self.ret_scaler.inverse_transform(all_returns_arr)
        port_returns = actual_unscaled @ self.portfolio_weights

        n_dates = min(len(self.test_dates), len(var_list))
        dates = self.test_dates[:n_dates]
        var_arr = np.array(var_list[:n_dates])
        es_arr = np.array(es_list[:n_dates])
        port_arr = port_returns[:n_dates]
        actual_arr = actual_unscaled[:n_dates]
        breaches = (port_arr < var_arr).astype(int)

        results = pd.DataFrame({
            "actual_port_return": port_arr,
            "var_99": var_arr,
            "es_99": es_arr,
            "breach": breaches,
            "var_99_pct": var_arr * 100,
            "es_99_pct": es_arr * 100,
        }, index=dates)

        for j, ticker in enumerate(self.tickers):
            results[f"{ticker}_actual"] = actual_arr[:, j]
        self.results = results

        breach_idx = np.where(breaches)[0]
        calm_idx = np.where(~breaches.astype(bool))[0]
        keep_idx = set()
        if len(breach_idx) > 0:
            keep_idx.update(breach_idx[:3].tolist())
        step = max(1, len(calm_idx) // 3)
        keep_idx.update(calm_idx[::step][:3].tolist())
        self._stored_port_samples = {
            dates[i]: _port_sample_store[i] for i in keep_idx if i < n_dates
        }

        n_breaches = int(breaches.sum())
        n_total = len(breaches)
        kupiec_result = kupiec_pof_test(n_breaches, n_total, self.alpha)

        logger.info("=== Backtest Results ===")
        logger.info(
            "Period: %s to %s (%d days)", dates[0].date(), dates[-1].date(), n_total,
        )
        logger.info(
            "VaR breaches: %d / %d (%.2f%% observed vs %.2f%% expected)",
            n_breaches, n_total, n_breaches / n_total * 100, self.alpha * 100,
        )
        logger.info(
            "Kupiec LR: %.4f, p-value: %.4f",
            kupiec_result.lr_statistic, kupiec_result.p_value,
        )
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
        ax1, ax2 = axes

        ax1.plot(
            df.index, df["actual_port_return"] * 100,
            color="#3a86ff", linewidth=0.8, alpha=0.9, label="Actual Portfolio Return",
        )
        ax1.fill_between(
            df.index, df["var_99_pct"], df["var_99_pct"] - 1.0,
            alpha=0.3, color="#ff006e", label="99% VaR",
        )
        ax1.plot(
            df.index, df["var_99_pct"],
            color="#ff006e", linewidth=1.2, linestyle="--",
        )
        ax1.plot(
            df.index, df["es_99_pct"],
            color="#fb8500", linewidth=1.0, linestyle=":",
            alpha=0.8, label="99% Expected Shortfall",
        )

        breach_dates = df.index[df["breach"] == 1]
        breach_returns = df.loc[df["breach"] == 1, "actual_port_return"] * 100
        ax1.scatter(
            breach_dates, breach_returns,
            color="#ff006e", s=30, zorder=5,
            label=f"VaR Breaches ({len(breach_dates)})", marker="v",
        )
        ax1.axhline(0, color="white", linewidth=0.5, alpha=0.5)
        ax1.set_ylabel("Daily Portfolio Return (%)", fontsize=11)
        ax1.set_title(title, fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

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

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        kupiec = self.kupiec_result
        ax1.annotate(
            (f"Kupiec's POF: LR={kupiec.lr_statistic:.3f}, "
             f"p={kupiec.p_value:.3f} "
             f"({'PASS' if not kupiec.reject_h0 else 'FAIL'})"),
            xy=(0.02, 0.04),
            xycoords="axes fraction",
            fontsize=9,
            color="#8ecae6" if not kupiec.reject_h0 else "#ff006e",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
        )

        plt.tight_layout()

        if output_path:
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("VaR plot saved to %s", output_path)

        return fig

    def plot_return_distributions(
        self,
        n_days: int = 5,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the predicted portfolio return distribution for selected test days.

        Shows how the model adapts to different macro regimes by displaying
        the MC histogram alongside VaR and actual return for breach days
        and calm days stored during run().

        Parameters
        ----------
        n_days : int
            Maximum number of days to plot.
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

            ax.hist(
                port_samples * 100, bins=80, density=True,
                color="#3a86ff", alpha=0.6, label="MC distribution",
            )
            ax.axvline(
                row["var_99"] * 100, color="#ff006e", linestyle="--",
                linewidth=1.8, label=f"VaR: {row['var_99'] * 100:.2f}%",
            )
            ax.axvline(
                row["actual_port_return"] * 100, color="#fb8500", linestyle="-",
                linewidth=2, label=f"Actual: {row['actual_port_return'] * 100:.2f}%",
            )
            is_breach = bool(row["breach"])
            ax.set_title(
                f"{date.date()}" + (" ← BREACH" if is_breach else ""),
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
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    def summary(self) -> pd.DataFrame:
        """
        Return a summary table of key backtest metrics.

        Returns
        -------
        pd.DataFrame
            Two columns: Metric and Value.
        """
        if self.results is None:
            raise RuntimeError("Must call run() before summary().")

        df = self.results
        kupiec = self.kupiec_result
        rows = [
            ("Backtest Period", f"{df.index[0].date()} to {df.index[-1].date()}"),
            ("Total Days", len(df)),
            ("VaR Confidence Level", f"{(1 - self.alpha) * 100:.0f}%"),
            ("Expected Breaches", f"{kupiec.expected_breaches:.1f}"),
            ("Observed Breaches", kupiec.n_breaches),
            ("Observed Breach Rate", f"{kupiec.breach_rate:.2%}"),
            ("Kupiec LR Statistic", f"{kupiec.lr_statistic:.4f}"),
            ("Kupiec p-value", f"{kupiec.p_value:.4f}"),
            ("Kupiec Result", "PASS" if not kupiec.reject_h0 else "FAIL"),
            ("Mean Daily VaR", f"{df['var_99'].mean() * 100:.3f}%"),
            ("Mean Daily ES", f"{df['es_99'].mean() * 100:.3f}%"),
            ("Mean Actual Daily Return", f"{df['actual_port_return'].mean() * 100:.3f}%"),
            ("Worst Day (actual)", f"{df['actual_port_return'].min() * 100:.3f}%"),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])