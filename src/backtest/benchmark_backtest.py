"""
benchmark_backtest.py
---------------------
Backtest utilities shared by all VaR benchmark models.

Provides:
- run_benchmark_backtest : builds results DataFrame + Kupiec test from
  any (port_returns, var_series) pair.
- plot_var_bands : identical visual style as Backtester.plot_var_bands,
  with ES as an optional column (benchmarks produce no ES).
"""

import os
from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.risk_metrics import KupiecResult, kupiec_pof_test


def run_benchmark_backtest(
    port_returns: np.ndarray,
    var_series: np.ndarray,
    dates: np.ndarray,
    alpha: float = 0.01,
) -> Tuple[pd.DataFrame, KupiecResult]:
    """
    Build a backtest results DataFrame and run Kupiec's POF test.

    Parameters
    ----------
    port_returns : np.ndarray of shape (T,)
        Actual equal-weighted portfolio returns (unscaled).
    var_series : np.ndarray of shape (T,)
        Predicted VaR for each test day.
    dates : np.ndarray
        Trading dates aligned with port_returns.
    alpha : float
        VaR tail probability (default: 0.01 for 99% VaR).

    Returns
    -------
    results : pd.DataFrame
        Columns: actual_port_return, var_99, breach, var_99_pct.
    kupiec : KupiecResult
    """
    breaches = (port_returns < var_series).astype(int)
    kupiec = kupiec_pof_test(int(breaches.sum()), len(breaches), alpha)

    results = pd.DataFrame({
        "actual_port_return": port_returns,
        "var_99": var_series,
        "var_99_pct": var_series * 100,
        "breach": breaches,
    }, index=pd.to_datetime(dates))

    return results, kupiec


def plot_var_bands(
    results: pd.DataFrame,
    kupiec: KupiecResult,
    title: str = "99% VaR vs Actual Portfolio Returns",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 7),
) -> plt.Figure:
    """
    Plot VaR bands overlaid on actual portfolio returns.

    Matches the style of Backtester.plot_var_bands. ES band is shown
    only if an 'es_99' column is present in results.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_benchmark_backtest (or backtester.results).
    kupiec : KupiecResult
    title : str
    output_path : str, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    ax1.plot(results.index, results["actual_port_return"] * 100,
             color="#3a86ff", linewidth=0.8, alpha=0.9, label="Actual Portfolio Return")
    ax1.fill_between(results.index, results["var_99_pct"], results["var_99_pct"] - 1.0,
                     alpha=0.3, color="#ff006e", label="99% VaR")
    ax1.plot(results.index, results["var_99_pct"],
             color="#ff006e", linewidth=1.2, linestyle="--")

    if "es_99_pct" in results.columns:
        ax1.plot(results.index, results["es_99_pct"],
                 color="#fb8500", linewidth=1.0, linestyle=":", alpha=0.8,
                 label="99% Expected Shortfall")

    breach_mask = results["breach"] == 1
    ax1.scatter(results.index[breach_mask], results.loc[breach_mask, "actual_port_return"] * 100,
                color="#ff006e", s=30, zorder=5,
                label=f"VaR Breaches ({breach_mask.sum()})", marker="v")
    ax1.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax1.set_ylabel("Daily Portfolio Return (%)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        f"Kupiec's POF: LR={kupiec.lr_statistic:.3f}, p={kupiec.p_value:.3f} "
        f"({'PASS' if not kupiec.reject_h0 else 'FAIL'})",
        xy=(0.02, 0.04), xycoords="axes fraction", fontsize=9,
        color="#8ecae6" if not kupiec.reject_h0 else "#ff006e",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )

    ax2.bar(results.index, results["breach"], color="#ff006e", alpha=0.8, width=1.0,
            label=f"VaR Breach (n={int(results['breach'].sum())})")
    ax2.set_ylabel("Breach", fontsize=10)
    ax2.set_yticks([0, 1])
    ax2.set_xlabel("Date", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.2)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
