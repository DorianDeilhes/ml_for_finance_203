"""
regime_backtester.py
--------------------
Regime-Specific Backtesting and Performance Analysis.

Combines regime detection with financial backtesting to answer:
  1. Does VaR accuracy vary across regimes?
  2. Are some regimes inherently harder to model (high volatility, tail events)?
  3. Does the model consistently under/over-estimate risk in specific regimes?

This is the KEY analysis for validating that the TFT genuinely captures
macro dynamics rather than just lucky test-period performance.

Methodology:
  1. Extract h_t embeddings for all test days
  2. Cluster embeddings to identify regimes
  3. Group backtest results by regime
  4. Compute regime-specific metrics: VaR breach rate, ES accuracy, Kupiec test
  5. Visualize: performance per regime, regime transitions

References:
  - Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series"
  - Ang & Timmermann (2012): "Regime Changes and Financial Markets"
"""

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import torch

from src.backtest.regime_detector import RegimeDetector
from src.backtest.risk_metrics import kupiec_pof_test, compute_var, compute_es

logger = logging.getLogger(__name__)


class RegimeBacktester:
    """
    Performs regime-conditional backtesting of VaR and ES predictions.

    Workflow:
      1. Run standard backtest to get VaR/ES predictions for all test days
      2. Extract TFT embeddings (h_t) for test days
      3. Cluster embeddings to detect macro regimes
      4. Group backtest results by regime
      5. Compute regime-specific performance metrics

    Parameters
    ----------
    baseline_backtester : Backtester
        A completed baseline backtest (with results available).
    n_regimes : int
        Number of regimes to detect (default: 3).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        baseline_backtester,
        n_regimes: int = 3,
        random_state: int = 42,
    ):
        self.baseline_backtester = baseline_backtester
        self.n_regimes = n_regimes
        self.random_state = random_state

        self.regime_detector = RegimeDetector(
            n_regimes=n_regimes,
            random_state=random_state,
            standardize_embeddings=True,
        )

        self.regime_labels_: Optional[np.ndarray] = None
        self.regime_results_: Optional[pd.DataFrame] = None
        self.regime_metrics_: Optional[Dict[int, Dict]] = None

    def run(self) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
        """
        Run regime-conditional backtesting.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, Dict]]
            (results_with_regimes_df, regime_specific_metrics)

        results_with_regimes_df: Backtest results with added 'regime' column
        regime_specific_metrics: Dict mapping regime_id -> metrics
            Metrics per regime:
            - n_days
            - n_breaches
            - breach_rate
            - expected_breach_rate
            - kupiec_result (KupiecResult object)
            - mean_var, mean_es
            - mean_actual_return
            - worst_day
        """
        logger.info("=" * 70)
        logger.info("REGIME-CONDITIONAL BACKTESTING")
        logger.info("=" * 70)

        # Extract TFT embeddings for test days
        logger.info("Extracting TFT embeddings for test days...")
        model = self.baseline_backtester.model
        test_loader = self.baseline_backtester.test_loader
        device = self.baseline_backtester.device

        embeddings = self.regime_detector.extract_embeddings(
            model, test_loader, device
        )

        # Detect regimes via clustering
        logger.info(f"Detecting {self.n_regimes} macro regimes via K-Means...")
        test_dates = self.baseline_backtester.test_dates
        self.regime_labels_ = self.regime_detector.fit(embeddings, dates=test_dates)

        # Add regime labels to backtest results
        results = self.baseline_backtester.results.copy()

        # Align regime labels with results (they should have same length)
        if len(self.regime_labels_) != len(results):
            logger.warning(
                f"Regime labels ({len(self.regime_labels_)}) and results ({len(results)}) "
                f"length mismatch. Truncating to minimum."
            )
            min_len = min(len(self.regime_labels_), len(results))
            self.regime_labels_ = self.regime_labels_[:min_len]
            results = results.iloc[:min_len]

        results['regime'] = self.regime_labels_
        self.regime_results_ = results

        # Compute regime-specific metrics
        logger.info("\nComputing regime-specific performance metrics...")
        self.regime_metrics_ = {}

        alpha = self.baseline_backtester.alpha

        for regime_id in range(self.n_regimes):
            regime_mask = results['regime'] == regime_id
            regime_data = results[regime_mask]

            if len(regime_data) == 0:
                logger.warning(f"Regime {regime_id} has no observations, skipping.")
                continue

            # VaR breaches
            breaches = regime_data['breach'].sum()
            n_obs = len(regime_data)

            # Kupiec test for this regime
            kupiec = kupiec_pof_test(
                breaches=int(breaches),
                n=n_obs,
                alpha=alpha,
            )

            self.regime_metrics_[regime_id] = {
                'n_days': n_obs,
                'n_breaches': int(breaches),
                'breach_rate': breaches / n_obs,
                'expected_breach_rate': alpha,
                'kupiec_result': kupiec,
                'mean_var': regime_data['var_99'].mean(),
                'mean_es': regime_data['es_99'].mean(),
                'mean_actual_return': regime_data['actual_port_return'].mean(),
                'std_actual_return': regime_data['actual_port_return'].std(),
                'worst_day': regime_data['actual_port_return'].min(),
                'best_day': regime_data['actual_port_return'].max(),
            }

            logger.info(f"\nRegime {regime_id}:")
            logger.info(f"  Days: {n_obs}")
            logger.info(f"  Breaches: {int(breaches)} ({breaches/n_obs:.2%} vs {alpha:.2%} expected)")
            logger.info(f"  Kupiec p-value: {kupiec.p_value:.4f} ({'PASS' if not kupiec.reject_h0 else 'FAIL'})")
            logger.info(f"  Mean VaR: {regime_data['var_99'].mean()*100:.3f}%")
            logger.info(f"  Mean actual return: {regime_data['actual_port_return'].mean()*100:.3f}%")

        logger.info("\n" + "=" * 70)
        logger.info("REGIME-CONDITIONAL SUMMARY")
        logger.info("=" * 70)

        # Overall statistics
        n_regimes_passed = sum(
            1 for m in self.regime_metrics_.values() if not m['kupiec_result'].reject_h0
        )
        logger.info(f"Regimes passed Kupiec test: {n_regimes_passed} / {self.n_regimes}")

        # Check if VaR accuracy varies significantly across regimes
        breach_rates = [m['breach_rate'] for m in self.regime_metrics_.values()]
        logger.info(f"Breach rate range: {min(breach_rates):.2%} to {max(breach_rates):.2%}")
        logger.info("=" * 70)

        return self.regime_results_, self.regime_metrics_

    def get_regime_summary(self) -> pd.DataFrame:
        """
        Return a summary table of regime-specific metrics.

        Returns
        -------
        pd.DataFrame with columns:
            - regime_id
            - n_days
            - breach_rate
            - kupiec_pass
            - mean_var
            - mean_es
            - mean_return
            - worst_day
        """
        if self.regime_metrics_ is None:
            raise RuntimeError("Must call run() before get_regime_summary()")

        rows = []
        for regime_id, metrics in self.regime_metrics_.items():
            kupiec = metrics['kupiec_result']
            rows.append({
                'regime_id': regime_id,
                'n_days': metrics['n_days'],
                'breach_rate': f"{metrics['breach_rate']:.2%}",
                'expected_rate': f"{metrics['expected_breach_rate']:.2%}",
                'kupiec_pass': 'PASS' if not kupiec.reject_h0 else 'FAIL',
                'kupiec_p_value': f"{kupiec.p_value:.4f}",
                'mean_var': f"{metrics['mean_var']*100:.3f}%",
                'mean_es': f"{metrics['mean_es']*100:.3f}%",
                'mean_return': f"{metrics['mean_actual_return']*100:.3f}%",
                'worst_day': f"{metrics['worst_day']*100:.3f}%",
            })

        return pd.DataFrame(rows)

    def plot_regime_performance(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Plot comprehensive regime-specific performance dashboard.

        Shows:
        - Panel 1: Breach rates per regime (vs expected)
        - Panel 2: Mean VaR/ES per regime
        - Panel 3: Mean return and volatility per regime

        Parameters
        ----------
        output_path : str, optional
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.regime_metrics_ is None:
            raise RuntimeError("Must call run() before plot_regime_performance()")

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        regime_ids = sorted(self.regime_metrics_.keys())
        colors = sns.color_palette("husl", self.n_regimes)

        # Panel 1: Breach rates
        ax1 = axes[0]
        breach_rates = [
            self.regime_metrics_[rid]['breach_rate'] * 100
            for rid in regime_ids
        ]
        expected_rate = self.regime_metrics_[regime_ids[0]]['expected_breach_rate'] * 100

        ax1.bar(regime_ids, breach_rates, color=colors, alpha=0.8)
        ax1.axhline(expected_rate, color='white', linestyle='--', linewidth=1.5,
                   label=f'Expected: {expected_rate:.1f}%')
        ax1.set_ylabel('Breach Rate (%)', fontsize=11)
        ax1.set_title('Regime-Specific Performance: VaR Accuracy and Risk Metrics',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Panel 2: VaR and ES
        ax2 = axes[1]
        mean_vars = [
            self.regime_metrics_[rid]['mean_var'] * 100
            for rid in regime_ids
        ]
        mean_es = [
            self.regime_metrics_[rid]['mean_es'] * 100
            for rid in regime_ids
        ]

        x = np.arange(len(regime_ids))
        width = 0.35
        ax2.bar(x - width/2, mean_vars, width, color='#ff006e', alpha=0.8, label='Mean VaR')
        ax2.bar(x + width/2, mean_es, width, color='#fb8500', alpha=0.8, label='Mean ES')
        ax2.set_ylabel('Risk Metric (%)', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Regime {rid}' for rid in regime_ids])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Panel 3: Return statistics
        ax3 = axes[2]
        mean_returns = [
            self.regime_metrics_[rid]['mean_actual_return'] * 100
            for rid in regime_ids
        ]
        std_returns = [
            self.regime_metrics_[rid]['std_actual_return'] * 100
            for rid in regime_ids
        ]

        ax3.bar(x - width/2, mean_returns, width, color='#3a86ff', alpha=0.8,
               label='Mean Return')
        ax3.bar(x + width/2, std_returns, width, color='#8ecae6', alpha=0.8,
               label='Volatility (Std)')
        ax3.set_xlabel('Regime', fontsize=11)
        ax3.set_ylabel('Return/Vol (%)', fontsize=11)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Regime {rid}' for rid in regime_ids])
        ax3.axhline(0, color='white', linewidth=0.5, alpha=0.5)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches='tight')
            logger.info(f"Saved regime performance plot to {output_path}")

        return fig

    def plot_var_bands_by_regime(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 6),
    ) -> plt.Figure:
        """
        Plot VaR bands timeline with regime-based background coloring.

        Shows how VaR predictions and breaches correspond to regime transitions.

        Parameters
        ----------
        output_path : str, optional
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.regime_results_ is None:
            raise RuntimeError("Must call run() before plot_var_bands_by_regime()")

        df = self.regime_results_
        fig, ax = plt.subplots(figsize=figsize)

        # Plot regime backgrounds
        colors = sns.color_palette("husl", self.n_regimes)
        for regime_id in range(self.n_regimes):
            mask = df['regime'] == regime_id
            ax.fill_between(
                df.index, -10, 10, where=mask,
                color=colors[regime_id], alpha=0.15,
                label=f'Regime {regime_id}',
            )

        # Plot actual returns
        ax.plot(
            df.index, df['actual_port_return'] * 100,
            color='#3a86ff', linewidth=0.8, alpha=0.9, label='Actual Return',
        )

        # Plot VaR
        ax.plot(
            df.index, df['var_99_pct'],
            color='#ff006e', linewidth=1.2, linestyle='--', label='99% VaR',
        )

        # Mark breaches
        breach_dates = df.index[df['breach'] == 1]
        breach_returns = df.loc[df['breach'] == 1, 'actual_port_return'] * 100
        ax.scatter(
            breach_dates, breach_returns,
            color='#ff006e', s=30, zorder=5, label=f'Breaches ({len(breach_dates)})',
            marker='v',
        )

        ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Daily Portfolio Return (%)', fontsize=11)
        ax.set_title('VaR Bands with Detected Macro Regimes', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches='tight')
            logger.info(f"Saved VaR bands by regime plot to {output_path}")

        return fig


if __name__ == "__main__":
    # This module requires a completed backtester
    # See notebook for usage example
    print("RegimeBacktester: See notebook for usage example")
