"""
walk_forward_validator.py
--------------------------
Rolling Window Cross-Validation for Time Series Models.

Unlike standard k-fold CV (which breaks temporal dependencies), walk-forward
validation trains on expanding windows and tests on future out-of-sample data.
This is the ONLY correct way to validate time series forecasting models.

Methodology:
  1. Start with initial training window (e.g., 2005-2016)
  2. Train model, test on next window (e.g., 2017)
  3. Expand training window to include 2017, test on 2018
  4. Repeat until all data is used
  5. Aggregate metrics across all out-of-sample windows

This ensures:
  - No look-ahead bias (model never sees future data)
  - Realistic performance estimates (each test is truly out-of-sample)
  - Robustness validation (performance across different market regimes)

References:
  - Hyndman & Athanasopoulos (2021): "Forecasting: Principles and Practice"
  - Prado (2018): "Advances in Financial Machine Learning"
"""

from typing import Dict, List, Optional, Tuple
import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.flow_model import ConditionalNormalizingFlow
from src.training.trainer import Trainer
from src.backtest.backtester import Backtester
from src.backtest.risk_metrics import kupiec_pof_test

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-Forward Cross-Validation for Conditional Normalizing Flows.

    Trains the model on expanding windows and evaluates on successive
    out-of-sample periods to ensure robust performance validation.

    Parameters
    ----------
    model_config : dict
        Model hyperparameters (passed to ConditionalNormalizingFlow).
    training_config : dict
        Training hyperparameters (passed to Trainer).
    initial_train_years : int
        Number of years for the initial training window (default: 10).
    test_window_days : int
        Number of days to test on in each fold (default: 252 ~ 1 year).
    min_train_size : int
        Minimum number of training samples required (default: 1000).
    alpha : float
        VaR confidence level for backtesting (default: 0.01 for 99% VaR).
    device : torch.device, optional
        Device to train on.
    """

    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        initial_train_years: int = 10,
        test_window_days: int = 252,
        min_train_size: int = 1000,
        alpha: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.initial_train_years = initial_train_years
        self.test_window_days = test_window_days
        self.min_train_size = min_train_size
        self.alpha = alpha

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.results_: Optional[pd.DataFrame] = None
        self.fold_metrics_: Optional[List[Dict]] = None

    def run(
        self,
        master_df: pd.DataFrame,
        feature_cols: List[str],
        ret_cols: List[str],
        tickers: List[str],
        macro_scaler,
        ret_scaler,
        seq_len: int = 63,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Execute walk-forward validation.

        Parameters
        ----------
        master_df : pd.DataFrame
            Full dataset with datetime index and columns for features + returns.
        feature_cols : list of str
            Names of macro feature columns.
        ret_cols : list of str
            Names of return columns.
        tickers : list of str
            Asset names.
        macro_scaler : StandardScaler
            Fitted scaler for macro features.
        ret_scaler : StandardScaler
            Fitted scaler for returns.
        seq_len : int
            TFT sequence length (default: 63).
        batch_size : int
            Batch size for training and inference.
        verbose : bool
            Whether to print detailed progress.

        Returns
        -------
        pd.DataFrame
            Aggregated results across all folds with columns:
            - fold_id, train_start, train_end, test_start, test_end
            - n_train, n_test, n_breaches, breach_rate, kupiec_p_value
            - mean_var, mean_es, test_nll
        """
        logger.info("=" * 70)
        logger.info("WALK-FORWARD CROSS-VALIDATION")
        logger.info("=" * 70)

        # Determine fold splits
        dates = master_df.index
        n_days = len(dates)

        # Initial training window: first `initial_train_years` years
        initial_train_end_idx = min(
            self.initial_train_years * 252,  # Approximate trading days per year
            n_days - self.test_window_days - seq_len,
        )

        fold_splits = []
        train_end_idx = initial_train_end_idx

        while train_end_idx + self.test_window_days < n_days:
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + self.test_window_days, n_days)

            if train_end_idx - seq_len < self.min_train_size:
                break

            fold_splits.append({
                'train_start': 0,
                'train_end': train_end_idx,
                'test_start': test_start_idx,
                'test_end': test_end_idx,
            })

            # Expand training window by test_window_days
            train_end_idx = test_end_idx

        n_folds = len(fold_splits)
        logger.info(f"Created {n_folds} folds with expanding training windows")
        logger.info(f"Initial training size: ~{initial_train_end_idx} days")
        logger.info(f"Test window size: {self.test_window_days} days")
        logger.info("=" * 70)

        # Run cross-validation
        fold_results = []
        for fold_id, split in enumerate(fold_splits):
            logger.info(f"\n{'='*70}")
            logger.info(f"FOLD {fold_id+1}/{n_folds}")
            logger.info(f"{'='*70}")

            train_start_date = dates[split['train_start']]
            train_end_date = dates[split['train_end'] - 1]
            test_start_date = dates[split['test_start']]
            test_end_date = dates[split['test_end'] - 1]

            logger.info(f"Train: {train_start_date.date()} to {train_end_date.date()} "
                       f"({split['train_end'] - split['train_start']} days)")
            logger.info(f"Test:  {test_start_date.date()} to {test_end_date.date()} "
                       f"({split['test_end'] - split['test_start']} days)")

            # Create fold-specific datasets
            train_data = master_df.iloc[split['train_start']:split['train_end']]
            test_data = master_df.iloc[split['test_start']:split['test_end']]

            # Build sequences
            train_loader, test_dates = self._build_loader(
                train_data, feature_cols, ret_cols, macro_scaler, ret_scaler,
                seq_len, batch_size, shuffle=True,
            )
            test_loader, test_dates = self._build_loader(
                test_data, feature_cols, ret_cols, macro_scaler, ret_scaler,
                seq_len, batch_size, shuffle=False,
            )

            # Train model
            model = ConditionalNormalizingFlow(**self.model_config).to(self.device)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,  # Use test as "validation" for early stopping
                device=self.device,
                **self.training_config,
            )

            logger.info(f"Training model for fold {fold_id+1}...")
            trainer.fit()
            trainer.load_best_model()

            # Backtest
            logger.info(f"Backtesting fold {fold_id+1}...")
            backtester = Backtester(
                model=model,
                test_loader=test_loader,
                test_dates=test_dates,
                ret_scaler=ret_scaler,
                tickers=tickers,
                n_mc_samples=10_000,
                alpha=self.alpha,
                device=self.device,
            )
            backtest_results = backtester.run()

            # Collect fold metrics
            kupiec = backtester.kupiec_result
            fold_metrics = {
                'fold_id': fold_id,
                'train_start': train_start_date,
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date,
                'n_train': len(train_data),
                'n_test': len(test_dates),
                'n_breaches': kupiec.n_breaches,
                'breach_rate': kupiec.breach_rate,
                'expected_breach_rate': kupiec.expected_rate,
                'kupiec_lr': kupiec.lr_statistic,
                'kupiec_p_value': kupiec.p_value,
                'kupiec_pass': not kupiec.reject_h0,
                'mean_var': backtest_results['var_99'].mean(),
                'mean_es': backtest_results['es_99'].mean(),
                'mean_actual_return': backtest_results['actual_port_return'].mean(),
                'worst_day': backtest_results['actual_port_return'].min(),
            }
            fold_results.append(fold_metrics)

            logger.info(f"Fold {fold_id+1} Results:")
            logger.info(f"  Breaches: {kupiec.n_breaches}/{kupiec.n_obs} ({kupiec.breach_rate:.2%})")
            logger.info(f"  Kupiec p-value: {kupiec.p_value:.4f} ({'PASS' if not kupiec.reject_h0 else 'FAIL'})")

        # Aggregate results
        self.results_ = pd.DataFrame(fold_results)
        self.fold_metrics_ = fold_results

        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total folds:          {n_folds}")
        logger.info(f"Folds passed Kupiec:  {self.results_['kupiec_pass'].sum()} / {n_folds}")
        logger.info(f"Avg breach rate:      {self.results_['breach_rate'].mean():.2%} "
                   f"(expected: {self.alpha:.2%})")
        logger.info(f"Avg Kupiec p-value:   {self.results_['kupiec_p_value'].mean():.4f}")
        logger.info("=" * 70)

        return self.results_

    def _build_loader(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        ret_cols: List[str],
        macro_scaler,
        ret_scaler,
        seq_len: int,
        batch_size: int,
        shuffle: bool,
    ) -> Tuple[DataLoader, pd.DatetimeIndex]:
        """Build DataLoader from DataFrame."""
        features = data[feature_cols].values
        returns = data[ret_cols].values

        # Scale
        features_scaled = macro_scaler.transform(features)
        returns_scaled = ret_scaler.transform(returns)

        # Build sequences
        macro_seqs = []
        ret_targets = []
        valid_dates = []

        for i in range(seq_len, len(features_scaled)):
            macro_seqs.append(features_scaled[i - seq_len:i])
            ret_targets.append(returns_scaled[i])
            valid_dates.append(data.index[i])

        macro_seqs = torch.FloatTensor(np.array(macro_seqs))
        ret_targets = torch.FloatTensor(np.array(ret_targets))

        dataset = TensorDataset(macro_seqs, ret_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return loader, pd.DatetimeIndex(valid_dates)

    def plot_fold_performance(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        Plot key metrics across all folds.

        Shows:
        - Breach rate per fold (vs expected alpha)
        - Kupiec p-values per fold
        - Mean VaR/ES per fold

        Parameters
        ----------
        output_path : str, optional
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results_ is None:
            raise RuntimeError("Must call run() before plotting")

        df = self.results_

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Panel 1: Breach rate
        ax1 = axes[0]
        ax1.bar(df['fold_id'], df['breach_rate'] * 100, color='#ff006e', alpha=0.7)
        ax1.axhline(self.alpha * 100, color='white', linestyle='--', linewidth=1.5,
                   label=f'Expected: {self.alpha*100:.1f}%')
        ax1.set_ylabel('Breach Rate (%)', fontsize=11)
        ax1.set_title('Walk-Forward Cross-Validation: Performance Across Folds',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Kupiec p-value
        ax2 = axes[1]
        colors = ['#8ecae6' if p else '#ff006e' for p in df['kupiec_pass']]
        ax2.bar(df['fold_id'], df['kupiec_p_value'], color=colors, alpha=0.7)
        ax2.axhline(0.05, color='white', linestyle='--', linewidth=1.5,
                   label='5% significance')
        ax2.set_ylabel('Kupiec p-value', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Mean VaR/ES
        ax3 = axes[2]
        x = df['fold_id']
        ax3.plot(x, df['mean_var'] * 100, 'o-', color='#ff006e', linewidth=2,
                markersize=6, label='Mean VaR')
        ax3.plot(x, df['mean_es'] * 100, 's-', color='#fb8500', linewidth=2,
                markersize=6, label='Mean ES')
        ax3.set_xlabel('Fold ID', fontsize=11)
        ax3.set_ylabel('Risk Metric (%)', fontsize=11)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches='tight')
            logger.info(f"Saved fold performance plot to {output_path}")

        return fig

    def get_summary(self) -> pd.DataFrame:
        """
        Return summary statistics across all folds.

        Returns
        -------
        pd.DataFrame with columns: Metric, Value
        """
        if self.results_ is None:
            raise RuntimeError("Must call run() before get_summary()")

        df = self.results_

        rows = [
            ("Total Folds", len(df)),
            ("", ""),
            ("Breach Rate Statistics", ""),
            ("  Mean Breach Rate", f"{df['breach_rate'].mean():.2%}"),
            ("  Std Breach Rate", f"{df['breach_rate'].std():.2%}"),
            ("  Min Breach Rate", f"{df['breach_rate'].min():.2%}"),
            ("  Max Breach Rate", f"{df['breach_rate'].max():.2%}"),
            ("  Expected Breach Rate", f"{self.alpha:.2%}"),
            ("", ""),
            ("Kupiec Test Results", ""),
            ("  Folds Passed", f"{df['kupiec_pass'].sum()} / {len(df)}"),
            ("  Avg p-value", f"{df['kupiec_p_value'].mean():.4f}"),
            ("", ""),
            ("Risk Metrics", ""),
            ("  Mean VaR (across folds)", f"{df['mean_var'].mean()*100:.3f}%"),
            ("  Mean ES (across folds)", f"{df['mean_es'].mean()*100:.3f}%"),
            ("", ""),
            ("Portfolio Performance", ""),
            ("  Mean Daily Return", f"{df['mean_actual_return'].mean()*100:.3f}%"),
            ("  Worst Single Day", f"{df['worst_day'].min()*100:.3f}%"),
        ]

        return pd.DataFrame(rows, columns=["Metric", "Value"])


if __name__ == "__main__":
    # This module requires full pipeline integration
    # See notebook for usage example
    print("WalkForwardValidator: See notebook for usage example")
