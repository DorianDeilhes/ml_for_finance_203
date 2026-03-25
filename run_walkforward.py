"""
run_walkforward.py
------------------
Executes the Macro-Conditional Normalizing Flow model using a 
Walk-Forward Expanding Window validation approach with Warm Starts.
"""

import logging
import os
import time

import pandas as pd
import torch
from dotenv import load_dotenv

from src.backtest.backtester import Backtester
from src.backtest.risk_metrics import kupiec_pof_test
from src.data.pipeline import build_walk_forward_pipeline, TICKERS
from src.models.flow_model import ConditionalNormalizingFlow
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        raise ValueError("FRED_API_KEY environment variable not set. Check your .env file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Initialize Walk-Forward Generator
    pipeline_generator = build_walk_forward_pipeline(
        fred_api_key=fred_api_key,
        seq_len=63,
        batch_size=64,
        start_year=2005,
        initial_train_end_year=2016,
        end_year=2023,
        val_years=1,
        test_years=1,
    )

    all_test_results = []
    
    model = None
    trainer = None
    fold = 1

    total_start_time = time.time()

    for train_loader, val_loader, test_loader, ret_scaler, info in pipeline_generator:
        logger.info("=" * 60)
        logger.info(f"Starting Walk-Forward Fold {fold}")
        logger.info(f"Train End: {info['train_end']} | Test: {info['test_start']} to {info['test_end']}")
        logger.info("=" * 60)

        # 2. Initialize Model on Fold 1
        if model is None:
            model = ConditionalNormalizingFlow(
                num_macro_features=info["num_macro_features"],
                num_assets=info["num_assets"],
                tft_d_model=64,
                tft_n_heads=4,
                tft_n_lstm_layers=2,
                flow_n_layers=6,
                flow_hidden_dim=64,
                flow_n_hidden=2,
                dropout=0.1,
            ).to(device)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=3e-4,
                weight_decay=1e-4,
                n_epochs=40,   # Keep epochs reasonable for full walk-forward runs
                patience=8,
                checkpoint_path=f"checkpoints/best_model_fold_{fold}.pt",
                device=device,
                warmup_epochs=2,
            )
        else:
            # Warm Start for subsequent folds
            logger.info("Warm-starting model from previous fold...")
            trainer.checkpoint_path = f"checkpoints/best_model_fold_{fold}.pt"
            trainer.reset_for_new_fold(new_train_loader=train_loader, new_val_loader=val_loader, reset_lr=True)

        # 3. Train the Model
        trainer.fit()
        trainer.load_best_model()

        # 4. Run Backtester Chunk
        backtester = Backtester(
            model=model,
            test_loader=test_loader,
            test_dates=info["dates_test"],
            ret_scaler=ret_scaler,
            tickers=info["tickers"],
            n_mc_samples=5_000,
            alpha=0.01,
            device=device,
        )

        logger.info("Running out-of-sample inference for the fold...")
        chunk_results = backtester.run_chunk()
        all_test_results.append(chunk_results)
        
        fold += 1

    elapsed = time.time() - total_start_time
    logger.info("=" * 60)
    logger.info(f"Walk-Forward Cross-Validation Complete in {elapsed/60:.2f} minutes.")
    logger.info("=" * 60)

    # 5. Aggregate Results
    df_results = pd.concat(all_test_results).sort_index()
    
    n_total = len(df_results)
    n_breaches = int(df_results["breach"].sum())
    alpha = 0.01

    logger.info(f"Aggregated Out-Of-Sample Period: {df_results.index[0].date()} to {df_results.index[-1].date()} ({n_total} days)")
    logger.info(f"Total aggregate VaR breaches: {n_breaches} / {n_total} ({(n_breaches/n_total)*100:.2f}%)")

    # 6. Global Kupiec Test
    kupiec = kupiec_pof_test(n_breaches, n_total, alpha)
    logger.info(f"Global Kupiec LR Statistic: {kupiec.lr_statistic:.4f}")
    logger.info(f"Global Kupiec p-value: {kupiec.p_value:.4f}")
    logger.info(f"Global Test Result: {'PASS' if not kupiec.reject_h0 else 'FAIL'}")

    # Optionally, we can inject these aggregated results back into a dummy backtester instance to reuse its plotting logic
    backtester.results = df_results
    backtester.kupiec_result = kupiec
    backtester.plot_var_bands(output_path="docs/walk_forward_var_bands.png", title="Walk-Forward Aggregated 99% VaR")

if __name__ == "__main__":
    main()
