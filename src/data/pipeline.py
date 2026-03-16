"""
pipeline.py
-----------
Point-in-Time Data Pipeline for Macro-Conditional Normalizing Flow.

This module:
1. Downloads market returns and macro indicators.
2. Aligns macro data to market trading days using publication dates (realtime_start),
   thus eliminating look-ahead bias via pd.merge_asof().
3. Splits into train (2005-01-01 to 2021-12-31) and test (2022-01-01 onwards).
4. Fits StandardScaler ONLY on training data, transforms both sets.
5. Creates sliding-window sequences of shape (N, seq_len, num_features).
6. Returns PyTorch DataLoaders for train and test.

Critical Design Decisions
--------------------------
- pd.merge_asof(..., direction='backward') ensures we only use macro values
  whose realtime_start <= market trading day t (i.e., "what was KNOWN by day t").
- Scalers are fit exclusively on training indices—no data leakage.
- Returns (macro_seq, returns) pairs where macro_seq is the past `seq_len` days
  of macro features and returns is the current-day asset returns vector.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.market_data import download_market_data, compute_rolling_realized_vol
from src.data.macro_data import download_macro_data

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = ["SPY", "TLT", "GLD"]
DATA_START = "2004-01-01"   # Extra buffer year for lagged feature computation
DATA_END   = "2024-01-01"
TRAIN_START = "2005-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
SEQ_LEN     = 63            # 3 months of trading days (lookback window for TFT)
BATCH_SIZE  = 64


def _pit_merge_macro(
    trading_index: pd.DatetimeIndex,
    macro_series: pd.DataFrame,
    value_col: str,
) -> pd.Series:
    """
    Perform a strict Point-in-Time merge of a single macro series onto the
    trading calendar using publication dates.

    For each trading day t, we find the most recent macro observation whose
    `realtime_start` (publication date) is <= t (i.e., data that was publicly
    available by market open on day t).

    Parameters
    ----------
    trading_index : DatetimeIndex
        The market trading day calendar.
    macro_series : pd.DataFrame
        Must contain columns: ['realtime_start', value_col].
    value_col : str
        Name of the value column to merge.

    Returns
    -------
    pd.Series
        Aligned values indexed by trading_index.
    """
    # Left DataFrame: trading calendar
    trading_df = pd.DataFrame({"date": trading_index})

    # Right DataFrame: macro data sorted by publication date
    macro_sorted = macro_series[["realtime_start", value_col]].copy()
    macro_sorted = macro_sorted.sort_values("realtime_start").dropna(subset=[value_col])

    # merge_asof: for each trading date, find the LAST macro record with
    # realtime_start <= trading date (backward search = no look-ahead)
    merged = pd.merge_asof(
        trading_df,
        macro_sorted.rename(columns={"realtime_start": "date"}),
        on="date",
        direction="backward",
    )
    merged = merged.set_index("date")[value_col]
    return merged


def build_master_dataset(
    fred_api_key: str,
    start: str = DATA_START,
    end: str = DATA_END,
) -> pd.DataFrame:
    """
    Build the master Point-in-Time dataset.

    Returns a DataFrame where each row t contains:
    - Asset log returns for day t (SPY_ret, TLT_ret, GLD_ret)
    - Macro features exactly as known at market open on day t
    - Rolling realized volatilities (high-frequency risk features)

    Parameters
    ----------
    fred_api_key : str
        FRED API key.
    start : str
        Start date (includes buffer for lagged feature computation).
    end : str
        End date.

    Returns
    -------
    pd.DataFrame
        Master dataset indexed by trading date.
    """
    logger.info("=== Building master Point-in-Time dataset ===")

    # 1. Download market returns
    returns = download_market_data(tickers=TICKERS, start=start, end=end)
    trading_index = returns.index
    ret_cols = [f"{t}_ret" for t in TICKERS]
    returns.columns = ret_cols

    # 2. Compute rolling realized volatilities (these are "safe" — no look-ahead
    #    as they only use past returns)
    realvol = compute_rolling_realized_vol(
        returns.rename(columns={v: k for k, v in zip(TICKERS, ret_cols)}),
        window=21,
    )
    realvol.index = trading_index

    # 3. Download macro data (with publication dates)
    macro_dict = download_macro_data(fred_api_key=fred_api_key, start=start, end=end)

    # 4. Point-in-Time align each macro variable onto trading calendar
    logger.info("Performing Point-in-Time alignment (pd.merge_asof)...")
    macro_aligned: Dict[str, pd.Series] = {}
    for name, df in macro_dict.items():
        aligned = _pit_merge_macro(trading_index, df, name)
        macro_aligned[name] = aligned
        logger.info("  %s: %d non-null values", name, aligned.notna().sum())

    # 5. Assemble master DataFrame
    master = returns.copy()
    for name, series in macro_aligned.items():
        master[name] = series.values
    master = pd.concat([master, realvol], axis=1)

    # Forward-fill macro variables (they update slowly; gaps are expected between
    # publications and should carry the last known value forward)
    macro_cols = list(macro_dict.keys())
    master[macro_cols] = master[macro_cols].ffill()

    # Drop rows with ALL macro values still NaN (beginning of dataset before
    # macro data is available)
    master = master.dropna(subset=macro_cols, how="all")

    logger.info(
        "Master dataset built. Shape: %s. Columns: %s",
        master.shape,
        list(master.columns),
    )
    return master


def verify_no_lookahead(
    master: pd.DataFrame,
    macro_dict: Dict[str, pd.DataFrame],
) -> None:
    """
    Sanity check: verify that no macro value in `master` was published AFTER
    the corresponding trading date.

    Parameters
    ----------
    master : pd.DataFrame
        The assembled master dataset.
    macro_dict : dict
        Raw macro DataFrames with 'realtime_start' column.

    Raises
    ------
    AssertionError
        If any look-ahead bias is detected.
    """
    logger.info("Verifying no look-ahead bias...")
    for name, df in macro_dict.items():
        if "realtime_start" not in df.columns:
            continue
        if name not in master.columns:
            continue
        # Build a lookup: observation value → earliest possible publication date
        pub_lookup = df.set_index("realtime_start")[name]
        for date in master.index:
            val = master.loc[date, name]
            if pd.isna(val):
                continue
            # Find the date(s) when this value was published
            pub_dates = pub_lookup[pub_lookup == val].index
            if len(pub_dates) == 0:
                continue  # Value from forward-fill, cannot trace exactly
            earliest_pub = pub_dates.min()
            assert date >= earliest_pub, (
                f"LOOK-AHEAD BIAS DETECTED: On trading day {date.date()}, "
                f"macro variable {name} has value {val:.4f} which was not "
                f"published until {earliest_pub.date()}."
            )
    logger.info("  No look-ahead bias detected. ✓")


def build_sequences(
    master: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for model training.

    For each time step t (from seq_len onward), we produce:
    - macro_seq[t]: shape (seq_len, num_macro_features) — past seq_len days of macro features
    - returns_t[t]: shape (D,) — asset returns on day t (the target)
    - dates[t]: the date corresponding to day t

    Parameters
    ----------
    master : pd.DataFrame
        Master dataset with return and macro columns.
    seq_len : int
        Number of past trading days in each sequence window.

    Returns
    -------
    macro_seqs : np.ndarray of shape (N, seq_len, num_macro_features)
    asset_returns : np.ndarray of shape (N, D)
    dates : np.ndarray of datetime64
    """
    ret_cols = [f"{t}_ret" for t in TICKERS]
    feature_cols = [c for c in master.columns if c not in ret_cols]

    X = master[feature_cols].values.astype(np.float32)
    y = master[ret_cols].values.astype(np.float32)
    dates = master.index.values

    macro_seqs = []
    asset_returns = []
    date_list = []

    for t in range(seq_len, len(master)):
        macro_seqs.append(X[t - seq_len: t])  # Past seq_len days of features
        asset_returns.append(y[t])             # Returns on day t (target)
        date_list.append(dates[t])

    return (
        np.array(macro_seqs, dtype=np.float32),
        np.array(asset_returns, dtype=np.float32),
        np.array(date_list),
    )


def build_pipeline(
    fred_api_key: str,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, StandardScaler, StandardScaler, Dict, int, int]:
    """
    Full pipeline: download → align → split → scale → sequence → DataLoader.

    Returns
    -------
    train_loader : DataLoader
    test_loader : DataLoader
    macro_scaler : StandardScaler (fitted on training macro features only)
    ret_scaler : StandardScaler (fitted on training returns only)
    info : dict with dataset metadata (dates, column names, shapes)
    num_macro_features : int
    num_assets : int
    """
    # ── Build master dataset ──────────────────────────────────────────────────
    master = build_master_dataset(fred_api_key=fred_api_key)

    ret_cols = [f"{t}_ret" for t in TICKERS]
    feature_cols = [c for c in master.columns if c not in ret_cols]
    num_macro_features = len(feature_cols)
    num_assets = len(ret_cols)

    # ── Temporal train/test split ─────────────────────────────────────────────
    train_mask = master.index <= pd.Timestamp(train_end)
    test_mask  = master.index >= pd.Timestamp(test_start)

    master_train = master[train_mask]
    master_test  = master[test_mask]

    logger.info(
        "Train: %s to %s (%d rows). Test: %s to %s (%d rows).",
        master_train.index[0].date(), master_train.index[-1].date(), len(master_train),
        master_test.index[0].date(),  master_test.index[-1].date(),  len(master_test),
    )

    # ── Fit scalers ONLY on training data (no leakage) ──────────────────────
    macro_scaler = StandardScaler()
    ret_scaler   = StandardScaler()

    # Fit on training portion only
    macro_scaler.fit(master_train[feature_cols].values)
    ret_scaler.fit(master_train[ret_cols].values)

    # Transform both train and test
    master_train_scaled = master_train.copy()
    master_test_scaled  = master_test.copy()

    master_train_scaled[feature_cols] = macro_scaler.transform(master_train[feature_cols].values)
    master_train_scaled[ret_cols]     = ret_scaler.transform(master_train[ret_cols].values)

    master_test_scaled[feature_cols] = macro_scaler.transform(master_test[feature_cols].values)
    master_test_scaled[ret_cols]     = ret_scaler.transform(master_test[ret_cols].values)

    # ── Build sliding-window sequences ────────────────────────────────────────
    X_train, y_train, dates_train = build_sequences(master_train_scaled, seq_len)
    X_test,  y_test,  dates_test  = build_sequences(master_test_scaled,  seq_len)

    logger.info(
        "Sequences built. Train: %s. Test: %s.",
        X_train.shape, X_test.shape,
    )

    # ── Create PyTorch DataLoaders ────────────────────────────────────────────
    dtype = torch.float32
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=dtype),
        torch.tensor(y_train, dtype=dtype),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=dtype),
        torch.tensor(y_test, dtype=dtype),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    info = {
        "feature_cols": feature_cols,
        "ret_cols": ret_cols,
        "tickers": TICKERS,
        "dates_train": dates_train,
        "dates_test": dates_test,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "master_train": master_train,   # Unscaled, for analysis
        "master_test": master_test,
    }

    return train_loader, test_loader, macro_scaler, ret_scaler, info, num_macro_features, num_assets


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    api_key = os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY")
    train_loader, test_loader, *_ = build_pipeline(fred_api_key=api_key)
    batch = next(iter(train_loader))
    print("Train batch shapes:", batch[0].shape, batch[1].shape)
