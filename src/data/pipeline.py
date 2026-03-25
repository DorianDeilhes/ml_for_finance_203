"""
pipeline.py
-----------
Point-in-time data pipeline for the macro-conditional normalizing flow.

Downloads market returns and macro indicators, aligns macro data to trading
days via publication dates (realtime_start) to avoid look-ahead bias, splits
into train/val/test, fits scalers on training data only, and returns PyTorch
DataLoaders of sliding-window sequences.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from macro_data import download_macro_data
from market_data import compute_rolling_realized_vol, download_market_data

logger = logging.getLogger(__name__)

TICKERS = ["SPY", "TLT", "GLD"]
DATA_START = "2004-01-01"
DATA_END = "2024-01-01"
TRAIN_START = "2005-01-01"
TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"
SEQ_LEN = 63
BATCH_SIZE = 64


def _pit_merge_macro(
    trading_index: pd.DatetimeIndex,
    macro_series: pd.DataFrame,
    value_col: str,
) -> pd.Series:
    """
    Align a single macro series onto the trading calendar using publication dates.

    For each trading day t, selects the most recent macro observation whose
    realtime_start <= t, ensuring only publicly available data is used.

    Parameters
    ----------
    trading_index : DatetimeIndex
        Market trading day calendar.
    macro_series : pd.DataFrame
        Must contain columns ['realtime_start', value_col].
    value_col : str
        Name of the value column to merge.

    Returns
    -------
    pd.Series
        Aligned values indexed by trading_index.
    """
    trading_df = pd.DataFrame({"date": trading_index})
    trading_df["date"] = trading_df["date"].astype("datetime64[us]")

    macro_sorted = (
        macro_series[["realtime_start", value_col]]
        .copy()
        .sort_values("realtime_start")
        .dropna(subset=[value_col])
    )
    macro_sorted["realtime_start"] = macro_sorted["realtime_start"].astype("datetime64[us]")

    merged = pd.merge_asof(
        trading_df,
        macro_sorted.rename(columns={"realtime_start": "date"}),
        on="date",
        direction="backward",
    )
    return merged.set_index("date")[value_col]


def build_master_dataset(
    fred_api_key: str,
    start: str = DATA_START,
    end: str = DATA_END,
) -> pd.DataFrame:
    """
    Build the master point-in-time dataset.

    Each row t contains asset log returns for day t and macro features as
    known at market open on day t (no look-ahead bias).

    Parameters
    ----------
    fred_api_key : str
        FRED API key.
    start : str
        Start date (should include a buffer year for rolling feature warmup).
    end : str
        End date.

    Returns
    -------
    pd.DataFrame
        Master dataset indexed by trading date. NaN rows are dropped.
    """
    logger.info("Building master point-in-time dataset...")

    returns = download_market_data(tickers=TICKERS, start=start, end=end)
    trading_index = returns.index
    ret_cols = [f"{t}_ret" for t in TICKERS]
    returns.columns = ret_cols

    realvol = compute_rolling_realized_vol(
        returns.rename(columns={v: k for k, v in zip(TICKERS, ret_cols)}),
        window=21,
    )
    realvol.index = trading_index

    macro_dict = download_macro_data(fred_api_key=fred_api_key, start=start, end=end)

    logger.info("Performing point-in-time alignment...")
    macro_aligned: Dict[str, pd.Series] = {}
    for name, df in macro_dict.items():
        aligned = _pit_merge_macro(trading_index, df, name)
        macro_aligned[name] = aligned
        logger.info("  %s: %d non-null values", name, aligned.notna().sum())

    master = returns.copy()
    for name, series in macro_aligned.items():
        master[name] = series.values
    master = pd.concat([master, realvol], axis=1)

    macro_cols = list(macro_dict.keys())
    master[macro_cols] = master[macro_cols].ffill()
    master = master.dropna()

    logger.info("Master dataset built. Shape: %s. Columns: %s", master.shape, list(master.columns))
    return master


def verify_no_lookahead(
    master: pd.DataFrame,
    macro_dict: Dict[str, pd.DataFrame],
) -> None:
    """
    Assert that no macro value in the master dataset was published after its trading date.

    Parameters
    ----------
    master : pd.DataFrame
        Assembled master dataset.
    macro_dict : dict
        Raw macro DataFrames containing a 'realtime_start' column.

    Raises
    ------
    AssertionError
        If any look-ahead bias is detected.
    """
    logger.info("Verifying no look-ahead bias...")
    for name, df in macro_dict.items():
        if "realtime_start" not in df.columns or name not in master.columns:
            continue

        pub_series = df.set_index("realtime_start")[name].dropna()
        val_to_earliest_pub = (
            pub_series.reset_index()
            .groupby(name)["realtime_start"]
            .min()
        )

        master_col = master[name].dropna()
        earliest_pubs = master_col.map(val_to_earliest_pub)
        traceable = earliest_pubs.dropna()
        violations = traceable.index[traceable.index < traceable]

        assert len(violations) == 0, (
            f"Look-ahead bias detected in '{name}': {len(violations)} trading days "
            f"have macro values published after the trading date. "
            f"First violation: {violations[0].date()}"
        )
        logger.info("  %s: OK", name)

    logger.info("No look-ahead bias detected.")


def build_sequences(
    master: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for model input.

    For each time step t >= seq_len, produces a window of seq_len past macro
    feature rows and the asset returns on day t as the target.

    Parameters
    ----------
    master : pd.DataFrame
        Master dataset with return and macro columns.
    seq_len : int
        Number of past trading days per sequence window.

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

    macro_seqs = np.array([X[t - seq_len: t] for t in range(seq_len, len(master))], dtype=np.float32)
    asset_returns = np.array([y[t] for t in range(seq_len, len(master))], dtype=np.float32)
    date_list = np.array([dates[t] for t in range(seq_len, len(master))])

    return macro_seqs, asset_returns, date_list


def build_pipeline(
    fred_api_key: str,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    train_end: str = TRAIN_END,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
    test_start: str = TEST_START,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler, Dict, int, int]:
    """
    Full pipeline: download → align → split → scale → sequence → DataLoader.

    Scalers are fit exclusively on training data to prevent leakage.

    Parameters
    ----------
    fred_api_key : str
        FRED API key.
    seq_len : int
        Sliding window length in trading days.
    batch_size : int
        DataLoader batch size.
    train_end : str
        Last date (inclusive) of the training set.
    val_start : str
        First date of the validation set.
    val_end : str
        Last date of the validation set.
    test_start : str
        First date of the test set.
    device : torch.device, optional
        Unused; reserved for future tensor placement.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    macro_scaler : StandardScaler
    ret_scaler : StandardScaler
    info : dict
        Dataset metadata (dates, column names, shapes, raw split DataFrames).
    num_macro_features : int
    num_assets : int
    """
    master = build_master_dataset(fred_api_key=fred_api_key)

    ret_cols = [f"{t}_ret" for t in TICKERS]
    feature_cols = [c for c in master.columns if c not in ret_cols]
    num_macro_features = len(feature_cols)
    num_assets = len(ret_cols)

    train_mask = master.index <= pd.Timestamp(train_end)
    val_mask = (master.index >= pd.Timestamp(val_start)) & (master.index <= pd.Timestamp(val_end))
    test_mask = master.index >= pd.Timestamp(test_start)

    master_train = master[train_mask]
    master_val = master[val_mask]
    master_test = master[test_mask]

    logger.info(
        "Train: %s to %s (%d rows). Val: %s to %s (%d rows). Test: %s to %s (%d rows).",
        master_train.index[0].date(), master_train.index[-1].date(), len(master_train),
        master_val.index[0].date(), master_val.index[-1].date(), len(master_val),
        master_test.index[0].date(), master_test.index[-1].date(), len(master_test),
    )

    macro_scaler = RobustScaler()
    ret_scaler = StandardScaler()
    macro_scaler.fit(master_train[feature_cols].values)
    ret_scaler.fit(master_train[ret_cols].values)

    def _scale(df: pd.DataFrame) -> pd.DataFrame:
        scaled = df.copy()
        scaled[feature_cols] = macro_scaler.transform(df[feature_cols].values)
        scaled[ret_cols] = ret_scaler.transform(df[ret_cols].values)
        return scaled

    master_train_scaled = _scale(master_train)
    master_val_scaled = _scale(master_val)
    master_test_scaled = _scale(master_test)

    X_train, y_train, dates_train = build_sequences(master_train_scaled, seq_len)
    X_val, y_val, dates_val = build_sequences(master_val_scaled, seq_len)
    X_test, y_test, dates_test = build_sequences(master_test_scaled, seq_len)

    logger.info(
        "Sequences built. Train: %s. Val: %s. Test: %s.",
        X_train.shape, X_val.shape, X_test.shape,
    )

    dtype = torch.float32
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=dtype), torch.tensor(y_train, dtype=dtype)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=dtype), torch.tensor(y_val, dtype=dtype)),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=dtype), torch.tensor(y_test, dtype=dtype)),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )

    info = {
        "feature_cols": feature_cols,
        "ret_cols": ret_cols,
        "tickers": TICKERS,
        "dates_train": dates_train,
        "dates_val": dates_val,
        "dates_test": dates_test,
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
        "master_train": master_train,
        "master_val": master_val,
        "master_test": master_test,
    }

    return train_loader, val_loader, test_loader, macro_scaler, ret_scaler, info, num_macro_features, num_assets



def build_walk_forward_pipeline(
    fred_api_key: Optional[str] = None,
    master: Optional[pd.DataFrame] = None,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    start_year: int = 2005,
    initial_train_end_year: int = 2016,
    end_year: int = 2023,
    val_years: int = 1,
    test_years: int = 1,
) -> Tuple:
    """
    Generator for Walk-Forward Expanding Window cross-validation.

    Yields:
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    ret_scaler : StandardScaler
    info : dict
        Fold metadata including dates and number of features/assets.
    """
    if master is None:
        if fred_api_key is None:
            raise ValueError("Must provide either 'master' DataFrame or 'fred_api_key'")
        master = build_master_dataset(fred_api_key=fred_api_key)

    ret_cols = [f"{t}_ret" for t in TICKERS]
    feature_cols = [c for c in master.columns if c not in ret_cols]
    num_macro_features = len(feature_cols)
    num_assets = len(ret_cols)

    current_train_end_year = initial_train_end_year

    while current_train_end_year + val_years + test_years <= end_year + 1:
        train_end = f"{current_train_end_year}-12-31"
        val_start = f"{current_train_end_year + 1}-01-01"
        val_end = f"{current_train_end_year + val_years}-12-31"
        test_start = f"{current_train_end_year + val_years + 1}-01-01"
        test_end = f"{current_train_end_year + val_years + test_years}-12-31"

        logger.info(
            "Fold: Train -> %s | Val %s -> %s | Test %s -> %s",
            train_end, val_start, val_end, test_start, test_end
        )

        train_mask = master.index <= pd.Timestamp(train_end)
        val_mask = (master.index >= pd.Timestamp(val_start)) & (master.index <= pd.Timestamp(val_end))
        test_mask = (master.index >= pd.Timestamp(test_start)) & (master.index <= pd.Timestamp(test_end))

        master_train = master[train_mask]
        master_val = master[val_mask]
        master_test = master[test_mask]

        if len(master_test) <= seq_len:
            logger.info("Not enough test data remaining. Stopping generator.")
            break

        macro_scaler = RobustScaler()
        ret_scaler = StandardScaler()
        macro_scaler.fit(master_train[feature_cols].values)
        ret_scaler.fit(master_train[ret_cols].values)

        def _scale(df: pd.DataFrame) -> pd.DataFrame:
            scaled = df.copy()
            scaled[feature_cols] = macro_scaler.transform(df[feature_cols].values)
            scaled[ret_cols] = ret_scaler.transform(df[ret_cols].values)
            return scaled

        X_train, y_train, dates_train = build_sequences(_scale(master_train), seq_len)
        X_val, y_val, dates_val = build_sequences(_scale(master_val), seq_len)
        X_test, y_test, dates_test = build_sequences(_scale(master_test), seq_len)

        dtype = torch.float32
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=dtype), torch.tensor(y_train, dtype=dtype)),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=dtype), torch.tensor(y_val, dtype=dtype)),
            batch_size=batch_size, shuffle=False, drop_last=False,
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=dtype), torch.tensor(y_test, dtype=dtype)),
            batch_size=batch_size, shuffle=False, drop_last=False,
        )

        info = {
            "feature_cols": feature_cols,
            "ret_cols": ret_cols,
            "tickers": TICKERS,
            "dates_train": dates_train,
            "dates_val": dates_val,
            "dates_test": dates_test,
            "num_macro_features": num_macro_features,
            "num_assets": num_assets,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        }

        yield train_loader, val_loader, test_loader, ret_scaler, info

        current_train_end_year += test_years


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    api_key = os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY")
    train_loader, val_loader, test_loader, *_ = build_pipeline(fred_api_key=api_key)
    batch = next(iter(train_loader))
    print("Train batch shapes:", batch[0].shape, batch[1].shape)