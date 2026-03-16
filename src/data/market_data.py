"""
market_data.py
--------------
Downloads daily adjusted close prices for SPY, TLT, and GLD from Yahoo Finance,
computes daily log returns, and returns a clean DataFrame indexed by trading date.

Usage:
    from src.data.market_data import download_market_data
    returns_df = download_market_data(start="2004-01-01", end="2024-01-01")
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default asset universe
DEFAULT_TICKERS: List[str] = ["SPY", "TLT", "GLD"]


def download_market_data(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = "2004-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Download daily adjusted close prices and compute log returns.

    Parameters
    ----------
    tickers : list of str
        List of Yahoo Finance ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format (inclusive).
    end : str
        End date in 'YYYY-MM-DD' format (exclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame of daily log returns for each ticker, indexed by date.
        Index is a DatetimeIndex of trading days (UTC-normalized).
        Shape: (T, D)  where D = len(tickers).
        Column names are the ticker symbols.

    Notes
    -----
    We compute log returns r_t = ln(P_t) - ln(P_{t-1}).
    The first row (which would be NaN) is dropped.
    """
    logger.info("Downloading market data for %s from %s to %s", tickers, start, end)

    # Download adjusted close prices
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # Use adjusted close prices
        progress=False,
    )

    # yfinance returns a MultiIndex if multiple tickers; extract 'Close'
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        # Single ticker case
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop rows where all prices are NaN (e.g., market holidays)
    prices = prices.dropna(how="all")

    # Forward-fill any remaining NaNs (minor gaps)
    prices = prices.ffill()

    # Compute daily log returns
    log_returns = np.log(prices / prices.shift(1))

    # Drop the first row (NaN from differencing)
    log_returns = log_returns.iloc[1:]

    # Normalize timezone information
    log_returns.index = log_returns.index.tz_localize(None)

    logger.info(
        "Market data downloaded. Shape: %s. Date range: %s to %s",
        log_returns.shape,
        log_returns.index[0].date(),
        log_returns.index[-1].date(),
    )

    return log_returns


def compute_rolling_realized_vol(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling realized volatility for each asset.

    Used as a high-frequency risk signal to supplement macro features.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns DataFrame from `download_market_data`.
    window : int
        Rolling window in trading days (default: 21 ~ 1 month).
    annualize : bool
        If True, multiply by sqrt(252) to annualize.

    Returns
    -------
    pd.DataFrame
        Rolling realized volatility, same index as `returns`.
    """
    vol = returns.rolling(window=window, min_periods=window // 2).std()
    if annualize:
        vol = vol * np.sqrt(252)
    vol.columns = [f"{col}_RealVol{window}d" for col in returns.columns]
    return vol


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = download_market_data()
    print(df.head())
    print(df.describe())
