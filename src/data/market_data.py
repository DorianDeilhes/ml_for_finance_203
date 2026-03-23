"""
market_data.py
--------------
Downloads daily adjusted close prices for SPY, TLT, and GLD from Yahoo Finance
and computes daily log returns.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

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
        Yahoo Finance ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format (inclusive).
    end : str
        End date in 'YYYY-MM-DD' format (exclusive).

    Returns
    -------
    pd.DataFrame
        Daily log returns r_t = ln(P_t) - ln(P_{t-1}), indexed by trading date.
        Shape: (T, D) where D = len(tickers). Columns are ticker symbols.
    """
    logger.info("Downloading market data for %s from %s to %s", tickers, start, end)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all").ffill()

    log_returns = np.log(prices / prices.shift(1)).iloc[1:]
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

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns from `download_market_data`.
    window : int
        Rolling window in trading days (default: 21 ~ 1 month).
    annualize : bool
        If True, multiply by sqrt(252) to annualize.

    Returns
    -------
    pd.DataFrame
        Rolling realized volatility with the same index as `returns`.
        Columns are renamed to '{ticker}_RealVol{window}d'.
    """
    vol = returns.rolling(window=window, min_periods=window // 2).std()
    if annualize:
        vol *= np.sqrt(252)
    vol.columns = [f"{col}_RealVol{window}d" for col in returns.columns]
    return vol


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = download_market_data()
    print(df.head())
    print(df.describe())