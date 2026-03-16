"""
macro_data.py
-------------
Downloads macroeconomic indicators from the FRED database and VIX from Yahoo Finance.
Returns DataFrames that include both the observation date AND publication date
(realtime_start), which is critical for point-in-time alignment to avoid look-ahead bias.

FRED series used:
    - CPIAUCSL : Consumer Price Index (monthly, all urban consumers)
    - PAYEMS   : Non-Farm Payrolls (monthly, thousands of persons)
    - DFF      : Federal Funds Effective Rate (daily)
    - BAMLH0A0HYM2 : ICE BofA US High Yield Option-Adjusted Spread (daily, risk sentiment)

VIX from Yahoo Finance:
    - ^VIX     : CBOE Volatility Index (daily close)

Usage:
    from src.data.macro_data import download_macro_data
    macro_dict = download_macro_data(fred_api_key="YOUR_KEY")
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

logger = logging.getLogger(__name__)

# FRED series identifiers and human-readable labels
FRED_SERIES: Dict[str, str] = {
    "CPIAUCSL": "CPI",           # Monthly CPI index level
    "PAYEMS": "NFP",             # Monthly non-farm payrolls (thousands)
    "DFF": "FedFundsRate",       # Daily federal funds effective rate
    "BAMLH0A0HYM2": "HYSpread", # Daily high-yield credit spread (risk sentiment)
}


def _download_fred_series_with_vintages(
    fred: Fred,
    series_id: str,
    label: str,
    start: str = "2003-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Download a FRED series including vintage (publication date) information.

    For monthly/quarterly series, FRED provides a `realtime_start` date indicating
    when each data point was first published—this is the key to avoiding look-ahead bias.

    Parameters
    ----------
    fred : Fred
        Authenticated fredapi.Fred instance.
    series_id : str
        FRED series identifier.
    label : str
        Human-readable column name for the value.
    start : str
        Start date string.
    end : str
        End date string.

    Returns
    -------
    pd.DataFrame
        Columns: ['observation_date', 'realtime_start', label]
        - observation_date: the date the economic event occurred
        - realtime_start:   the date this data point was first PUBLISHED
    """
    logger.info("Downloading FRED series %s (%s) with vintage dates...", series_id, label)

    try:
        # Download the release dates for this series (publication history)
        releases = fred.get_series_all_releases(series_id)
        # releases is a pd.DataFrame with ['date', 'realtime_start', 'value']
        releases = releases.rename(columns={"date": "observation_date", "value": label})
        releases["observation_date"] = pd.to_datetime(releases["observation_date"])
        releases["realtime_start"] = pd.to_datetime(releases["realtime_start"])

        # Filter to our date range
        mask = (releases["observation_date"] >= start) & (releases["observation_date"] < end)
        releases = releases[mask].copy()

        # Keep only the FIRST release for each observation date
        # (the first time the data point was published—no revisions)
        releases = releases.sort_values(["observation_date", "realtime_start"])
        releases = releases.drop_duplicates(subset=["observation_date"], keep="first")
        releases = releases.reset_index(drop=True)

        logger.info(
            "  Downloaded %d observations for %s. Publication dates range: %s to %s",
            len(releases),
            series_id,
            releases["realtime_start"].min().date() if len(releases) > 0 else "N/A",
            releases["realtime_start"].max().date() if len(releases) > 0 else "N/A",
        )
        return releases

    except Exception as e:
        logger.warning(
            "Could not download vintage data for %s: %s. Falling back to basic download.",
            series_id,
            str(e),
        )
        # Fallback: download standard series and use a fixed reporting lag
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        df = pd.DataFrame({"observation_date": series.index, label: series.values})
        df["observation_date"] = pd.to_datetime(df["observation_date"])

        # Apply a conservative reporting lag per series type
        # Monthly series (CPI, NFP) are typically published ~2 weeks after month-end
        # Daily series (DFF) are available the next business day
        is_monthly = series_id in ("CPIAUCSL", "PAYEMS", "BAMLH0A0HYM2")
        lag_days = 45 if is_monthly else 1
        df["realtime_start"] = df["observation_date"] + pd.Timedelta(days=lag_days)

        mask = df["observation_date"] >= start
        return df[mask].copy()


def _transform_cpi(df: pd.DataFrame, label: str = "CPI") -> pd.DataFrame:
    """
    Compute year-over-year percentage change for CPI.

    Raw CPI is non-stationary (trending upward). The YoY % change (inflation rate)
    is approximately stationary and economically meaningful.

    Returns the DataFrame with the value column replaced by YoY % change.
    """
    df = df.copy().sort_values("observation_date")
    # Sort and compute YoY % change using a 12-month lag
    df[label] = df[label].pct_change(periods=12) * 100  # In percent
    df = df.dropna(subset=[label])
    return df


def _transform_nfp(df: pd.DataFrame, label: str = "NFP") -> pd.DataFrame:
    """
    Compute month-over-month change in Non-Farm Payrolls.

    Raw NFP is non-stationary. The monthly change (new jobs) is stationary.
    """
    df = df.copy().sort_values("observation_date")
    df[label] = df[label].diff()  # Monthly change in thousands
    df = df.dropna(subset=[label])
    return df


def _transform_fed_funds_rate(df: pd.DataFrame, label: str = "FedFundsRate") -> pd.DataFrame:
    """
    Compute first difference of the Federal Funds Rate.

    The rate level is not stationary; the daily or monthly change is approximately stationary.
    """
    df = df.copy().sort_values("observation_date")
    # Resample to monthly first for daily series to reduce noise
    df_monthly = df.set_index("observation_date")[[label]].resample("MS").last()
    df_monthly[label] = df_monthly[label].diff()
    df_monthly = df_monthly.dropna()
    # Re-attach the realtime_start from original df
    df_out = df_monthly.reset_index().rename(columns={"observation_date": "observation_date"})
    # Use observation_date + 1 day as realtime_start (daily rate, near-instant publication)
    df_out["realtime_start"] = df_out["observation_date"] + pd.Timedelta(days=1)
    return df_out


def download_vix(start: str = "2003-01-01", end: str = "2024-01-01") -> pd.DataFrame:
    """
    Download the CBOE VIX index from Yahoo Finance.

    VIX is a daily series published the same day, so no publication lag adjustment needed.
    It is already approximately stationary (mean-reverting), so no differencing required.

    Returns
    -------
    pd.DataFrame
        Columns: ['observation_date', 'realtime_start', 'VIX']
    """
    logger.info("Downloading VIX from Yahoo Finance...")
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix = vix_raw["Close"].values.flatten()
    else:
        vix = vix_raw["Close"].values

    df = pd.DataFrame({
        "observation_date": pd.to_datetime(vix_raw.index),
        "VIX": vix,
    })
    df["observation_date"] = df["observation_date"].dt.tz_localize(None)
    # VIX is published intraday; treat publication date = observation date
    df["realtime_start"] = df["observation_date"]
    df = df.dropna(subset=["VIX"])
    logger.info("VIX downloaded. %d observations.", len(df))
    return df


def download_macro_data(
    fred_api_key: str,
    start: str = "2003-01-01",
    end: str = "2024-01-01",
) -> Dict[str, pd.DataFrame]:
    """
    Download and stationarize all macroeconomic indicators.

    Each returned DataFrame has columns:
        ['observation_date', 'realtime_start', <value_label>]

    where `realtime_start` is the date the data was publicly available
    (used for point-in-time alignment in the pipeline).

    Parameters
    ----------
    fred_api_key : str
        Your FRED API key (free from https://fred.stlouisfed.org/docs/api/api_key.html).
    start : str
        Start date.
    end : str
        End date.

    Returns
    -------
    dict
        Keys: 'CPI', 'NFP', 'FedFundsRate', 'HYSpread', 'VIX'
        Values: DataFrames as described above.
    """
    fred = Fred(api_key=fred_api_key)
    results: Dict[str, pd.DataFrame] = {}

    # --- CPI (monthly) ---
    cpi_raw = _download_fred_series_with_vintages(fred, "CPIAUCSL", "CPI", start, end)
    cpi = _transform_cpi(cpi_raw, "CPI")
    results["CPI"] = cpi

    # --- NFP (monthly) ---
    nfp_raw = _download_fred_series_with_vintages(fred, "PAYEMS", "NFP", start, end)
    nfp = _transform_nfp(nfp_raw, "NFP")
    results["NFP"] = nfp

    # --- Fed Funds Rate (daily → monthly differenced) ---
    ffr_raw = _download_fred_series_with_vintages(fred, "DFF", "FedFundsRate", start, end)
    ffr = _transform_fed_funds_rate(ffr_raw, "FedFundsRate")
    results["FedFundsRate"] = ffr

    # --- HY Credit Spread (daily, already roughly stationary) ---
    hy_raw = _download_fred_series_with_vintages(fred, "BAMLH0A0HYM2", "HYSpread", start, end)
    results["HYSpread"] = hy_raw  # No transformation needed

    # --- VIX (daily from Yahoo Finance) ---
    results["VIX"] = download_vix(start, end)

    logger.info("All macro data downloaded. Variables: %s", list(results.keys()))
    return results


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    api_key = os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY")
    macro = download_macro_data(fred_api_key=api_key)
    for name, df in macro.items():
        print(f"\n{name}: {df.shape}")
        print(df.head(3))
