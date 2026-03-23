"""
macro_data.py
-------------
Downloads macroeconomic indicators from FRED and VIX from Yahoo Finance.
Each series retains its publication date (realtime_start) for point-in-time
alignment to avoid look-ahead bias.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

logger = logging.getLogger(__name__)

FRED_SERIES: Dict[str, str] = {
    "CPIAUCSL": "CPI",
    "PAYEMS": "NFP",
    "DFF": "FedFundsRate",
    "BAMLH0A0HYM2": "HYSpread",
}


def _download_fred_series_with_vintages(
    fred: Fred,
    series_id: str,
    label: str,
    start: str = "2003-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Download a FRED series with vintage (publication date) information.

    Parameters
    ----------
    fred : Fred
        Authenticated fredapi.Fred instance.
    series_id : str
        FRED series identifier.
    label : str
        Column name for the value.
    start : str
        Start date string.
    end : str
        End date string.

    Returns
    -------
    pd.DataFrame
        Columns: ['observation_date', 'realtime_start', label].
        Only the first release for each observation date is kept.
    """
    logger.info("Downloading FRED series %s (%s) with vintage dates...", series_id, label)

    try:
        releases = fred.get_series_all_releases(series_id)
        releases = releases.rename(columns={"date": "observation_date", "value": label})
        releases["observation_date"] = pd.to_datetime(releases["observation_date"])
        releases["realtime_start"] = pd.to_datetime(releases["realtime_start"])

        mask = (releases["observation_date"] >= start) & (releases["observation_date"] < end)
        releases = releases[mask].copy()

        releases = (
            releases
            .sort_values(["observation_date", "realtime_start"])
            .drop_duplicates(subset=["observation_date"], keep="first")
            .reset_index(drop=True)
        )

        logger.info(
            "  %d observations for %s. Publication range: %s to %s",
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
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        df = pd.DataFrame({"observation_date": pd.to_datetime(series.index), label: series.values})

        # Monthly series lag ~45 days; daily series lag 1 day
        lag_days = 45 if series_id in ("CPIAUCSL", "PAYEMS", "BAMLH0A0HYM2") else 1
        df["realtime_start"] = df["observation_date"] + pd.Timedelta(days=lag_days)

        return df[df["observation_date"] >= start].copy()


def _transform_cpi(df: pd.DataFrame, label: str = "CPI") -> pd.DataFrame:
    """
    Replace raw CPI levels with year-over-year percentage change.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CPI DataFrame with columns ['observation_date', 'realtime_start', label].
    label : str
        Column name for the CPI value.

    Returns
    -------
    pd.DataFrame
        Same structure with label replaced by YoY % change.
    """
    df = df.copy().sort_values("observation_date")
    df[label] = df[label].pct_change(periods=12) * 100
    return df.dropna(subset=[label])


def _transform_nfp(df: pd.DataFrame, label: str = "NFP") -> pd.DataFrame:
    """
    Replace raw NFP levels with month-over-month change.

    Parameters
    ----------
    df : pd.DataFrame
        Raw NFP DataFrame with columns ['observation_date', 'realtime_start', label].
    label : str
        Column name for the NFP value.

    Returns
    -------
    pd.DataFrame
        Same structure with label replaced by monthly change (thousands of jobs).
    """
    df = df.copy().sort_values("observation_date")
    df[label] = df[label].diff()
    return df.dropna(subset=[label])


def _transform_fed_funds_rate(df: pd.DataFrame, label: str = "FedFundsRate") -> pd.DataFrame:
    """
    Resample the Fed Funds Rate to monthly frequency and first-difference it.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DFF DataFrame with columns ['observation_date', 'realtime_start', label].
    label : str
        Column name for the rate value.

    Returns
    -------
    pd.DataFrame
        Monthly first-differenced rate with columns ['observation_date', 'realtime_start', label].
    """
    df = df.copy().sort_values("observation_date").set_index("observation_date")

    df_monthly = pd.concat([
        df[[label]].resample("MS").last(),
        df[["realtime_start"]].resample("MS").max(),
    ], axis=1)

    df_monthly[label] = df_monthly[label].diff()
    return df_monthly.dropna().reset_index().rename(columns={"index": "observation_date"})


def download_vix(start: str = "2003-01-01", end: str = "2024-01-01") -> pd.DataFrame:
    """
    Download the CBOE VIX index from Yahoo Finance.

    Parameters
    ----------
    start : str
        Start date string.
    end : str
        End date string.

    Returns
    -------
    pd.DataFrame
        Columns: ['observation_date', 'realtime_start', 'VIX'].
        realtime_start equals observation_date (VIX is published intraday).
    """
    logger.info("Downloading VIX from Yahoo Finance...")
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)

    vix_values = (
        vix_raw["Close"].values.flatten()
        if isinstance(vix_raw.columns, pd.MultiIndex)
        else vix_raw["Close"].values
    )

    df = pd.DataFrame({
        "observation_date": pd.to_datetime(vix_raw.index).tz_localize(None),
        "VIX": vix_values,
    })
    df["realtime_start"] = df["observation_date"]
    return df.dropna(subset=["VIX"])


def download_macro_data(
    fred_api_key: str,
    start: str = "2003-01-01",
    end: str = "2024-01-01",
) -> Dict[str, pd.DataFrame]:
    """
    Download and stationarize all macroeconomic indicators.

    Parameters
    ----------
    fred_api_key : str
        FRED API key (free from https://fred.stlouisfed.org/docs/api/api_key.html).
    start : str
        Start date string.
    end : str
        End date string.

    Returns
    -------
    dict
        Keys: 'CPI', 'NFP', 'FedFundsRate', 'HYSpread', 'VIX'.
        Values: DataFrames with columns ['observation_date', 'realtime_start', <label>].
    """
    fred = Fred(api_key=fred_api_key)
    results: Dict[str, pd.DataFrame] = {}

    # --- CPI (monthly, YoY % change) ---
    results["CPI"] = _transform_cpi(
        _download_fred_series_with_vintages(fred, "CPIAUCSL", "CPI", start, end)
    )

    # --- NFP (monthly, MoM change) ---
    results["NFP"] = _transform_nfp(
        _download_fred_series_with_vintages(fred, "PAYEMS", "NFP", start, end)
    )

    # --- Fed Funds Rate (daily → monthly, first-differenced) ---
    results["FedFundsRate"] = _transform_fed_funds_rate(
        _download_fred_series_with_vintages(fred, "DFF", "FedFundsRate", start, end)
    )

    # --- HY Credit Spread (daily, level — already stationary) ---
    results["HYSpread"] = _download_fred_series_with_vintages(
        fred, "BAMLH0A0HYM2", "HYSpread", start, end
    )

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