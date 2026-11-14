"""Price data: download from Yahoo Finance, cache on disk, validate.

Every other module assumes the frame produced here: a DataFrame indexed by
trading day (ascending, unique, timezone-naive) with float columns
``open, high, low, close`` and ``volume``. Validating once at the boundary
means the engine never has to defend against bad rows in its hot loop.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

Downloader = Callable[[str, str, str], pd.DataFrame]


class DataError(ValueError):
    """Raised when price data is missing or fails validation."""


def _yfinance_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Imported lazily so the rest of the package (and the test suite) works
    # without network access or yfinance installed.
    import yfinance as yf

    # auto_adjust=True rewrites OHLC for splits and dividends. Without it a
    # 4-for-1 split looks like a 75% overnight crash, and buy-and-hold is
    # understated because dividends are missing from the price series.
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a downloader's frame into the canonical shape and validate it."""
    if raw is None or raw.empty:
        raise DataError("no rows returned")

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Recent yfinance versions return (field, ticker) column pairs even
        # for a single ticker. Keep whichever level holds the field names.
        for level in range(df.columns.nlevels):
            if "Close" in df.columns.get_level_values(level):
                df.columns = df.columns.get_level_values(level)
                break

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"missing columns: {missing}")

    df = df[PRICE_COLUMNS].astype("float64")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df.index.name = "date"
    df = df.sort_index()

    # A day with no quote can't be traded. Dropping it is safer than filling:
    # a forward-filled price would let the simulator trade at a stale level.
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)

    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise DataError if ``df`` is not a clean daily price frame."""
    if df.empty:
        raise DataError("no usable rows")
    if not df.index.is_monotonic_increasing:
        raise DataError("dates are not sorted")
    if df.index.has_duplicates:
        raise DataError("duplicate dates")
    prices = df[["open", "high", "low", "close"]].to_numpy()
    if not np.isfinite(prices).all():
        raise DataError("non-finite prices")
    if (prices <= 0).any():
        raise DataError("non-positive prices")


def load_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path = ".cache",
    refresh: bool = False,
    downloader: Downloader = _yfinance_download,
) -> pd.DataFrame:
    """Daily adjusted OHLCV for ``ticker`` over ``[start, end)``.

    The first call downloads and writes a CSV to ``cache_dir``; later calls
    read that file. This makes runs reproducible (Yahoo occasionally revises
    history) and keeps a parameter sweep from hitting the network repeatedly.
    ``downloader`` is injectable so tests can run offline.
    """
    ticker = ticker.upper()
    path = Path(cache_dir) / f"{ticker}_{start}_{end}_1d.csv"

    if path.exists() and not refresh:
        return normalize(pd.read_csv(path, index_col="date", parse_dates=True))

    try:
        df = normalize(downloader(ticker, start, end))
    except DataError as exc:
        raise DataError(f"{ticker} {start}..{end}: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def synthetic_prices(
    n_days: int = 2520,
    seed: int = 0,
    annual_drift: float = 0.07,
    annual_vol: float = 0.18,
    start: str = "2015-01-02",
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Seeded random-walk OHLCV (geometric Brownian motion).

    Used by the tests and the offline demo. It has no exploitable structure,
    so no strategy should reliably beat buy-and-hold on it; that makes it a
    useful sanity check as well as a fixture.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    log_ret = rng.normal(
        (annual_drift - 0.5 * annual_vol**2) * dt, annual_vol * np.sqrt(dt), n_days
    )
    close = start_price * np.exp(np.cumsum(log_ret))

    # Open gaps away from the previous close; high/low bracket the day's range.
    prev_close = np.concatenate(([start_price], close[:-1]))
    open_ = prev_close * np.exp(rng.normal(0.0, annual_vol * np.sqrt(dt) * 0.3, n_days))
    spread = np.abs(rng.normal(0.0, annual_vol * np.sqrt(dt) * 0.5, (2, n_days)))
    high = np.maximum(open_, close) * (1.0 + spread[0])
    low = np.minimum(open_, close) * (1.0 - spread[1])
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype("float64")

    index = pd.bdate_range(start=start, periods=n_days, name="date")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    validate(df)
    return df
