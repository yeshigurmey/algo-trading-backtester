import numpy as np
import pandas as pd
import pytest

from backtester.data import DataError, load_prices, normalize, synthetic_prices


def _yahoo_frame(multi_level: bool) -> pd.DataFrame:
    """A frame shaped like yfinance.download output."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B", name="Date")
    df = pd.DataFrame(
        {
            "Close": [101.0, 102.0, 103.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Open": [100.0, 101.0, 102.0],
            "Volume": [10, 20, 30],
        },
        index=idx,
    )
    if multi_level:
        df.columns = pd.MultiIndex.from_product([df.columns, ["SPY"]], names=["Price", "Ticker"])
    return df


@pytest.mark.parametrize("multi_level", [False, True])
def test_normalize_handles_both_yfinance_column_layouts(multi_level):
    df = normalize(_yahoo_frame(multi_level))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert df["close"].tolist() == [101.0, 102.0, 103.0]


def test_normalize_sorts_and_drops_unquoted_days():
    raw = _yahoo_frame(False).iloc[::-1].copy()
    raw.iloc[0, raw.columns.get_loc("Close")] = np.nan
    df = normalize(raw)
    assert df.index.is_monotonic_increasing
    assert len(df) == 2


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: d.drop(columns=["Open"]),
        lambda d: d.assign(Close=[101.0, -5.0, 103.0]),
        lambda d: pd.concat([d, d.iloc[[0]]]),
        lambda d: d.iloc[0:0],
    ],
    ids=["missing-column", "negative-price", "duplicate-date", "empty"],
)
def test_normalize_rejects_bad_data(corrupt):
    with pytest.raises(DataError):
        normalize(corrupt(_yahoo_frame(False)))


def test_load_prices_downloads_once_then_reads_cache(tmp_path):
    calls = []

    def fake_download(ticker, start, end):
        calls.append((ticker, start, end))
        return _yahoo_frame(True)

    first = load_prices("spy", "2024-01-01", "2024-02-01", tmp_path, downloader=fake_download)
    second = load_prices("spy", "2024-01-01", "2024-02-01", tmp_path, downloader=fake_download)

    assert calls == [("SPY", "2024-01-01", "2024-02-01")]
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_load_prices_refresh_bypasses_cache(tmp_path):
    calls = []

    def fake_download(ticker, start, end):
        calls.append(ticker)
        return _yahoo_frame(False)

    for _ in range(2):
        load_prices(
            "SPY", "2024-01-01", "2024-02-01", tmp_path, refresh=True, downloader=fake_download
        )
    assert len(calls) == 2


def test_load_prices_reports_ticker_on_empty_download(tmp_path):
    with pytest.raises(DataError, match="NOPE"):
        load_prices(
            "nope", "2024-01-01", "2024-02-01", tmp_path, downloader=lambda *a: pd.DataFrame()
        )
    assert not list(tmp_path.iterdir()), "a failed download must not leave a cache file"


def test_synthetic_prices_are_deterministic_and_well_formed():
    a, b = synthetic_prices(500, seed=7), synthetic_prices(500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(synthetic_prices(500, seed=8))
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
