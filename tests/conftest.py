from dataclasses import dataclass, field

import pandas as pd
import pytest

from backtester.data import synthetic_prices
from backtester.strategies import Strategy


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    """About four years of seeded random-walk prices."""
    return synthetic_prices(1000, seed=42)


@dataclass(frozen=True)
class FixedWeights(Strategy):
    """Test double: replays a hand-written list of target weights."""

    values: tuple = field(default_factory=tuple)

    def target_weights(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(list(self.values), index=prices.index, dtype="float64")


def make_prices(open_: list[float], close: list[float]) -> pd.DataFrame:
    """Tiny hand-written price frame for exact, by-hand expected values."""
    index = pd.bdate_range("2024-01-01", periods=len(close), name="date")
    hi = [max(o, c) for o, c in zip(open_, close, strict=True)]
    lo = [min(o, c) for o, c in zip(open_, close, strict=True)]
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": 1e6},
        index=index,
        dtype="float64",
    )
