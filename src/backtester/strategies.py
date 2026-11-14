"""Trading strategies.

A strategy maps a price history to a *target weight* per day: the fraction of
account equity it wants invested in the asset, between 0 (all cash) and 1
(fully invested). The value at date ``t`` may use data up to and including the
close of ``t`` and nothing later. The engine fills that decision at the open of
``t + 1``, so a strategy can never trade on a price it has not yet seen.

Indicators are computed for the whole history in one vectorised pandas pass
(O(n)), not recomputed inside the simulation loop (which would be O(n * window)).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Base class. Subclasses are frozen dataclasses whose fields are the parameters."""

    @abstractmethod
    def target_weights(self, prices: pd.DataFrame) -> pd.Series:
        """Target weight in [0, 1] for each row of ``prices``."""

    @property
    def params(self) -> dict:
        return asdict(self) if is_dataclass(self) else {}

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


@dataclass(frozen=True)
class BuyAndHold(Strategy):
    """Always fully invested. The benchmark every other strategy is judged against."""

    def target_weights(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index)


@dataclass(frozen=True)
class SmaCrossover(Strategy):
    """Trend following: invested while the fast moving average is above the slow one."""

    fast: int = 50
    slow: int = 200

    def __post_init__(self) -> None:
        if not 0 < self.fast < self.slow:
            raise ValueError(f"need 0 < fast < slow, got fast={self.fast}, slow={self.slow}")

    def target_weights(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        # Until `slow` days exist the averages are NaN, and NaN > NaN is False,
        # so the strategy sits in cash during warm-up instead of guessing.
        return (fast > slow).astype("float64")


@dataclass(frozen=True)
class Momentum(Strategy):
    """Time-series momentum: invested while the trailing ``lookback``-day return is positive."""

    lookback: int = 126

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self.lookback}")

    def target_weights(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        return (close > close.shift(self.lookback)).astype("float64")


@dataclass(frozen=True)
class MeanReversion(Strategy):
    """Bollinger-band mean reversion.

    Buy when the close drops more than ``num_std`` standard deviations below its
    ``window``-day average; sell once it recovers to the average.
    """

    window: int = 20
    num_std: float = 2.0

    def __post_init__(self) -> None:
        if self.window < 2 or self.num_std <= 0:
            raise ValueError("need window >= 2 and num_std > 0")

    def target_weights(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        mid = close.rolling(self.window).mean()
        lower = mid - self.num_std * close.rolling(self.window).std()
        # Entry and exit use different thresholds, so the position depends on
        # the previous state. Mark the days that change state (1 = enter,
        # 0 = exit), leave the rest NaN, and forward-fill to carry the position.
        events = np.where(close < lower, 1.0, np.where(close >= mid, 0.0, np.nan))
        return pd.Series(events, index=prices.index).ffill().fillna(0.0)


STRATEGIES: dict[str, type[Strategy]] = {
    "buyhold": BuyAndHold,
    "sma": SmaCrossover,
    "momentum": Momentum,
    "meanrev": MeanReversion,
}


def find_lookahead(strategy: Strategy, prices: pd.DataFrame, n_checks: int = 25) -> int | None:
    """Detect look-ahead bias by replaying history.

    A causal strategy must reach the same decision for day ``t`` whether or not
    later days exist. This recomputes the weights on truncated histories and
    compares them with the full run. Returns the first row position where they
    differ, or ``None`` if the strategy is clean at every checked cut.
    """
    full = strategy.target_weights(prices).to_numpy()
    n = len(prices)
    for cut in np.unique(np.linspace(2, n - 1, n_checks).astype(int)):
        part = strategy.target_weights(prices.iloc[:cut]).to_numpy()
        diff = np.flatnonzero(~np.isclose(part, full[:cut], equal_nan=True))
        if diff.size:
            return int(diff[0])
    return None
