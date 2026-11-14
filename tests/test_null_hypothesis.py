"""Statistical sanity check of the whole pipeline.

On a random walk with no drift, tomorrow's move is independent of everything
known today, so no strategy that only uses past data can make money on average
before costs. If the simulator leaked future prices into decisions or fills,
strategies would show a profit here. This is the test that would catch it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from backtester.data import synthetic_prices
from backtester.engine import BacktestConfig
from backtester.strategies import MeanReversion, Momentum, SmaCrossover, Strategy
from backtester.vectorized import run_vectorized

FRICTIONLESS = BacktestConfig(commission_bps=0, slippage_bps=0, fractional_shares=True)


def _t_statistic(strategy: Strategy, n_paths: int = 200) -> float:
    """Mean total return across independent random walks, in standard errors from zero."""
    totals = np.empty(n_paths)
    for seed in range(n_paths):
        prices = synthetic_prices(500, seed=seed, annual_drift=0.0)
        equity = run_vectorized(prices, strategy, FRICTIONLESS)
        totals[seed] = equity.iloc[-1] / equity.iloc[0] - 1.0
    return float(totals.mean() / (totals.std(ddof=1) / np.sqrt(n_paths)))


@pytest.mark.parametrize(
    "strategy", [SmaCrossover(10, 30), Momentum(20), MeanReversion(20, 1.5)], ids=str
)
def test_no_strategy_profits_from_a_driftless_random_walk(strategy):
    assert abs(_t_statistic(strategy)) < 3.0


def test_a_strategy_that_sees_tomorrow_does_profit():
    @dataclass(frozen=True)
    class PeeksAtTomorrow(Strategy):
        def target_weights(self, prices: pd.DataFrame) -> pd.Series:
            close = prices["close"]
            return (close.shift(-1) > close).astype("float64")

    assert _t_statistic(PeeksAtTomorrow()) > 10.0
