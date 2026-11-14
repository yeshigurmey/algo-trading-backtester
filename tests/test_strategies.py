from dataclasses import dataclass

import pandas as pd
import pytest

from backtester.strategies import (
    STRATEGIES,
    BuyAndHold,
    MeanReversion,
    Momentum,
    SmaCrossover,
    Strategy,
    find_lookahead,
)
from tests.conftest import make_prices

ALL = [BuyAndHold(), SmaCrossover(10, 30), Momentum(20), MeanReversion(20, 1.5)]


@pytest.mark.parametrize("strategy", ALL, ids=str)
def test_weights_are_valid_and_aligned(strategy, prices):
    w = strategy.target_weights(prices)
    assert w.index.equals(prices.index)
    assert w.notna().all()
    assert w.between(0.0, 1.0).all()


@pytest.mark.parametrize("strategy", ALL, ids=str)
def test_no_strategy_looks_ahead(strategy, prices):
    assert find_lookahead(strategy, prices) is None


def test_lookahead_detector_catches_a_leaky_strategy(prices):
    @dataclass(frozen=True)
    class PeeksAtTomorrow(Strategy):
        def target_weights(self, prices: pd.DataFrame) -> pd.Series:
            close = prices["close"]
            return (close.shift(-1) > close).astype("float64")

    assert find_lookahead(PeeksAtTomorrow(), prices) is not None


def test_sma_crossover_by_hand():
    close = [10, 10, 10, 13, 16, 10, 4, 4]
    w = SmaCrossover(fast=2, slow=3).target_weights(make_prices(close, close))
    # fast: -, 10, 10, 11.5, 14.5, 13, 7, 4
    # slow: -, -, 10, 11, 13, 13, 10, 6
    assert w.tolist() == [0, 0, 0, 1, 1, 0, 0, 0]


def test_sma_crossover_stays_flat_during_warmup(prices):
    w = SmaCrossover(10, 30).target_weights(prices)
    assert (w.iloc[:29] == 0).all()


def test_momentum_by_hand():
    close = [10, 11, 12, 11, 10, 12]
    w = Momentum(lookback=2).target_weights(make_prices(close, close))
    assert w.tolist() == [0, 0, 1, 0, 0, 1]


def test_mean_reversion_holds_from_dip_until_recovery():
    close = [100.0] * 5 + [90.0, 91.0, 92.0, 99.0, 100.0]
    w = MeanReversion(window=5, num_std=1.0).target_weights(make_prices(close, close))
    # Enters on the 90 dip, holds through the partial recovery, exits once the
    # close is back above its 5-day average.
    assert w.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]


@pytest.mark.parametrize(
    "build",
    [
        lambda: SmaCrossover(fast=50, slow=50),
        lambda: SmaCrossover(fast=0, slow=10),
        lambda: Momentum(lookback=0),
        lambda: MeanReversion(window=1),
    ],
)
def test_invalid_parameters_are_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_registry_and_readable_names():
    assert set(STRATEGIES) == {"buyhold", "sma", "momentum", "meanrev"}
    assert str(SmaCrossover(20, 100)) == "SmaCrossover(fast=20, slow=100)"
