import numpy as np
import pytest

from backtester.engine import BacktestConfig, run_backtest
from backtester.strategies import BuyAndHold, MeanReversion, Momentum, SmaCrossover
from backtester.vectorized import run_vectorized

STRATEGIES = [BuyAndHold(), SmaCrossover(10, 30), Momentum(20), MeanReversion(20, 1.5)]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=str)
def test_matches_event_engine_exactly_without_frictions(strategy, prices):
    cfg = BacktestConfig(commission_bps=0, slippage_bps=0, fractional_shares=True)
    event = run_backtest(prices, strategy, cfg).equity
    np.testing.assert_allclose(run_vectorized(prices, strategy, cfg), event, rtol=1e-9)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=str)
def test_cost_model_agrees_with_event_engine(strategy, prices):
    cfg = BacktestConfig(commission_bps=2, slippage_bps=5, fractional_shares=True)
    event = run_backtest(prices, strategy, cfg).equity
    np.testing.assert_allclose(run_vectorized(prices, strategy, cfg), event, rtol=1e-4)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=str)
def test_whole_share_rounding_is_a_small_effect(strategy, prices):
    cfg = BacktestConfig()
    event = run_backtest(prices, strategy, cfg).equity
    np.testing.assert_allclose(run_vectorized(prices, strategy, cfg), event, rtol=5e-3)
