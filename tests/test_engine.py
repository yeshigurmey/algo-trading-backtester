import numpy as np
import pytest

from backtester.engine import BacktestConfig, run_backtest
from backtester.strategies import BuyAndHold, MeanReversion, Momentum, SmaCrossover
from tests.conftest import FixedWeights, make_prices

FREE = BacktestConfig(initial_cash=1_000.0, commission_bps=0.0, slippage_bps=0.0)


def test_trade_by_hand_with_commission():
    px = make_prices(open_=[10, 10, 11, 12, 12], close=[10, 11, 12, 12, 13])
    cfg = BacktestConfig(initial_cash=1_000.0, commission_bps=10.0, slippage_bps=0.0)
    res = run_backtest(px, FixedWeights((1, 1, 0, 0, 0)), cfg)

    # Day 1 open: 100 shares would cost 1000 + 1.00 commission, which is more
    # than the account holds, so 99 are bought: 990 + 0.99 -> cash 9.01.
    # Day 3 open: sell 99 @ 12 = 1188 - 1.188 commission -> cash 1195.822.
    assert res.shares.tolist() == [0, 99, 99, 0, 0]
    assert res.equity.tolist() == pytest.approx([1000, 1098.01, 1197.01, 1195.822, 1195.822])
    assert res.trades["side"].tolist() == ["BUY", "SELL"]
    assert res.trades["commission"].tolist() == pytest.approx([0.99, 1.188])


def test_signal_at_close_fills_at_next_open_not_same_close():
    px = make_prices(open_=[10, 20, 20], close=[10, 20, 20])
    res = run_backtest(px, FixedWeights((1, 1, 1)), FREE)
    # Decided on day 0 (price 10) but filled on day 1 at 20: the jump is missed.
    assert res.trades["date"].tolist() == [px.index[1]]
    assert res.trades["price"].tolist() == [20.0]
    assert res.equity.iloc[-1] == 1_000.0


def test_decision_on_last_day_is_never_filled():
    px = make_prices(open_=[10, 10, 10], close=[10, 10, 10])
    res = run_backtest(px, FixedWeights((0, 0, 1)), FREE)
    assert res.trades.empty
    assert (res.equity == 1_000.0).all()


def test_slippage_moves_fills_against_the_trader():
    px = make_prices(open_=[100, 100, 100, 100], close=[100, 100, 100, 100])
    cfg = BacktestConfig(initial_cash=10_000.0, commission_bps=0.0, slippage_bps=100.0)
    res = run_backtest(px, FixedWeights((1, 0, 0, 0)), cfg)
    assert res.trades["price"].tolist() == pytest.approx([101.0, 99.0])
    # 99 shares bought at 101 and sold at 99: a flat market still loses money.
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 - 99 * 2.0)


def test_partial_weight_and_resize():
    px = make_prices(open_=[10, 10, 10, 10], close=[10, 10, 10, 10])
    res = run_backtest(px, FixedWeights((0.5, 0.25, 0.25, 0.25)), FREE)
    assert res.shares.tolist() == [0, 50, 25, 25]


def test_orders_only_when_target_changes(prices):
    res = run_backtest(prices, BuyAndHold())
    assert len(res.trades) == 1


@pytest.mark.parametrize(
    "strategy", [BuyAndHold(), SmaCrossover(10, 30), Momentum(20), MeanReversion(20, 1.5)], ids=str
)
def test_accounting_invariants(strategy, prices):
    res = run_backtest(prices, strategy)
    assert (res.cash >= 0).all(), "long-only account must never borrow"
    assert (res.shares >= 0).all()
    assert (res.shares % 1 == 0).all(), "whole shares by default"
    np.testing.assert_allclose(res.equity, res.cash + res.shares * prices["close"])

    # Cash must reconcile exactly with the trade log.
    t = res.trades
    signed = np.where(t["side"] == "BUY", -1.0, 1.0)
    cash_flow = (signed * t["shares"] * t["price"] - t["commission"]).sum()
    assert res.cash.iloc[-1] == pytest.approx(res.config.initial_cash + cash_flow)


def test_frictionless_buy_and_hold_tracks_the_asset(prices):
    cfg = BacktestConfig(commission_bps=0, slippage_bps=0, fractional_shares=True)
    res = run_backtest(prices, BuyAndHold(), cfg)
    expected = cfg.initial_cash * prices["close"].iloc[-1] / prices["open"].iloc[1]
    assert res.equity.iloc[-1] == pytest.approx(expected)


def test_costs_only_ever_reduce_equity(prices):
    strategy = SmaCrossover(10, 30)
    free = run_backtest(prices, strategy, BacktestConfig(commission_bps=0, slippage_bps=0))
    costly = run_backtest(prices, strategy, BacktestConfig(commission_bps=5, slippage_bps=10))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


@pytest.mark.parametrize("bad", [(1.5, 0, 0), (-0.1, 0, 0), (float("nan"), 0, 0)])
def test_rejects_leverage_shorts_and_nan(bad):
    px = make_prices(open_=[10, 10, 10], close=[10, 10, 10])
    with pytest.raises(ValueError):
        run_backtest(px, FixedWeights(bad), FREE)


def test_config_validation():
    with pytest.raises(ValueError):
        BacktestConfig(initial_cash=0)
    with pytest.raises(ValueError):
        BacktestConfig(slippage_bps=-1)
