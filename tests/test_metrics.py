import math

import pandas as pd
import pytest

from backtester import metrics
from backtester.engine import BacktestConfig, run_backtest
from tests.conftest import FixedWeights, make_prices


def test_total_return_and_cagr():
    # 2019-01-01 to 2023-01-01 is 1461 days, exactly 4 years of 365.25 days.
    equity = pd.Series([100.0, 200.0], index=pd.to_datetime(["2019-01-01", "2023-01-01"]))
    assert metrics.total_return(equity) == pytest.approx(1.0)
    assert metrics.cagr(equity) == pytest.approx(2 ** (1 / 4) - 1)


def test_sharpe_by_hand():
    # mean 0.02, sample std 0.02 / sqrt(2)  ->  sqrt(2) * sqrt(252)
    assert metrics.sharpe_ratio([0.01, 0.03]) == pytest.approx(math.sqrt(504))


def test_sharpe_subtracts_risk_free_rate():
    r = [0.01, 0.03, -0.02, 0.005]
    assert metrics.sharpe_ratio(r, risk_free_rate=0.05) < metrics.sharpe_ratio(r)


def test_sharpe_is_nan_without_variation():
    assert math.isnan(metrics.sharpe_ratio([0.0, 0.0, 0.0]))
    assert math.isnan(metrics.sharpe_ratio([0.01]))


def test_sortino_by_hand():
    # mean 0.005, downside deviation sqrt(mean([0, 0.01^2]))  ->  sqrt(126)
    assert metrics.sortino_ratio([0.02, -0.01]) == pytest.approx(math.sqrt(126))


def test_volatility_annualises_by_sqrt_252():
    assert metrics.annual_volatility([0.01, 0.03]) == pytest.approx(0.02 / math.sqrt(2) * 252**0.5)


def test_drawdown_depth_and_length():
    equity = pd.Series([100.0, 120.0, 90.0, 108.0, 130.0, 117.0])
    assert metrics.max_drawdown(equity) == pytest.approx(-0.25)
    assert metrics.drawdown(equity).tolist() == pytest.approx([0, 0, -0.25, -0.10, 0, -0.10])
    assert metrics.max_drawdown_days(equity) == 2
    assert metrics.max_drawdown_days(pd.Series([1.0, 2.0, 3.0])) == 0


def _by_hand(weights):
    px = make_prices(open_=[10, 10, 11, 12, 12], close=[10, 11, 12, 12, 13])
    cfg = BacktestConfig(initial_cash=1_000.0, commission_bps=10.0, slippage_bps=0.0)
    return run_backtest(px, FixedWeights(weights), cfg)


def test_round_trip_pnl_matches_equity_change():
    trips = metrics.round_trips(_by_hand((1, 1, 0, 0, 0)))
    assert len(trips) == 1
    assert trips["pnl"].iloc[0] == pytest.approx(195.822)
    assert trips["return"].iloc[0] == pytest.approx(1186.812 / 990.99 - 1)
    assert not trips["open"].iloc[0]


def test_open_position_is_marked_to_final_close():
    res = _by_hand((1, 1, 1, 1, 1))
    trips = metrics.round_trips(res)
    assert trips["open"].tolist() == [True]
    assert trips["pnl"].iloc[0] == pytest.approx(res.equity.iloc[-1] - 1_000.0)


def test_summarize_by_hand():
    s = metrics.summarize(_by_hand((1, 1, 0, 0, 0)))
    assert s["final_equity"] == pytest.approx(1195.822)
    assert s["total_return"] == pytest.approx(0.195822)
    assert s["exposure"] == pytest.approx(0.4)
    assert s["round_trips"] == 1
    assert s["win_rate"] == 1.0
    assert s["costs_paid"] == pytest.approx(0.99 + 1.188)


def test_summarize_handles_a_strategy_that_never_trades():
    s = metrics.summarize(_by_hand((0, 0, 0, 0, 0)))
    assert s["total_return"] == 0.0
    assert math.isnan(s["sharpe"]) and math.isnan(s["win_rate"]) and math.isnan(s["calmar"])


def test_compare_builds_one_column_per_result():
    table = metrics.compare({"a": _by_hand((1, 1, 0, 0, 0)), "b": _by_hand((0, 0, 0, 0, 0))})
    assert list(table.columns) == ["a", "b"]
    assert table.loc["final_equity", "b"] == 1_000.0
