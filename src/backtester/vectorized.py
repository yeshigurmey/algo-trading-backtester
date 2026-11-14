"""Vectorised backtest: the whole simulation as array arithmetic, no Python loop.

Same timing rules as the event-driven engine (decide at the close, fill at the
next open), but it works on portfolio *weights* instead of cash and share
counts. That makes it several times faster (see benchmarks/bench.py), which
adds up when a parameter search runs hundreds of backtests, at the price of
two approximations: it ignores whole-share rounding, and it charges costs as
a flat fraction of the amount traded.

The event-driven engine stays the source of truth. The test suite checks the
two agree, so a bug in either one shows up as a disagreement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import BacktestConfig
from backtester.strategies import Strategy


def weight_returns(
    open_: np.ndarray, close: np.ndarray, weights: np.ndarray, cost_bps: float
) -> np.ndarray:
    """Daily portfolio returns for target ``weights`` decided at each close."""
    held = np.empty_like(weights)  # weight held from today's open onward
    held[0] = 0.0
    held[1:] = weights[:-1]
    carried = np.empty_like(held)  # weight carried in from yesterday
    carried[0] = 0.0
    carried[1:] = held[:-1]

    overnight = np.empty_like(close)  # yesterday's close -> today's open
    overnight[0] = 0.0
    overnight[1:] = open_[1:] / close[:-1] - 1.0
    intraday = close / open_ - 1.0  # today's open -> today's close

    # The old weight earns the overnight move, the trade happens at the open
    # and pays costs on the amount traded, then the new weight earns the rest
    # of the day.
    cost = np.abs(held - carried) * (cost_bps / 1e4)
    return (1.0 + carried * overnight) * (1.0 - cost) * (1.0 + held * intraday) - 1.0


def run_vectorized(
    prices: pd.DataFrame, strategy: Strategy, config: BacktestConfig | None = None
) -> pd.Series:
    """Equity curve for ``strategy``, computed without a per-day loop."""
    config = config or BacktestConfig()
    returns = weight_returns(
        prices["open"].to_numpy(dtype="float64"),
        prices["close"].to_numpy(dtype="float64"),
        strategy.target_weights(prices).to_numpy(dtype="float64"),
        config.commission_bps + config.slippage_bps,
    )
    equity = config.initial_cash * np.cumprod(1.0 + returns)
    return pd.Series(equity, index=prices.index, name="equity")
