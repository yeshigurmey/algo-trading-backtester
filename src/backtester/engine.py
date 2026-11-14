"""Event-driven simulator: steps through history one trading day at a time.

Timeline for each day ``t``:

1. At the open, fill the order implied by the decision made at the close of
   ``t - 1``. Buys pay ``open * (1 + slippage)``, sells receive
   ``open * (1 - slippage)``, and both pay commission on the traded value.
2. At the close, mark the account to market: ``equity = cash + shares * close``.

Deciding on one bar and filling on the next is what keeps the simulation
honest. Filling at the same close that produced the signal assumes you can
observe the closing price and trade at it simultaneously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtester.strategies import Strategy

TRADE_COLUMNS = ["date", "side", "shares", "price", "commission", "slippage_cost"]


@dataclass(frozen=True)
class BacktestConfig:
    """Account and cost assumptions.

    Costs are in basis points of traded value (1 bp = 0.01%) rather than
    dollars per share, because prices here are split/dividend adjusted: an
    adjusted price from years ago is far below what actually traded, so a
    per-share fee would be charged on the wrong number of shares. A
    percentage of value is unaffected by the adjustment.
    """

    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    fractional_shares: bool = False

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")


@dataclass
class BacktestResult:
    strategy: str
    config: BacktestConfig
    equity: pd.Series  # account value at each close
    cash: pd.Series
    shares: pd.Series
    weights: pd.Series  # the strategy's target weights (decision at each close)
    trades: pd.DataFrame = field(repr=False)

    @property
    def returns(self) -> pd.Series:
        """Daily simple returns of the equity curve."""
        return self.equity.pct_change().fillna(0.0)


def run_backtest(
    prices: pd.DataFrame, strategy: Strategy, config: BacktestConfig | None = None
) -> BacktestResult:
    config = config or BacktestConfig()
    weights = strategy.target_weights(prices)
    if not weights.index.equals(prices.index):
        raise ValueError("strategy weights must be indexed like prices")

    w = weights.to_numpy(dtype="float64")
    if np.isnan(w).any() or (w < 0).any() or (w > 1).any():
        raise ValueError("target weights must lie in [0, 1] (long-only, no leverage)")

    # Plain NumPy arrays and Python floats in the loop: indexing a pandas
    # object per row costs microseconds each and would dominate the runtime.
    open_ = prices["open"].to_numpy(dtype="float64")
    close = prices["close"].to_numpy(dtype="float64")
    n = len(prices)
    commission = config.commission_bps / 1e4
    slippage = config.slippage_bps / 1e4
    whole_shares = not config.fractional_shares

    cash = config.initial_cash
    shares = 0.0
    current_target = 0.0
    equity_hist = np.empty(n)
    cash_hist = np.empty(n)
    shares_hist = np.empty(n)
    trades: list[tuple] = []

    for t in range(n):
        # Yesterday's decision, filled at today's open. Orders are only sent
        # when the target changes, so a held position is not churned daily to
        # correct small drifts in its weight.
        if t > 0 and w[t - 1] != current_target:
            current_target = w[t - 1]
            px = open_[t]
            wanted = current_target * (cash + shares * px) / px
            if whole_shares:
                wanted = math.floor(wanted + 1e-9)
            delta = wanted - shares

            if delta > 0:
                fill = px * (1.0 + slippage)
                unit_cost = fill * (1.0 + commission)
                # Never spend more cash than the account holds once costs are included.
                affordable = cash / unit_cost
                if whole_shares:
                    affordable = math.floor(affordable + 1e-9)
                qty = min(delta, affordable)
                if qty > 0:
                    fee = qty * fill * commission
                    cash = max(cash - qty * fill - fee, 0.0)
                    shares += qty
                    trades.append((t, "BUY", qty, fill, fee, qty * px * slippage))
            elif delta < 0:
                qty = -delta
                fill = px * (1.0 - slippage)
                fee = qty * fill * commission
                cash += qty * fill - fee
                shares -= qty
                trades.append((t, "SELL", qty, fill, fee, qty * px * slippage))

        equity_hist[t] = cash + shares * close[t]
        cash_hist[t] = cash
        shares_hist[t] = shares

    index = prices.index
    trade_log = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    trade_log["date"] = index[trade_log["date"].to_numpy(dtype=int)]
    return BacktestResult(
        strategy=str(strategy),
        config=config,
        equity=pd.Series(equity_hist, index=index, name="equity"),
        cash=pd.Series(cash_hist, index=index, name="cash"),
        shares=pd.Series(shares_hist, index=index, name="shares"),
        weights=weights.rename("target_weight"),
        trades=trade_log,
    )
