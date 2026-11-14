"""Parameter search, and walk-forward analysis to measure how much of it is overfitting.

Trying many parameter sets and reporting the best one is biased: the winner is
partly the one that best fits noise in that sample. Walk-forward analysis
measures what survives. Parameters are chosen on a training window, then
scored on the following window they have never seen, and the windows roll
forward through history. Only the unseen-window results are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from backtester.engine import BacktestConfig
from backtester.metrics import max_drawdown, sharpe_ratio
from backtester.strategies import Strategy
from backtester.vectorized import weight_returns

ParamGrid = dict[str, list]


def expand_grid(strategy_cls: type[Strategy], grid: ParamGrid) -> list[Strategy]:
    """Every valid combination in ``grid`` (e.g. fast >= slow is skipped)."""
    strategies = []
    for values in product(*grid.values()):
        try:
            strategies.append(strategy_cls(**dict(zip(grid, values, strict=True))))
        except ValueError:
            continue
    if not strategies:
        raise ValueError("parameter grid contains no valid combination")
    return strategies


def _returns_matrix(
    prices: pd.DataFrame, strategies: list[Strategy], config: BacktestConfig
) -> np.ndarray:
    """Daily returns for every candidate: shape (n_candidates, n_days)."""
    open_ = prices["open"].to_numpy(dtype="float64")
    close = prices["close"].to_numpy(dtype="float64")
    cost_bps = config.commission_bps + config.slippage_bps
    out = np.empty((len(strategies), len(prices)))
    for i, strategy in enumerate(strategies):
        weights = strategy.target_weights(prices).to_numpy(dtype="float64")
        out[i] = weight_returns(open_, close, weights, cost_bps)
    return out


def grid_search(
    prices: pd.DataFrame,
    strategy_cls: type[Strategy],
    grid: ParamGrid,
    config: BacktestConfig | None = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Backtest every parameter combination; best Sharpe ratio first."""
    config = config or BacktestConfig()
    strategies = expand_grid(strategy_cls, grid)
    returns = _returns_matrix(prices, strategies, config)
    rows = []
    for strategy, r in zip(strategies, returns, strict=True):
        equity = pd.Series(np.cumprod(1.0 + r))
        rows.append(
            {
                **strategy.params,
                "sharpe": sharpe_ratio(r, risk_free_rate),
                "total_return": float(equity.iloc[-1] - 1.0),
                "max_drawdown": max_drawdown(equity),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("sharpe", ascending=False, na_position="last", ignore_index=True)


@dataclass
class WalkForwardResult:
    returns: pd.Series  # out-of-sample daily returns, all test windows joined
    folds: pd.DataFrame  # one row per window: dates, chosen parameters, both Sharpes

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def walk_forward(
    prices: pd.DataFrame,
    strategy_cls: type[Strategy],
    grid: ParamGrid,
    train_days: int = 756,
    test_days: int = 252,
    config: BacktestConfig | None = None,
    risk_free_rate: float = 0.0,
) -> WalkForwardResult:
    """Rolling out-of-sample test.

    For each window: pick the parameters with the best Sharpe over the previous
    ``train_days``, then record their returns over the next ``test_days``.

    Each candidate is simulated once over the full history and the windows are
    sliced from that. This is only valid because strategies are causal (their
    weight for day t ignores later days, see ``find_lookahead``), so the
    full-history returns inside a window are identical to what would have been
    computed standing at the end of that window.

    Simplification: when the chosen parameters change between windows, the new
    candidate's existing position is adopted without charging a switching trade.
    """
    config = config or BacktestConfig()
    n = len(prices)
    if train_days < 2 or test_days < 1 or train_days + test_days > n:
        raise ValueError("need train_days >= 2, test_days >= 1, and enough history for one window")

    strategies = expand_grid(strategy_cls, grid)
    returns = _returns_matrix(prices, strategies, config)

    folds, pieces = [], []
    for split in range(train_days, n, test_days):
        train = slice(split - train_days, split)
        test = slice(split, min(split + test_days, n))
        train_scores = np.array([sharpe_ratio(r[train], risk_free_rate) for r in returns])
        # If nothing traded during training every score is NaN and no candidate
        # is better than another, so fall back to the first.
        best = 0 if np.isnan(train_scores).all() else int(np.nanargmax(train_scores))
        pieces.append(pd.Series(returns[best, test], index=prices.index[test]))
        folds.append(
            {
                "train_start": prices.index[train.start],
                "test_start": prices.index[test.start],
                "test_end": prices.index[test.stop - 1],
                **strategies[best].params,
                "train_sharpe": train_scores[best],
                "test_sharpe": sharpe_ratio(returns[best, test], risk_free_rate),
            }
        )
    return WalkForwardResult(returns=pd.concat(pieces).rename("returns"), folds=pd.DataFrame(folds))
