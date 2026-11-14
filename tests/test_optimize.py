import numpy as np
import pandas as pd
import pytest

from backtester.metrics import sharpe_ratio
from backtester.optimize import expand_grid, grid_search, walk_forward
from backtester.strategies import SmaCrossover
from backtester.vectorized import run_vectorized

GRID = {"fast": [5, 10, 20], "slow": [10, 20, 40]}


def test_expand_grid_skips_invalid_combinations():
    combos = [(s.fast, s.slow) for s in expand_grid(SmaCrossover, GRID)]
    assert combos == [(5, 10), (5, 20), (5, 40), (10, 20), (10, 40), (20, 40)]
    with pytest.raises(ValueError):
        expand_grid(SmaCrossover, {"fast": [50], "slow": [10]})


def test_grid_search_ranks_by_sharpe_and_matches_a_direct_run(prices):
    table = grid_search(prices, SmaCrossover, GRID)
    assert len(table) == 6
    assert table["sharpe"].is_monotonic_decreasing

    best = table.iloc[0]
    equity = run_vectorized(prices, SmaCrossover(int(best["fast"]), int(best["slow"])))
    direct = sharpe_ratio(equity.pct_change().fillna(0.0))
    assert best["sharpe"] == pytest.approx(direct)


def test_walk_forward_windows_tile_the_history_after_training(prices):
    wf = walk_forward(prices, SmaCrossover, GRID, train_days=300, test_days=100)
    assert wf.returns.index.equals(prices.index[300:])
    assert len(wf.folds) == 7
    assert wf.folds["test_start"].iloc[0] == prices.index[300]
    assert wf.folds["test_end"].iloc[-1] == prices.index[-1]
    # Each test window starts the day after the previous one ends.
    gaps = wf.folds["test_start"].iloc[1:].to_numpy() > wf.folds["test_end"].iloc[:-1].to_numpy()
    assert gaps.all()
    assert wf.equity.iloc[-1] == pytest.approx(np.prod(1 + wf.returns))


def test_walk_forward_is_blind_to_the_future(prices):
    """Appending later data must not change any earlier fold's choice or result."""
    full = walk_forward(prices, SmaCrossover, GRID, train_days=300, test_days=100)
    early = walk_forward(prices.iloc[:700], SmaCrossover, GRID, train_days=300, test_days=100)
    pd.testing.assert_frame_equal(full.folds.iloc[: len(early.folds)], early.folds)
    pd.testing.assert_series_equal(full.returns.iloc[:400], early.returns)


def test_walk_forward_picks_the_best_training_candidate(prices):
    wf = walk_forward(prices, SmaCrossover, GRID, train_days=300, test_days=100)
    fold = wf.folds.iloc[2]
    start = prices.index.get_loc(fold["test_start"])
    candidates = expand_grid(SmaCrossover, GRID)
    scores = []
    for s in candidates:
        r = run_vectorized(prices, s).pct_change().to_numpy()
        scores.append(sharpe_ratio(r[start - 300 : start]))
    winner = candidates[int(np.nanargmax(scores))]
    assert (fold["fast"], fold["slow"]) == (winner.fast, winner.slow)
    assert fold["train_sharpe"] == pytest.approx(max(scores))


def test_walk_forward_needs_enough_history(prices):
    with pytest.raises(ValueError):
        walk_forward(prices, SmaCrossover, GRID, train_days=900, test_days=200)
