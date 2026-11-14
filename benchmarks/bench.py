"""Time both engines on 20 years of daily bars.  Usage: python benchmarks/bench.py"""

from __future__ import annotations

import time
from collections.abc import Callable

from backtester.data import synthetic_prices
from backtester.engine import run_backtest
from backtester.optimize import grid_search
from backtester.strategies import SmaCrossover
from backtester.vectorized import run_vectorized


def best_of(fn: Callable[[], object], repeats: int = 5, loops: int = 20) -> float:
    """Best average seconds per call; the minimum is the least noisy estimate."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(loops):
            fn()
        timings.append((time.perf_counter() - start) / loops)
    return min(timings)


def main() -> None:
    prices = synthetic_prices(5040, seed=1)
    strategy = SmaCrossover(50, 200)
    grid = {"fast": list(range(5, 105, 5)), "slow": list(range(110, 310, 10))}

    event = best_of(lambda: run_backtest(prices, strategy))
    vector = best_of(lambda: run_vectorized(prices, strategy))
    sweep = best_of(lambda: grid_search(prices, SmaCrossover, grid), repeats=3, loops=1)

    print(f"bars per backtest:        {len(prices):,}")
    print(
        f"event-driven engine:      {event * 1e3:7.2f} ms  ({event / len(prices) * 1e9:.0f} ns/bar)"
    )
    print(f"vectorised engine:        {vector * 1e3:7.2f} ms  ({event / vector:.1f}x faster)")
    print(f"400-combination sweep:    {sweep * 1e3:7.0f} ms")


if __name__ == "__main__":
    main()
