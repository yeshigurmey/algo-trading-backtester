"""Command-line interface.

backtester run         --ticker SPY --strategy sma -p fast=50 -p slow=200 --plot results/spy.png
backtester sweep       --ticker SPY --strategy sma -g fast=10,20,50 -g slow=100,150,200
backtester walkforward --ticker SPY --strategy sma -g fast=10,20,50 -g slow=100,150,200
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from backtester.data import DataError, load_prices, synthetic_prices
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compare, max_drawdown, sharpe_ratio
from backtester.optimize import grid_search, walk_forward
from backtester.report import format_table, plot_report
from backtester.strategies import STRATEGIES, BuyAndHold
from backtester.vectorized import run_vectorized


def _number(text: str) -> int | float:
    try:
        return int(text)
    except ValueError:
        return float(text)


def _param(text: str) -> tuple[str, int | float]:
    """Parse ``name=value``."""
    name, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    try:
        return name, _number(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None


def _grid_axis(text: str) -> tuple[str, list[int | float]]:
    """Parse ``name=v1,v2,v3``."""
    name, sep, values = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected name=v1,v2,..., got {text!r}")
    try:
        return name, [_number(v) for v in values.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{values!r} is not a list of numbers") from None


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    data = common.add_argument_group("data")
    data.add_argument("--ticker", default="SPY", help="Yahoo Finance symbol (default: SPY)")
    data.add_argument("--start", default="2010-01-01", help="first date, YYYY-MM-DD")
    data.add_argument("--end", default="2025-01-01", help="end date (exclusive), YYYY-MM-DD")
    data.add_argument("--cache-dir", default=".cache", help="where downloads are cached")
    data.add_argument("--refresh", action="store_true", help="re-download even if cached")
    data.add_argument(
        "--synthetic",
        type=int,
        metavar="SEED",
        default=None,
        help="use a seeded random walk instead of downloading (works offline)",
    )
    account = common.add_argument_group("account and costs")
    account.add_argument("--cash", type=float, default=100_000.0, help="starting cash")
    account.add_argument("--commission-bps", type=float, default=1.0)
    account.add_argument("--slippage-bps", type=float, default=2.0)
    account.add_argument("--fractional", action="store_true", help="allow fractional shares")
    account.add_argument(
        "--risk-free",
        type=float,
        default=0.0,
        help="annual risk-free rate for Sharpe/Sortino, e.g. 0.04",
    )
    common.add_argument("--strategy", choices=sorted(STRATEGIES), default="sma")

    parser = argparse.ArgumentParser(prog="backtester", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="backtest one strategy vs buy-and-hold")
    run.add_argument(
        "-p",
        "--param",
        type=_param,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="strategy parameter, repeatable",
    )
    run.add_argument("--plot", metavar="PNG", help="write the chart report to this file")
    run.add_argument("--trades", metavar="CSV", help="write the trade log to this file")

    grid_help = "values to try for one parameter, repeatable"
    sweep = sub.add_parser("sweep", parents=[common], help="grid-search strategy parameters")
    sweep.add_argument(
        "-g",
        "--grid",
        type=_grid_axis,
        action="append",
        required=True,
        metavar="NAME=V1,V2,...",
        help=grid_help,
    )
    sweep.add_argument("--top", type=int, default=10, help="rows to show")

    wf = sub.add_parser("walkforward", parents=[common], help="rolling out-of-sample test")
    wf.add_argument(
        "-g",
        "--grid",
        type=_grid_axis,
        action="append",
        required=True,
        metavar="NAME=V1,V2,...",
        help=grid_help,
    )
    wf.add_argument("--train-days", type=int, default=756, help="training window (trading days)")
    wf.add_argument("--test-days", type=int, default=252, help="test window (trading days)")
    return parser


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.synthetic is not None:
        return synthetic_prices(seed=args.synthetic), f"Random walk (seed {args.synthetic})"
    prices = load_prices(args.ticker, args.start, args.end, args.cache_dir, args.refresh)
    return prices, args.ticker.upper()


def _config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(args.cash, args.commission_bps, args.slippage_bps, args.fractional)


def _cmd_run(args: argparse.Namespace) -> None:
    prices, label = _load(args)
    config = _config(args)
    strategy = STRATEGIES[args.strategy](**dict(args.param))
    result = run_backtest(prices, strategy, config)
    benchmark = run_backtest(prices, BuyAndHold(), config)

    print(
        f"{label}: {len(prices):,} trading days, "
        f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}"
    )
    print(format_table(compare({str(strategy): result, "Buy and hold": benchmark}, args.risk_free)))
    if args.plot:
        plot_report(result, benchmark, f"{label}: {strategy} vs buy and hold", args.plot)
        print(f"\nchart written to {args.plot}")
    if args.trades:
        result.trades.to_csv(args.trades, index=False)
        print(f"trade log written to {args.trades}")


def _cmd_sweep(args: argparse.Namespace) -> None:
    prices, label = _load(args)
    table = grid_search(
        prices, STRATEGIES[args.strategy], dict(args.grid), _config(args), args.risk_free
    )
    print(f"{label}: {len(table)} parameter sets, best Sharpe first (in-sample)")
    print(table.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nThe top row is fitted to this sample. Use `walkforward` for an honest estimate.")


def _cmd_walkforward(args: argparse.Namespace) -> None:
    prices, label = _load(args)
    config = _config(args)
    wf = walk_forward(
        prices,
        STRATEGIES[args.strategy],
        dict(args.grid),
        args.train_days,
        args.test_days,
        config,
        args.risk_free,
    )
    folds = wf.folds.copy()
    for col in ("train_start", "test_start", "test_end"):
        folds[col] = folds[col].dt.strftime("%Y-%m-%d")
    print(f"{label}: {len(folds)} walk-forward windows")
    print(folds.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # Buy-and-hold over exactly the out-of-sample days, for a like-for-like comparison.
    bench = run_vectorized(prices, BuyAndHold(), config).pct_change().loc[wf.returns.index]
    bench_equity = (1.0 + bench.fillna(0.0)).cumprod()
    print("\nOut-of-sample, all windows joined:")
    for name, returns, equity in (
        ("strategy", wf.returns, wf.equity),
        ("buy and hold", bench.fillna(0.0), bench_equity),
    ):
        print(
            f"  {name:<13} Sharpe {sharpe_ratio(returns, args.risk_free):5.2f}   "
            f"return {equity.iloc[-1] - 1:8.2%}   max drawdown {max_drawdown(equity):7.2%}"
        )
    print(
        f"  mean Sharpe in training {folds['train_sharpe'].mean():.2f} "
        f"vs out-of-sample {folds['test_sharpe'].mean():.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"run": _cmd_run, "sweep": _cmd_sweep, "walkforward": _cmd_walkforward}
    try:
        commands[args.command](args)
    except (DataError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
