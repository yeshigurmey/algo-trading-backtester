"""Text and chart reports for a backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from backtester.engine import BacktestResult
from backtester.metrics import drawdown

# Colours: a blue/orange pair that stays distinguishable under the common
# forms of colour blindness. Text is always neutral ink, never a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
STRATEGY_COLOR = "#2a78d6"
BENCHMARK_COLOR = "#eb6834"

_PERCENT_ROWS = {
    "total_return",
    "cagr",
    "volatility",
    "max_drawdown",
    "exposure",
    "win_rate",
}
_MONEY_ROWS = {"final_equity", "costs_paid"}
_COUNT_ROWS = {"max_drawdown_days", "round_trips"}


def format_table(table: pd.DataFrame) -> str:
    """Render a ``metrics.compare`` table with units a person can read."""

    def fmt(metric: str, value: float) -> str:
        if pd.isna(value):
            return "n/a"
        if metric in _PERCENT_ROWS:
            return f"{value:.2%}"
        if metric in _MONEY_ROWS:
            return f"${value:,.2f}"
        if metric in _COUNT_ROWS:
            return f"{int(value):,}"
        return f"{value:.2f}"

    text = pd.DataFrame(
        {col: [fmt(m, v) for m, v in table[col].items()] for col in table.columns},
        index=table.index,
    )
    return text.to_string()


def _money(value: float, _pos=None) -> str:
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    if abs(value) >= 1e4:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


def _dates(series: pd.Series):
    """Plain datetime64 array: every matplotlib version plots it without pandas converters."""
    return series.index.to_numpy()


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    ax.margins(x=0)


def _end_labels(ax, series: dict[str, tuple[pd.Series, str]]) -> None:
    """Dot and value at the end of each line, nudged apart if they would overlap."""
    ends = sorted(
        ((s.iloc[-1], s.index[-1].to_datetime64(), color) for s, color in series.values()),
        key=lambda end: end[0],
        reverse=True,
    )
    low, high = ax.get_ylim()
    crowded = len(ends) == 2 and abs(ends[0][0] - ends[1][0]) < 0.05 * (high - low)
    for rank, (value, when, color) in enumerate(ends):
        ax.plot(
            [when],
            [value],
            marker="o",
            markersize=7,
            color=color,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            clip_on=False,
            zorder=5,
        )
        nudge = (6 if rank == 0 else -6) if crowded else 0
        ax.annotate(
            _money(value),
            (when, value),
            xytext=(8, nudge),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
            annotation_clip=False,
        )


def plot_report(
    result: BacktestResult,
    benchmark: BacktestResult,
    title: str,
    path: str | Path | None = None,
) -> Figure:
    """Three stacked panels on a shared time axis: account value, drawdown, exposure."""
    fig = Figure(figsize=(11, 8), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[3, 1.4, 0.8],
        hspace=0.32,
        left=0.07,
        right=0.91,
        top=0.865,
        bottom=0.06,
    )
    ax_eq = fig.add_subplot(grid[0])
    ax_dd = fig.add_subplot(grid[1], sharex=ax_eq)
    ax_ex = fig.add_subplot(grid[2], sharex=ax_eq)
    for ax in (ax_eq, ax_dd, ax_ex):
        _style(ax)

    fig.text(0.07, 0.955, title, fontsize=14, fontweight="bold", color=INK)
    subtitle = (
        f"{result.equity.index[0]:%b %Y} to {result.equity.index[-1]:%b %Y}"
        f"   |   start ${result.config.initial_cash:,.0f}"
        f"   |   costs {result.config.commission_bps:g} bp commission"
        f" + {result.config.slippage_bps:g} bp slippage per trade"
    )
    fig.text(0.07, 0.925, subtitle, fontsize=9.5, color=INK_SOFT)

    lines = {
        "Strategy": (result.equity, STRATEGY_COLOR),
        "Buy and hold": (benchmark.equity, BENCHMARK_COLOR),
    }
    for label, (equity, color) in lines.items():
        ax_eq.plot(
            _dates(equity),
            equity.to_numpy(),
            color=color,
            linewidth=2,
            label=label,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    _end_labels(ax_eq, lines)
    ax_eq.yaxis.set_major_formatter(FuncFormatter(_money))
    ax_eq.set_title("Account value", loc="left", fontsize=10, color=INK_SOFT)
    ax_eq.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)

    dd, bench_dd = drawdown(result.equity), drawdown(benchmark.equity)
    ax_dd.fill_between(_dates(dd), dd.to_numpy(), 0, color=STRATEGY_COLOR, alpha=0.10, linewidth=0)
    ax_dd.plot(_dates(dd), dd.to_numpy(), color=STRATEGY_COLOR, linewidth=1.5)
    ax_dd.plot(_dates(bench_dd), bench_dd.to_numpy(), color=BENCHMARK_COLOR, linewidth=1.5)
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_dd.set_title("Drawdown from previous peak", loc="left", fontsize=10, color=INK_SOFT)

    invested = 1.0 - result.cash / result.equity
    ax_ex.fill_between(
        _dates(invested),
        invested.to_numpy(),
        0,
        step="post",
        color=STRATEGY_COLOR,
        alpha=0.35,
        linewidth=0,
    )
    ax_ex.set_ylim(0, 1.05)
    ax_ex.set_yticks([0, 1])
    ax_ex.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_ex.set_title("Strategy: share of account invested", loc="left", fontsize=10, color=INK_SOFT)

    ax_eq.tick_params(labelbottom=False)
    ax_dd.tick_params(labelbottom=False)

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig
