"""Performance metrics computed from an equity curve and a trade log."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import BacktestResult

TRADING_DAYS = 252  # US equity sessions per year, used to annualise daily figures


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate: the constant yearly return that gives the same result."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annual_volatility(returns) -> float:
    r = np.asarray(returns, dtype="float64")
    if r.size < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns, risk_free_rate: float = 0.0) -> float:
    """Annualised mean excess return per unit of volatility.

    ``risk_free_rate`` is annual. Daily mean and standard deviation scale by
    252 and sqrt(252), so the ratio scales by sqrt(252).
    """
    r = np.asarray(returns, dtype="float64") - risk_free_rate / TRADING_DAYS
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd < 1e-12:  # e.g. the strategy never traded
        return float("nan")
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe, but only downside moves count as risk."""
    r = np.asarray(returns, dtype="float64") - risk_free_rate / TRADING_DAYS
    if r.size < 2:
        return float("nan")
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside < 1e-12:
        return float("nan")
    return float(r.mean() / downside * np.sqrt(TRADING_DAYS))


def drawdown(equity: pd.Series) -> pd.Series:
    """Fractional loss from the running peak (0 at a new high, negative below it)."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def max_drawdown_days(equity: pd.Series) -> int:
    """Longest run of consecutive trading days spent below a previous peak."""
    underwater = (drawdown(equity).to_numpy() < 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], underwater, [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if starts.size else 0


def round_trips(result: BacktestResult) -> pd.DataFrame:
    """Group fills into trades that run from flat, to invested, back to flat.

    A position still open on the last day is valued at the final close and
    flagged ``open``.
    """
    rows = []
    position = cost = proceeds = 0.0
    entry = None
    for fill in result.trades.itertuples(index=False):
        if position == 0.0:
            entry, cost, proceeds = fill.date, 0.0, 0.0
        if fill.side == "BUY":
            position += fill.shares
            cost += fill.shares * fill.price + fill.commission
        else:
            position -= fill.shares
            proceeds += fill.shares * fill.price - fill.commission
        if position < 1e-9:
            position = 0.0
            rows.append((entry, fill.date, proceeds - cost, proceeds / cost - 1.0, False))
    if position > 0.0:
        final_value = result.equity.iloc[-1] - result.cash.iloc[-1]
        proceeds += final_value
        rows.append((entry, result.equity.index[-1], proceeds - cost, proceeds / cost - 1.0, True))
    return pd.DataFrame(rows, columns=["entry", "exit", "pnl", "return", "open"])


def summarize(result: BacktestResult, risk_free_rate: float = 0.0) -> dict[str, float]:
    equity = result.equity
    returns = equity.pct_change().to_numpy()[1:]
    trips = round_trips(result)
    mdd = max_drawdown(equity)
    growth = cagr(equity)
    costs = result.trades["commission"].sum() + result.trades["slippage_cost"].sum()
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return(equity),
        "cagr": growth,
        "volatility": annual_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate),
        "sortino": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": mdd,
        "max_drawdown_days": max_drawdown_days(equity),
        "calmar": growth / abs(mdd) if mdd < 0 else float("nan"),
        "exposure": float((result.shares > 0).mean()),
        "round_trips": len(trips),
        "win_rate": float((trips["pnl"] > 0).mean()) if len(trips) else float("nan"),
        "costs_paid": float(costs),
    }


def compare(results: dict[str, BacktestResult], risk_free_rate: float = 0.0) -> pd.DataFrame:
    """One column per named result, one row per metric."""
    return pd.DataFrame({name: summarize(r, risk_free_rate) for name, r in results.items()})
