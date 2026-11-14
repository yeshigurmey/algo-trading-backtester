# Design notes

Why the backtester is built the way it is. Each section states a decision, the
alternative that was rejected, and the reason.

## 1. Decide at the close, fill at the next open

**Decision.** A strategy's output for day `t` may use prices up to and
including the close of `t`. The engine fills the resulting order at the open of
`t + 1`.

**Rejected.** Filling at the close of `t`, the same bar that produced the
signal. This is the most common backtesting shortcut and it is optimistic: the
closing price is only known once the market has closed, so you cannot both
observe it and trade at it.

**Consequence.** Overnight gaps hurt, as they would in real trading. If a
stock jumps 5% overnight after a buy signal, the simulated account buys after
the jump, not before it.

**How it is enforced.** Three layers:

1. The engine reads `weights[t - 1]` when processing day `t`. The strategy
   never sees the loop, so it cannot fill early by accident.
2. `find_lookahead()` recomputes a strategy on truncated histories. If the
   weight for some past day changes when later rows are removed, the strategy
   used future data (for example a centred rolling window or `shift(-1)`).
3. The random-walk test (section 6) checks the end-to-end result.

## 2. Strategies return weights, not orders

**Decision.** A strategy returns one number per day: the target fraction of
the account to hold in the asset, from 0 to 1.

**Rejected.** Having strategies emit buy/sell orders with share counts.

**Reason.** It separates *what to hold* from *how to trade into it*. Position
sizing, whole-share rounding, cash limits and costs live in one place (the
engine), so every strategy gets them right for free, and a strategy is a few
lines of pandas. It also means the same strategy code feeds both engines.

Indicators are computed once for the whole history with vectorised pandas
operations. A rolling mean is O(n) for the full series; recomputing a 200-day
average inside a day-by-day loop would be O(n x 200).

## 3. Costs in basis points of traded value

**Decision.** Slippage and commission are each a percentage of the trade's
value (defaults: 2 bp and 1 bp, where 1 bp = 0.01%).

**Rejected.** A per-share commission, which is how many brokers actually quote.

**Reason.** The price series is adjusted for splits and dividends, which
rescales old prices. A stock that traded at $400 before a 4-for-1 split shows
as $100 or less in adjusted history, so the simulated share count is several
times the real one and a per-share fee would be overcharged by the same
factor. A percentage of value does not change under that rescaling.

Slippage moves the fill price against the trader (buys fill higher, sells fill
lower), which models crossing the bid-ask spread. The defaults are meant to be
conservative for a liquid ETF and should be raised for anything less liquid.

## 4. Whole shares and a hard cash limit

The engine buys `floor(target value / price)` shares, then reduces the order if
the cash needed *including costs* exceeds the cash available. Being fully
invested in a $500 stock with a $100,000 account leaves some cash idle, as it
would in a real account. `fractional_shares=True` turns this off, which is used
by the tests that compare the two engines exactly.

Orders are sent only when the target weight changes. Rebalancing every day to
correct small drifts would generate many tiny trades and pay costs on each.

## 5. Two engines that must agree

**Event-driven (`engine.py`).** A loop over days that tracks cash and shares.
It is the source of truth: it models whole shares, the cash limit and exact
costs, and produces the trade log.

**Vectorised (`vectorized.py`).** The same timing rules written as array
arithmetic on weights. Each day's return is split in two: the previous weight
earns the overnight move (close to open), the trade happens at the open, and
the new weight earns the intraday move (open to close).

**Why both.** Speed is part of it: the vectorised engine is about 6x faster,
which matters for sweeps. The larger benefit is verification. The two
implementations share almost no code, so a timing or accounting bug in one
shows up as a disagreement. The tests require agreement to 1e-9 with no costs
and fractional shares, where the two models are mathematically identical, and
to 1e-4 with costs, where the vectorised model is a first-order approximation.

## 6. Testing against a known answer: the random walk

Unit tests can show the code does what was intended. They cannot show the
intention was right. For that the simulator needs an input where the correct
answer is known in advance.

A random walk with zero drift is that input. Tomorrow's move is independent of
all past prices, so any strategy that only uses past data has an expected
profit of zero before costs. The test simulates 200 independent paths per
strategy and requires the mean return to be within three standard errors of
zero. The same test run on a strategy that reads tomorrow's close gives a mean
more than 80 standard errors above zero, so the test has the power to detect a
leak.

## 7. Walk-forward analysis instead of trusting a grid search

A grid search over 400 parameter sets will always produce a winner, and the
winner's Sharpe ratio is biased upward because it was selected for fitting
that sample, noise included. Walk-forward analysis measures what survives:

1. Choose the best parameters on a training window (default 3 years).
2. Record how they perform on the next window (default 1 year), which played
   no part in choosing them.
3. Slide both windows forward and repeat. Join only the test windows.

The command prints mean training Sharpe next to mean out-of-sample Sharpe. The
gap between them is the overfitting.

**Efficiency.** A naive implementation reruns every candidate for every
window. Because strategies are causal (section 1), a candidate's returns inside
any window are the same whether computed from the full history or from the
history up to that window. So each candidate is simulated once and the windows
are sliced from the result: `P` backtests instead of `P x windows`.

## 8. Data handling

- **Adjusted prices.** `auto_adjust=True` folds splits and dividends into the
  price series. Without it a split looks like a crash, and buy-and-hold is
  understated because dividends are missing.
- **Cache.** Downloads are saved as CSV keyed by ticker and date range. Runs
  are reproducible, and a sweep does not make repeated network requests.
- **Validate at the boundary.** Sorted unique dates, finite positive prices.
  Days with a missing quote are dropped, not forward-filled, because a filled
  price would let the simulator trade at a level that was not available.
- **Dependency injection.** `load_prices` takes the download function as an
  argument, so tests substitute a fake and run offline.

## 9. Metrics

| Metric | Definition |
|---|---|
| Total return | `final / initial - 1` |
| CAGR | `(final / initial) ^ (1 / years) - 1` |
| Volatility | standard deviation of daily returns x sqrt(252) |
| Sharpe | mean daily excess return / standard deviation x sqrt(252) |
| Sortino | as Sharpe, but dividing by downside deviation only |
| Max drawdown | largest fall from a previous peak, `min(equity / running max - 1)` |
| Calmar | CAGR / abs(max drawdown) |
| Exposure | fraction of days holding a position |

Daily figures are annualised with 252 trading days. The mean scales with time
and the standard deviation with its square root, which is where the sqrt(252)
in the Sharpe ratio comes from. That scaling assumes daily returns are
independent, which is an approximation.

## 10. What would come next

- Multiple assets with portfolio weights, and short positions with borrow costs.
- Stop-loss and limit orders using the daily high and low.
- Slippage that grows with order size relative to volume.
- Bootstrap confidence intervals for the Sharpe ratio.
