import pandas as pd
import pytest

from backtester import cli
from backtester.engine import run_backtest
from backtester.metrics import compare
from backtester.report import format_table, plot_report
from backtester.strategies import BuyAndHold, SmaCrossover

GRID = ["-g", "fast=5,10", "-g", "slow=20,40"]


def test_format_table_uses_readable_units(prices):
    text = format_table(compare({"bh": run_backtest(prices, BuyAndHold())}))
    rows = {line.split()[0]: line.split()[-1] for line in text.splitlines()[1:]}
    assert rows["final_equity"].startswith("$")
    assert rows["total_return"].endswith("%")
    assert rows["round_trips"] == "1"


def test_plot_report_writes_a_png(prices, tmp_path):
    out = tmp_path / "nested" / "report.png"
    result = run_backtest(prices, SmaCrossover(10, 30))
    fig = plot_report(result, run_backtest(prices, BuyAndHold()), "title", out)
    assert out.read_bytes().startswith(b"\x89PNG")
    assert len(fig.axes) == 3


def test_run_command_prints_metrics_and_writes_outputs(tmp_path, capsys):
    plot, trades = tmp_path / "r.png", tmp_path / "t.csv"
    argv = [
        "run",
        "--synthetic",
        "1",
        "--strategy",
        "sma",
        "-p",
        "fast=10",
        "-p",
        "slow=30",
        "--plot",
        str(plot),
        "--trades",
        str(trades),
    ]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "SmaCrossover(fast=10, slow=30)" in out and "Buy and hold" in out and "sharpe" in out
    assert plot.exists()
    assert set(pd.read_csv(trades)["side"]) == {"BUY", "SELL"}


def test_run_command_downloads_through_the_data_layer(monkeypatch, prices, capsys):
    seen = {}

    def fake_load(ticker, start, end, cache_dir, refresh):
        seen.update(ticker=ticker, start=start, end=end)
        return prices

    monkeypatch.setattr(cli, "load_prices", fake_load)
    assert cli.main(["run", "--ticker", "qqq", "--start", "2020-01-01", "--end", "2024-01-01"]) == 0
    assert seen == {"ticker": "qqq", "start": "2020-01-01", "end": "2024-01-01"}
    assert capsys.readouterr().out.startswith("QQQ:")


def test_sweep_and_walkforward_commands(capsys):
    assert cli.main(["sweep", "--synthetic", "1", *GRID, "--top", "3"]) == 0
    assert "4 parameter sets" in capsys.readouterr().out

    argv = ["walkforward", "--synthetic", "1", *GRID, "--train-days", "500", "--test-days", "250"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "walk-forward windows" in out and "Out-of-sample" in out


@pytest.mark.parametrize(
    "params", [["-p", "fast=200", "-p", "slow=50"], ["-p", "unknown=1"]], ids=["invalid", "unknown"]
)
def test_bad_strategy_parameters_exit_with_an_error(params, capsys):
    assert cli.main(["run", "--synthetic", "1", *params]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_malformed_parameter_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        cli.main(["run", "--synthetic", "1", "-p", "fast"])
