"""qtrade 命令行入口（typer）。

子命令：
- ``init``       初始化数据库
- ``download``   下载历史数据到本地缓存
- ``backtest``   跑一个策略的回测，输出指标
- ``ui``         启动桌面 UI
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True, help="qtrade 量化平台")
console = Console()


@app.command()
def init() -> None:
    """初始化数据库（创建所有表）。幂等。"""
    from qtrade.persistence import init_db
    init_db()
    console.print("[green]✓[/green] 数据库已初始化")


@app.command()
def download(
    symbols: list[str] = typer.Argument(..., help="标的列表，如 600519 000001"),
    start: str = typer.Option("2024-01-01", help="起始日 YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="终止日 YYYY-MM-DD，默认今天"),
) -> None:
    """下载日线到本地 Parquet 缓存。"""
    from qtrade.data.loader import get_loader
    loader = get_loader()
    end_d = date.fromisoformat(end) if end else date.today()
    for sym in symbols:
        df = loader.get_daily_bars(sym, start=start, end=end_d, force_refresh=True)
        console.print(f"[green]✓[/green] {sym}: {len(df)} 行")


@app.command()
def backtest(
    strategy: str = typer.Option("ma_cross", help="策略类型 ma_cross | llm"),
    symbols: list[str] = typer.Option(["600519"], "--symbol", "-s"),
    start: str = typer.Option("2024-01-01"),
    end: str | None = typer.Option(None),
    initial_cash: float = typer.Option(100_000.0),
) -> None:
    """跑回测并打印关键指标。"""
    from qtrade.backtest.engine import BacktestEngine
    from qtrade.data.schema import normalize_symbol
    from qtrade.strategies.technical.ma_cross import MACrossStrategy

    syms = [normalize_symbol(s) for s in symbols]
    if strategy == "ma_cross":
        strat = MACrossStrategy(universe=syms, params={"fast_window": 5, "slow_window": 20})
    elif strategy == "llm":
        from qtrade.strategies.llm.graph import LLMMultiAgentStrategy
        strat = LLMMultiAgentStrategy(universe=syms)
    else:
        raise typer.BadParameter(f"未知策略：{strategy}")

    engine = BacktestEngine(
        strategy=strat,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end) if end else date.today(),
        initial_cash=initial_cash,
    )
    result = engine.run()
    _print_backtest(result)


def _print_backtest(result) -> None:
    m = result.metrics
    table = Table(title="回测结果")
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="white")
    table.add_row("总收益", f"{m.total_return:.2%}")
    table.add_row("年化收益", f"{m.annual_return:.2%}")
    table.add_row("年化波动", f"{m.annual_volatility:.2%}")
    table.add_row("Sharpe", f"{m.sharpe:.2f}")
    table.add_row("Sortino", f"{m.sortino:.2f}")
    table.add_row("Calmar", f"{m.calmar:.2f}")
    table.add_row("最大回撤", f"{m.max_drawdown:.2%}")
    table.add_row("最大回撤天数", f"{m.max_drawdown_duration_days}")
    table.add_row("交易次数", f"{m.n_trades}")
    table.add_row("胜率", f"{m.win_rate:.2%}")
    table.add_row("盈亏比", f"{m.profit_loss_ratio:.2f}")
    table.add_row("初始权益", f"{m.initial_equity:,.2f}")
    table.add_row("最终权益", f"{m.final_equity:,.2f}")
    console.print(table)


@app.command()
def ui() -> None:
    """启动桌面应用。"""
    from qtrade.ui.main import run
    run()


if __name__ == "__main__":
    app()
