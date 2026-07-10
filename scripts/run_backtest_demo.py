#!/usr/bin/env python3
"""跑双均线策略的端到端回测 demo。

不依赖 LLM、不依赖券商，只验证：数据→策略→风控→Broker→指标 链路通畅。

用法::

    python scripts/run_backtest_demo.py
    python scripts/run_backtest_demo.py --symbol 600519 --start 2023-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table

from qtrade.backtest.engine import BacktestEngine
from qtrade.core.logging import setup_logging
from qtrade.data.schema import normalize_symbol
from qtrade.strategies.technical.ma_cross import MACrossStrategy

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fast", type=int, default=5)
    parser.add_argument("--slow", type=int, default=20)
    parser.add_argument("--out-json", default=None, help="把结果导出为 JSON")
    args = parser.parse_args()

    setup_logging()

    syms = [normalize_symbol(s) for s in (args.symbol or ["600519"])]
    console.rule(f"[bold]MA Cross Backtest[/bold]  {syms}  {args.start} → {args.end or 'today'}")

    strat = MACrossStrategy(
        universe=syms,
        params={"fast_window": args.fast, "slow_window": args.slow, "position_pct": 0.3},
    )
    engine = BacktestEngine(
        strategy=strat,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end) if args.end else date.today(),
        initial_cash=args.initial_cash,
    )
    result = engine.run()
    _print(result)

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "metrics": result.metrics.__dict__,
                    "trades": result.trades,
                    "final_account": result.final_account,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        console.print(f"\n[green]✓[/green] result written to {args.out_json}")


def _print(result) -> None:
    m = result.metrics
    table = Table(title="回测结果")
    table.add_column("指标", style="cyan", no_wrap=True)
    table.add_column("数值", style="white")
    rows = [
        ("总收益", f"{m.total_return:.2%}"),
        ("年化收益", f"{m.annual_return:.2%}"),
        ("年化波动", f"{m.annual_volatility:.2%}"),
        ("Sharpe", f"{m.sharpe:.2f}"),
        ("Sortino", f"{m.sortino:.2f}"),
        ("Calmar", f"{m.calmar:.2f}"),
        ("最大回撤", f"{m.max_drawdown:.2%}"),
        ("最大回撤天数", str(m.max_drawdown_duration_days)),
        ("交易次数", str(m.n_trades)),
        ("胜率", f"{m.win_rate:.2%}"),
        ("盈亏比", f"{m.profit_loss_ratio:.2f}"),
        ("初始权益", f"{m.initial_equity:,.2f}"),
        ("最终权益", f"{m.final_equity:,.2f}"),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)

    if result.trades:
        console.print("\n[bold]最近 5 笔交易：[/bold]")
        for t in result.trades[-5:]:
            console.print(
                f"  [{t['ts']}] {t['side']:<4} {t['symbol']} x{t['quantity']:<6}"
                f" @ {t['price']:.2f}  fees={t['commission'] + t['stamp_tax']:.2f}"
                f"  reason={t['reason']}"
            )


if __name__ == "__main__":
    main()
