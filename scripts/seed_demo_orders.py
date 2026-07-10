#!/usr/bin/env python3
"""一次性脚本：往 SQLite 里注入几笔示意成交，让 UI 有真实数据可看。

用法::

    python scripts/seed_demo_orders.py
    python scripts/seed_demo_orders.py --reset       # 先清掉旧数据再注入
"""
from __future__ import annotations

import argparse
from decimal import Decimal

from rich.console import Console

from qtrade.execution.account import Account
from qtrade.execution.broker.paper import PaperBroker
from qtrade.execution.oms import OMS
from qtrade.persistence import init_db, session_scope
from qtrade.persistence.models import OrderSide, OrderType
from qtrade.risk.engine import RiskEngine
from qtrade.risk.policies import RiskPolicies
from qtrade.runtime.state import StateManager

console = Console()

DEMO_TRADES = [
    # (symbol, side, qty, price)
    ("000001.SZ", OrderSide.BUY, 2000, 11.05),
    ("000001.SZ", OrderSide.BUY, 1000, 11.20),     # 加仓
    ("600519.SH", OrderSide.BUY, 100, 1680.00),     # 茅台 1 手
]


def _maybe_reset() -> None:
    """删掉旧 demo run 的所有相关记录。"""
    from qtrade.persistence.models import (
        AccountSnapshot,
        AgentDecision,
        OrderRecord,
        PositionSnapshot,
        StrategyRun,
        TradeFill,
    )

    with session_scope() as s:
        runs = s.query(StrategyRun).filter(StrategyRun.strategy_id == "demo_seed").all()
        for r in runs:
            s.query(TradeFill).filter(
                TradeFill.order_id.in_(s.query(OrderRecord.id).filter(OrderRecord.run_id == r.id))
            ).delete(synchronize_session=False)
            s.query(OrderRecord).filter(OrderRecord.run_id == r.id).delete()
            s.query(AccountSnapshot).filter(AccountSnapshot.run_id == r.id).delete()
            s.query(PositionSnapshot).filter(PositionSnapshot.run_id == r.id).delete()
            s.query(AgentDecision).filter(AgentDecision.run_id == r.id).delete()
            s.delete(r)
    console.print("[yellow]已清掉所有 demo_seed run 的旧数据[/yellow]")


def _make_oms(broker: PaperBroker, session, risk: RiskEngine, account: Account) -> OMS:
    def risk_adapter(ctx: dict) -> tuple[bool, str | None]:
        d = risk.check_pre_trade(
            symbol=ctx["symbol"],
            side=ctx["side"],
            quantity=int(ctx["quantity"]),
            price=float(ctx.get("limit_price") or 0)
                  or float(broker.last_prices.get(ctx["symbol"], Decimal("1"))),
            account=account,
        )
        return d.passed, d.detail

    return OMS(
        broker=broker,
        strategy_id="demo_seed",
        run_id=session.run_id,
        risk_check=risk_adapter,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="先清掉旧 demo 数据")
    parser.add_argument("--initial-cash", type=float, default=500_000.0)
    args = parser.parse_args()

    init_db()
    if args.reset:
        _maybe_reset()

    state = StateManager(strategy_id="demo_seed", mode="paper")
    session = state.load_or_create(initial_cash=args.initial_cash)
    console.print(f"[green]✓[/green] 启动 demo run | id={session.run_id} uid={session.run_uid}")

    broker = PaperBroker(account=session.account)
    risk = RiskEngine(policies=RiskPolicies())
    oms = _make_oms(broker, session, risk, session.account)

    # 推一遍价格让 broker 知道每个 symbol 的市场价
    broker.update_market({sym: float(price) for sym, _, _, price in DEMO_TRADES})

    n_ok, n_fail = 0, 0
    for sym, side, qty, price in DEMO_TRADES:
        broker.update_market({sym: float(price)})
        try:
            order = oms.place_order(
                symbol=sym, side=side, quantity=qty, order_type=OrderType.MARKET,
                salt=f"demo-{sym}-{side.value}-{qty}",
            )
            console.print(f"  [green]✓[/green] {side.value:<4} {sym} x{qty} @ {price:.2f} → {order.status.value}")
            n_ok += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] {side.value:<4} {sym} x{qty} @ {price:.2f} → {e}")
            n_fail += 1

    state.snapshot(session)
    a = session.account
    console.print()
    console.print(f"[bold]最终账户[/bold]: 现金 {float(a.cash):,.2f} | "
                  f"市值 {float(a.market_value):,.2f} | 总资产 {float(a.total_assets):,.2f}")
    console.print(f"成功 {n_ok} 笔 / 失败 {n_fail} 笔")
    console.print("\n现在打开 UI：[cyan]qtrade ui[/cyan]")


if __name__ == "__main__":
    main()
