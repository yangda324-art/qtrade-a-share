"""账户/持仓的持久化与恢复。

核心约束：进程崩了重启后，模拟盘的钱、持仓、累计盈亏必须能完整恢复。
否则就成了"每次重启都重置"的玩具。

实现：
- 每个长跑会话对应一条 ``StrategyRun`` 记录
- ``LiveRunner`` 每个 tick 完成后，写一条 ``AccountSnapshot`` 和每个标的的 ``PositionSnapshot``
- 启动时，``StateManager.load_or_create()`` 找最近的活跃 run：
    - 找到 → 用最近一条 snapshot 重建 ``Account`` 和 ``PositionBook``，复用 run_id
    - 没找到 → 创建新 run、空账户

注意：
- 这里恢复的是"账户层面"的状态。每笔订单/成交本来就是已落库的，不需要重放
- T+1 锁仓信息是恢复的盲点（snapshot 只存了 quantity / available 数字，不知道 today_buy）
  → 启动时如果是同一交易日，依赖恢复出的 quantity/available 差值近似；跨日则交给 ``settle_t1_all``
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select

from qtrade.core.logging import get_logger
from qtrade.core.time import market_date_from_utc_naive, market_today, utcnow
from qtrade.execution.account import Account
from qtrade.execution.fees import AStockFeeModel
from qtrade.execution.position import Position
from qtrade.persistence import session_scope
from qtrade.persistence.models import (
    AccountSnapshot,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSnapshot,
    StrategyRun,
)

log = get_logger(__name__)


@dataclass
class RunSession:
    """活跃运行会话句柄。"""

    run_id: int
    run_uid: str
    strategy_id: str
    mode: str
    account: Account


class StateManager:
    """状态加载 / 持久化。"""

    def __init__(self, *, strategy_id: str, mode: str = "paper") -> None:
        self.strategy_id = strategy_id
        self.mode = mode

    # ---------------- 加载 / 创建 ----------------

    def load_or_create(self, *, initial_cash: float = 100_000.0, config: dict | None = None) -> RunSession:
        """找最近未结束的同名 run；若无则新建。"""
        with session_scope() as s:
            stmt = (
                select(StrategyRun)
                .where(
                    StrategyRun.strategy_id == self.strategy_id,
                    StrategyRun.mode == self.mode,
                    StrategyRun.ended_at.is_(None),
                )
                .order_by(StrategyRun.started_at.desc())
            )
            existing = s.scalars(stmt).first()
            if existing:
                run_id = existing.id
                run_uid = existing.run_uid
                log.info("resume existing run | id={} uid={}", run_id, run_uid)
                account = self._restore_account(s, run_id, default_cash=initial_cash)
                return RunSession(
                    run_id=run_id, run_uid=run_uid,
                    strategy_id=self.strategy_id, mode=self.mode,
                    account=account,
                )

            # 新建
            run_uid = uuid.uuid4().hex[:16]
            run = StrategyRun(
                run_uid=run_uid,
                strategy_id=self.strategy_id,
                strategy_version="dev",        # TODO: 接 git sha
                mode=self.mode,
                started_at=utcnow(),
                config_json=config or {},
            )
            s.add(run)
            s.flush()
            run_id = run.id
            log.info("create new run | id={} uid={}", run_id, run_uid)

            account = Account(initial_cash=Decimal(str(initial_cash)))
            return RunSession(
                run_id=run_id, run_uid=run_uid,
                strategy_id=self.strategy_id, mode=self.mode,
                account=account,
            )

    def end(self, run: RunSession, note: str | None = None) -> None:
        """标记 run 结束（写 ended_at）。"""
        with session_scope() as s:
            obj = s.get(StrategyRun, run.run_id)
            if obj and obj.ended_at is None:
                obj.ended_at = utcnow()
                if note:
                    obj.notes = (obj.notes or "") + f"\n[{utcnow().isoformat()}] {note}"

    # ---------------- 持久化 snapshot ----------------

    def snapshot(self, run: RunSession) -> None:
        """把当前账户 + 持仓写一条 snapshot。每个 tick 末尾或盘后调一次。"""
        ts = utcnow()
        with session_scope() as s:
            s.add(
                AccountSnapshot(
                    run_id=run.run_id,
                    cash=run.account.cash,
                    frozen_cash=run.account.frozen_cash,
                    market_value=run.account.market_value,
                    total_assets=run.account.total_assets,
                    realized_pnl=run.account.realized_pnl,
                    unrealized_pnl=run.account.unrealized_pnl,
                    ts=ts,
                )
            )
            for pos in run.account.positions:
                if pos.quantity == 0:
                    continue
                s.add(
                    PositionSnapshot(
                        run_id=run.run_id,
                        symbol=pos.symbol,
                        quantity=pos.quantity,
                        available_quantity=pos.available,
                        avg_cost=pos.avg_cost,
                        market_price=pos.market_price,
                        market_value=pos.market_value,
                        unrealized_pnl=pos.unrealized_pnl,
                        ts=ts,
                    )
                )

    # ---------------- 内部：从 snapshot 重建 ----------------

    def _restore_account(self, s, run_id: int, *, default_cash: float) -> Account:
        latest = s.scalars(
            select(AccountSnapshot)
            .where(AccountSnapshot.run_id == run_id)
            .order_by(AccountSnapshot.ts.desc())
        ).first()

        if not latest:
            log.info("no prior snapshot for run_id={}, init cash={}", run_id, default_cash)
            return Account(initial_cash=Decimal(str(default_cash)))

        account = Account(initial_cash=latest.total_assets)  # 占位，之后覆盖
        account.cash = Decimal(str(latest.cash))
        account.frozen_cash = Decimal(str(latest.frozen_cash))
        account.realized_pnl = Decimal(str(latest.realized_pnl))

        snapshot_is_today = market_date_from_utc_naive(latest.ts) == market_today()

        # 重建持仓：拿每个 symbol 的最近 snapshot
        # SQLite 没有 partition by，分两步取
        symbols = [
            row[0]
            for row in s.execute(
                select(PositionSnapshot.symbol)
                .where(PositionSnapshot.run_id == run_id)
                .distinct()
            )
        ]
        for sym in symbols:
            ps = s.scalars(
                select(PositionSnapshot)
                .where(PositionSnapshot.run_id == run_id, PositionSnapshot.symbol == sym)
                .order_by(PositionSnapshot.ts.desc())
            ).first()
            if not ps or ps.quantity == 0:
                continue
            if snapshot_is_today:
                available = ps.available_quantity
                today_buy = max(0, ps.quantity - ps.available_quantity)
            else:
                # 上一个交易日及之前的锁仓，重启后视为已过 T+1。
                available = ps.quantity
                today_buy = 0
            p = Position(
                symbol=ps.symbol,
                quantity=ps.quantity,
                available=available,
                today_buy=today_buy,
                avg_cost=Decimal(str(ps.avg_cost)),
                market_price=Decimal(str(ps.market_price)),
            )
            account.positions.positions[sym] = p

        self._reconcile_open_limit_orders(
            s,
            run_id,
            account,
            expected_total_assets=Decimal(str(latest.total_assets)),
        )

        log.info(
            "restored | run_id={} cash={:.2f} positions={} mv={:.2f}",
            run_id, float(account.cash), len(account.positions.positions),
            float(account.market_value),
        )
        return account

    def _reconcile_open_limit_orders(
        self,
        s,
        run_id: int,
        account: Account,
        *,
        expected_total_assets: Decimal,
    ) -> None:
        """用数据库中的未成交限价单修正账户冻结/可卖状态。

        这处理一个实际故障场景：进程在写入 SUBMITTED LIMIT 后、写 snapshot 前崩溃。
        重启时如果只读旧 snapshot，会丢失冻结资金或可卖锁定。
        """
        orders = s.scalars(
            select(OrderRecord).where(
                OrderRecord.run_id == run_id,
                OrderRecord.order_type == OrderType.LIMIT,
                OrderRecord.status == OrderStatus.SUBMITTED,
            )
        ).all()
        if not orders:
            return

        fee_model = AStockFeeModel()
        expected_frozen = Decimal("0")
        pending_sell: dict[str, int] = {}

        for order in orders:
            if order.limit_price is None:
                continue
            price = Decimal(str(order.limit_price))
            if order.side == OrderSide.BUY:
                fees = fee_model.calc(
                    symbol=order.symbol,
                    side=OrderSide.BUY.value,
                    quantity=order.quantity,
                    price=float(price),
                )
                expected_frozen += price * Decimal(order.quantity) + fees.total
            else:
                pending_sell[order.symbol] = pending_sell.get(order.symbol, 0) + order.quantity

        if expected_frozen > account.frozen_cash:
            account.frozen_cash = expected_frozen
            excess_assets = account.total_assets - expected_total_assets
            if excess_assets > 0:
                account.cash -= min(account.cash, excess_assets)

        for sym, locked_qty in pending_sell.items():
            pos = account.positions.get(sym)
            max_available = max(0, pos.quantity - locked_qty)
            pos.available = min(pos.available, max_available)
