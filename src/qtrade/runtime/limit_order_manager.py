"""持久化限价单管理器。

``PaperBroker`` 里已有内存版 LIMIT 簿子，但模拟实盘需要跨 UI/进程重启恢复，
所以自动策略的限价单必须以数据库订单为准：

- 提交 LIMIT：风控 -> 冻结资金/锁定可卖 -> orders.SUBMITTED
- 每个 tick：用最新行情扫描 SUBMITTED LIMIT，满足条件则成交并写 TradeFill
- 账户状态通过 ``StateManager.snapshot`` 持久化
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select

from qtrade.core.exceptions import QtradeError, RiskRejected
from qtrade.core.time import utcnow
from qtrade.execution.broker.paper import PaperBroker
from qtrade.persistence import session_scope
from qtrade.persistence.models import (
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeFill,
)
from qtrade.runtime.state import RunSession

RiskCheck = Callable[[dict], tuple[bool, str | None]]


@dataclass
class PersistentLimitOrderManager:
    """基于数据库订单的限价单提交和撮合。"""

    broker: PaperBroker
    strategy_id: str
    run_id: int
    risk_check: RiskCheck

    def submit_limit_order(
        self,
        *,
        run: RunSession,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: Decimal,
        decision_id: int | None = None,
        ts=None,
        match_price: Decimal | None = None,
    ) -> OrderRecord:
        if quantity <= 0:
            raise QtradeError("quantity must be positive")
        if limit_price <= 0:
            raise QtradeError("limit_price must be positive")

        ctx = {
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "order_type": OrderType.LIMIT.value,
            "limit_price": float(limit_price),
        }
        passed, reason = self.risk_check(ctx)
        if not passed:
            raise RiskRejected("risk_check", reason or "rejected")

        reserved_cash = self._reserve(run=run, symbol=symbol, side=side, quantity=quantity, limit_price=limit_price)
        try:
            order = self._write_submitted_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                decision_id=decision_id,
                ts=ts,
            )
        except Exception:
            self._release_reserve(
                run=run,
                symbol=symbol,
                side=side,
                quantity=quantity,
                reserved_cash=reserved_cash,
            )
            raise

        if match_price is not None and self._is_marketable(
            side=side,
            limit_price=limit_price,
            price=match_price,
        ):
            matched = self.match_open_orders(run=run, prices={symbol: float(match_price)})
            if matched:
                return matched[-1]
        return order

    def match_open_orders(self, *, run: RunSession, prices: dict[str, float]) -> list[OrderRecord]:
        """用最新行情撮合当前 run 的未成交限价单。"""
        if not prices:
            return []

        rows = self._open_limit_orders(symbols=set(prices))
        matched: list[OrderRecord] = []
        for row in rows:
            mkt_price = Decimal(str(prices.get(row.symbol, 0)))
            if mkt_price <= 0:
                continue
            limit_price = Decimal(str(row.limit_price))
            if not self._is_marketable(side=row.side, limit_price=limit_price, price=mkt_price):
                continue

            if row.side == OrderSide.BUY:
                run.account.unfreeze(
                    self._buy_reserve(
                        symbol=row.symbol,
                        quantity=row.quantity,
                        price=limit_price,
                    )
                )
            else:
                pos = run.account.positions.get(row.symbol)
                pos.available = min(pos.quantity, pos.available + row.quantity)

            result = self.broker.execute_order_at_price(
                broker_order_id=row.broker_order_id,
                client_order_id=row.client_order_id,
                symbol=row.symbol,
                side=row.side,
                quantity=row.quantity,
                price=limit_price,
            )
            matched.append(self._update_order(order_id=row.id, result=result))
        return matched

    def open_symbols(self) -> set[str]:
        """当前 run 仍有未成交限价单的标的集合。"""
        return {order.symbol for order in self._open_limit_orders(symbols=set())}

    def _reserve(
        self,
        *,
        run: RunSession,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: Decimal,
    ) -> Decimal:
        if side == OrderSide.BUY:
            reserved_cash = self._buy_reserve(symbol=symbol, quantity=quantity, price=limit_price)
            run.account.freeze(reserved_cash)
            return reserved_cash

        pos = run.account.positions.get(symbol)
        if pos.available < quantity:
            raise QtradeError(f"insufficient available shares ({pos.available} < {quantity})")
        pos.available -= quantity
        return Decimal("0")

    def _release_reserve(
        self,
        *,
        run: RunSession,
        symbol: str,
        side: OrderSide,
        quantity: int,
        reserved_cash: Decimal,
    ) -> None:
        if side == OrderSide.BUY:
            run.account.unfreeze(reserved_cash)
            return

        pos = run.account.positions.get(symbol)
        pos.available = min(pos.quantity, pos.available + quantity)

    def _write_submitted_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: Decimal,
        decision_id: int | None,
        ts,
    ) -> OrderRecord:
        with session_scope() as s:
            order = OrderRecord(
                run_id=self.run_id,
                strategy_id=self.strategy_id,
                client_order_id=f"COID-LIM-{uuid.uuid4().hex[:14]}",
                broker_order_id=f"PB-{uuid.uuid4().hex[:12]}",
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.SUBMITTED,
                decision_id=decision_id,
                created_at=ts or utcnow(),
            )
            s.add(order)
            s.flush()
            return s.get(OrderRecord, order.id)

    def _open_limit_orders(self, *, symbols: set[str]) -> list[OrderRecord]:
        with session_scope() as s:
            stmt = select(OrderRecord).where(
                OrderRecord.run_id == self.run_id,
                OrderRecord.strategy_id == self.strategy_id,
                OrderRecord.order_type == OrderType.LIMIT,
                OrderRecord.status == OrderStatus.SUBMITTED,
            )
            if symbols:
                stmt = stmt.where(OrderRecord.symbol.in_(symbols))
            rows = s.scalars(stmt).all()
            return list(rows)

    def _update_order(self, *, order_id: int, result) -> OrderRecord:
        with session_scope() as s:
            order = s.get(OrderRecord, order_id)
            order.status = result.status
            order.filled_quantity = result.filled_quantity
            order.avg_fill_price = result.avg_fill_price
            order.commission = result.commission
            order.stamp_tax = result.stamp_tax
            order.reject_reason = result.reject_reason
            if result.status == OrderStatus.FILLED and result.filled_quantity > 0:
                s.add(
                    TradeFill(
                        order_id=order_id,
                        quantity=result.filled_quantity,
                        price=result.avg_fill_price or Decimal("0"),
                        commission=result.commission,
                        stamp_tax=result.stamp_tax,
                        filled_at=utcnow(),
                    )
                )
            s.flush()
            return s.get(OrderRecord, order_id)

    def _buy_reserve(self, *, symbol: str, quantity: int, price: Decimal) -> Decimal:
        fees = self.broker.fee_model.calc(
            symbol=symbol,
            side=OrderSide.BUY.value,
            quantity=quantity,
            price=float(price),
        )
        return price * Decimal(quantity) + fees.total

    @staticmethod
    def _is_marketable(*, side: OrderSide, limit_price: Decimal, price: Decimal) -> bool:
        if side == OrderSide.BUY:
            return price <= limit_price
        return price >= limit_price
