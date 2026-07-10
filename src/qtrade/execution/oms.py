"""订单管理系统（OMS）。

职责：
1. 接收策略的"目标动作"（buy/sell + 数量），生成 ``client_order_id``（幂等键）。
2. 走风控检查 —— 任何风控否决都直接 ``RiskRejected``，不下单。
3. 落库一条 PENDING 订单，再调 ``Broker.submit_order`` 拿券商响应，更新订单状态。
4. 把成交回报写到 ``TradeFill`` 表。

幂等保证：``client_order_id`` 是 SQLite 唯一索引，重复信号会插入失败而不是双下单。
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from qtrade.core.exceptions import BrokerError, RiskRejected
from qtrade.core.logging import get_logger
from qtrade.core.time import utcnow
from qtrade.execution.broker.base import Broker
from qtrade.persistence import session_scope
from qtrade.persistence.models import (
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeFill,
)

log = get_logger(__name__)

# 风控钩子：调用方传入；返回 (passed: bool, reject_reason: str | None)
RiskCheck = Callable[[dict], tuple[bool, str | None]]


def _make_client_order_id(strategy_id: str, symbol: str, side: str, ts: datetime, salt: str = "") -> str:
    """生成幂等 client_order_id：同一策略/标的/方向/分钟最多一笔。

    salt 让"同一分钟下两笔不同决策"成为可能，但默认要求策略主动传 salt。
    """
    raw = f"{strategy_id}|{symbol}|{side}|{ts.strftime('%Y%m%d%H%M')}|{salt}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"COID-{h}"


class OMS:
    def __init__(
        self,
        broker: Broker,
        *,
        strategy_id: str,
        run_id: int | None = None,
        risk_check: RiskCheck | None = None,
    ) -> None:
        self.broker = broker
        self.strategy_id = strategy_id
        self.run_id = run_id
        self.risk_check = risk_check or (lambda _ctx: (True, None))

    def place_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | None = None,
        decision_id: int | None = None,
        ts: datetime | None = None,
        salt: str = "",
    ) -> OrderRecord:
        """提交一个订单。返回最终 ORM 记录（已 commit）。"""
        ts = ts or utcnow()

        # 1) 风控
        ctx = {
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "order_type": order_type.value,
            "limit_price": float(limit_price) if limit_price else None,
        }
        passed, reason = self.risk_check(ctx)
        if not passed:
            log.warning("risk rejected | {}", reason)
            raise RiskRejected("risk_check", reason or "rejected")

        coid = _make_client_order_id(self.strategy_id, symbol, side.value, ts, salt)

        # 2) 落 PENDING 记录（幂等：同 coid 已存在则直接返回）
        with session_scope() as s:
            existing = s.query(OrderRecord).filter_by(client_order_id=coid).one_or_none()
            if existing:
                log.info("idempotent hit | coid={} | reusing existing order", coid)
                return existing

            order = OrderRecord(
                run_id=self.run_id,
                strategy_id=self.strategy_id,
                client_order_id=coid,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                decision_id=decision_id,
            )
            s.add(order)
            s.flush()
            order_id = order.id

        # 3) 调 broker
        try:
            res = self.broker.submit_order(
                client_order_id=coid,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
            )
        except Exception as e:
            with session_scope() as s:
                obj = s.get(OrderRecord, order_id)
                obj.status = OrderStatus.REJECTED
                obj.reject_reason = f"broker exception: {e}"
            raise BrokerError(str(e)) from e

        # 4) 写回订单状态 + 成交回报
        with session_scope() as s:
            obj = s.get(OrderRecord, order_id)
            obj.broker_order_id = res.broker_order_id
            obj.status = res.status
            obj.filled_quantity = res.filled_quantity
            obj.avg_fill_price = res.avg_fill_price
            obj.commission = res.commission
            obj.stamp_tax = res.stamp_tax
            obj.reject_reason = res.reject_reason
            if res.status == OrderStatus.FILLED and res.filled_quantity > 0:
                s.add(
                    TradeFill(
                        order_id=order_id,
                        quantity=res.filled_quantity,
                        price=res.avg_fill_price or Decimal("0"),
                        commission=res.commission,
                        stamp_tax=res.stamp_tax,
                        filled_at=utcnow(),
                    )
                )
            s.flush()
            return s.get(OrderRecord, order_id)
