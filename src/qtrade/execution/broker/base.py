"""Broker 抽象。

每个真实券商接口（paper / 掘金 / Ptrade / IBKR …）都实现这个最小接口。
``OMS`` 持有 ``Broker`` 的依赖注入，业务代码不直接调券商 API。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from qtrade.persistence.models import OrderSide, OrderStatus, OrderType


@dataclass
class BrokerOrderResult:
    """券商 ``submit`` 的同步返回。"""

    broker_order_id: str
    status: OrderStatus
    filled_quantity: int = 0
    avg_fill_price: Decimal | None = None
    commission: Decimal = Decimal("0")
    stamp_tax: Decimal = Decimal("0")
    reject_reason: str | None = None


class Broker(ABC):
    """所有券商接入要实现的接口。

    生产实现需要异步 + 回调，但日级策略下同步阻塞接口足够清晰。
    """

    name: str = "base"

    @abstractmethod
    def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        limit_price: Decimal | None = None,
    ) -> BrokerOrderResult: ...

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool: ...

    @abstractmethod
    def get_account_balance(self) -> dict: ...

    @abstractmethod
    def get_positions(self) -> dict[str, dict]: ...
