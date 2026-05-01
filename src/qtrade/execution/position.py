"""持仓账本。

A 股的关键约束：
- T+1：今天买入的股数 today_buy 当天不可卖；明天交易日开盘后才进入 ``available``
- 整手交易：买卖必须 100 股的整数倍（科创板 200 起步、零股可卖但不能买）

设计：
- ``Position`` 是单只股票的持仓。
- ``PositionBook`` 是整账户的持仓集合 + 一致性维护。
- ``settle_t1()`` 在每日开盘前调用，把 ``today_buy`` 移到 ``available``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterator


@dataclass
class Position:
    symbol: str
    quantity: int = 0                  # 总持仓股数
    available: int = 0                 # 可卖股数（已过 T+1）
    today_buy: int = 0                 # 今日买入股数（未解锁）
    avg_cost: Decimal = Decimal("0")   # 持仓均价
    market_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        return self.market_price * Decimal(self.quantity)

    @property
    def unrealized_pnl(self) -> Decimal:
        return (self.market_price - self.avg_cost) * Decimal(self.quantity)

    def update_market_price(self, price: float | Decimal) -> None:
        self.market_price = Decimal(str(price))

    def add_buy(self, quantity: int, price: Decimal) -> None:
        if quantity <= 0:
            return
        new_total = self.quantity + quantity
        # 加权平均成本
        if new_total > 0:
            self.avg_cost = (
                (self.avg_cost * Decimal(self.quantity)) + price * Decimal(quantity)
            ) / Decimal(new_total)
        self.quantity = new_total
        self.today_buy += quantity
        # available 不变 —— T+1 还没解锁

    def add_sell(self, quantity: int, price: Decimal) -> None:
        if quantity <= 0:
            return
        if quantity > self.available:
            raise ValueError(f"sell exceeds available: {quantity} > {self.available}")
        # 实现盈亏（基于成本价）
        self.realized_pnl += (price - self.avg_cost) * Decimal(quantity)
        self.quantity -= quantity
        self.available -= quantity
        if self.quantity == 0:
            # 清仓后归零成本，避免 next round 用旧均价
            self.avg_cost = Decimal("0")

    def settle_t1(self) -> None:
        """每日开盘前调用：把昨天买入的股数解锁。"""
        self.available += self.today_buy
        self.today_buy = 0


@dataclass
class PositionBook:
    positions: dict[str, Position] = field(default_factory=dict)

    def get(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def __iter__(self) -> Iterator[Position]:
        return iter(self.positions.values())

    def __contains__(self, sym: str) -> bool:
        return sym in self.positions and self.positions[sym].quantity > 0

    def total_market_value(self) -> Decimal:
        return sum((p.market_value for p in self.positions.values()), Decimal("0"))

    def settle_t1_all(self) -> None:
        for p in self.positions.values():
            p.settle_t1()

    def update_prices(self, prices: dict[str, float]) -> None:
        for sym, price in prices.items():
            if sym in self.positions:
                self.positions[sym].update_market_price(price)

    def to_dict(self) -> dict:
        return {
            sym: {
                "quantity": p.quantity,
                "available": p.available,
                "today_buy": p.today_buy,
                "avg_cost": float(p.avg_cost),
                "market_price": float(p.market_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pnl),
                "realized_pnl": float(p.realized_pnl),
            }
            for sym, p in self.positions.items()
        }
