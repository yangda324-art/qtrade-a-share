"""账户：现金 + 持仓 + 总资产。

回测和模拟盘共用同一个 ``Account``，区别只在 ``Broker`` 实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from qtrade.execution.position import PositionBook


@dataclass
class Account:
    initial_cash: Decimal = Decimal("100000")
    cash: Decimal = field(init=False)
    frozen_cash: Decimal = Decimal("0")           # 已挂单冻结
    positions: PositionBook = field(default_factory=PositionBook)
    realized_pnl: Decimal = Decimal("0")          # 累计已实现
    equity_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = Decimal(str(self.initial_cash))

    # ---- 资金动作 ----

    def freeze(self, amount: Decimal) -> None:
        if amount > self.cash:
            raise ValueError(f"insufficient cash to freeze: {amount} > {self.cash}")
        self.cash -= amount
        self.frozen_cash += amount

    def unfreeze(self, amount: Decimal) -> None:
        amount = min(amount, self.frozen_cash)
        self.frozen_cash -= amount
        self.cash += amount

    def settle_buy(self, *, frozen: Decimal, actual: Decimal) -> None:
        """买入成交：冻结金额 → 实际扣除，差额返还现金。"""
        self.frozen_cash -= frozen
        # 实际花费 actual；多冻结的部分回到 cash
        self.cash += frozen - actual

    def settle_sell(self, proceeds: Decimal, realized_delta: Decimal) -> None:
        """卖出成交：增加现金，并累计已实现盈亏。"""
        self.cash += proceeds
        self.realized_pnl += realized_delta

    # ---- 派生指标 ----

    @property
    def market_value(self) -> Decimal:
        return self.positions.total_market_value()

    @property
    def total_assets(self) -> Decimal:
        return self.cash + self.frozen_cash + self.market_value

    @property
    def unrealized_pnl(self) -> Decimal:
        return sum((p.unrealized_pnl for p in self.positions), Decimal("0"))

    def snapshot(self, ts: datetime) -> None:
        """记录权益曲线点（每日收盘或每个 tick）。"""
        self.equity_curve.append((ts, self.total_assets))

    def to_dict(self) -> dict:
        return {
            "cash": float(self.cash),
            "frozen_cash": float(self.frozen_cash),
            "market_value": float(self.market_value),
            "total_assets": float(self.total_assets),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "positions": self.positions.to_dict(),
        }
