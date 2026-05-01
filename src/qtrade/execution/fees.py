"""A 股交易费用模型。

A 股费用结构：
- 佣金：万 2.5 起步（券商可议价），单笔最低 5 元
- 印花税：仅卖出收取，目前 0.05%（2023 年下调后）
- 过户费：沪市股票（60 / 68 开头）双向收 0.001%
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from qtrade.data.schema import normalize_symbol


@dataclass(frozen=True)
class FeeBreakdown:
    commission: Decimal
    stamp_tax: Decimal
    transfer_fee: Decimal

    @property
    def total(self) -> Decimal:
        return self.commission + self.stamp_tax + self.transfer_fee


class FeeModel(ABC):
    @abstractmethod
    def calc(self, *, symbol: str, side: str, quantity: int, price: float) -> FeeBreakdown: ...


@dataclass
class AStockFeeModel(FeeModel):
    commission_rate: float = 0.00025
    commission_min: float = 5.0
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001

    def calc(self, *, symbol: str, side: str, quantity: int, price: float) -> FeeBreakdown:
        notional = Decimal(str(price)) * Decimal(quantity)
        commission = max(notional * Decimal(str(self.commission_rate)), Decimal(str(self.commission_min)))
        stamp_tax = (
            notional * Decimal(str(self.stamp_tax_rate)) if side.upper() == "SELL" else Decimal("0")
        )

        # 沪市过户费（60 / 68 / 9 开头），深市为 0
        sym = normalize_symbol(symbol)
        is_sh = sym.endswith(".SH")
        transfer_fee = (
            notional * Decimal(str(self.transfer_fee_rate)) if is_sh else Decimal("0")
        )

        return FeeBreakdown(
            commission=commission.quantize(Decimal("0.0001")),
            stamp_tax=stamp_tax.quantize(Decimal("0.0001")),
            transfer_fee=transfer_fee.quantize(Decimal("0.0001")),
        )
