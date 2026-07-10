"""MACD 策略：DIF 上穿 DEA 买入，下穿卖出。

参数（默认）：
- fast=12, slow=26, signal=9
- DIF = EMA(close, fast) - EMA(close, slow)
- DEA = EMA(DIF, signal)
- 金叉 = DIF 上穿 DEA；死叉 = DIF 下穿 DEA
"""
from __future__ import annotations

from decimal import Decimal

from qtrade.core.logging import get_logger
from qtrade.strategies.base import Signal, Strategy, StrategyContext

log = get_logger(__name__)


class MACDStrategy(Strategy):
    id = "macd_v1"

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        super().__init__(params=params, universe=universe)
        self.fast = int(self.params.get("fast", 12))
        self.slow = int(self.params.get("slow", 26))
        self.signal = int(self.params.get("signal", 9))
        self.position_pct = float(self.params.get("position_pct", 0.3))
        if self.fast >= self.slow:
            raise ValueError("MACD fast 必须小于 slow")

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.history
        # 至少要 slow + signal 才能算稳定的 DEA
        if len(df) < self.slow + self.signal + 2:
            return []

        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()

        prev = dif.iloc[-2] - dea.iloc[-2]
        curr = dif.iloc[-1] - dea.iloc[-1]
        pos = ctx.account.positions.get(ctx.symbol)

        if prev <= 0 and curr > 0 and pos.quantity == 0:
            target_value = ctx.account.total_assets * Decimal(str(self.position_pct))
            price = Decimal(str(close.iloc[-1]))
            shares = int(target_value / price // 100) * 100
            if shares >= 100:
                return [Signal(
                    symbol=ctx.symbol, action="BUY", quantity=shares,
                    confidence=0.6,
                    reason=f"MACD 金叉：DIF({curr:.4f}) 上穿 DEA",
                )]
        elif prev >= 0 and curr < 0 and pos.available > 0:
            return [Signal(
                symbol=ctx.symbol, action="SELL", quantity=pos.available,
                confidence=0.6,
                reason=f"MACD 死叉：DIF({curr:.4f}) 下穿 DEA",
            )]
        return []
