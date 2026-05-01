"""双均线策略：fast 上穿 slow 买入，下穿卖出。

这个策略**仅作基线**，目的是验证整条链路（数据→策略→风控→OMS→Broker）通畅，
不要指望它真赚钱。
"""
from __future__ import annotations

from decimal import Decimal

from qtrade.strategies.base import Signal, Strategy, StrategyContext


class MACrossStrategy(Strategy):
    id = "ma_cross_v1"

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        super().__init__(params=params, universe=universe)
        self.fast = int(self.params.get("fast_window", 5))
        self.slow = int(self.params.get("slow_window", 20))
        self.position_pct = float(self.params.get("position_pct", 0.3))
        if self.fast >= self.slow:
            raise ValueError("fast_window must be < slow_window")

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.history
        if len(df) < self.slow + 1:
            return []

        close = df["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        # 取最后两根 K 线判断金/死叉
        prev_diff = fast_ma.iloc[-2] - slow_ma.iloc[-2]
        curr_diff = fast_ma.iloc[-1] - slow_ma.iloc[-1]

        pos = ctx.account.positions.get(ctx.symbol)
        signals: list[Signal] = []

        if prev_diff <= 0 and curr_diff > 0 and pos.quantity == 0:
            # 金叉买入：按目标仓位比例分配
            target_value = ctx.account.total_assets * Decimal(str(self.position_pct))
            price = Decimal(str(df["close"].iloc[-1]))
            shares = int(target_value / price // 100) * 100   # 整百股
            if shares >= 100:
                signals.append(
                    Signal(
                        symbol=ctx.symbol,
                        action="BUY",
                        quantity=shares,
                        confidence=0.6,
                        reason=f"金叉：MA{self.fast} 上穿 MA{self.slow} | 目标仓位 {self.position_pct:.0%}",
                    )
                )
        elif prev_diff >= 0 and curr_diff < 0 and pos.available > 0:
            signals.append(
                Signal(
                    symbol=ctx.symbol,
                    action="SELL",
                    quantity=pos.available,
                    confidence=0.6,
                    reason=f"死叉：MA{self.fast} 下穿 MA{self.slow} | 平仓",
                )
            )
        return signals
