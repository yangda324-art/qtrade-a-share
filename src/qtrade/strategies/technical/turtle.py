"""海龟交易策略（简化版）：N 日突破入场，N 日新低退出。

参数：
- entry_window=20（20 日新高入场）
- exit_window=10（10 日新低退出）
- atr_window=14（用 ATR 估波动率，决定仓位）
- atr_risk=0.01（单位风险预算 = 账户 1%）

逻辑：
- 当日 high 突破前 20 日 high 且空仓 → 买入
- 当日 low 跌破前 10 日 low 且持仓 → 卖出

仓位管理：
- 单股最大持仓金额 = 账户总资产 × atr_risk / (ATR / price)
- 这是简化版：海龟原版还有金字塔加仓和多市场分配，这里只做基础

参考：Curtis Faith《海龟交易法则》
"""
from __future__ import annotations

from decimal import Decimal

from qtrade.core.logging import get_logger
from qtrade.strategies.base import Signal, Strategy, StrategyContext

log = get_logger(__name__)


class TurtleStrategy(Strategy):
    id = "turtle_v1"

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        super().__init__(params=params, universe=universe)
        self.entry_window = int(self.params.get("entry_window", 20))
        self.exit_window = int(self.params.get("exit_window", 10))
        self.atr_window = int(self.params.get("atr_window", 14))
        self.atr_risk = float(self.params.get("atr_risk", 0.01))
        self.max_position_pct = float(self.params.get("max_position_pct", 0.3))

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.history
        need = max(self.entry_window, self.exit_window, self.atr_window) + 2
        if len(df) < need:
            return []

        # 用前 N 日（不含今日）的 high/low 判突破，避免今日参考自己
        prev_high = float(df["high"].iloc[-self.entry_window - 1:-1].max())
        prev_low = float(df["low"].iloc[-self.exit_window - 1:-1].min())

        today = df.iloc[-1]
        high, low, price = float(today["high"]), float(today["low"]), float(today["close"])

        # 简化 ATR：(high - low) 的 N 日均值
        tr = (df["high"] - df["low"]).rolling(self.atr_window).mean()
        atr = float(tr.iloc[-1])

        pos = ctx.account.positions.get(ctx.symbol)

        if high > prev_high and pos.quantity == 0:
            # 仓位 = 账户资产 × atr_risk / (ATR/price)
            if atr > 0:
                risk_capital = float(ctx.account.total_assets) * self.atr_risk
                shares_by_risk = int((risk_capital / atr) // 100) * 100
            else:
                shares_by_risk = 0
            # 同时受单标的最大仓位约束
            max_value = ctx.account.total_assets * Decimal(str(self.max_position_pct))
            shares_by_cap = int(max_value / Decimal(str(price)) // 100) * 100
            shares = min(shares_by_risk, shares_by_cap) if shares_by_risk > 0 else shares_by_cap
            if shares >= 100:
                return [Signal(
                    symbol=ctx.symbol, action="BUY", quantity=shares,
                    confidence=0.65,
                    reason=f"海龟突破：今日高 {high:.2f} > 前 {self.entry_window} 日高 {prev_high:.2f}（ATR={atr:.3f}）",
                )]
        elif low < prev_low and pos.available > 0:
            return [Signal(
                symbol=ctx.symbol, action="SELL", quantity=pos.available,
                confidence=0.65,
                reason=f"海龟退出：今日低 {low:.2f} < 前 {self.exit_window} 日低 {prev_low:.2f}",
            )]
        return []
