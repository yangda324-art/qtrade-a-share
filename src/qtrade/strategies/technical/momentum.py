"""动量策略：N 日涨幅超阈值买入，跌破 MA 卖出。

参数：
- lookback=60（动量窗口）
- entry_return=0.05（60 日涨幅 ≥ 5% 触发买入）
- exit_ma=20（跌破 MA20 平仓）

逻辑：
- 60 日累计收益 ≥ 5% 且空仓 → 买入
- 持仓时收盘 < MA20 → 卖出

经典动量因子的简化实现。区别于 MA cross：动量看绝对涨幅，cross 看趋势方向。
"""
from __future__ import annotations

from decimal import Decimal

from qtrade.core.logging import get_logger
from qtrade.strategies.base import Signal, Strategy, StrategyContext

log = get_logger(__name__)


class MomentumStrategy(Strategy):
    id = "momentum_v1"

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        super().__init__(params=params, universe=universe)
        self.lookback = int(self.params.get("lookback", 60))
        self.entry_return = float(self.params.get("entry_return", 0.05))
        self.exit_ma = int(self.params.get("exit_ma", 20))
        self.position_pct = float(self.params.get("position_pct", 0.3))

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.history
        if len(df) < max(self.lookback, self.exit_ma) + 1:
            return []

        close = df["close"]
        n_return = float(close.iloc[-1] / close.iloc[-1 - self.lookback] - 1)
        ma_exit = close.rolling(self.exit_ma).mean().iloc[-1]
        price = float(close.iloc[-1])
        pos = ctx.account.positions.get(ctx.symbol)

        if pos.quantity == 0 and n_return >= self.entry_return:
            target_value = ctx.account.total_assets * Decimal(str(self.position_pct))
            shares = int(target_value / Decimal(str(price)) // 100) * 100
            if shares >= 100:
                return [Signal(
                    symbol=ctx.symbol, action="BUY", quantity=shares,
                    confidence=0.6,
                    reason=f"动量：{self.lookback}日涨幅 {n_return:+.2%} ≥ {self.entry_return:.0%}",
                )]
        elif pos.available > 0 and price < ma_exit:
            return [Signal(
                symbol=ctx.symbol, action="SELL", quantity=pos.available,
                confidence=0.6,
                reason=f"动量退出：价 {price:.2f} < MA{self.exit_ma} {ma_exit:.2f}",
            )]
        return []
