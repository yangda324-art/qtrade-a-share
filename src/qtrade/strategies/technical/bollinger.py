"""布林带策略（均值回归）：价格触下轨买入，触上轨卖出。

参数：
- window=20, num_std=2.0
- 上轨 = MA + num_std × σ；下轨 = MA - num_std × σ

逻辑：
- 收盘价 ≤ 下轨 且空仓 → 买入
- 收盘价 ≥ 上轨 且持仓 → 卖出
- 介于两轨之间 → 持有不动

适用场景：震荡市表现好；趋势市容易"逆势抄底被埋"。所以这个策略的回测结果
应该和趋势型（MA cross / 海龟）有显著相关性差异，是组合的好补充。
"""
from __future__ import annotations

from decimal import Decimal

from qtrade.core.logging import get_logger
from qtrade.strategies.base import Signal, Strategy, StrategyContext

log = get_logger(__name__)


class BollingerStrategy(Strategy):
    id = "bollinger_v1"

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        super().__init__(params=params, universe=universe)
        self.window = int(self.params.get("window", 20))
        self.num_std = float(self.params.get("num_std", 2.0))
        self.position_pct = float(self.params.get("position_pct", 0.3))

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.history
        if len(df) < self.window + 1:
            return []

        close = df["close"]
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper = ma.iloc[-1] + self.num_std * std.iloc[-1]
        lower = ma.iloc[-1] - self.num_std * std.iloc[-1]
        price = float(close.iloc[-1])
        pos = ctx.account.positions.get(ctx.symbol)

        if price <= lower and pos.quantity == 0:
            target_value = ctx.account.total_assets * Decimal(str(self.position_pct))
            shares = int(target_value / Decimal(str(price)) // 100) * 100
            if shares >= 100:
                return [Signal(
                    symbol=ctx.symbol, action="BUY", quantity=shares,
                    confidence=0.55,
                    reason=f"布林：价 {price:.2f} ≤ 下轨 {lower:.2f}（MA20={ma.iloc[-1]:.2f}, σ={std.iloc[-1]:.3f}）",
                )]
        elif price >= upper and pos.available > 0:
            return [Signal(
                symbol=ctx.symbol, action="SELL", quantity=pos.available,
                confidence=0.55,
                reason=f"布林：价 {price:.2f} ≥ 上轨 {upper:.2f}",
            )]
        return []
