"""策略抽象基类。

约定：
- 策略**只产出信号**，不直接下单。下单交给 ``OMS``，让风控/审计统一介入。
- ``on_bar`` 在每根 K 线推进时调用；只能看 ``ctx.history``（历史数据），
  不能看未来——这是回测正确性的根本。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from qtrade.execution.account import Account


@dataclass
class Signal:
    """策略输出的目标动作。"""

    symbol: str
    action: str                      # "BUY" / "SELL" / "HOLD"
    quantity: int = 0                # 0 表示由组合管理层决定
    target_pct: float | None = None  # 目标仓位占比（与 quantity 二选一）
    confidence: float = 1.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyContext:
    """策略可读上下文。"""

    ts: datetime
    symbol: str
    history: pd.DataFrame             # ≤ ts 的所有 OHLCV，索引为 date
    account: Account
    extras: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    id: str = "base"
    universe: list[str] = []

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        self.params = params or {}
        if universe is not None:
            self.universe = list(universe)

    def on_start(self) -> None:
        """运行开始时回调。可用于初始化指标缓存等。"""

    def on_finish(self) -> None:
        """运行结束时回调。可用于清理或落库。"""

    @abstractmethod
    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        """每个标的、每根 K 线触发一次，返回信号列表。"""
