"""数据源适配器抽象。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from qtrade.core.time import utcnow
from qtrade.data.schema import AdjustMode, NewsItem

CACHE_QUOTE_SOURCE = "cache_daily_close"
DEFAULT_REALTIME_MAX_QUOTE_AGE_SECONDS = 15

QUOTE_REJECT_INVALID = "invalid_quote"
QUOTE_REJECT_FALLBACK = "fallback_quote"
QUOTE_REJECT_STALE = "stale_quote"
QUOTE_REJECT_AMBIGUOUS = "ambiguous_quote"


@dataclass
class Quote:
    """实时快照行情（盯盘用）。"""

    symbol: str
    price: float           # 最新价
    open: float            # 今开
    high: float            # 今高
    low: float             # 今低
    prev_close: float      # 昨收
    volume: float          # 累计成交量（股）
    amount: float          # 累计成交金额（元）
    ts: datetime           # 数据时间戳
    source: str = "unknown"  # 数据源；缓存兜底行情不能用于自动撮合限价单
    fetched_at: datetime | None = None
    stale_reason: str | None = None

    def __post_init__(self) -> None:
        if self.fetched_at is None:
            self.fetched_at = self.ts

    @property
    def is_cache_fallback(self) -> bool:
        return self.source == CACHE_QUOTE_SOURCE

    @property
    def is_tradeable(self) -> bool:
        """基础可交易标记：价格有效且不是缓存兜底。

        注意：实时模拟盘/实盘还必须检查行情年龄，使用 ``is_realtime_tradeable``。
        """
        return self.price > 0 and not self.is_cache_fallback

    def realtime_reject_reason(
        self,
        *,
        max_age_seconds: int = DEFAULT_REALTIME_MAX_QUOTE_AGE_SECONDS,
        now: datetime | None = None,
    ) -> str | None:
        """实时 paper/live 下单前的行情质量拒绝原因。

        历史回测可以使用缓存历史 K 线；这个检查只用于实时订单路径。
        """
        if self.price <= 0:
            return QUOTE_REJECT_INVALID
        if self.is_cache_fallback:
            return QUOTE_REJECT_FALLBACK
        if self.stale_reason:
            return QUOTE_REJECT_AMBIGUOUS
        if self.age_seconds(now) > max_age_seconds:
            return QUOTE_REJECT_STALE
        return None

    def is_realtime_tradeable(
        self,
        *,
        max_age_seconds: int = DEFAULT_REALTIME_MAX_QUOTE_AGE_SECONDS,
        now: datetime | None = None,
    ) -> bool:
        return self.realtime_reject_reason(
            max_age_seconds=max_age_seconds,
            now=now,
        ) is None

    def age_seconds(self, now: datetime | None = None) -> float:
        """返回本地获取行情后的秒数；主要用于 UI/CLI 显示新鲜度。"""
        if self.fetched_at is None:
            return 0.0
        ref = now or utcnow()
        return max((ref - self.fetched_at).total_seconds(), 0.0)


class DataAdapter(ABC):
    """所有数据源必须实现的最小接口。"""

    name: str = "base"

    @abstractmethod
    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: AdjustMode = "qfq",
    ) -> pd.DataFrame:
        """日线 OHLCV，返回标准 schema DataFrame。"""

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: AdjustMode = "qfq",
        period: str = "daily",
    ) -> pd.DataFrame:
        """通用周期 K 线。默认实现只支持 daily（fallback 到 ``fetch_daily_bars``）。

        想支持分钟级的 adapter 必须 override 这个方法。
        """
        if period == "daily":
            return self.fetch_daily_bars(symbol, start, end, adjust)
        raise NotImplementedError(f"{self.name} 不支持周期 {period}")

    def fetch_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """批量拉一组标的的最新快照。盯盘用，默认空实现，需要的 adapter override。"""
        return {}

    def fetch_news(
        self,
        symbol: str | None = None,
        *,
        limit: int = 20,
    ) -> list[NewsItem]:
        """新闻抽取。默认空实现，需要的 adapter override。"""
        return []
