"""时间工具。

Python 3.12 起 ``datetime.utcnow()`` 被弃用（理由：返回的是 naive datetime，
但语义上是 UTC，容易导致时区 bug）。我们这里统一用 ``utcnow()`` 工具函数包一层：

- 业务上仍然用 naive UTC datetime（SQLite 没有时区类型，保持一致更简单）
- 内部实现用 ``datetime.now(timezone.utc).replace(tzinfo=None)``，未来想全切 tz-aware
  也只用改这一处
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("Asia/Shanghai")


def utcnow() -> datetime:
    """返回当前 UTC 时间的 naive datetime（兼容现有 SQLite schema）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_naive_to_market_time(ts: datetime | None) -> datetime | None:
    """把数据库里的 naive UTC 时间转成 A 股市场本地时间。"""
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc).astimezone(MARKET_TZ).replace(tzinfo=None)


def market_today() -> date:
    """返回 A 股市场本地日期。"""
    return datetime.now(MARKET_TZ).date()


def market_date_from_utc_naive(ts: datetime | None) -> date | None:
    """返回 naive UTC 时间对应的 A 股市场日期。"""
    local = utc_naive_to_market_time(ts)
    return local.date() if local else None
