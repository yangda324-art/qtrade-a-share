"""A 股交易日历。

依赖 ``chinese-calendar`` 拿到法定节假日，再叠加周末过滤即可。
对外 API：

- ``is_trading_day(date)``    判断某天是否交易日
- ``next_trading_day(date)``  下一交易日（不含当天）
- ``prev_trading_day(date)``  上一交易日（不含当天）
- ``trading_days_between(start, end)``   左闭右闭区间内所有交易日
- ``is_in_trading_session(dt)`` 判断时刻是否在盘中
- ``current_trading_session(dt)`` 返回当前所在交易段名（"morning" / "afternoon" / None）

设计取舍：
- A 股每年节前会发布节假日，``chinese-calendar`` 库会跟进更新。
  如果要跑久远的回测（>2030），可能要自己叠加额外节假日表。
- 不缓存，调用本身已经很快；如果需要批量也可以一次性 ``trading_days_between``。
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import Literal

import chinese_calendar as cn_cal

# A 股标准交易段
_MORNING = (time(9, 30), time(11, 30))
_AFTERNOON = (time(13, 0), time(15, 0))

Session = Literal["morning", "afternoon"]


def _to_date(d: date | datetime) -> date:
    return d.date() if isinstance(d, datetime) else d


def is_trading_day(d: date | datetime) -> bool:
    """是否为 A 股交易日。"""
    day = _to_date(d)
    if day.weekday() >= 5:
        return False
    try:
        # ``is_workday`` 已经把法定节假日过滤掉了
        return cn_cal.is_workday(day) and not cn_cal.is_holiday(day)
    except NotImplementedError:
        # 超出 chinese-calendar 数据范围时降级：仅排除周末
        return True


def next_trading_day(d: date | datetime, *, n: int = 1) -> date:
    """``d`` 之后第 ``n`` 个交易日（不含 ``d``）。"""
    if n <= 0:
        raise ValueError("n must be >= 1")
    cur = _to_date(d)
    count = 0
    while count < n:
        cur += timedelta(days=1)
        if is_trading_day(cur):
            count += 1
    return cur


def prev_trading_day(d: date | datetime, *, n: int = 1) -> date:
    """``d`` 之前第 ``n`` 个交易日（不含 ``d``）。"""
    if n <= 0:
        raise ValueError("n must be >= 1")
    cur = _to_date(d)
    count = 0
    while count < n:
        cur -= timedelta(days=1)
        if is_trading_day(cur):
            count += 1
    return cur


def trading_days_between(start: date | datetime, end: date | datetime) -> list[date]:
    """左闭右闭区间内的所有交易日。"""
    s, e = _to_date(start), _to_date(end)
    if s > e:
        return []
    out: list[date] = []
    cur = s
    while cur <= e:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def current_trading_session(dt: datetime) -> Session | None:
    """返回当前时刻所在交易段名，盘外返回 None。"""
    if not is_trading_day(dt):
        return None
    t = dt.time()
    if _MORNING[0] <= t <= _MORNING[1]:
        return "morning"
    if _AFTERNOON[0] <= t <= _AFTERNOON[1]:
        return "afternoon"
    return None


def is_in_trading_session(dt: datetime) -> bool:
    return current_trading_session(dt) is not None


def iter_trading_days(start: date | datetime, end: date | datetime) -> Iterable[date]:
    """生成器版的 ``trading_days_between``，大区间下省内存。"""
    s, e = _to_date(start), _to_date(end)
    cur = s
    while cur <= e:
        if is_trading_day(cur):
            yield cur
        cur += timedelta(days=1)
