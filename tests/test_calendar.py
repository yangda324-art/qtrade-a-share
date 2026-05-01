"""交易日历测试。"""
from __future__ import annotations

from datetime import date, datetime

from qtrade.core.calendar import (
    current_trading_session,
    is_in_trading_session,
    is_trading_day,
    next_trading_day,
    prev_trading_day,
    trading_days_between,
)


def test_weekend_not_trading():
    # 2024-01-06 周六、2024-01-07 周日
    assert is_trading_day(date(2024, 1, 6)) is False
    assert is_trading_day(date(2024, 1, 7)) is False


def test_weekday_typical_trading():
    # 2024-01-08 周一
    assert is_trading_day(date(2024, 1, 8)) is True


def test_chinese_holiday_not_trading():
    # 2024 国庆 10-1
    assert is_trading_day(date(2024, 10, 1)) is False


def test_next_prev_trading_day():
    # 2024-01-05 周五 → next=2024-01-08 周一
    assert next_trading_day(date(2024, 1, 5)) == date(2024, 1, 8)
    assert prev_trading_day(date(2024, 1, 8)) == date(2024, 1, 5)


def test_trading_days_between():
    days = trading_days_between(date(2024, 1, 1), date(2024, 1, 12))
    # 2024-01-01 元旦休、1/2~1/5 交易日、1/8~1/12 交易日
    assert date(2024, 1, 1) not in days
    assert date(2024, 1, 8) in days
    assert all(d.weekday() < 5 for d in days)


def test_session_detection():
    morning = datetime(2024, 1, 8, 10, 0)
    lunch = datetime(2024, 1, 8, 12, 0)
    weekend = datetime(2024, 1, 6, 10, 0)

    assert current_trading_session(morning) == "morning"
    assert current_trading_session(lunch) is None
    assert is_in_trading_session(morning) is True
    assert is_in_trading_session(weekend) is False
