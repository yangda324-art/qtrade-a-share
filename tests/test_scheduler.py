"""SchedulerService 调度参数测试。"""
from __future__ import annotations

import pytest

from qtrade.runtime.scheduler import SchedulerService


class _DummyRunner:
    def pre_market(self):
        return None

    def post_market(self):
        return None

    def tick(self):
        return None


def test_scheduler_prefers_seconds_interval_label():
    scheduler = SchedulerService(runner=_DummyRunner(), interval_minutes=5, interval_seconds=10)

    scheduler._tick_trigger()

    assert scheduler._tick_label() == "*/10s"


def test_scheduler_rejects_invalid_seconds_interval():
    scheduler = SchedulerService(runner=_DummyRunner(), interval_seconds=60)

    with pytest.raises(ValueError, match="interval_seconds"):
        scheduler._tick_trigger()
