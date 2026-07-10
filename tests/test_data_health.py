"""Market-data health check tests."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from qtrade.core.exceptions import DataError
from qtrade.core.time import utcnow
from qtrade.data.adapters.base import CACHE_QUOTE_SOURCE, Quote
from qtrade.data.health import _coverage_status, run_data_health_check
from qtrade.data.loader import BAR_SOURCE_ATTR, BAR_STATUS_ATTR


class _FakeLoader:
    def __init__(self, *, quote_source: str = "fake_live", bar_start: date | None = None) -> None:
        self.quote_source = quote_source
        self.bar_start = bar_start or date(2026, 1, 1)
        self.calls: list[dict] = []

    def get_quotes(self, symbols):
        now = utcnow()
        return {
            symbol: Quote(
                symbol=symbol,
                price=10.0,
                open=9.8,
                high=10.2,
                low=9.7,
                prev_close=9.9,
                volume=1000,
                amount=10_000,
                ts=now,
                fetched_at=now,
                source=self.quote_source,
            )
            for symbol in symbols
        }

    def get_bars(self, symbol, *, start, end, adjust="qfq", period="daily", force_refresh=False):
        self.calls.append(
            {
                "symbol": symbol,
                "start": start,
                "end": end,
                "adjust": adjust,
                "period": period,
                "force_refresh": force_refresh,
            }
        )
        first = max(pd.to_datetime(self.bar_start), pd.to_datetime(start))
        dates = pd.date_range(first, pd.to_datetime(end), freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.2,
                "volume": 1000.0,
                "amount": 10_000.0,
                "symbol": symbol,
                "adjust": adjust,
            }
        )


def test_data_health_check_passes_with_live_quote_and_bars():
    loader = _FakeLoader()

    report = run_data_health_check(
        ["000001"],
        periods=["daily", "5min"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
        loader=loader,
    )

    assert report.overall_status == "PASS"
    assert report.symbols == ["000001.SZ"]
    assert report.periods == ["daily", "5min"]
    assert len(report.passed) == 3
    assert all(call["force_refresh"] is False for call in loader.calls)


def test_data_health_check_warns_for_fallback_quote_and_short_coverage():
    loader = _FakeLoader(
        quote_source=CACHE_QUOTE_SOURCE,
        bar_start=date(2026, 1, 20),
    )

    report = run_data_health_check(
        ["000001"],
        periods=["daily"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 31),
        loader=loader,
    )

    assert report.overall_status == "WARN"
    assert len(report.warnings) == 2
    assert "不可用于实时模拟" in report.warnings[0].detail
    assert "起点覆盖不足" in report.warnings[1].detail


def test_data_health_check_warns_when_using_cache_after_refresh_error():
    class _CacheFallbackLoader(_FakeLoader):
        def get_bars(self, *args, **kwargs):
            df = super().get_bars(*args, **kwargs)
            df.attrs[BAR_STATUS_ATTR] = "cache_fallback"
            df.attrs[BAR_SOURCE_ATTR] = "local_cache"
            return df

    report = run_data_health_check(
        ["000001"],
        periods=["daily"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
        include_quotes=False,
        loader=_CacheFallbackLoader(),
    )

    assert report.overall_status == "WARN"
    assert "刷新失败，使用本地缓存" in report.warnings[0].detail
    assert report.warnings[0].data["data_status"] == "cache_fallback"


def test_data_health_warns_when_intraday_minute_bars_stop_yesterday():
    status, note = _coverage_status(
        "5min",
        start=date(2026, 6, 17),
        end=date(2026, 6, 24),
        first=datetime(2026, 6, 17, 9, 35),
        last=datetime(2026, 6, 23, 15, 0),
        now=datetime(2026, 6, 24, 13, 0),
    )

    assert status == "WARN"
    assert "交易日分钟K线未包含今日数据" in note


def test_data_health_check_fails_on_missing_bars():
    class _EmptyBarsLoader(_FakeLoader):
        def get_bars(self, *args, **kwargs):
            return pd.DataFrame()

    report = run_data_health_check(
        ["000001"],
        periods=["daily"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
        include_quotes=False,
        loader=_EmptyBarsLoader(),
    )

    assert report.overall_status == "FAIL"
    assert report.failures[0].name == "K 线 000001.SZ 日"


def test_data_health_check_shortens_proxy_errors():
    class _ProxyErrorLoader(_FakeLoader):
        def get_bars(self, *args, **kwargs):
            raise DataError(
                "all adapters failed: HTTPSConnectionPool(host='push2his.eastmoney.com') "
                "(Caused by ProxyError('Unable to connect to proxy', "
                "RemoteDisconnected('Remote end closed connection without response')))"
            )

    report = run_data_health_check(
        ["000001"],
        periods=["5min"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
        include_quotes=False,
        loader=_ProxyErrorLoader(),
    )

    assert report.overall_status == "FAIL"
    assert "外部 K 线接口连接失败" in report.failures[0].detail
    assert "HTTPSConnectionPool" not in report.failures[0].detail
    assert "HTTPSConnectionPool" in report.failures[0].data["error"]


def test_data_health_check_warns_for_stale_quote():
    class _StaleQuoteLoader(_FakeLoader):
        def get_quotes(self, symbols):
            old = utcnow() - timedelta(seconds=120)
            return {
                symbol: Quote(
                    symbol=symbol,
                    price=10.0,
                    open=9.8,
                    high=10.2,
                    low=9.7,
                    prev_close=9.9,
                    volume=1000,
                    amount=10_000,
                    ts=old,
                    fetched_at=old,
                    source="fake_live",
                )
                for symbol in symbols
            }

    report = run_data_health_check(
        ["000001"],
        periods=["daily"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
        loader=_StaleQuoteLoader(),
        max_quote_age_seconds=15,
    )

    assert report.overall_status == "WARN"
    assert report.warnings[0].data["reject_reason"] == "stale_quote"
