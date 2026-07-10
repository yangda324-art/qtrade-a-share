from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from qtrade.core.time import utcnow
from qtrade.data.adapters.base import CACHE_QUOTE_SOURCE, Quote
from qtrade.persistence import init_db
from qtrade.runtime.paper_check import (
    PAPER_CHECK_PERIODS,
    report_to_markdown,
    run_paper_check,
    write_report_json,
    write_report_markdown,
)


def _bars(symbol: str = "000001.SZ", rows: int = 5) -> pd.DataFrame:
    dates = pd.date_range(end=date.today(), periods=rows, freq="D")
    prices = np.linspace(10.0, 10.4, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": np.full(rows, 1_000_000.0),
            "amount": prices * 1_000_000.0,
            "symbol": symbol,
            "adjust": "qfq",
        }
    )


class _GoodLoader:
    def get_quotes(self, symbols):
        now = utcnow()
        return {
            sym: Quote(
                symbol=sym,
                price=10.0,
                open=9.9,
                high=10.2,
                low=9.8,
                prev_close=9.95,
                volume=1_000_000.0,
                amount=10_000_000.0,
                ts=now,
                fetched_at=now,
                source="fake_realtime",
            )
            for sym in symbols
        }

    def get_bars(self, symbol, *, start, end, period="daily", adjust="qfq", force_refresh=False):
        return _bars(symbol)


class _FallbackQuoteLoader(_GoodLoader):
    def get_quotes(self, symbols):
        now = utcnow()
        return {
            sym: Quote(
                symbol=sym,
                price=10.0,
                open=10.0,
                high=10.0,
                low=10.0,
                prev_close=10.0,
                volume=1_000_000.0,
                amount=10_000_000.0,
                ts=now - timedelta(minutes=1),
                fetched_at=now,
                source=CACHE_QUOTE_SOURCE,
                stale_reason="test fallback",
            )
            for sym in symbols
        }


def test_paper_check_validates_quotes_and_all_bar_periods():
    init_db()

    report = run_paper_check(["000001"], loader=_GoodLoader())

    assert report.overall_status == "WARN"
    assert not report.has_failures
    names = [item.name for item in report.items]
    assert "实时行情" in names
    assert all(item.status == "PASS" for item in report.items if item.name.startswith("K 线"))
    assert len([item for item in report.items if item.name.startswith("K 线")]) == len(PAPER_CHECK_PERIODS)


def test_paper_check_fails_when_realtime_quote_is_fallback():
    init_db()

    report = run_paper_check(["000001"], loader=_FallbackQuoteLoader(), include_bars=False)

    quote_item = next(item for item in report.items if item.name == "实时行情")
    assert quote_item.status == "FAIL"
    assert "fallback_quote" in quote_item.detail


def test_paper_check_rejects_unknown_period():
    init_db()

    report = run_paper_check(["000001"], loader=_GoodLoader(), periods=["daily", "bad"])

    bad_period = next(item for item in report.items if item.name == "K 线 bad")
    assert bad_period.status == "FAIL"


def test_paper_check_report_exports_json_and_markdown(tmp_path):
    init_db()
    report = run_paper_check(["000001"], loader=_GoodLoader(), include_bars=False)

    json_path = write_report_json(report, tmp_path / "paper-check.json")
    md_path = write_report_markdown(report, tmp_path / "paper-check.md")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["symbols"] == ["000001.SZ"]
    assert payload["overall_status"] == report.overall_status
    assert payload["summary"]["warn"] == len(report.warnings)
    assert payload["items"][0]["name"] == "配置"

    markdown = md_path.read_text(encoding="utf-8")
    assert "# 模拟实盘验收报告" in markdown
    assert "| 状态 | 检查项 | 结果 |" in markdown
    assert report_to_markdown(report) == markdown
