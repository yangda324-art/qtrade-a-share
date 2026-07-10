from __future__ import annotations

from qtrade.core import config as cfg
from qtrade.core.time import utcnow
from qtrade.persistence import init_db, session_scope
from qtrade.persistence.models import TickLog
from qtrade.runtime.llm_preflight import run_llm_paper_preflight


def test_llm_preflight_fails_without_deepseek_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("QTRADE_DEEPSEEK_API_KEY", "")
    cfg.get_settings.cache_clear()
    init_db()

    report = run_llm_paper_preflight(symbols=["000001"])

    key_item = next(item for item in report.items if item.name == "LLM Key")
    assert key_item.status == "FAIL"
    assert report.has_failures


def test_llm_preflight_passes_with_key_and_recent_tradeable_tick(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg.get_settings.cache_clear()
    init_db()
    now = utcnow()
    with session_scope() as s:
        s.add(
            TickLog(
                run_id=None,
                strategy_id="test",
                ts=now,
                quotes_count=1,
                tradeable_quotes=1,
                fallback_quotes=0,
                signals_generated=0,
                strategy_skipped=0,
                orders_placed=0,
                risk_rejected=0,
                symbols_json=["000001.SZ"],
                per_symbol_json=[{"symbol": "000001.SZ", "source": "fake"}],
                errors_json=[],
            )
        )

    report = run_llm_paper_preflight(symbols=["000001"])

    assert report.overall_status == "PASS"
    assert next(item for item in report.items if item.name == "最近行情").status == "PASS"


def test_llm_preflight_warns_when_position_exceeds_cap(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg.get_settings.cache_clear()
    init_db()

    report = run_llm_paper_preflight(
        symbols=["000001"],
        position_pct=0.5,
        max_position_pct=0.3,
    )

    item = next(item for item in report.items if item.name == "目标仓位")
    assert item.status == "WARN"
    assert "裁剪" in item.detail
