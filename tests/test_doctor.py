"""Doctor readiness check tests."""
from __future__ import annotations

from datetime import date

from qtrade.data.health import DataHealthItem, DataHealthReport
from qtrade.data.network_check import NetworkCheckItem, NetworkCheckReport
from qtrade.runtime import doctor as doctor_mod
from qtrade.runtime.doctor import run_doctor


def _fake_network_report() -> NetworkCheckReport:
    return NetworkCheckReport(
        symbol="000001.SZ",
        timeout_s=1.0,
        periods=["daily"],
        items=[NetworkCheckItem("EastMoney K线 daily", "PASS", "OK", "接口可达")],
    )


def _fake_data_report() -> DataHealthReport:
    return DataHealthReport(
        symbols=["000001.SZ"],
        periods=["daily"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
        adjust="qfq",
        include_quotes=True,
        force_refresh=False,
        items=[DataHealthItem("K 线 000001.SZ 日", "PASS", "2 根")],
    )


def test_doctor_reports_missing_db_without_creating_file(monkeypatch):
    from qtrade.core.config import get_settings

    monkeypatch.setattr(doctor_mod, "run_network_check", lambda **kwargs: _fake_network_report())
    monkeypatch.setattr(doctor_mod, "run_data_health_check", lambda **kwargs: _fake_data_report())
    db_path = get_settings().paths.db_path
    assert not db_path.exists()

    report = run_doctor(symbol="000001", periods=["daily"], run_smoke=False)

    db_item = next(item for item in report.items if item.name == "SQLite 数据库")
    assert db_item.status == "FAIL"
    assert "数据库不存在" in db_item.detail
    assert not db_path.exists()
    assert next(item for item in report.items if item.name == "LLM/DeepSeek").status == "PASS"


def test_doctor_database_item_passes_after_init(monkeypatch):
    from qtrade.persistence import init_db

    monkeypatch.setattr(doctor_mod, "run_network_check", lambda **kwargs: _fake_network_report())
    monkeypatch.setattr(doctor_mod, "run_data_health_check", lambda **kwargs: _fake_data_report())
    init_db()

    report = run_doctor(symbol="000001", periods=["daily"], run_smoke=False)

    db_item = next(item for item in report.items if item.name == "SQLite 数据库")
    assert db_item.status == "PASS"
    assert "schema 可读" in db_item.detail
    assert any(item.name == "PaperBroker smoke" and item.status == "WARN" for item in report.items)
