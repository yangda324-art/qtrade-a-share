"""One-command readiness checks for the basic paper-trading loop."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from qtrade.core.config import get_settings
from qtrade.core.time import utcnow
from qtrade.data.health import run_data_health_check
from qtrade.data.network_check import run_network_check
from qtrade.data.schema import Period, normalize_symbol

DoctorStatus = Literal["PASS", "WARN", "FAIL"]

DEFAULT_DOCTOR_PERIODS: tuple[Period, ...] = ("daily", "5min", "15min")


@dataclass(frozen=True)
class DoctorItem:
    """One doctor check result."""

    name: str
    status: DoctorStatus
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DoctorReport:
    """Aggregate doctor result."""

    symbol: str
    periods: list[str]
    items: list[DoctorItem]
    generated_at: datetime = field(default_factory=utcnow)

    @property
    def failures(self) -> list[DoctorItem]:
        return [item for item in self.items if item.status == "FAIL"]

    @property
    def warnings(self) -> list[DoctorItem]:
        return [item for item in self.items if item.status == "WARN"]

    @property
    def passed(self) -> list[DoctorItem]:
        return [item for item in self.items if item.status == "PASS"]

    @property
    def overall_status(self) -> DoctorStatus:
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "symbol": self.symbol,
            "periods": list(self.periods),
            "overall_status": self.overall_status,
            "summary": {
                "pass": len(self.passed),
                "warn": len(self.warnings),
                "fail": len(self.failures),
            },
            "items": [
                {
                    "name": item.name,
                    "status": item.status,
                    "detail": item.detail,
                    "data": _jsonable(item.data),
                }
                for item in self.items
            ],
        }


def run_doctor(
    *,
    symbol: str = "000001",
    periods: list[str] | tuple[str, ...] | None = None,
    timeout_s: float = 5.0,
    start: date | str | None = None,
    end: date | str | None = None,
    run_smoke: bool = True,
) -> DoctorReport:
    """Run the basic readiness checks.

    This does not call LLM APIs and does not place real orders.  The paper smoke
    test runs inside a temporary isolated SQLite database.
    """
    normalized = normalize_symbol(symbol)
    selected_periods = list(periods or DEFAULT_DOCTOR_PERIODS)
    items: list[DoctorItem] = []

    items.append(DoctorItem("LLM/DeepSeek", "PASS", "doctor 不调用 LLM API，不产生模型费用"))
    items.extend(_network_items(normalized, selected_periods, timeout_s=timeout_s))
    items.extend(_data_items(normalized, selected_periods, start=start, end=end))
    items.append(_database_item())
    items.append(_ui_import_item())
    items.append(_paper_state_item())
    if run_smoke:
        items.extend(_paper_smoke_items(normalized))
    else:
        items.append(DoctorItem("PaperBroker smoke", "WARN", "已跳过；去掉 --skip-smoke 可检查临时模拟成交链路"))

    return DoctorReport(symbol=normalized, periods=selected_periods, items=items)


def write_doctor_json(report: DoctorReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _network_items(symbol: str, periods: list[str], *, timeout_s: float) -> list[DoctorItem]:
    try:
        report = run_network_check(symbol=symbol, periods=periods, timeout_s=timeout_s)
    except Exception as exc:
        return [DoctorItem("网络诊断", "FAIL", f"网络诊断异常：{_short(exc)}")]
    return [
        DoctorItem(
            f"网络 · {item.name}",
            _status(item.status),
            f"{item.category} | {item.detail}",
            data={"category": item.category, "latency_ms": item.latency_ms, **item.data},
        )
        for item in report.items
    ]


def _data_items(
    symbol: str,
    periods: list[str],
    *,
    start: date | str | None,
    end: date | str | None,
) -> list[DoctorItem]:
    try:
        report = run_data_health_check(
            symbols=[symbol],
            periods=periods,
            start=start,
            end=end,
            include_quotes=True,
            force_refresh=False,
        )
    except Exception as exc:
        return [DoctorItem("行情/K线", "FAIL", f"行情检查异常：{_short(exc)}")]
    return [
        DoctorItem(
            f"行情/K线 · {item.name}",
            _status(item.status),
            item.detail,
            data=item.data,
        )
        for item in report.items
    ]


def _database_item() -> DoctorItem:
    from qtrade.persistence.models import Base

    settings = get_settings()
    db_path = settings.paths.db_path
    if not db_path.exists():
        return DoctorItem(
            "SQLite 数据库",
            "FAIL",
            f"数据库不存在：{db_path}；先运行 qtrade init 或打开 UI 初始化",
            data={"db_path": str(db_path)},
        )
    expected_tables = set(Base.metadata.tables)
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    except Exception as exc:
        return DoctorItem("SQLite 数据库", "FAIL", f"只读打开失败：{_short(exc)}", data={"db_path": str(db_path)})
    existing = {str(row[0]) for row in rows}
    missing = sorted(expected_tables - existing)
    if missing:
        return DoctorItem(
            "SQLite 数据库",
            "FAIL",
            f"缺少表：{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}",
            data={"db_path": str(db_path), "missing_tables": missing},
        )
    return DoctorItem(
        "SQLite 数据库",
        "PASS",
        f"schema 可读，表数 {len(existing)}",
        data={"db_path": str(db_path), "table_count": len(existing)},
    )


def _ui_import_item() -> DoctorItem:
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401

        from qtrade.ui.windows.kline import KlineWidget  # noqa: F401
        from qtrade.ui.windows.market_watch import MarketWatchWidget  # noqa: F401
    except Exception as exc:
        return DoctorItem("UI 组件", "FAIL", f"PySide6/UI 组件不可导入：{_short(exc)}")
    return DoctorItem("UI 组件", "PASS", "PySide6 与核心 UI 组件可导入")


def _paper_state_item() -> DoctorItem:
    settings = get_settings()
    if not settings.paths.db_path.exists():
        return DoctorItem("模拟盘状态", "WARN", "数据库不存在，无法读取已有模拟盘状态")
    try:
        from qtrade.ui.data_access import paper_run_health

        health = paper_run_health()
    except Exception as exc:
        return DoctorItem("模拟盘状态", "FAIL", f"读取模拟盘状态失败：{_short(exc)}")
    if not health:
        return DoctorItem("模拟盘状态", "WARN", "没有可读的模拟盘状态")
    severity = str(health.get("severity") or "muted")
    text = str(health.get("status_text") or health.get("status") or "-")
    if severity == "ok":
        status: DoctorStatus = "PASS"
    elif severity == "error":
        status = "FAIL"
    else:
        status = "WARN"
    run = health.get("run") or {}
    run_desc = f"run_id={run.get('run_id')}" if run.get("run_id") else "无 active run"
    return DoctorItem("模拟盘状态", status, f"{run_desc} | {text}", data={"health": health})


def _paper_smoke_items(symbol: str) -> list[DoctorItem]:
    try:
        with _isolated_runtime_paths():
            from qtrade.runtime.paper_smoke import run_paper_smoke_check

            report = run_paper_smoke_check(symbol=symbol, isolated=True)
    except Exception as exc:
        return [DoctorItem("PaperBroker smoke", "FAIL", f"临时模拟盘 smoke 异常：{_short(exc)}")]
    return [
        DoctorItem(
            f"PaperBroker smoke · {item.name}",
            _status(item.status),
            item.detail,
            data=item.data,
        )
        for item in report.items
    ]


@contextmanager
def _isolated_runtime_paths():
    keys = {
        "QTRADE_CONFIG_DIR",
        "QTRADE_PATHS__DATA_DIR",
        "QTRADE_PATHS__LOG_DIR",
        "QTRADE_PATHS__CACHE_DIR",
        "QTRADE_PATHS__DB_PATH",
    }
    old_env = {key: os.environ.get(key) for key in keys}
    with tempfile.TemporaryDirectory(prefix="qtrade-doctor-") as tmp:
        base = Path(tmp)
        data_dir = base / "data"
        log_dir = base / "logs"
        cache_dir = data_dir / "cache"
        cfg_dir = base / "cfg"
        for path in (data_dir, log_dir, cache_dir, cfg_dir):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["QTRADE_CONFIG_DIR"] = str(cfg_dir)
        os.environ["QTRADE_PATHS__DATA_DIR"] = str(data_dir)
        os.environ["QTRADE_PATHS__LOG_DIR"] = str(log_dir)
        os.environ["QTRADE_PATHS__CACHE_DIR"] = str(cache_dir)
        os.environ["QTRADE_PATHS__DB_PATH"] = str(data_dir / "qtrade.sqlite")
        _reconfigure_runtime_paths()
        try:
            yield base
        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
            _reconfigure_runtime_paths()


def _reconfigure_runtime_paths() -> None:
    from qtrade.core import config as cfg

    cfg.get_settings.cache_clear()
    try:
        from qtrade.data.loader import get_loader
        from qtrade.persistence import db as dbmod
    except Exception:
        return
    get_loader.cache_clear()
    dbmod.engine.dispose()
    dbmod.engine = dbmod._build_engine()
    dbmod.SessionLocal.configure(bind=dbmod.engine)


def _status(status: str) -> DoctorStatus:
    if status == "FAIL":
        return "FAIL"
    if status == "WARN":
        return "WARN"
    return "PASS"


def _short(value: Any, *, max_len: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return f"{text[:max_len - 1]}..."


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value
