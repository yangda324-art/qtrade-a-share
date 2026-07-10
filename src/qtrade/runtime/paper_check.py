"""Paper trading readiness checks.

This module powers ``qtrade paper-check``.  The default path validates data and
paper-run state without executing a strategy tick.  ``run_once=True`` is
explicitly opt-in because it writes paper-trading audit records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from qtrade.core.config import get_settings
from qtrade.core.time import market_today, utcnow
from qtrade.data.loader import DataLoader, get_loader
from qtrade.data.schema import PERIOD_LABELS, Period, normalize_symbol

PaperCheckStatus = Literal["PASS", "WARN", "FAIL"]

PAPER_CHECK_PERIODS: tuple[Period, ...] = (
    "5min",
    "15min",
    "30min",
    "60min",
    "daily",
    "weekly",
    "monthly",
)

_PERIOD_LOOKBACK_DAYS: dict[str, int] = {
    "5min": 10,
    "15min": 30,
    "30min": 45,
    "60min": 60,
    "daily": 365,
    "weekly": 365 * 3,
    "monthly": 365 * 5,
}


@dataclass(frozen=True)
class PaperCheckItem:
    """One readiness check result."""

    name: str
    status: PaperCheckStatus
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperCheckReport:
    """Aggregate readiness check report."""

    symbols: list[str]
    items: list[PaperCheckItem]
    generated_at: datetime = field(default_factory=utcnow)

    @property
    def failures(self) -> list[PaperCheckItem]:
        return [item for item in self.items if item.status == "FAIL"]

    @property
    def warnings(self) -> list[PaperCheckItem]:
        return [item for item in self.items if item.status == "WARN"]

    @property
    def passed(self) -> list[PaperCheckItem]:
        return [item for item in self.items if item.status == "PASS"]

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    @property
    def overall_status(self) -> PaperCheckStatus:
        if self.has_failures:
            return "FAIL"
        if self.has_warnings:
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "market_day": market_today().isoformat(),
            "symbols": list(self.symbols),
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


def run_paper_check(
    symbols: list[str],
    *,
    periods: list[str] | tuple[str, ...] | None = None,
    loader: DataLoader | None = None,
    include_bars: bool = True,
    run_once: bool = False,
    initial_cash: float = 100_000.0,
) -> PaperCheckReport:
    """Run paper-trading readiness checks.

    By default this does not run strategy ticks or write paper-trading records.
    Market-data checks may refresh local bar cache through the normal loader.
    """
    items: list[PaperCheckItem] = []
    normalized = _normalize_symbols(symbols)
    if not normalized:
        return PaperCheckReport(
            symbols=[],
            items=[PaperCheckItem("标的", "FAIL", "没有有效标的")],
        )

    loader = loader or get_loader()
    settings = get_settings()
    max_age_seconds = settings.trading.realtime.max_quote_age_seconds

    items.append(_settings_item(max_age_seconds=max_age_seconds))
    items.append(_quotes_item(loader, normalized, max_age_seconds=max_age_seconds))

    if include_bars:
        selected_periods = tuple(periods or PAPER_CHECK_PERIODS)
        items.extend(_bar_items(loader, normalized[0], selected_periods))
    else:
        items.append(PaperCheckItem("K 线周期", "WARN", "已跳过；使用 --period 可指定周期，去掉 --skip-bars 可检查"))

    items.append(_paper_health_item())
    items.append(_performance_item())
    items.append(_llm_budget_item())

    if run_once:
        items.append(_run_once_item(loader, normalized, initial_cash=initial_cash))

    return PaperCheckReport(symbols=normalized, items=items)


def write_report_json(report: PaperCheckReport, path: str | Path) -> Path:
    """Write a paper-check report as JSON."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def write_report_markdown(report: PaperCheckReport, path: str | Path) -> Path:
    """Write a paper-check report as Markdown."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_to_markdown(report), encoding="utf-8")
    return out


def report_to_markdown(report: PaperCheckReport) -> str:
    """Render a paper-check report as Markdown."""
    lines = [
        "# 模拟实盘验收报告",
        "",
        f"- 生成时间: {report.generated_at.isoformat()}",
        f"- 市场日期: {market_today().isoformat()}",
        f"- 标的: {', '.join(report.symbols) or '-'}",
        f"- 总状态: {report.overall_status}",
        f"- 汇总: PASS={len(report.passed)} / WARN={len(report.warnings)} / FAIL={len(report.failures)}",
        "",
        "| 状态 | 检查项 | 结果 |",
        "| --- | --- | --- |",
    ]
    for item in report.items:
        lines.append(f"| {item.status} | {_md_cell(item.name)} | {_md_cell(item.detail)} |")
    lines.append("")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _normalize_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    for raw in symbols:
        try:
            sym = normalize_symbol(raw)
        except Exception:
            continue
        if sym not in out:
            out.append(sym)
    return out


def _settings_item(*, max_age_seconds: int) -> PaperCheckItem:
    settings = get_settings()
    return PaperCheckItem(
        "配置",
        "PASS",
        (
            f"db={settings.paths.db_path} | cache={settings.paths.cache_dir} | "
            f"实时行情最大年龄={max_age_seconds}s"
        ),
        data={
            "db_path": str(settings.paths.db_path),
            "cache_dir": str(settings.paths.cache_dir),
            "max_quote_age_seconds": max_age_seconds,
        },
    )


def _quotes_item(
    loader: DataLoader,
    symbols: list[str],
    *,
    max_age_seconds: int,
) -> PaperCheckItem:
    now = utcnow()
    try:
        quotes = loader.get_quotes(symbols)
    except Exception as exc:
        return PaperCheckItem("实时行情", "FAIL", f"行情拉取异常：{exc}")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    rejected: list[str] = []
    tradeable_count = 0

    for sym in symbols:
        quote = quotes.get(sym)
        if quote is None:
            missing.append(sym)
            rows.append({"symbol": sym, "status": "missing"})
            continue
        reject_reason = quote.realtime_reject_reason(max_age_seconds=max_age_seconds, now=now)
        if reject_reason is None:
            tradeable_count += 1
        else:
            rejected.append(f"{sym}:{reject_reason}")
        rows.append(
            {
                "symbol": sym,
                "price": quote.price,
                "source": quote.source,
                "age_seconds": quote.age_seconds(now),
                "reject_reason": reject_reason,
            }
        )

    if tradeable_count == len(symbols):
        return PaperCheckItem(
            "实时行情",
            "PASS",
            f"{tradeable_count}/{len(symbols)} 个标的可用于实时模拟",
            data={"quotes": rows},
        )
    if tradeable_count == 0:
        status: PaperCheckStatus = "FAIL"
    else:
        status = "WARN"
    problems = []
    if missing:
        problems.append(f"缺失 {','.join(missing)}")
    if rejected:
        problems.append(f"不可模拟 {','.join(rejected)}")
    return PaperCheckItem(
        "实时行情",
        status,
        f"{tradeable_count}/{len(symbols)} 个标的可用于实时模拟；" + "；".join(problems),
        data={"quotes": rows},
    )


def _bar_items(
    loader: DataLoader,
    symbol: str,
    periods: tuple[str, ...],
) -> list[PaperCheckItem]:
    today = market_today()
    items: list[PaperCheckItem] = []
    for period in periods:
        if period not in PAPER_CHECK_PERIODS:
            items.append(PaperCheckItem(f"K 线 {period}", "FAIL", f"不支持的周期：{period}"))
            continue
        start = today - timedelta(days=_PERIOD_LOOKBACK_DAYS.get(period, 365))
        label = PERIOD_LABELS.get(period, period)
        try:
            df = loader.get_bars(symbol, start=start, end=today, period=period)
        except Exception as exc:
            items.append(PaperCheckItem(f"K 线 {label}", "FAIL", f"{symbol} 拉取失败：{exc}"))
            continue
        if df.empty:
            items.append(PaperCheckItem(f"K 线 {label}", "FAIL", f"{symbol} 无数据"))
            continue
        first, last = _bar_date_range(df)
        items.append(
            PaperCheckItem(
                f"K 线 {label}",
                "PASS",
                f"{symbol} {len(df)} 根 | {first} -> {last}",
                data={"symbol": symbol, "period": period, "rows": len(df)},
            )
        )
    return items


def _bar_date_range(df: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(df["date"])
    first = dates.min()
    last = dates.max()
    return first.strftime("%Y-%m-%d %H:%M"), last.strftime("%Y-%m-%d %H:%M")


def _paper_health_item() -> PaperCheckItem:
    from qtrade.ui.data_access import paper_run_health

    health = paper_run_health()
    if not health:
        return PaperCheckItem("模拟盘健康", "WARN", "无法读取健康状态")
    severity = str(health.get("severity") or "muted")
    raw_status = str(health.get("status") or "-")
    text = str(health.get("status_text") or raw_status)
    run = health.get("run") or {}
    run_desc = f"run_id={run.get('run_id')}" if run.get("run_id") else "无 run"
    if severity == "ok":
        status: PaperCheckStatus = "PASS"
    elif severity == "error":
        status = "FAIL"
    else:
        status = "WARN"
    detail = (
        f"{text} | {run_desc} | 最新tick行情={health.get('latest_quotes_count', 0)} | "
        f"可模拟={health.get('latest_tradeable_quotes', 0)} | 原因={health.get('latest_reason', '-')}"
    )
    return PaperCheckItem("模拟盘健康", status, detail, data={"health": health})


def _performance_item() -> PaperCheckItem:
    from qtrade.ui.data_access import run_performance_report

    report = run_performance_report()
    if not report:
        return PaperCheckItem("绩效报告", "WARN", "暂无可生成绩效报告的模拟盘快照")
    metrics = report.get("metrics") or {}
    curve = report.get("equity_curve") or []
    detail = (
        f"权益点={len(curve)} | 总收益={float(metrics.get('total_return', 0.0)):.2%} | "
        f"最大回撤={float(metrics.get('max_drawdown', 0.0)):.2%} | "
        f"成交={int(metrics.get('filled_order_count', 0))}"
    )
    status: PaperCheckStatus = "PASS" if curve else "WARN"
    return PaperCheckItem("绩效报告", status, detail, data={"metrics": metrics})


def _llm_budget_item() -> PaperCheckItem:
    from qtrade.ui.data_access import llm_usage_today

    usage = llm_usage_today()
    if not usage:
        return PaperCheckItem("AI 预算", "WARN", "无法读取今日 AI 调用预算")

    problems: list[str] = []
    near: list[str] = []
    _budget_state(
        label="决策",
        used=float(usage.get("decision_count") or 0),
        limit=usage.get("daily_max_decisions"),
        problems=problems,
        near=near,
    )
    _budget_state(
        label="tokens",
        used=float(usage.get("total_tokens") or 0),
        limit=usage.get("daily_max_tokens"),
        problems=problems,
        near=near,
    )
    _budget_state(
        label="cost",
        used=float(usage.get("total_cost_usd") or 0.0),
        limit=usage.get("daily_max_cost_usd"),
        problems=problems,
        near=near,
    )

    status: PaperCheckStatus = "PASS"
    if problems or near:
        status = "WARN"
    detail = (
        f"{usage.get('day')} | decisions={usage.get('decision_count', 0)}/"
        f"{_limit_text(usage.get('daily_max_decisions'))} | tokens={usage.get('total_tokens', 0)}/"
        f"{_limit_text(usage.get('daily_max_tokens'))} | cost=${float(usage.get('total_cost_usd') or 0.0):.4f}/"
        f"{_limit_text(usage.get('daily_max_cost_usd'), money=True)}"
    )
    if problems:
        detail += " | 已达上限：" + ",".join(problems)
    elif near:
        detail += " | 接近上限：" + ",".join(near)
    return PaperCheckItem("AI 预算", status, detail, data={"usage": usage})


def _budget_state(
    *,
    label: str,
    used: float,
    limit: int | float | None,
    problems: list[str],
    near: list[str],
) -> None:
    if limit is None or float(limit) <= 0:
        return
    ratio = used / float(limit)
    if ratio >= 1:
        problems.append(label)
    elif ratio >= 0.8:
        near.append(label)


def _limit_text(value: int | float | None, *, money: bool = False) -> str:
    if value is None or float(value) <= 0:
        return "不限"
    if money:
        return f"${float(value):.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return f"{value:,}"


def _run_once_item(
    loader: DataLoader,
    symbols: list[str],
    *,
    initial_cash: float,
) -> PaperCheckItem:
    from qtrade.execution.account import Account
    from qtrade.execution.broker.paper import PaperBroker
    from qtrade.persistence import init_db
    from qtrade.risk.engine import RiskEngine
    from qtrade.risk.policies import load_risk_policies
    from qtrade.runtime.live_runner import LiveRunner
    from qtrade.runtime.state import StateManager
    from qtrade.strategies.technical.ma_cross import MACrossStrategy

    class _PaperCheckMACrossStrategy(MACrossStrategy):
        id = "paper_check_ma_cross_v1"

    init_db()
    strategy = _PaperCheckMACrossStrategy(
        universe=symbols,
        params={"fast_window": 5, "slow_window": 20, "position_pct": 0.3},
    )
    runner = LiveRunner(
        strategy=strategy,
        state=StateManager(strategy_id=strategy.id, mode="paper"),
        risk=RiskEngine(policies=load_risk_policies()),
        broker=PaperBroker(account=Account()),
        loader=loader,
    )
    session = runner.start(initial_cash=initial_cash)
    try:
        tick = runner.tick()
    finally:
        runner.stop(note="paper-check run-once")

    detail = (
        f"run_id={session.run_id} | quotes={tick.quotes_count} | signals={tick.signals_generated} | "
        f"orders={tick.orders_placed} | skipped={tick.strategy_skipped} | "
        f"risk_rejected={tick.risk_rejected}"
    )
    if tick.errors:
        return PaperCheckItem("跑一次 tick", "FAIL", detail + f" | errors={tick.errors[:3]}")
    if tick.quotes_count == 0:
        return PaperCheckItem("跑一次 tick", "FAIL", detail + " | 无行情")
    if tick.strategy_skipped or tick.risk_rejected:
        return PaperCheckItem("跑一次 tick", "WARN", detail)
    return PaperCheckItem("跑一次 tick", "PASS", detail)
