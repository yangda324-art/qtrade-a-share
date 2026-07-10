"""Market-data health checks and warmup helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from qtrade.core.calendar import is_trading_day
from qtrade.core.config import get_settings
from qtrade.core.time import MARKET_TZ, market_today, utcnow
from qtrade.data.loader import BAR_SOURCE_ATTR, BAR_STATUS_ATTR, DataLoader, get_loader
from qtrade.data.schema import PERIOD_LABELS, AdjustMode, Period, normalize_symbol

DataHealthStatus = Literal["PASS", "WARN", "FAIL"]

DATA_CHECK_PERIODS: tuple[Period, ...] = (
    "5min",
    "15min",
    "30min",
    "60min",
    "daily",
    "weekly",
    "monthly",
)
DEFAULT_DATA_CHECK_PERIODS: tuple[Period, ...] = ("daily", "5min", "15min")

_START_TOLERANCE_DAYS: dict[str, int] = {
    "5min": 7,
    "15min": 7,
    "30min": 7,
    "60min": 7,
    "daily": 7,
    "weekly": 21,
    "monthly": 45,
}
_END_TOLERANCE_DAYS: dict[str, int] = {
    "5min": 7,
    "15min": 7,
    "30min": 7,
    "60min": 7,
    "daily": 7,
    "weekly": 21,
    "monthly": 45,
}


@dataclass(frozen=True)
class DataHealthItem:
    """One market-data health check result."""

    name: str
    status: DataHealthStatus
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataHealthReport:
    """Aggregate market-data health check report."""

    symbols: list[str]
    periods: list[str]
    start: date
    end: date
    adjust: AdjustMode
    include_quotes: bool
    force_refresh: bool
    items: list[DataHealthItem]
    generated_at: datetime = field(default_factory=utcnow)

    @property
    def failures(self) -> list[DataHealthItem]:
        return [item for item in self.items if item.status == "FAIL"]

    @property
    def warnings(self) -> list[DataHealthItem]:
        return [item for item in self.items if item.status == "WARN"]

    @property
    def passed(self) -> list[DataHealthItem]:
        return [item for item in self.items if item.status == "PASS"]

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    @property
    def overall_status(self) -> DataHealthStatus:
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
            "periods": list(self.periods),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "adjust": self.adjust,
            "include_quotes": self.include_quotes,
            "force_refresh": self.force_refresh,
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


def run_data_health_check(
    symbols: list[str],
    *,
    periods: list[str] | tuple[str, ...] | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
    adjust: AdjustMode = "qfq",
    include_quotes: bool = True,
    force_refresh: bool = False,
    loader: DataLoader | None = None,
    max_quote_age_seconds: int | None = None,
) -> DataHealthReport:
    """Check quote and K-line availability.

    Bar checks use the normal ``DataLoader`` path, so they may warm local cache
    when cache coverage is missing or stale.  Quote checks are read-only.
    """
    normalized = _normalize_symbols(symbols)
    start_d = date.fromisoformat(start) if isinstance(start, str) else start
    end_d = date.fromisoformat(end) if isinstance(end, str) else end
    end_d = end_d or market_today()
    start_d = start_d or (end_d - timedelta(days=30))
    if start_d > end_d:
        raise ValueError("start 不能晚于 end")

    selected_periods = _normalize_periods(periods)
    loader = loader or get_loader()
    max_age = max_quote_age_seconds or get_settings().trading.realtime.max_quote_age_seconds

    items: list[DataHealthItem] = []
    if not normalized:
        items.append(DataHealthItem("标的", "FAIL", "没有有效标的"))
        return DataHealthReport(
            symbols=[],
            periods=list(selected_periods),
            start=start_d,
            end=end_d,
            adjust=adjust,
            include_quotes=include_quotes,
            force_refresh=force_refresh,
            items=items,
        )

    if include_quotes:
        items.extend(_quote_items(loader, normalized, max_quote_age_seconds=max_age))

    for symbol in normalized:
        for period in selected_periods:
            items.append(
                _bar_item(
                    loader,
                    symbol,
                    period,
                    start=start_d,
                    end=end_d,
                    adjust=adjust,
                    force_refresh=force_refresh,
                )
            )

    return DataHealthReport(
        symbols=normalized,
        periods=list(selected_periods),
        start=start_d,
        end=end_d,
        adjust=adjust,
        include_quotes=include_quotes,
        force_refresh=force_refresh,
        items=items,
    )


def write_data_health_json(report: DataHealthReport, path: str | Path) -> Path:
    """Write a data health report as JSON."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_data_health_markdown(report: DataHealthReport, path: str | Path) -> Path:
    """Write a data health report as Markdown."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data_health_to_markdown(report), encoding="utf-8")
    return out


def data_health_to_markdown(report: DataHealthReport) -> str:
    """Render a data health report as Markdown."""
    lines = [
        "# 行情源健康检查报告",
        "",
        f"- 生成时间: {report.generated_at.isoformat()}",
        f"- 市场日期: {market_today().isoformat()}",
        f"- 标的: {', '.join(report.symbols) or '-'}",
        f"- 周期: {', '.join(report.periods) or '-'}",
        f"- 范围: {report.start.isoformat()} -> {report.end.isoformat()}",
        f"- 复权: {report.adjust}",
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


def _quote_items(
    loader: DataLoader,
    symbols: list[str],
    *,
    max_quote_age_seconds: int,
) -> list[DataHealthItem]:
    try:
        quotes = loader.get_quotes(symbols)
    except Exception as exc:
        return [DataHealthItem("实时行情", "FAIL", f"行情拉取异常：{exc}")]

    now = utcnow()
    items: list[DataHealthItem] = []
    for symbol in symbols:
        quote = quotes.get(symbol)
        name = f"实时行情 {symbol}"
        if quote is None:
            items.append(DataHealthItem(name, "FAIL", "无行情返回"))
            continue
        reject_reason = quote.realtime_reject_reason(max_age_seconds=max_quote_age_seconds, now=now)
        age = quote.age_seconds(now)
        detail = f"price={quote.price:.3f} | source={quote.source} | age={age:.1f}s"
        data = {
            "symbol": symbol,
            "price": quote.price,
            "source": quote.source,
            "age_seconds": age,
            "reject_reason": reject_reason,
            "stale_reason": quote.stale_reason,
        }
        if reject_reason is None:
            items.append(DataHealthItem(name, "PASS", f"{detail} | 可用于实时模拟", data=data))
        else:
            items.append(
                DataHealthItem(
                    name,
                    "WARN",
                    f"{detail} | 仅展示，不可用于实时模拟：{reject_reason}",
                    data=data,
                )
            )
    return items


def _bar_item(
    loader: DataLoader,
    symbol: str,
    period: str,
    *,
    start: date,
    end: date,
    adjust: AdjustMode,
    force_refresh: bool,
) -> DataHealthItem:
    label = PERIOD_LABELS.get(period, period)
    name = f"K 线 {symbol} {label}"
    try:
        df = loader.get_bars(symbol, start=start, end=end, adjust=adjust, period=period, force_refresh=force_refresh)
    except Exception as exc:
        return DataHealthItem(
            name,
            "FAIL",
            f"拉取失败：{_short_error(exc)}",
            data={"symbol": symbol, "period": period, "error": str(exc)},
        )
    if df.empty:
        return DataHealthItem(name, "FAIL", "无数据", data={"symbol": symbol, "period": period})

    first, last = _bar_date_range(df)
    coverage_status, coverage_note = _coverage_status(period, start=start, end=end, first=first, last=last)
    detail = f"{len(df)} 根 | {first:%Y-%m-%d %H:%M} -> {last:%Y-%m-%d %H:%M}"
    data_status = str(df.attrs.get(BAR_STATUS_ATTR) or "unknown")
    data_source = str(df.attrs.get(BAR_SOURCE_ATTR) or "unknown")
    source_note = _bar_source_note(data_status, data_source)
    if source_note:
        detail = f"{detail} | {source_note}"
    if coverage_note:
        detail = f"{detail} | {coverage_note}"
    status = _combine_status(coverage_status, _bar_source_status(data_status))
    return DataHealthItem(
        name,
        status,
        detail,
        data={
            "symbol": symbol,
            "period": period,
            "rows": len(df),
            "first": first.isoformat(),
            "last": last.isoformat(),
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "data_status": data_status,
            "data_source": data_source,
        },
    )


def _coverage_status(
    period: str,
    *,
    start: date,
    end: date,
    first: datetime,
    last: datetime,
    now: datetime | None = None,
) -> tuple[DataHealthStatus, str]:
    start_gap_days = (first.date() - start).days
    end_gap_days = (end - last.date()).days
    start_tolerance = _START_TOLERANCE_DAYS.get(period, 7)
    end_tolerance = _END_TOLERANCE_DAYS.get(period, 7)
    notes: list[str] = []
    if start_gap_days > start_tolerance:
        notes.append(f"起点覆盖不足 {start_gap_days} 天")
    if end_gap_days > end_tolerance:
        notes.append(f"终点覆盖不足 {end_gap_days} 天")
    intraday_note = _intraday_minute_note(period, end=end, last=last, now=now)
    if intraday_note:
        notes.append(intraday_note)
    if notes:
        return "WARN", "；".join(notes)
    return "PASS", ""


def _intraday_minute_note(
    period: str,
    *,
    end: date,
    last: datetime,
    now: datetime | None = None,
) -> str:
    if not period.endswith("min"):
        return ""
    local_now = now or datetime.now(MARKET_TZ).replace(tzinfo=None)
    today = local_now.date()
    if end < today or last.date() >= today:
        return ""
    if not is_trading_day(today):
        return ""
    if local_now.time() < time(9, 35):
        return ""
    return "交易日分钟K线未包含今日数据"


def _bar_source_status(data_status: str) -> DataHealthStatus:
    if data_status in {"cache_fallback", "derived_15min_from_5min"}:
        return "WARN"
    return "PASS"


def _bar_source_note(data_status: str, data_source: str) -> str:
    if data_status == "refreshed":
        return f"已刷新({data_source})"
    if data_status == "cache_hit":
        return "本地缓存"
    if data_status == "cache_fallback":
        return "刷新失败，使用本地缓存"
    if data_status == "derived_15min_from_5min":
        return "15分由5分重采样"
    return ""


def _combine_status(*statuses: DataHealthStatus) -> DataHealthStatus:
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _bar_date_range(df: pd.DataFrame) -> tuple[datetime, datetime]:
    dates = pd.to_datetime(df["date"])
    return dates.min().to_pydatetime(), dates.max().to_pydatetime()


def _normalize_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    for raw in symbols:
        try:
            symbol = normalize_symbol(raw)
        except Exception:
            continue
        if symbol not in out:
            out.append(symbol)
    return out


def _normalize_periods(periods: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    selected = tuple(periods or DEFAULT_DATA_CHECK_PERIODS)
    if not selected:
        return DEFAULT_DATA_CHECK_PERIODS
    invalid = [period for period in selected if period not in DATA_CHECK_PERIODS]
    if invalid:
        raise ValueError(f"不支持的周期：{', '.join(invalid)}")
    return selected


def _short_error(exc: Exception, *, max_len: int = 220) -> str:
    text = " ".join(str(exc).split())
    if "ProxyError" in text or "RemoteDisconnected" in text:
        return "外部 K 线接口连接失败/代理断连；请稍后重试，或先使用已有缓存"
    if len(text) <= max_len:
        return text
    return f"{text[:max_len - 1]}…"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")
