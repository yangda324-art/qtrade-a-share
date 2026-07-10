"""Network diagnostics for market data endpoints."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import requests

from qtrade.core.time import market_today, utcnow
from qtrade.data.schema import normalize_symbol, split_symbol

NetworkStatus = Literal["PASS", "WARN", "FAIL"]

_QUOTE_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_QUOTE_FIELDS = ",".join(["f12", "f13", "f14", "f2", "f17", "f15", "f16", "f18", "f5", "f6"])
_KLINE_FIELDS1 = "f1,f2,f3,f4,f5,f6"
_KLINE_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57"
_UT = "7eea3edcaed734bea9cbfc24409ed989"
_KLT = {
    "5min": "5",
    "15min": "15",
    "30min": "30",
    "60min": "60",
    "daily": "101",
    "weekly": "102",
    "monthly": "103",
}


@dataclass(frozen=True)
class NetworkCheckItem:
    """One network diagnostic result."""

    name: str
    status: NetworkStatus
    category: str
    detail: str
    latency_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NetworkCheckReport:
    """Aggregate network diagnostic result."""

    symbol: str
    timeout_s: float
    periods: list[str]
    items: list[NetworkCheckItem]
    generated_at: str = field(default_factory=lambda: utcnow().isoformat())

    @property
    def failures(self) -> list[NetworkCheckItem]:
        return [item for item in self.items if item.status == "FAIL"]

    @property
    def warnings(self) -> list[NetworkCheckItem]:
        return [item for item in self.items if item.status == "WARN"]

    @property
    def passed(self) -> list[NetworkCheckItem]:
        return [item for item in self.items if item.status == "PASS"]

    @property
    def overall_status(self) -> NetworkStatus:
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "symbol": self.symbol,
            "timeout_s": self.timeout_s,
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
                    "category": item.category,
                    "detail": item.detail,
                    "latency_ms": item.latency_ms,
                    "data": item.data,
                }
                for item in self.items
            ],
        }


def run_network_check(
    *,
    symbol: str = "000001",
    periods: list[str] | tuple[str, ...] | None = None,
    timeout_s: float = 5.0,
    session: requests.Session | None = None,
    end: date | None = None,
) -> NetworkCheckReport:
    """Run non-persistent market-data network diagnostics."""
    if timeout_s <= 0:
        raise ValueError("timeout_s 必须为正数")
    normalized = normalize_symbol(symbol)
    selected_periods = list(periods or ["daily", "5min"])
    invalid = [period for period in selected_periods if period not in _KLT]
    if invalid:
        raise ValueError(f"不支持的周期：{', '.join(invalid)}")

    http = session or requests.Session()
    end_d = end or market_today()
    start_d = end_d - timedelta(days=7)

    items = [
        _proxy_env_item(),
        _akshare_import_item(),
        _eastmoney_quote_item(http, normalized, timeout_s=timeout_s),
    ]
    for period in selected_periods:
        items.append(
            _eastmoney_kline_item(
                http,
                normalized,
                period=period,
                start=start_d,
                end=end_d,
                timeout_s=timeout_s,
            )
        )

    return NetworkCheckReport(
        symbol=normalized,
        timeout_s=timeout_s,
        periods=selected_periods,
        items=items,
    )


def write_network_check_json(report: NetworkCheckReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _proxy_env_item() -> NetworkCheckItem:
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")
    values = {key: os.environ[key] for key in keys if os.environ.get(key)}
    if values:
        shown = ", ".join(sorted(values))
        return NetworkCheckItem(
            "代理环境",
            "WARN",
            "PROXY_ENV",
            f"检测到代理环境变量：{shown}",
            data={"proxy_env_keys": sorted(values)},
        )
    return NetworkCheckItem("代理环境", "PASS", "NO_PROXY_ENV", "未检测到代理环境变量；requests 不读取 macOS 系统代理")


def _akshare_import_item() -> NetworkCheckItem:
    try:
        import akshare as ak
    except Exception as exc:
        return NetworkCheckItem("AkShare 包", "FAIL", "IMPORT_ERROR", f"AkShare 不可导入：{exc}")
    version = str(getattr(ak, "__version__", "unknown"))
    return NetworkCheckItem("AkShare 包", "PASS", "IMPORT_OK", f"已安装 akshare {version}", data={"version": version})


def _eastmoney_quote_item(
    session: requests.Session,
    symbol: str,
    *,
    timeout_s: float,
) -> NetworkCheckItem:
    return _json_endpoint_item(
        name="EastMoney 快照",
        session=session,
        url=_QUOTE_URL,
        params={
            "fltt": "2",
            "invt": "2",
            "fields": _QUOTE_FIELDS,
            "secids": _to_secid(symbol),
        },
        timeout_s=timeout_s,
        response_kind="quote",
    )


def _eastmoney_kline_item(
    session: requests.Session,
    symbol: str,
    *,
    period: str,
    start: date,
    end: date,
    timeout_s: float,
) -> NetworkCheckItem:
    return _json_endpoint_item(
        name=f"EastMoney K线 {period}",
        session=session,
        url=_KLINE_URL,
        params={
            "fields1": _KLINE_FIELDS1,
            "fields2": _KLINE_FIELDS2,
            "ut": _UT,
            "klt": _KLT[period],
            "fqt": "1",
            "secid": _to_secid(symbol),
            "beg": _request_start(start, period),
            "end": _request_end(end, period),
        },
        timeout_s=timeout_s,
        response_kind="kline",
    )


def _json_endpoint_item(
    *,
    name: str,
    session: requests.Session,
    url: str,
    params: dict[str, str],
    timeout_s: float,
    response_kind: Literal["quote", "kline"],
) -> NetworkCheckItem:
    started = time.perf_counter()
    try:
        resp = session.get(
            url,
            params=params,
            headers={
                "User-Agent": "Mozilla/5.0 qtrade/0.1",
                "Referer": "https://quote.eastmoney.com/",
            },
            timeout=timeout_s,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        resp.raise_for_status()
        try:
            payload = resp.json()
        except Exception as exc:
            return NetworkCheckItem(name, "FAIL", "JSON_ERROR", f"HTTP 可达，但 JSON 解析失败：{exc}", latency_ms)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        category, detail = _classify_request_error(exc)
        return NetworkCheckItem(name, "FAIL", category, detail, latency_ms)

    count = _payload_count(payload, response_kind)
    if count <= 0:
        return NetworkCheckItem(
            name,
            "FAIL",
            "EMPTY_RESPONSE",
            "接口可达，但返回数据为空",
            latency_ms,
            data={"url": url, "count": count},
        )
    return NetworkCheckItem(
        name,
        "PASS",
        "OK",
        f"接口可达，返回 {count} 条",
        latency_ms,
        data={"url": url, "count": count},
    )


def _payload_count(payload: Any, response_kind: Literal["quote", "kline"]) -> int:
    if not isinstance(payload, dict):
        return 0
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        return 0
    rows = (data.get("diff") if response_kind == "quote" else data.get("klines")) or []
    return len(rows) if isinstance(rows, list) else 0


def _classify_request_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, requests.exceptions.ProxyError):
        return "PROXY_ERROR", "代理连接失败/远端断连；检查 VPN、网络出口、透明代理，或稍后重试"
    if isinstance(exc, requests.exceptions.Timeout):
        return "TIMEOUT", f"请求超时：{_short_text(exc)}"
    if isinstance(exc, requests.exceptions.SSLError):
        return "SSL_ERROR", f"SSL 连接失败：{_short_text(exc)}"
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return "HTTP_ERROR", f"HTTP 状态异常：{status_code or _short_text(exc)}"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "CONNECTION_ERROR", f"连接失败：{_short_text(exc)}"
    if isinstance(exc, requests.exceptions.RequestException):
        return "REQUEST_ERROR", f"请求失败：{_short_text(exc)}"
    return "UNKNOWN_ERROR", f"未知异常：{_short_text(exc)}"


def _short_text(value: Any, *, max_len: int = 160) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return f"{text[:max_len - 1]}…"


def _request_start(value: date, period: str) -> str:
    if period.endswith("min"):
        return "0"
    return value.strftime("%Y%m%d")


def _request_end(value: date, period: str) -> str:
    if period.endswith("min"):
        return "20500000"
    return value.strftime("%Y%m%d")


def _to_secid(symbol: str) -> str:
    code, exchange = split_symbol(normalize_symbol(symbol))
    market = "1" if exchange == "SH" else "0"
    return f"{market}.{code}"
