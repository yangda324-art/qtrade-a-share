"""EastMoney K-line data adapter.

This adapter is a direct fallback for AkShare K-line outages.  It uses the same
EastMoney historical kline endpoint that AkShare wraps, but keeps the fallback
inside our adapter boundary so UI/runtime code does not need source-specific
logic.
"""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import pandas as pd
import requests

from qtrade.core.exceptions import DataError
from qtrade.data.adapters.base import DataAdapter
from qtrade.data.schema import AdjustMode, normalize_symbol, split_symbol

_EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_FIELDS1 = "f1,f2,f3,f4,f5,f6"
_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57"
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
_FQT = {"none": "0", "qfq": "1", "hfq": "2"}


class EastMoneyKlineAdapter(DataAdapter):
    """Direct EastMoney K-line adapter."""

    name = "eastmoney_kline"

    def __init__(self, *, timeout_s: float = 8.0) -> None:
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: AdjustMode = "qfq",
    ) -> pd.DataFrame:
        return self.fetch_bars(symbol, start, end, adjust=adjust, period="daily")

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: AdjustMode = "qfq",
        period: str = "daily",
    ) -> pd.DataFrame:
        sym = normalize_symbol(symbol)
        klt = _KLT.get(period)
        if klt is None:
            raise DataError(f"{self.name} 不支持周期 {period}")

        try:
            resp = self._session.get(
                _EASTMONEY_KLINE_URL,
                params={
                    "fields1": _FIELDS1,
                    "fields2": _FIELDS2,
                    "ut": _UT,
                    "klt": klt,
                    "fqt": _FQT[adjust],
                    "secid": _to_secid(sym),
                    "beg": _request_start(start, period),
                    "end": _request_end(end, period),
                },
                headers={
                    "User-Agent": "Mozilla/5.0 qtrade/0.1",
                    "Referer": "https://quote.eastmoney.com/",
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise DataError(f"{self.name} {period} fetch failed for {sym}: {exc}") from exc

        rows = (payload.get("data") or {}).get("klines") or []
        parsed = [_parse_kline(row) for row in rows if isinstance(row, str)]
        parsed = [row for row in parsed if row is not None]
        if not parsed:
            return _empty_bars()

        df = pd.DataFrame(parsed, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= _start_ts(start)) & (df["date"] <= _end_ts(end, period))]
        if df.empty:
            return _empty_bars()
        df["volume"] = df["volume"].astype(float) * 100
        df["symbol"] = sym
        df["adjust"] = adjust
        return df[["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]].reset_index(
            drop=True
        )


def _parse_kline(raw: str) -> tuple[Any, float, float, float, float, float, float] | None:
    parts = raw.split(",")
    if len(parts) < 7:
        return None
    try:
        ts = pd.to_datetime(parts[0])
        open_ = _num(parts[1])
        close = _num(parts[2])
        high = _num(parts[3])
        low = _num(parts[4])
        volume = _num(parts[5])
        amount = _num(parts[6])
    except Exception:
        return None
    return ts, open_, high, low, close, volume, amount


def _request_start(value: date, period: str) -> str:
    if period.endswith("min"):
        return "0"
    return value.strftime("%Y%m%d")


def _request_end(value: date, period: str) -> str:
    if period.endswith("min"):
        return "20500000"
    return value.strftime("%Y%m%d")


def _start_ts(value: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(value, time.min))


def _end_ts(value: date, period: str) -> pd.Timestamp:
    if period.endswith("min"):
        return pd.Timestamp(datetime.combine(value, time.max))
    return pd.Timestamp(value)


def _to_secid(symbol: str) -> str:
    code, exchange = split_symbol(normalize_symbol(symbol))
    market = "1" if exchange == "SH" else "0"
    return f"{market}.{code}"


def _num(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "-", "--"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"])
