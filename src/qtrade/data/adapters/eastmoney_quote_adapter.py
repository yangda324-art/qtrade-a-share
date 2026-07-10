"""东方财富快照行情兜底适配器。

用途只限实时快照：AkShare 的单股快照接口偶发 JSON/SSL 失败时，
直接请求东方财富 push2 API，避免 UI 和模拟盘只能退到缓存日线收盘价。
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests

from qtrade.core.exceptions import DataError
from qtrade.core.time import utcnow
from qtrade.data.adapters.base import DataAdapter, Quote
from qtrade.data.schema import AdjustMode, normalize_symbol, split_symbol

_EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
_FIELDS = ",".join(
    [
        "f12",  # code
        "f13",  # market id
        "f14",  # name
        "f2",  # latest price
        "f17",  # open
        "f15",  # high
        "f16",  # low
        "f18",  # previous close
        "f5",  # volume, hand
        "f6",  # amount, CNY
    ]
)


class EastMoneyQuoteAdapter(DataAdapter):
    """东方财富实时快照适配器。"""

    name = "eastmoney_quote"

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: AdjustMode = "qfq",
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]
        )

    def fetch_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        normalized = [normalize_symbol(sym) for sym in symbols]
        if not normalized:
            return {}

        secids = ",".join(_to_secid(sym) for sym in normalized)
        try:
            resp = self._session.get(
                _EASTMONEY_QUOTE_URL,
                params={
                    "fltt": "2",
                    "invt": "2",
                    "fields": _FIELDS,
                    "secids": secids,
                },
                headers={
                    "User-Agent": "Mozilla/5.0 qtrade/0.1",
                    "Referer": "https://quote.eastmoney.com/",
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            raise DataError(f"eastmoney quote fetch failed: {e}") from e

        rows = (payload.get("data") or {}).get("diff") or []
        if not rows:
            return {}

        wanted = {split_symbol(sym)[0]: sym for sym in normalized}
        ts = utcnow()
        out: dict[str, Quote] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("f12") or "")
            symbol = wanted.get(code)
            if symbol is None:
                continue
            price = _num(row.get("f2"))
            if price <= 0:
                continue
            prev_close = _num(row.get("f18"), price)
            out[symbol] = Quote(
                symbol=symbol,
                price=price,
                open=_num(row.get("f17"), price),
                high=_num(row.get("f15"), price),
                low=_num(row.get("f16"), price),
                prev_close=prev_close,
                volume=_num(row.get("f5")) * 100,
                amount=_num(row.get("f6")),
                ts=ts,
                fetched_at=ts,
                source=self.name,
            )
        return out


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
