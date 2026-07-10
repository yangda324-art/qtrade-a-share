"""数据 schema 标准化。

不同源的字段命名千奇百怪：akshare 用中文列名，tushare 用 ``trade_date`` + ``ts_code``，
掘金又用 ``eob`` + ``symbol``。我们在 adapter 层全部转成统一格式，下游只对一种 schema 负责。

行情 DataFrame 标准列（小写）::

    date  open  high  low  close  volume  amount  symbol  adjust

- ``date``：``pandas.Timestamp``，盘后日线统一用收盘日 0 点
- ``volume``：股数
- ``amount``：成交金额（元）
- ``adjust``：``"none"`` / ``"qfq"`` / ``"hfq"``  数据复权方式
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]

AdjustMode = Literal["none", "qfq", "hfq"]

# 支持的 K 线周期
Period = Literal["5min", "15min", "30min", "60min", "daily", "weekly", "monthly"]
PERIOD_LABELS = {
    "5min": "5 分", "15min": "15 分", "30min": "30 分", "60min": "60 分",
    "daily": "日", "weekly": "周", "monthly": "月",
}


@dataclass(frozen=True)
class Bar:
    """单根 K 线（用于事件驱动的回测和实时推送）。"""

    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    adjust: AdjustMode = "qfq"


@dataclass(frozen=True)
class NewsItem:
    """新闻条目。"""

    symbol: str | None        # 关联标的；纯宏观新闻可为 None
    title: str
    content: str
    source: str
    url: str
    published_at: datetime


# -------------------- 标的代码标准化 --------------------

_SH_PREFIXES = ("60", "68", "9")
_SZ_PREFIXES = ("00", "30", "20")
_BJ_PREFIXES = ("4", "8", "92")


def normalize_symbol(raw: str) -> str:
    """转成内部标准格式 ``"600519.SH"``。

    支持的输入：
    - ``600519`` → ``600519.SH``
    - ``600519.SH`` / ``sh.600519`` / ``SH600519`` → ``600519.SH``
    - ``000001`` → ``000001.SZ``
    """
    s = raw.strip().upper().replace("SH.", "").replace("SZ.", "").replace("BJ.", "")
    s = s.replace("SH", "").replace("SZ", "").replace("BJ", "")
    s = re.sub(r"[^\d.]", "", s)

    if "." in s:
        code, suffix = s.split(".", 1)
        if suffix in {"SH", "SZ", "BJ"}:
            return f"{code}.{suffix}"
        # 后缀不规范，按代码前缀重新判
        s = code

    if not s.isdigit() or len(s) != 6:
        raise ValueError(f"unrecognized symbol: {raw!r}")

    if s.startswith(_SH_PREFIXES):
        return f"{s}.SH"
    if s.startswith(_SZ_PREFIXES):
        return f"{s}.SZ"
    if s.startswith(_BJ_PREFIXES):
        return f"{s}.BJ"
    raise ValueError(f"cannot infer exchange for symbol: {raw!r}")


def split_symbol(sym: str) -> tuple[str, str]:
    """``"600519.SH"`` → ``("600519", "SH")``。"""
    s = normalize_symbol(sym)
    code, exch = s.split(".")
    return code, exch
