"""本地 Parquet 缓存。

行情数据按 ``{cache_dir}/bars/{symbol}/{adjust}.parquet`` 落盘。
读写都是列存格式，一只票几年的日线 ~几十 KB，按需 ``pyarrow`` 直读，秒级。

为什么不用数据库存行情：行情是 append-only 的纯数值表，列存 + 文件分区比数据库更香——
回测扫描时直接 mmap，不走 SQL 解析；备份就是 ``cp``。
"""
from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from qtrade.core.config import get_settings
from qtrade.core.logging import get_logger
from qtrade.data.schema import BAR_COLUMNS, AdjustMode, normalize_symbol

log = get_logger(__name__)


def _bar_path(symbol: str, adjust: AdjustMode, period: str = "daily") -> Path:
    settings = get_settings()
    sym = normalize_symbol(symbol)
    # 路径布局：bars/{symbol}/{period}/{adjust}.parquet
    # 兼容旧布局 bars/{symbol}/{adjust}.parquet —— load_bars 自动迁移识别
    p = settings.paths.cache_dir / "bars" / sym / period / f"{adjust}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _legacy_bar_path(symbol: str, adjust: AdjustMode) -> Path:
    """老路径（无 period 子目录），仅 daily 用。"""
    settings = get_settings()
    sym = normalize_symbol(symbol)
    return settings.paths.cache_dir / "bars" / sym / f"{adjust}.parquet"


def save_bars(df: pd.DataFrame, symbol: str, adjust: AdjustMode = "qfq", period: str = "daily") -> Path:
    """保存 K 线 DataFrame 到本地 Parquet。会和已有数据合并去重。"""
    if df.empty:
        log.warning("save_bars called with empty df | symbol={}", symbol)
        return _bar_path(symbol, adjust, period)

    df = df.copy()
    missing = set(BAR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    path = _bar_path(symbol, adjust, period)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    df.to_parquet(path, index=False)
    log.debug("cached bars | symbol={} | period={} | rows={} | path={}", symbol, period, len(df), path)
    return path


def load_bars(
    symbol: str,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    adjust: AdjustMode = "qfq",
    period: str = "daily",
) -> pd.DataFrame:
    """从本地 Parquet 读取 K 线，按 ``[start, end]`` 闭区间过滤。文件不存在时返回空 DF。"""
    path = _bar_path(symbol, adjust, period)
    # daily 周期 + 新路径不存在时回退到旧路径（向后兼容已有缓存）
    if not path.exists() and period == "daily":
        legacy = _legacy_bar_path(symbol, adjust)
        if legacy.exists():
            path = legacy
    if not path.exists():
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= _end_timestamp(end, period)]
    return df.reset_index(drop=True)


def has_cache(symbol: str, adjust: AdjustMode = "qfq", period: str = "daily") -> bool:
    return _bar_path(symbol, adjust, period).exists() or (
        period == "daily" and _legacy_bar_path(symbol, adjust).exists()
    )


def cache_coverage(symbol: str, adjust: AdjustMode = "qfq", period: str = "daily") -> tuple[date, date] | None:
    """返回缓存中的 ``(min_date, max_date)``，无缓存返回 ``None``。"""
    df = load_bars(symbol, adjust=adjust, period=period)
    if df.empty:
        return None
    return df["date"].min().date(), df["date"].max().date()


def _end_timestamp(value: date | datetime | str, period: str) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if period in {"5min", "15min", "30min", "60min"}:
        if isinstance(value, date) and not isinstance(value, datetime):
            return pd.Timestamp(datetime.combine(value, time.max))
        if isinstance(value, str) and len(value.strip()) <= 10:
            return pd.Timestamp(datetime.combine(ts.date(), time.max))
    return ts
