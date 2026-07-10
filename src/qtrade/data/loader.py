"""统一数据加载入口。

业务代码只通过 ``DataLoader`` 拿数据，背后是"缓存命中→源拉取→落库→返回"的全链路。
默认源 = akshare；以后接 Tushare 时只需在 ``__init__`` 里加，并设置 fallback 顺序。

使用::

    loader = get_loader()
    df = loader.get_daily_bars("600519", start="2024-01-01", end="2024-12-31")
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd

from qtrade.core.exceptions import DataError
from qtrade.core.logging import get_logger
from qtrade.data import cache as bar_cache
from qtrade.data.adapters import (
    AkshareAdapter,
    DataAdapter,
    EastMoneyKlineAdapter,
    EastMoneyQuoteAdapter,
)
from qtrade.data.schema import AdjustMode, NewsItem, normalize_symbol

# 缓存终点离请求终点超过这个天数才算"需要刷新"。3 天涵盖周末 + 长假常见情况。
_CACHE_END_TOLERANCE_DAYS = 3
_QUOTE_FAILURE_COOLDOWN = timedelta(seconds=60)
BAR_STATUS_ATTR = "qtrade_bar_status"
BAR_SOURCE_ATTR = "qtrade_bar_source"
BAR_WARNING_ATTR = "qtrade_bar_warning"

log = get_logger(__name__)


class DataLoader:
    """带本地缓存的数据加载器。"""

    def __init__(self, adapters: list[DataAdapter] | None = None) -> None:
        if adapters is not None:
            self.adapters = adapters
        else:
            self.adapters = self._build_default_adapters()
        self._quote_cooldowns: dict[str, datetime] = {}

    @staticmethod
    def _build_default_adapters() -> list[DataAdapter]:
        """默认 adapter 链：akshare 优先；如有 TUSHARE_TOKEN 自动加 tushare 作为备援。"""
        from qtrade.core.config import get_settings
        from qtrade.data.adapters import TushareAdapter
        out: list[DataAdapter] = [AkshareAdapter(), EastMoneyKlineAdapter(), EastMoneyQuoteAdapter()]
        if TushareAdapter is not None and get_settings().tushare_token:
            try:
                out.append(TushareAdapter())
                log.info("tushare adapter enabled as fallback")
            except Exception as e:
                log.warning("tushare adapter init failed: {}", e)
        return out

    # ---------------- 行情 ----------------

    def get_daily_bars(
        self,
        symbol: str,
        *,
        start: date | datetime | str,
        end: date | datetime | str | None = None,
        adjust: AdjustMode = "qfq",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """日线获取（向后兼容入口，等价于 ``get_bars(period='daily')``）。"""
        return self.get_bars(
            symbol, start=start, end=end, adjust=adjust,
            period="daily", force_refresh=force_refresh,
        )

    def get_bars(
        self,
        symbol: str,
        *,
        start: date | datetime | str,
        end: date | datetime | str | None = None,
        adjust: AdjustMode = "qfq",
        period: str = "daily",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """通用周期 K 线获取。优先读本地缓存，缺失部分再向数据源拉取并补缓存。"""
        sym = normalize_symbol(symbol)
        s = pd.to_datetime(start).date()
        e = pd.to_datetime(end).date() if end else date.today()

        if force_refresh:
            try:
                source = self._fetch_and_cache(sym, s, e, adjust, period)
            except DataError as err:
                if period == "15min":
                    try:
                        fallback = self._resample_and_cache_15min_from_5min(sym, s, e, adjust)
                    except DataError as fallback_err:
                        raise err from fallback_err
                    if not fallback.empty:
                        log.warning(
                            "15min refresh failed for {}, using 5min resample fallback: {}",
                            sym, err,
                        )
                        return _mark_bars(
                            fallback,
                            status="derived_15min_from_5min",
                            source="local_resample",
                            warning=str(err),
                        )
                raise
            return _mark_bars(
                bar_cache.load_bars(sym, start=s, end=e, adjust=adjust, period=period),
                status="refreshed",
                source=source,
            )

        cached = bar_cache.load_bars(sym, start=s, end=e, adjust=adjust, period=period)
        coverage = bar_cache.cache_coverage(sym, adjust=adjust, period=period)

        # 不同周期用不同的"过时容差"——分钟级数据更敏感，每天都需要刷新
        if period in ("5min", "15min", "30min", "60min"):
            tolerance_days = 0
        else:
            tolerance_days = _CACHE_END_TOLERANCE_DAYS

        # 三种触发刷新的情况：
        # 1) 没缓存
        # 2) 缓存终点过时（用户想看更新数据）
        # 3) 缓存起点比用户请求起点晚太多（用户想看更多历史，之前缓存覆盖不到）
        #    容差 7 天：处理"用户写 1 月 1 日但首个交易日是 1 月 3 日"这种节假日情况
        start_gap_days = (coverage[0] - s).days if coverage else 0
        need_refresh = (
            coverage is None
            or coverage[1] < e - timedelta(days=tolerance_days)
            or start_gap_days > 7
        )

        if need_refresh:
            try:
                source = self._fetch_and_cache(sym, s, e, adjust, period)
                cached = bar_cache.load_bars(sym, start=s, end=e, adjust=adjust, period=period)
                return _mark_bars(cached, status="refreshed", source=source)
            except DataError as err:
                if period == "15min":
                    try:
                        fallback = self._resample_and_cache_15min_from_5min(sym, s, e, adjust)
                    except DataError as fallback_err:
                        if not cached.empty:
                            log.warning(
                                "15min refresh failed for {} and 5min resample fallback failed; "
                                "using cached 15min data: {}; fallback={}",
                                sym,
                                err,
                                fallback_err,
                            )
                            return _mark_bars(
                                cached,
                                status="cache_fallback",
                                source="local_cache",
                                warning=f"{err}; fallback={fallback_err}",
                            )
                        raise err from fallback_err
                    if not fallback.empty:
                        log.warning(
                            "15min refresh failed for {}, using 5min resample fallback: {}",
                            sym, err,
                        )
                        cached = fallback
                        return _mark_bars(
                            cached,
                            status="derived_15min_from_5min",
                            source="local_resample",
                            warning=str(err),
                        )
                    elif not cached.empty:
                        log.warning(
                            "network refresh failed for {} ({}), falling back to cached: {}",
                            sym, period, err,
                        )
                        return _mark_bars(
                            cached,
                            status="cache_fallback",
                            source="local_cache",
                            warning=str(err),
                        )
                    else:
                        raise
                elif not cached.empty:
                    log.warning(
                        "network refresh failed for {} ({}), falling back to cached: {}",
                        sym, period, err,
                    )
                    return _mark_bars(
                        cached,
                        status="cache_fallback",
                        source="local_cache",
                        warning=str(err),
                    )
                else:
                    raise

        return _mark_bars(cached, status="cache_hit", source="local_cache")

    def _resample_and_cache_15min_from_5min(
        self, symbol: str, start: date, end: date, adjust: AdjustMode
    ) -> pd.DataFrame:
        five_min = self.get_bars(symbol, start=start, end=end, adjust=adjust, period="5min")
        derived = _resample_minute_bars(five_min, symbol=symbol, adjust=adjust, target_minutes=15)
        if derived.empty:
            return derived
        bar_cache.save_bars(derived, symbol, adjust, period="15min")
        return bar_cache.load_bars(symbol, start=start, end=end, adjust=adjust, period="15min")

    def _fetch_and_cache(
        self, symbol: str, start: date, end: date, adjust: AdjustMode, period: str = "daily"
    ) -> str:
        last_err: Exception | None = None
        errors: list[str] = []
        for adapter in self.adapters:
            if period != "daily" and type(adapter).fetch_bars is DataAdapter.fetch_bars:
                log.debug("{} skipped for {}: unsupported period", adapter.name, period)
                continue
            try:
                if hasattr(adapter, "fetch_bars"):
                    df = adapter.fetch_bars(symbol, start, end, adjust=adjust, period=period)
                elif period == "daily" and hasattr(adapter, "fetch_daily_bars"):
                    df = adapter.fetch_daily_bars(symbol, start, end, adjust=adjust)
                else:
                    raise DataError(f"{adapter.name} does not support period {period}")
                if df.empty:
                    log.warning("{} returned empty | symbol={} period={}", adapter.name, symbol, period)
                    continue
                bar_cache.save_bars(df, symbol, adjust, period=period)
                log.info(
                    "fetched {} rows | symbol={} period={} source={}",
                    len(df), symbol, period, adapter.name,
                )
                return adapter.name
            except Exception as e:
                last_err = e
                errors.append(f"{adapter.name}: {e}")
                log.warning("{} failed for {} {}: {}", adapter.name, symbol, period, e)
        if last_err:
            detail = "; ".join(errors[-4:]) if errors else str(last_err)
            raise DataError(f"all adapters failed for {symbol} {period}: {detail}") from last_err
        raise DataError(f"no adapter supports {symbol} {period}")

    # ---------------- 实时快照 ----------------

    def get_quotes(self, symbols: list[str]) -> dict:
        """批量拉一组标的的实时快照。返回 dict[symbol -> Quote]。

        失败回退顺序：
        1) 各 adapter 的 fetch_quotes（实时接口）
        2) 本地最新日线收盘价拼一个 Quote（盘外/网络异常时合理）
        """
        normalized = [normalize_symbol(s) for s in symbols]
        now = datetime.now()
        out: dict = {}
        missing = set(normalized)
        for adapter in self.adapters:
            if not missing:
                break
            cooldown_until = self._quote_cooldowns.get(adapter.name)
            if cooldown_until and cooldown_until > now:
                log.debug("{} quote fetch skipped until {}", adapter.name, cooldown_until.isoformat())
                continue
            try:
                quotes = adapter.fetch_quotes(sorted(missing))
                if quotes:
                    self._quote_cooldowns.pop(adapter.name, None)
                    for sym, quote in quotes.items():
                        full = normalize_symbol(sym)
                        if full in missing:
                            out[full] = quote
                            missing.discard(full)
            except Exception as e:
                self._quote_cooldowns[adapter.name] = now + _QUOTE_FAILURE_COOLDOWN
                log.warning(
                    "{} quote fetch failed; using fallback quotes for {}s: {}",
                    adapter.name,
                    int(_QUOTE_FAILURE_COOLDOWN.total_seconds()),
                    e,
                )

        # ---- fallback：本地缓存最新一根 K 线 ----
        for sym in sorted(missing):
            df = bar_cache.load_bars(sym)
            if df.empty:
                continue
            last = df.iloc[-1]
            from qtrade.core.time import utcnow
            from qtrade.data.adapters.base import CACHE_QUOTE_SOURCE, Quote
            close = float(last["close"])
            now_ts = utcnow()
            out[sym] = Quote(
                symbol=sym, price=close,
                open=float(last["open"]), high=float(last["high"]),
                low=float(last["low"]), prev_close=close,
                volume=float(last["volume"]), amount=float(last["amount"]),
                ts=now_ts,
                source=CACHE_QUOTE_SOURCE,
                fetched_at=now_ts,
                stale_reason="实时行情不可用，使用本地日线收盘价",
            )
        if out:
            fallback_count = sum(1 for q in out.values() if q.source == "cache_daily_close")
            if fallback_count:
                log.debug("using cached daily-close as fallback quotes for {} symbols", fallback_count)
        return out

    # ---------------- 新闻 ----------------

    def get_news(self, symbol: str | None = None, *, limit: int = 20) -> list[NewsItem]:
        for adapter in self.adapters:
            try:
                items = adapter.fetch_news(symbol, limit=limit)
                if items:
                    return items
            except Exception as e:
                log.warning("{} news fetch failed: {}", adapter.name, e)
        return []

    # ---------------- 工具 ----------------

    @staticmethod
    def latest_close(df: pd.DataFrame) -> float | None:
        if df.empty:
            return None
        return float(df.iloc[-1]["close"])


def _resample_minute_bars(
    df: pd.DataFrame,
    *,
    symbol: str,
    adjust: AdjustMode,
    target_minutes: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"])
    source = df.copy()
    source["date"] = pd.to_datetime(source["date"])
    source = source.sort_values("date").set_index("date")
    rule = f"{target_minutes}min"
    out = source.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    if out.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"])
    out["symbol"] = normalize_symbol(symbol)
    out["adjust"] = adjust
    return out[["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]]


def _mark_bars(
    df: pd.DataFrame,
    *,
    status: str,
    source: str,
    warning: str | None = None,
) -> pd.DataFrame:
    df.attrs[BAR_STATUS_ATTR] = status
    df.attrs[BAR_SOURCE_ATTR] = source
    if warning:
        df.attrs[BAR_WARNING_ATTR] = warning
    else:
        df.attrs.pop(BAR_WARNING_ATTR, None)
    return df


@lru_cache(maxsize=1)
def get_loader() -> DataLoader:
    return DataLoader()
