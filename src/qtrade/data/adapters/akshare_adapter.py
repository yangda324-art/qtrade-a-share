"""akshare 适配器。

为什么默认用 akshare：
- 完全免费，不用 token
- A 股日线、新闻、财务、指数都覆盖
- 缺点是接口偶尔会变、限流不固定，所以加重试

如果你后面接入 Tushare Pro，复制本文件改成 ``tushare_adapter.py`` 即可，
``DataLoader`` 那一层会自动多源 fallback。
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from qtrade.core.exceptions import DataError
from qtrade.core.logging import get_logger
from qtrade.core.time import utcnow
from qtrade.data.adapters.base import DataAdapter, Quote
from qtrade.data.schema import AdjustMode, NewsItem, normalize_symbol, split_symbol

log = get_logger(__name__)

# akshare 把 adjust 映射到这些字符串
_AK_ADJUST = {"none": "", "qfq": "qfq", "hfq": "hfq"}


class AkshareAdapter(DataAdapter):
    name = "akshare"
    SINGLE_QUOTE_THRESHOLD = 20

    def __init__(self) -> None:
        try:
            import akshare  # noqa: F401  仅检测依赖
        except ImportError as e:
            raise DataError("akshare 未安装：pip install akshare") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: AdjustMode = "qfq",
    ) -> pd.DataFrame:
        import akshare as ak

        sym = normalize_symbol(symbol)
        code, _ = split_symbol(sym)

        log.debug("akshare fetch | symbol={} | {} → {} | adjust={}", sym, start, end, adjust)

        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust=_AK_ADJUST[adjust],
            )
        except Exception as e:
            raise DataError(f"akshare fetch failed for {sym}: {e}") from e

        if df is None or df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]
            )

        # akshare 中文列 → 标准列
        rename_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",        # 单位：手
            "成交额": "amount",        # 单位：元
        }
        df = df.rename(columns=rename_map)
        df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = df["volume"].astype(float) * 100  # 手 → 股
        df["symbol"] = sym
        df["adjust"] = adjust
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: AdjustMode = "qfq",
        period: str = "daily",
    ) -> pd.DataFrame:
        """通用周期 K 线。

        - daily/weekly/monthly：用 ``stock_zh_a_hist``
        - 5min/15min/30min/60min：用 ``stock_zh_a_hist_min_em``
        """
        if period in ("daily", "weekly", "monthly"):
            return self._fetch_period_bars(symbol, start, end, adjust, period)
        if period in ("5min", "15min", "30min", "60min"):
            return self._fetch_minute_bars(symbol, start, end, adjust, period)
        raise DataError(f"akshare 不支持周期 {period}")

    def _fetch_period_bars(self, symbol, start, end, adjust, period) -> pd.DataFrame:
        """daily/weekly/monthly 共用 stock_zh_a_hist。"""
        import akshare as ak

        sym = normalize_symbol(symbol)
        code, _ = split_symbol(sym)

        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period=period,
                start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
                adjust=_AK_ADJUST[adjust],
            )
        except Exception as e:
            raise DataError(f"akshare {period} fetch failed for {sym}: {e}") from e

        if df is None or df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]
            )

        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        })
        df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = df["volume"].astype(float) * 100
        df["symbol"] = sym
        df["adjust"] = adjust
        return df

    def _fetch_minute_bars(self, symbol, start, end, adjust, period) -> pd.DataFrame:
        """5/15/30/60 分钟线。注意 akshare 分钟接口对历史范围有限制（一般近 60 天）。"""
        import akshare as ak

        sym = normalize_symbol(symbol)
        code, _ = split_symbol(sym)
        period_num = period.replace("min", "")     # "5min" -> "5"

        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=code, period=period_num,
                start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                adjust=_AK_ADJUST[adjust],
            )
        except Exception as e:
            raise DataError(f"akshare {period} min fetch failed for {sym}: {e}") from e

        if df is None or df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]
            )

        df = df.rename(columns={
            "时间": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        })
        df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = df["volume"].astype(float) * 100
        df["symbol"] = sym
        df["adjust"] = adjust
        return df

    def fetch_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """批量拉 A 股快照。

        少量标的用东方财富单股报价接口，避免每个 tick 为 1–2 只股票拉全市场 58 页。
        当股票池很大时再回退到全市场快照，然后筛出需要的标的。

        ``stock_zh_a_spot_em()`` 返回所有 A 股的当前快照；筛选用代码 6 位前缀匹配。
        盘外调用会拿到上次收盘的数据。
        """
        import akshare as ak

        if not symbols:
            return {}

        normalized = [normalize_symbol(sym) for sym in symbols]
        if len(normalized) <= self.SINGLE_QUOTE_THRESHOLD:
            out: dict[str, Quote] = {}
            failures: list[str] = []
            for sym in normalized:
                try:
                    quote = self._fetch_single_quote(ak, sym)
                except Exception as e:
                    failures.append(f"{sym}: {e}")
                    log.debug("akshare single quote failed for {}: {}", sym, e)
                    continue
                if quote is not None:
                    out[sym] = quote
            if out:
                return out
            failure_hint = "; ".join(failures[-3:]) if failures else "empty single quote response"
            raise DataError(f"akshare single quote unavailable: {failure_hint}")

        try:
            df = ak.stock_zh_a_spot_em()
        except Exception as e:
            raise DataError(f"akshare snapshot failed: {e}") from e

        if df is None or df.empty:
            return {}

        # 标准化目标代码
        wanted: dict[str, str] = {}     # 6位代码 → 标准化的 symbol
        for sym in symbols:
            full = normalize_symbol(sym)
            code, _ = split_symbol(full)
            wanted[code] = full

        df = df[df["代码"].isin(wanted.keys())].copy()
        if df.empty:
            return {}

        ts = utcnow()
        out: dict[str, Quote] = {}
        for _, row in df.iterrows():
            code = str(row["代码"])
            sym = wanted.get(code)
            if sym is None:
                continue
            try:
                out[sym] = Quote(
                    symbol=sym,
                    price=float(row.get("最新价") or row.get("现价") or 0),
                    open=float(row.get("今开") or 0),
                    high=float(row.get("最高") or 0),
                    low=float(row.get("最低") or 0),
                    prev_close=float(row.get("昨收") or 0),
                    volume=float(row.get("成交量") or 0) * 100,    # 手 → 股
                    amount=float(row.get("成交额") or 0),
                    ts=ts,
                    source=self.name,
                )
            except (TypeError, ValueError) as e:
                log.warning("parse quote failed for {}: {}", code, e)
        return out

    def _fetch_single_quote(self, ak, symbol: str) -> Quote | None:
        """拉单只股票快照。``stock_bid_ask_em`` 返回 item/value 两列。"""
        code, _ = split_symbol(symbol)
        df = ak.stock_bid_ask_em(symbol=code)
        if df is None or df.empty:
            return None

        raw = {str(row["item"]): row["value"] for _, row in df.iterrows()}

        def num(key: str, default: float = 0.0) -> float:
            val = raw.get(key, default)
            if val in ("-", "--", None, ""):
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        price = num("最新")
        if price <= 0:
            return None

        return Quote(
            symbol=symbol,
            price=price,
            open=num("今开", price),
            high=num("最高", price),
            low=num("最低", price),
            prev_close=num("昨收", price),
            volume=num("总手") * 100,
            amount=num("金额"),
            ts=utcnow(),
            source=self.name,
        )

    def fetch_news(self, symbol: str | None = None, *, limit: int = 20) -> list[NewsItem]:
        """akshare 的新闻接口偶发不稳，这里给最简实现，错误时返回空。"""
        import akshare as ak

        items: list[NewsItem] = []
        try:
            if symbol:
                code, _ = split_symbol(normalize_symbol(symbol))
                df = ak.stock_news_em(symbol=code)
            else:
                df = ak.news_economic_baidu()
        except Exception as e:
            log.warning("akshare news fetch failed: {}", e)
            return items

        if df is None or df.empty:
            return items

        for _, row in df.head(limit).iterrows():
            try:
                items.append(
                    NewsItem(
                        symbol=symbol,
                        title=str(row.get("新闻标题") or row.get("title") or ""),
                        content=str(row.get("新闻内容") or row.get("content") or ""),
                        source=str(row.get("文章来源") or row.get("source") or "akshare"),
                        url=str(row.get("新闻链接") or row.get("url") or ""),
                        published_at=pd.to_datetime(
                            row.get("发布时间") or row.get("date") or row.get("发布日期")
                        ).to_pydatetime(),
                    )
                )
            except Exception:
                continue
        return items
