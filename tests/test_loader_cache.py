"""数据 loader 的缓存命中逻辑测试。

主要回归保护：
- 不能因为 "请求 start 早于缓存 start"（节假日情况）就强制刷新
- 网络失败时若有缓存应该降级使用，不应直接抛
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from qtrade.core.exceptions import DataError
from qtrade.data import cache as bar_cache
from qtrade.data.adapters.akshare_adapter import AkshareAdapter
from qtrade.data.adapters.base import DataAdapter, Quote
from qtrade.data.adapters.eastmoney_kline_adapter import EastMoneyKlineAdapter
from qtrade.data.adapters.eastmoney_quote_adapter import EastMoneyQuoteAdapter
from qtrade.data.loader import BAR_SOURCE_ATTR, BAR_STATUS_ATTR, BAR_WARNING_ATTR, DataLoader
from qtrade.data.schema import AdjustMode


def _make_df(start: str, n: int, symbol: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=n, freq="B")
    prices = np.linspace(10, 20, n)
    return pd.DataFrame({
        "date": dates,
        "open": prices, "high": prices * 1.01, "low": prices * 0.99,
        "close": prices, "volume": np.full(n, 1e6), "amount": prices * 1e6,
        "symbol": symbol, "adjust": "qfq",
    })


class _CountingAdapter(DataAdapter):
    name = "counting"

    def __init__(self, df: pd.DataFrame, *, fail: bool = False) -> None:
        self.df = df
        self.fail = fail
        self.call_count = 0

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        self.call_count += 1
        if self.fail:
            raise DataError("network down")
        return self.df


class _FailingQuoteAdapter(DataAdapter):
    name = "failing_quote"

    def __init__(self) -> None:
        self.quote_call_count = 0

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        return pd.DataFrame()

    def fetch_quotes(self, symbols):
        self.quote_call_count += 1
        raise DataError("quote endpoint down")


class _PartialQuoteAdapter(DataAdapter):
    name = "partial_quote"

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        return pd.DataFrame()

    def fetch_quotes(self, symbols):
        sym = "000001.SZ"
        if sym not in symbols:
            return {}
        return {
            sym: Quote(
                symbol=sym,
                price=10.0,
                open=10.0,
                high=10.2,
                low=9.9,
                prev_close=9.8,
                volume=1_000_000,
                amount=10_000_000,
                ts=pd.Timestamp("2026-05-26").to_pydatetime(),
                source=self.name,
            )
        }


class _SecondQuoteAdapter(DataAdapter):
    name = "second_quote"

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        return pd.DataFrame()

    def fetch_quotes(self, symbols):
        sym = "600519.SH"
        if sym not in symbols:
            return {}
        return {
            sym: Quote(
                symbol=sym,
                price=1500.0,
                open=1490.0,
                high=1510.0,
                low=1480.0,
                prev_close=1495.0,
                volume=2_000_000,
                amount=3_000_000_000,
                ts=pd.Timestamp("2026-05-26").to_pydatetime(),
                source=self.name,
            )
        }


class _MinuteFallbackAdapter(DataAdapter):
    name = "minute_fallback"

    def __init__(self, five_min_df: pd.DataFrame, *, fail_5min: bool = False) -> None:
        self.five_min_df = five_min_df
        self.fail_5min = fail_5min
        self.calls: list[str] = []

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        return pd.DataFrame()

    def fetch_bars(self, symbol, start, end, adjust: AdjustMode = "qfq", period: str = "daily"):
        self.calls.append(period)
        if period == "15min":
            raise DataError("15min endpoint down")
        if period == "5min":
            if self.fail_5min:
                raise DataError("5min endpoint down")
            return self.five_min_df
        return pd.DataFrame()


def test_cache_hit_when_start_is_holiday(tmp_path):
    """缓存第一天 = 2024-01-02，请求 start = 2024-01-01（元旦休市）应直接命中缓存。"""
    sym = "000001.SZ"
    bar_cache.save_bars(_make_df("2024-01-02", 200, sym), sym)

    adapter = _CountingAdapter(pd.DataFrame())
    loader = DataLoader(adapters=[adapter])
    df = loader.get_daily_bars(sym, start="2024-01-01", end=date(2024, 9, 30))
    assert not df.empty
    assert adapter.call_count == 0, "节假日不应触发刷新"


def test_network_failure_falls_back_to_cache(tmp_path):
    """缓存有数据 + 网络失败 + 请求新日期 → 用缓存数据，不抛异常。"""
    sym = "600519.SH"
    bar_cache.save_bars(_make_df("2024-01-02", 100, sym), sym)

    adapter = _CountingAdapter(pd.DataFrame(), fail=True)
    loader = DataLoader(adapters=[adapter])
    # 请求一个远超缓存的 end，触发刷新；网络失败但有缓存 → 应降级
    df = loader.get_daily_bars(sym, start="2024-01-02", end=date(2026, 12, 31))
    assert not df.empty
    assert adapter.call_count >= 1
    assert df.attrs[BAR_STATUS_ATTR] == "cache_fallback"
    assert df.attrs[BAR_SOURCE_ATTR] == "local_cache"
    assert "network down" in df.attrs[BAR_WARNING_ATTR]


def test_network_failure_without_cache_raises(tmp_path):
    """没缓存 + 网络失败 → 必须抛错（用户得知道）。"""
    sym = "300750.SZ"
    adapter = _CountingAdapter(pd.DataFrame(), fail=True)
    loader = DataLoader(adapters=[adapter])
    with pytest.raises(DataError):
        loader.get_daily_bars(sym, start="2024-01-01", end=date(2024, 9, 30))


def test_get_quotes_falls_back_to_cached_close(tmp_path):
    """实时接口失败时，应该用本地最新一根 K 线的收盘价拼一个 Quote 返回。"""
    sym = "000001.SZ"
    bar_cache.save_bars(_make_df("2024-01-02", 100, sym), sym)

    # adapter 的 fetch_quotes 默认是空（base 类的实现），等价"实时不可用"
    adapter = _CountingAdapter(pd.DataFrame())
    loader = DataLoader(adapters=[adapter])
    quotes = loader.get_quotes([sym])
    assert sym in quotes
    assert quotes[sym].price > 0
    # 来源应该是缓存的最后一根，价格应该等于线性序列末端
    assert abs(quotes[sym].price - 20.0) < 1e-6
    assert quotes[sym].source == "cache_daily_close"
    assert not quotes[sym].is_tradeable
    assert quotes[sym].stale_reason


def test_get_quotes_empty_when_no_cache(tmp_path):
    """无缓存 + 实时不可用 → 返回空 dict（runner 自己 skip）。"""
    adapter = _CountingAdapter(pd.DataFrame())
    loader = DataLoader(adapters=[adapter])
    assert loader.get_quotes(["999999.SH"]) == {}


def test_akshare_small_quote_uses_single_symbol_endpoint(monkeypatch):
    """少量标的快照应走单股接口，避免每个 tick 拉全市场。"""
    calls = {"single": 0, "market": 0}

    def fake_single(symbol):
        calls["single"] += 1
        assert symbol == "000001"
        return pd.DataFrame(
            {
                "item": ["最新", "今开", "最高", "最低", "昨收", "总手", "金额"],
                "value": [11.52, 11.48, 11.60, 11.40, 11.50, 12345, 142214400],
            }
        )

    def fake_market():
        calls["market"] += 1
        return pd.DataFrame()

    import akshare as ak

    monkeypatch.setattr(ak, "stock_bid_ask_em", fake_single)
    monkeypatch.setattr(ak, "stock_zh_a_spot_em", fake_market)

    adapter = AkshareAdapter()
    quotes = adapter.fetch_quotes(["000001"])

    assert calls == {"single": 1, "market": 0}
    q = quotes["000001.SZ"]
    assert q.price == 11.52
    assert q.volume == 1_234_500


def test_akshare_small_quote_failure_skips_full_market_and_uses_cache(monkeypatch):
    """单股实时接口失败时不应退到全市场 58 页抓取，loader 应快速用缓存收盘价降级。"""
    sym = "000001.SZ"
    bar_cache.save_bars(_make_df("2024-01-02", 100, sym), sym)
    calls = {"single": 0, "market": 0}

    def fake_single(symbol):
        calls["single"] += 1
        assert symbol == "000001"
        raise RuntimeError("ssl eof")

    def fake_market():
        calls["market"] += 1
        raise AssertionError("small quote failure should not fetch full market snapshot")

    import akshare as ak

    monkeypatch.setattr(ak, "stock_bid_ask_em", fake_single)
    monkeypatch.setattr(ak, "stock_zh_a_spot_em", fake_market)

    loader = DataLoader(adapters=[AkshareAdapter()])
    quotes = loader.get_quotes([sym])

    assert calls == {"single": 1, "market": 0}
    assert quotes[sym].price == 20.0


def test_quote_failure_enters_cooldown_and_uses_cache(tmp_path):
    """实时行情源失败后短时间内不应重复打外部接口，避免 UI 刷屏和卡顿。"""
    sym = "000001.SZ"
    bar_cache.save_bars(_make_df("2024-01-02", 100, sym), sym)

    adapter = _FailingQuoteAdapter()
    loader = DataLoader(adapters=[adapter])

    first = loader.get_quotes([sym])
    second = loader.get_quotes([sym])

    assert first[sym].price == 20.0
    assert second[sym].price == 20.0
    assert adapter.quote_call_count == 1


def test_get_quotes_merges_multiple_realtime_adapters():
    """第一个实时源只返回部分标的时，应继续用下一个实时源补齐。"""
    loader = DataLoader(adapters=[_PartialQuoteAdapter(), _SecondQuoteAdapter()])

    quotes = loader.get_quotes(["000001", "600519"])

    assert quotes["000001.SZ"].source == "partial_quote"
    assert quotes["600519.SH"].source == "second_quote"
    assert quotes["600519.SH"].price == 1500.0


def test_15min_bars_fallback_to_resampled_5min(tmp_path):
    """15 分钟接口失败时，应能用 5 分钟 K 线重采样补齐。"""
    sym = "000001.SZ"
    dates = pd.date_range("2026-06-23 09:35", periods=6, freq="5min")
    prices = np.array([10, 11, 12, 13, 14, 15], dtype=float)
    five_min = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 0.2,
            "low": prices - 0.2,
            "close": prices + 0.1,
            "volume": np.arange(1, 7, dtype=float) * 100,
            "amount": np.arange(1, 7, dtype=float) * 1000,
            "symbol": sym,
            "adjust": "qfq",
        }
    )
    adapter = _MinuteFallbackAdapter(five_min)
    loader = DataLoader(adapters=[adapter])

    df = loader.get_bars(sym, start=date(2026, 6, 23), end=date(2026, 6, 24), period="15min")

    assert adapter.calls == ["15min", "5min"]
    assert df.attrs[BAR_STATUS_ATTR] == "derived_15min_from_5min"
    assert df.attrs[BAR_SOURCE_ATTR] == "local_resample"
    assert len(df) == 2
    assert list(df["date"]) == [pd.Timestamp("2026-06-23 09:45"), pd.Timestamp("2026-06-23 10:00")]
    assert list(df["open"]) == [10, 13]
    assert list(df["close"]) == [12.1, 15.1]
    assert list(df["high"]) == [12.2, 15.2]
    assert list(df["low"]) == [9.8, 12.8]
    assert list(df["volume"]) == [600.0, 1500.0]
    cached = bar_cache.load_bars(sym, start=date(2026, 6, 23), end=date(2026, 6, 24), period="15min")
    assert len(cached) == 2


def test_15min_fallback_failure_preserves_original_15min_error(tmp_path):
    sym = "000001.SZ"
    adapter = _MinuteFallbackAdapter(pd.DataFrame(), fail_5min=True)
    loader = DataLoader(adapters=[adapter])

    with pytest.raises(DataError, match="15min endpoint down"):
        loader.get_bars(sym, start=date(2026, 6, 23), end=date(2026, 6, 24), period="15min")


def test_minute_bars_skip_quote_only_adapter_and_report_source_error(tmp_path):
    sym = "000001.SZ"

    class _FailingMinuteAdapter(DataAdapter):
        name = "failing_minute"

        def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
            return pd.DataFrame()

        def fetch_bars(self, symbol, start, end, adjust: AdjustMode = "qfq", period: str = "daily"):
            raise DataError("minute endpoint down")

    loader = DataLoader(adapters=[_FailingMinuteAdapter(), EastMoneyQuoteAdapter()])

    with pytest.raises(DataError) as exc:
        loader.get_bars(sym, start=date(2026, 6, 23), end=date(2026, 6, 23), period="5min")

    message = str(exc.value)
    assert "failing_minute: minute endpoint down" in message
    assert "eastmoney_quote 不支持周期" not in message


def test_minute_cache_end_date_includes_same_day_intraday_bars(tmp_path):
    sym = "000001.SZ"
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-23 09:35", "2026-06-23 15:00"]),
            "open": [10.0, 10.5],
            "high": [10.2, 10.7],
            "low": [9.9, 10.4],
            "close": [10.1, 10.6],
            "volume": [1000.0, 1200.0],
            "amount": [10_000.0, 12_000.0],
            "symbol": sym,
            "adjust": "qfq",
        }
    )

    bar_cache.save_bars(df, sym, period="5min")

    loaded = bar_cache.load_bars(sym, start=date(2026, 6, 23), end=date(2026, 6, 23), period="5min")

    assert len(loaded) == 2
    assert loaded["date"].max() == pd.Timestamp("2026-06-23 15:00")


def test_eastmoney_kline_adapter_parses_minute_response(monkeypatch):
    calls = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "klines": [
                        "2026-06-23 09:35,10.00,10.20,10.30,9.90,123,123456",
                        "2026-06-23 09:40,10.20,10.10,10.40,10.00,456,456789",
                    ]
                }
            }

    def fake_get(self, url, *, params, headers, timeout):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("requests.Session.get", fake_get)

    adapter = EastMoneyKlineAdapter(timeout_s=3)
    df = adapter.fetch_bars("000001", date(2026, 6, 23), date(2026, 6, 23), period="5min")

    assert calls["params"]["secid"] == "0.000001"
    assert calls["params"]["klt"] == "5"
    assert calls["params"]["fqt"] == "1"
    assert calls["params"]["beg"] == "0"
    assert calls["params"]["end"] == "20500000"
    assert calls["timeout"] == 3
    assert len(df) == 2
    assert list(df["date"]) == [pd.Timestamp("2026-06-23 09:35"), pd.Timestamp("2026-06-23 09:40")]
    assert list(df["open"]) == [10.0, 10.2]
    assert list(df["close"]) == [10.2, 10.1]
    assert list(df["high"]) == [10.3, 10.4]
    assert list(df["low"]) == [9.9, 10.0]
    assert list(df["volume"]) == [12_300.0, 45_600.0]
    assert list(df["amount"]) == [123_456.0, 456_789.0]
    assert set(df["symbol"]) == {"000001.SZ"}
    assert set(df["adjust"]) == {"qfq"}


def test_data_loader_uses_second_adapter_for_minute_bars(tmp_path):
    sym = "000001.SZ"
    dates = pd.date_range("2026-06-23 09:35", periods=2, freq="5min")
    fallback_df = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.2],
            "high": [10.3, 10.4],
            "low": [9.9, 10.0],
            "close": [10.2, 10.1],
            "volume": [1000.0, 1200.0],
            "amount": [10_000.0, 12_000.0],
            "symbol": sym,
            "adjust": "qfq",
        }
    )

    class _FailingMinuteAdapter(DataAdapter):
        name = "failing_minute"

        def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
            return pd.DataFrame()

        def fetch_bars(self, symbol, start, end, adjust: AdjustMode = "qfq", period: str = "daily"):
            raise DataError("minute endpoint down")

    class _FallbackMinuteAdapter(DataAdapter):
        name = "fallback_minute"

        def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
            return pd.DataFrame()

        def fetch_bars(self, symbol, start, end, adjust: AdjustMode = "qfq", period: str = "daily"):
            assert period == "5min"
            return fallback_df

    loader = DataLoader(adapters=[_FailingMinuteAdapter(), _FallbackMinuteAdapter()])

    df = loader.get_bars(sym, start=date(2026, 6, 23), end=date(2026, 6, 23), period="5min")

    assert len(df) == 2
    assert df.attrs[BAR_STATUS_ATTR] == "refreshed"
    assert df.attrs[BAR_SOURCE_ATTR] == "fallback_minute"
    assert list(df["close"]) == [10.2, 10.1]
    cached = bar_cache.load_bars(sym, start=date(2026, 6, 23), end=date(2026, 6, 23), period="5min")
    assert len(cached) == 2


def test_eastmoney_quote_adapter_parses_push2_response(monkeypatch):
    calls = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "diff": [
                        {
                            "f12": "000001",
                            "f2": 11.52,
                            "f17": 11.48,
                            "f15": 11.60,
                            "f16": 11.40,
                            "f18": 11.50,
                            "f5": 12345,
                            "f6": 142214400,
                        },
                        {
                            "f12": "600519",
                            "f2": 1500.0,
                            "f17": 1490.0,
                            "f15": 1510.0,
                            "f16": 1480.0,
                            "f18": 1495.0,
                            "f5": 20000,
                            "f6": 3000000000,
                        },
                    ]
                }
            }

    def fake_get(self, url, *, params, headers, timeout):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("requests.Session.get", fake_get)

    adapter = EastMoneyQuoteAdapter(timeout_s=3)
    quotes = adapter.fetch_quotes(["000001", "600519"])

    assert calls["params"]["secids"] == "0.000001,1.600519"
    assert calls["timeout"] == 3
    assert quotes["000001.SZ"].source == "eastmoney_quote"
    assert quotes["000001.SZ"].price == 11.52
    assert quotes["000001.SZ"].volume == 1_234_500
    assert quotes["600519.SH"].price == 1500.0
