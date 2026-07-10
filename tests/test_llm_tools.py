"""LLM 工具层测试：用本地缓存数据验证计算正确，不依赖网络。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from qtrade.data import cache as bar_cache
from qtrade.strategies.llm import tools

SYM = "000001.SZ"


def _seed_cache(prices: list[float]) -> None:
    n = len(prices)
    dates = pd.date_range(end=date.today(), periods=n, freq="D")
    arr = np.asarray(prices, dtype=float)
    df = pd.DataFrame({
        "date": dates,
        "open": arr, "high": arr * 1.005, "low": arr * 0.995,
        "close": arr, "volume": np.full(n, 1e7), "amount": arr * 1e7,
        "symbol": SYM, "adjust": "qfq",
    })
    bar_cache.save_bars(df, SYM)


def test_multi_period_returns_basic():
    _seed_cache(list(np.linspace(10.0, 12.0, 300)))
    out = tools.get_multi_period_returns(SYM)
    assert "1d" in out and "20d" in out and "60d" in out and "252d" in out
    # 单调上行，所有窗口收益都应为正
    assert all(v > 0 for v in out.values())


def test_multi_period_returns_no_data_returns_empty(monkeypatch):
    """无缓存 + 网络拉不到 → 应该返回 {}，不抛。"""
    sym = "999999.SH"
    # loader 必然失败（没数据 + 没适配器能给）
    monkeypatch.setattr(tools, "get_loader", lambda: type("L", (), {
        "get_daily_bars": lambda s, sym, start, end: pd.DataFrame()
    })())
    assert tools.get_multi_period_returns(sym) == {}


def test_volatility_and_drawdown():
    """构造单调下跌 → MDD 应等于总跌幅。"""
    prices = list(np.linspace(20, 10, 100))   # 跌 50%
    _seed_cache(prices)
    out = tools.get_volatility_metrics(SYM)
    assert out["annual_volatility"] > 0
    assert out["max_drawdown"] < 0
    # 最低点正是末日，cur_dd ≈ mdd
    assert abs(out["current_drawdown"] - out["max_drawdown"]) < 1e-6


def test_volatility_too_few_bars_returns_empty():
    _seed_cache([10.0, 10.1])
    assert tools.get_volatility_metrics(SYM) == {}


def test_prepare_research_data_returns_partial_on_failures(monkeypatch):
    """即使 vs_index / fundamentals 都失败，多周期收益和波动率还是要有。"""
    _seed_cache(list(np.linspace(10.0, 12.0, 300)))
    monkeypatch.setattr(tools, "get_relative_to_index", lambda *a, **k: {})
    monkeypatch.setattr(tools, "get_fundamentals", lambda *a, **k: {})

    out = tools.prepare_research_data(SYM)
    assert out["symbol"] == SYM
    assert out["multi_period_returns"]
    assert out["volatility"]
    assert out["vs_csi300"] == {}
    assert out["fundamentals"] == {}
