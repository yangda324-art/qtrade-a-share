"""4 个传统策略的最小烟测：构造能触发条件的合成数据，确认能产 BUY 信号。

不验证赚钱与否（那是回测的事），只验证策略本身的"接到正确条件 → 产正确信号"。
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from qtrade.execution.account import Account
from qtrade.strategies.base import StrategyContext
from qtrade.strategies.factor import FactorThresholdStrategy
from qtrade.strategies.technical import (
    BollingerStrategy,
    MACDStrategy,
    MomentumStrategy,
    TurtleStrategy,
)

SYM = "000001.SZ"


def _ctx_from_prices(prices: list[float] | np.ndarray) -> StrategyContext:
    n = len(prices)
    # 用 freq="D" 而非 "B"：避免某些 pandas 版本 freq="B" + (end+periods) 返回 != n 项
    dates = pd.date_range(end=date.today(), periods=n, freq="D")
    arr = np.asarray(prices, dtype=float)
    df = pd.DataFrame({
        "date": dates,
        "open": arr, "high": arr * 1.005, "low": arr * 0.995,
        "close": arr, "volume": np.full(n, 1e7), "amount": arr * 1e7,
        "symbol": SYM, "adjust": "qfq",
    })
    return StrategyContext(
        ts=datetime.now(), symbol=SYM, history=df,
        account=Account(initial_cash=Decimal("100000")),
    )


# -------------------- MACD --------------------


def test_macd_buys_on_golden_cross():
    """先长期下跌使 DIF<DEA，再持续反弹形成金叉。

    on_bar 只看最后两根 K 线，金叉具体发生在反弹的哪一天不可预测，
    所以滑窗喂给策略：只要序列中某一天能触发金叉买入就算通过。
    """
    n = 60
    down = np.linspace(20, 10, n)
    up = np.linspace(10, 16, 30)            # 后段长反弹保证金叉一定能在中段发生
    prices = np.concatenate([down, up])
    strat = MACDStrategy(universe=[SYM])

    # 从能产指标的最早窗口开始往后逐根喂
    min_required = strat.slow + strat.signal + 2
    saw_buy = False
    for end_idx in range(min_required, len(prices) + 1):
        window = prices[:end_idx]
        sigs = strat.on_bar(_ctx_from_prices(window))
        if any(s.action == "BUY" for s in sigs):
            saw_buy = True
            break
    assert saw_buy, "下跌后持续反弹应至少在某根 K 线触发 MACD 金叉买入"


def test_macd_no_signal_on_flat():
    """平盘无趋势，不应产生信号。"""
    strat = MACDStrategy(universe=[SYM])
    sigs = strat.on_bar(_ctx_from_prices(np.full(80, 10.0)))
    assert sigs == []


# -------------------- Bollinger --------------------


def test_bollinger_buys_on_lower_band_touch():
    """构造价格刚好砸到下轨。"""
    n = 30
    prices = list(np.full(n - 1, 10.0)) + [9.0]   # 末根大跌打到下轨之外
    strat = BollingerStrategy(universe=[SYM], params={"window": 20, "num_std": 2.0})
    sigs = strat.on_bar(_ctx_from_prices(prices))
    assert any(s.action == "BUY" for s in sigs)


# -------------------- Momentum --------------------


def test_momentum_buys_on_strong_rally():
    """近 60 日涨 20% > 阈值 5%。"""
    prices = np.linspace(10, 12, 70)
    strat = MomentumStrategy(universe=[SYM])
    sigs = strat.on_bar(_ctx_from_prices(prices))
    assert any(s.action == "BUY" for s in sigs)


def test_momentum_no_buy_on_weak_rally():
    """涨幅低于阈值不应触发。"""
    prices = np.linspace(10, 10.2, 70)        # 60 日仅涨 2% < 5%
    strat = MomentumStrategy(universe=[SYM])
    sigs = strat.on_bar(_ctx_from_prices(prices))
    assert all(s.action != "BUY" for s in sigs)


# -------------------- Turtle --------------------


def test_turtle_buys_on_n_day_breakout():
    """20 日新高突破。"""
    n = 30
    prices = list(np.full(n - 1, 10.0)) + [12.0]   # 末根创新高
    strat = TurtleStrategy(universe=[SYM])
    sigs = strat.on_bar(_ctx_from_prices(prices))
    assert any(s.action == "BUY" for s in sigs)


def test_turtle_no_buy_below_breakout():
    """未突破不应买。"""
    strat = TurtleStrategy(universe=[SYM])
    sigs = strat.on_bar(_ctx_from_prices(np.full(40, 10.0)))
    assert all(s.action != "BUY" for s in sigs)


# -------------------- Factor threshold --------------------


def test_factor_threshold_buys_on_high_factor_percentile():
    prices = np.concatenate([np.full(55, 10.0), [10.0, 10.0, 10.0, 10.0, 16.0]])
    strat = FactorThresholdStrategy(
        universe=[SYM],
        params={
            "factor_name": "ret_5",
            "windows": [5],
            "lookback": 20,
            "entry_quantile": 0.8,
            "exit_quantile": 0.5,
            "position_pct": 0.3,
        },
    )

    sigs = strat.on_bar(_ctx_from_prices(prices))

    assert any(s.action == "BUY" for s in sigs)


def test_factor_threshold_sells_when_factor_exits_band():
    prices = np.concatenate([np.linspace(10, 16, 30), np.linspace(16, 15, 30)])
    ctx = _ctx_from_prices(prices)
    pos = ctx.account.positions.get(SYM)
    pos.add_buy(1000, Decimal("10"))
    pos.settle_t1()
    strat = FactorThresholdStrategy(
        universe=[SYM],
        params={
            "factor_name": "ret_5",
            "windows": [5],
            "lookback": 20,
            "entry_quantile": 0.8,
            "exit_quantile": 0.5,
        },
    )

    sigs = strat.on_bar(ctx)

    assert any(s.action == "SELL" for s in sigs)
