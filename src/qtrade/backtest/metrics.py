"""回测绩效指标。

实现的指标：
- 总收益、年化收益、年化波动率、Sharpe、Sortino、Calmar
- 最大回撤（MDD）、回撤持续天数
- 胜率、盈亏比

口径：
- 无风险利率默认 2%（A 股 10 年期国债大致水平，可调）
- 年化天数 252（A 股交易日数）
- 收益序列基于权益曲线的日 close
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

ANNUAL_DAYS = 252


@dataclass
class Metrics:
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_loss_ratio: float
    n_trades: int
    final_equity: float
    initial_equity: float
    raw: dict = field(default_factory=dict)


def _equity_to_series(equity_curve: list[tuple[datetime, Decimal]]) -> pd.Series:
    if not equity_curve:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([t for t, _ in equity_curve])
    vals = [float(v) for _, v in equity_curve]
    s = pd.Series(vals, index=idx).sort_index()
    # 同一天多 snapshot 时取最后一条
    return s.groupby(s.index.date).last()


def compute_metrics(
    equity_curve: list[tuple[datetime, Decimal]],
    *,
    trade_pnls: list[float] | None = None,
    risk_free: float = 0.02,
) -> Metrics:
    s = _equity_to_series(equity_curve)
    if len(s) < 2:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       float(s.iloc[-1]) if len(s) else 0,
                       float(s.iloc[0]) if len(s) else 0)

    init_eq = float(s.iloc[0])
    final_eq = float(s.iloc[-1])
    total_return = final_eq / init_eq - 1

    rets = s.pct_change().dropna()
    n_days = len(rets)
    annual_return = (1 + total_return) ** (ANNUAL_DAYS / max(n_days, 1)) - 1 if n_days else 0.0
    annual_vol = float(rets.std(ddof=1)) * np.sqrt(ANNUAL_DAYS) if n_days > 1 else 0.0

    excess = annual_return - risk_free
    sharpe = excess / annual_vol if annual_vol > 1e-12 else 0.0

    downside = rets[rets < 0]
    downside_vol = float(downside.std(ddof=1)) * np.sqrt(ANNUAL_DAYS) if len(downside) > 1 else 0.0
    sortino = excess / downside_vol if downside_vol > 1e-12 else 0.0

    # 最大回撤
    running_max = s.cummax()
    drawdown = (s - running_max) / running_max
    mdd = float(drawdown.min()) if len(drawdown) else 0.0

    # 回撤持续天数：从峰值到最低
    if mdd < 0:
        end = drawdown.idxmin()
        peak_idx = s.loc[:end].idxmax()
        # 索引可能是 date 或 timestamp，统一转 datetime 算差
        peak = pd.Timestamp(peak_idx)
        trough = pd.Timestamp(end)
        mdd_days = (trough - peak).days
    else:
        mdd_days = 0

    calmar = annual_return / abs(mdd) if mdd < 0 else 0.0

    if trade_pnls:
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        plr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    else:
        win_rate = 0.0
        plr = 0.0

    return Metrics(
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        max_drawdown_duration_days=int(mdd_days),
        win_rate=win_rate,
        profit_loss_ratio=plr,
        n_trades=len(trade_pnls or []),
        final_equity=final_eq,
        initial_equity=init_eq,
    )
