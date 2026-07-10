"""Agent 可用的"工具"——提供决策所需的丰富市场上下文。

设计原则：
- **不调用外部 LLM**：纯数据查询和计算，所有 Agent 共享，每次决策只算一次
- **失败容忍**：每个工具失败时返回空 dict，不抛异常，Agent 看到字段缺失自动调低置信度
- **足够便宜**：尽量用本地缓存的日线计算（多周期收益、波动率），减少 akshare 调用

工具清单：
- ``prepare_research_data(symbol)``  一次性拉齐所有可用数据，返回扁平 dict
   - multi_period_returns：1日/5日/20日/60日/252日 累计收益
   - volatility：年化波动率、最大回撤
   - vs_index：相对沪深 300 的超额收益
   - fundamentals：行业、PE/PB（best-effort）
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from qtrade.core.logging import get_logger
from qtrade.data import cache as bar_cache
from qtrade.data.loader import get_loader
from qtrade.data.schema import normalize_symbol, split_symbol

log = get_logger(__name__)

# 沪深 300 指数代码（akshare）
INDEX_CSI300 = "sh000300"


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ 多周期收益


def get_multi_period_returns(symbol: str) -> dict[str, float]:
    """计算 1/5/20/60/252 日累计收益率。"""
    sym = normalize_symbol(symbol)
    df = bar_cache.load_bars(sym)
    if df.empty:
        try:
            df = get_loader().get_daily_bars(sym, start=date.today() - timedelta(days=400), end=date.today())
        except Exception as e:
            log.warning("multi_period_returns: load failed for {}: {}", sym, e)
            return {}
    if df.empty or len(df) < 2:
        return {}

    df = df.sort_values("date").reset_index(drop=True)
    closes = df["close"].astype(float).values
    last = closes[-1]
    out: dict[str, float] = {}
    for label, n in (("1d", 1), ("5d", 5), ("20d", 20), ("60d", 60), ("252d", 252)):
        if len(closes) > n:
            ref = closes[-1 - n]
            if ref > 0:
                out[label] = float(last / ref - 1)
    return out


# ------------------------------------------------------------------ 波动率 & 回撤


def get_volatility_metrics(symbol: str, *, lookback: int = 252) -> dict[str, float]:
    """年化波动率 + 最大回撤 + 当前距离峰值的回撤。"""
    sym = normalize_symbol(symbol)
    df = bar_cache.load_bars(sym)
    if df.empty or len(df) < 20:
        return {}
    df = df.sort_values("date").tail(lookback)
    closes = df["close"].astype(float).values
    rets = np.diff(closes) / closes[:-1]
    if len(rets) < 5:
        return {}
    annual_vol = float(np.std(rets, ddof=1)) * np.sqrt(252)

    running_max = np.maximum.accumulate(closes)
    dd = (closes - running_max) / running_max
    mdd = float(dd.min())
    cur_dd = float(dd[-1])
    return {
        "annual_volatility": annual_vol,
        "max_drawdown": mdd,
        "current_drawdown": cur_dd,
    }


# ------------------------------------------------------------------ 相对指数


def get_relative_to_index(symbol: str, *, periods: tuple[int, ...] = (20, 60)) -> dict[str, float]:
    """计算相对沪深 300 的超额收益（标的收益 - 指数收益）。"""
    sym = normalize_symbol(symbol)
    out: dict[str, float] = {}

    sym_df = bar_cache.load_bars(sym)
    if sym_df.empty:
        return {}

    # 拉指数（用 akshare 的指数日线接口；缓存路径走 bar_cache 通用逻辑）
    try:
        import akshare as ak
        idx_df = ak.stock_zh_index_daily_em(symbol=INDEX_CSI300)
        if idx_df is None or idx_df.empty:
            return {}
        idx_df = idx_df.rename(columns={"date": "date", "close": "close"})
        idx_df["date"] = pd.to_datetime(idx_df["date"])
    except Exception as e:
        log.warning("relative_to_index: index fetch failed: {}", e)
        return {}

    sym_df = sym_df.sort_values("date").reset_index(drop=True)
    idx_df = idx_df.sort_values("date").reset_index(drop=True)

    sym_closes = sym_df["close"].astype(float).values
    idx_closes = idx_df["close"].astype(float).values

    for n in periods:
        if len(sym_closes) > n and len(idx_closes) > n:
            sym_ret = sym_closes[-1] / sym_closes[-1 - n] - 1
            idx_ret = idx_closes[-1] / idx_closes[-1 - n] - 1
            out[f"alpha_{n}d"] = float(sym_ret - idx_ret)
    return out


# ------------------------------------------------------------------ 基本面（best-effort）


def get_fundamentals(symbol: str) -> dict[str, str | float]:
    """基本面信息：行业、所属板块、市值（如能拿到）。akshare 接口偶发失败，失败返回 {}。"""
    sym = normalize_symbol(symbol)
    code, _ = split_symbol(sym)
    try:
        import akshare as ak
        df = ak.stock_individual_info_em(symbol=code)
        if df is None or df.empty:
            return {}
        # akshare 返回的格式是 item / value 两列
        info = {row["item"]: row["value"] for _, row in df.iterrows()}
        # 标准化几个常用字段
        out: dict[str, str | float] = {}
        for key, std in (("行业", "industry"), ("总市值", "total_mcap"), ("流通市值", "float_mcap"), ("市盈率(动)", "pe_ttm"), ("市净率", "pb")):
            if key in info:
                v = info[key]
                if std in ("total_mcap", "float_mcap", "pe_ttm", "pb"):
                    out[std] = _safe_float(v)
                else:
                    out[std] = str(v)
        return out
    except Exception as e:
        log.warning("fundamentals fetch failed for {}: {}", sym, e)
        return {}


# ------------------------------------------------------------------ 一站式打包


def prepare_research_data(symbol: str) -> dict:
    """一次性拉齐所有 research 数据，返回打包好的 dict 给 Agent。

    所有子调用都失败容忍，缺哪部分就少哪部分。Agent 拿到后该字段是 None / 空就降低置信度。
    """
    out = {
        "symbol": normalize_symbol(symbol),
        "as_of": datetime.now().isoformat(),
        "multi_period_returns": get_multi_period_returns(symbol),
        "volatility": get_volatility_metrics(symbol),
        "vs_csi300": get_relative_to_index(symbol),
        "fundamentals": get_fundamentals(symbol),
    }
    log.info(
        "research data | {} | mp_ret={} vol={} idx={} fund={}",
        out["symbol"],
        bool(out["multi_period_returns"]),
        bool(out["volatility"]),
        bool(out["vs_csi300"]),
        bool(out["fundamentals"]),
    )
    return out
