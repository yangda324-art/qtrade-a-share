"""多 Agent 决策的共享状态。

LangGraph 节点之间通过这个 ``TypedDict`` 传递状态。我们故意保持字段扁平、
全部可序列化（JSON-able），方便 checkpoint 和事后回放。

注意 ``trace`` / ``errors`` 两个字段用 ``Annotated[..., operator.add]`` 标注：
多空研究员是并行节点，如果不用 reducer，两个分支返回的 trace 会互相覆盖。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Action = Literal["BUY", "SELL", "HOLD"]


class TechnicalView(TypedDict, total=False):
    trend: Literal["bullish", "bearish", "sideways"]
    signals: list[str]
    confidence: float
    reasoning: str


class SentimentView(TypedDict, total=False):
    sentiment: Literal["positive", "negative", "neutral"]
    key_news: list[str]
    confidence: float
    reasoning: str


class ResearcherView(TypedDict, total=False):
    stance: Literal["bull", "bear"]
    arguments: list[str]
    confidence: float
    rebuttals: list[str]


class PortfolioDecision(TypedDict, total=False):
    action: Action
    target_position_pct: float
    confidence: float
    reasoning: str
    risks: list[str]


class AgentState(TypedDict, total=False):
    # ---- 输入（运行时填）----
    symbol: str
    symbol_name: str
    as_of_date: str                  # YYYY-MM-DD
    market_summary: dict[str, Any]   # 行情切片：最近若干 K 线 + 基础指标
    news_items: list[dict]           # 舆情原料
    fundamentals: dict[str, Any]     # 财报 / 基本面（可选）

    # ---- Agent 输出 ----
    technical_view: TechnicalView
    sentiment_view: SentimentView
    bull_view: ResearcherView
    bear_view: ResearcherView
    decision: PortfolioDecision

    # ---- 元数据 ----
    debate_round: int                # 当前辩论轮数
    trace: Annotated[list[dict], operator.add]   # 节点并行写入时按 list 追加
    errors: Annotated[list[str], operator.add]
