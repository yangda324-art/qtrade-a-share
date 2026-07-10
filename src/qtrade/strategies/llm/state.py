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
    evidence: list[str]
    order_type: Literal["MARKET", "LIMIT"]
    limit_price: float | None
    time_in_force: Literal["DAY", "GTC"]
    horizon_days: int
    max_loss_pct: float | None
    take_profit_pct: float | None
    schema_version: str


class AgentState(TypedDict, total=False):
    # ---- 输入（运行时填）----
    symbol: str
    symbol_name: str
    as_of_date: str                  # YYYY-MM-DD
    market_summary: dict[str, Any]   # 行情切片：最近若干 K 线 + 基础指标
    news_items: list[dict]           # 舆情原料
    fundamentals: dict[str, Any]     # 财报 / 基本面（可选）
    decision_memory: dict[str, Any]   # 历史相似决策与结果摘要

    # ---- V2 新增：research_data 拉齐多周期收益、波动率、相对指数等 ----
    research_data: dict[str, Any]
    prompt_version: str              # "v1" 或 "v2"，决定用哪套 prompt

    # ---- Agent 输出 ----
    technical_view: TechnicalView
    sentiment_view: SentimentView
    bull_view: ResearcherView         # 仅最后一轮（向后兼容）
    bear_view: ResearcherView
    decision: PortfolioDecision

    # ---- V2 新增：多轮辩论历史（每轮一条） ----
    bull_history: Annotated[list[ResearcherView], operator.add]
    bear_history: Annotated[list[ResearcherView], operator.add]
    current_round: int               # 当前进行到第几轮（1-indexed）
    max_debate_rounds: int

    # ---- 元数据 ----
    debate_round: int                # 兼容字段（v1 用）
    trace: Annotated[list[dict], operator.add]   # 节点并行写入时按 list 追加
    errors: Annotated[list[str], operator.add]
