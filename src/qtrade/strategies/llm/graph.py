"""LangGraph 多 Agent 决策图 + 适配 Strategy 接口。

图结构（v1，简单线性 + 多空并行）::

      ┌──────────────────────┐
      │ technical_analyst    │
      └────────┬─────────────┘
               │
      ┌────────▼─────────────┐
      │ sentiment_analyst    │
      └────────┬─────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  bull_researcher  bear_researcher       (并行)
       │                │
       └───────┬────────┘
               │
      ┌────────▼─────────────┐
      │ portfolio_manager    │  ← 最终决策
      └──────────────────────┘

为什么先做线性版而不是多轮辩论：
- 一轮就够调试整条链路了，多轮辩论 token 翻倍但收益不明显
- 配置 ``debate_rounds > 1`` 时再启用循环边（已留 TODO）
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from qtrade.core.exceptions import LLMError
from qtrade.core.logging import get_logger
from qtrade.persistence import session_scope
from qtrade.persistence.models import AgentDecision, DecisionAction
from qtrade.strategies.base import Signal, Strategy, StrategyContext
from qtrade.strategies.llm.agents import (
    bear_researcher_node,
    bull_researcher_node,
    portfolio_manager_node,
    sentiment_analyst_node,
    technical_analyst_node,
)
from qtrade.strategies.llm.state import AgentState

log = get_logger(__name__)


def build_decision_graph():
    """构建并编译 LangGraph 决策图。"""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as e:
        raise LLMError("langgraph 未安装：pip install langgraph") from e

    g = StateGraph(AgentState)

    g.add_node("technical_analyst", technical_analyst_node)
    g.add_node("sentiment_analyst", sentiment_analyst_node)
    g.add_node("bull_researcher", bull_researcher_node)
    g.add_node("bear_researcher", bear_researcher_node)
    g.add_node("portfolio_manager", portfolio_manager_node)

    g.add_edge(START, "technical_analyst")
    g.add_edge("technical_analyst", "sentiment_analyst")
    # 并行多空辩论：sentiment 完成后 fan-out
    g.add_edge("sentiment_analyst", "bull_researcher")
    g.add_edge("sentiment_analyst", "bear_researcher")
    g.add_edge("bull_researcher", "portfolio_manager")
    g.add_edge("bear_researcher", "portfolio_manager")
    g.add_edge("portfolio_manager", END)

    return g.compile()


# ---------------- 适配 Strategy 接口 ----------------


def _market_summary_from_history(df: pd.DataFrame, lookback: int = 60) -> dict[str, Any]:
    """把最近 lookback 根 K 线压缩成 LLM 可读的紧凑摘要。"""
    if df.empty:
        return {}
    sub = df.tail(lookback)
    closes = sub["close"].tolist()
    return {
        "latest_close": float(closes[-1]),
        "lookback_days": len(sub),
        "return_pct": float((closes[-1] / closes[0] - 1) * 100) if closes[0] else 0.0,
        "avg_volume": float(sub["volume"].mean()),
        "ma5": float(sub["close"].rolling(5).mean().iloc[-1]) if len(sub) >= 5 else None,
        "ma20": float(sub["close"].rolling(20).mean().iloc[-1]) if len(sub) >= 20 else None,
        "ma60": float(sub["close"].rolling(60).mean().iloc[-1]) if len(sub) >= 60 else None,
        "high_60": float(sub["high"].max()),
        "low_60": float(sub["low"].min()),
        "recent_5_closes": closes[-5:],
    }


class LLMMultiAgentStrategy(Strategy):
    id = "llm_multi_agent_v1"

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        super().__init__(params=params, universe=universe)
        self.decision_threshold = float(self.params.get("decision_threshold", 0.6))
        self.position_pct = float(self.params.get("position_pct", 0.2))
        self.lookback = int(self.params.get("lookback", 60))
        self._graph = None
        self._news_provider = None

    def on_start(self) -> None:
        self._graph = build_decision_graph()
        # 懒加载 loader 以拿新闻
        from qtrade.data.loader import get_loader
        self._news_provider = get_loader()

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        if self._graph is None:
            self.on_start()

        df = ctx.history
        if len(df) < 30:
            return []

        market_summary = _market_summary_from_history(df, self.lookback)
        news = self._news_provider.get_news(ctx.symbol, limit=10) if self._news_provider else []

        state: AgentState = {
            "symbol": ctx.symbol,
            "symbol_name": ctx.symbol,            # TODO: 接入名称映射
            "as_of_date": ctx.ts.strftime("%Y-%m-%d"),
            "market_summary": market_summary,
            "news_items": [
                {"title": n.title, "source": n.source, "published_at": n.published_at.isoformat()}
                for n in news
            ],
            "fundamentals": {},
            "trace": [],
            "errors": [],
            "debate_round": 0,
        }

        try:
            final_state: AgentState = self._graph.invoke(state)
        except Exception as e:
            log.exception("decision graph failed: {}", e)
            return []

        decision = final_state.get("decision", {})
        action = decision.get("action", "HOLD")
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        target_pct = float(decision.get("target_position_pct", 0.0) or 0.0)

        # 落库审计
        decision_id = self._persist_decision(ctx, final_state, decision)

        if action == "HOLD" or confidence < self.decision_threshold:
            return []

        price = Decimal(str(df["close"].iloc[-1]))
        target_value = ctx.account.total_assets * Decimal(str(target_pct or self.position_pct))
        target_shares = int(target_value / price // 100) * 100

        pos = ctx.account.positions.get(ctx.symbol)
        if action == "BUY" and target_shares > pos.quantity:
            qty = target_shares - pos.quantity
            return [Signal(
                symbol=ctx.symbol,
                action="BUY",
                quantity=qty,
                target_pct=target_pct,
                confidence=confidence,
                reason=decision.get("reasoning", "")[:300],
                metadata={"decision_id": decision_id},
            )]
        if action == "SELL" and pos.available > 0:
            qty = pos.available if target_pct == 0 else max(0, pos.quantity - target_shares)
            qty = min(qty, pos.available) // 100 * 100
            if qty >= 100:
                return [Signal(
                    symbol=ctx.symbol,
                    action="SELL",
                    quantity=qty,
                    target_pct=target_pct,
                    confidence=confidence,
                    reason=decision.get("reasoning", "")[:300],
                    metadata={"decision_id": decision_id},
                )]
        return []

    # ---------------- 持久化 ----------------

    def _persist_decision(
        self,
        ctx: StrategyContext,
        final_state: AgentState,
        decision: dict,
    ) -> int | None:
        try:
            from qtrade.llm.cost import get_cost_tracker
            tracker = get_cost_tracker()
            with session_scope() as s:
                row = AgentDecision(
                    strategy_id=self.id,
                    symbol=ctx.symbol,
                    action=DecisionAction(decision.get("action", "HOLD")),
                    confidence=float(decision.get("confidence", 0.0) or 0.0),
                    target_position_pct=float(decision.get("target_position_pct", 0.0) or 0.0),
                    reasoning=decision.get("reasoning", "")[:5000],
                    trace_json=final_state.get("trace", []),
                    total_tokens=tracker.total_prompt_tokens + tracker.total_completion_tokens,
                    total_cost_usd=Decimal(str(tracker.total_cost_usd)),
                    risk_passed=True,
                    ts=datetime.utcnow(),
                )
                s.add(row)
                s.flush()
                return row.id
        except Exception as e:
            log.warning("persist decision failed: {}", e)
            return None
