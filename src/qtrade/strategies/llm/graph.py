"""LangGraph 多 Agent 决策图（V1 + V2 / 多轮辩论）。

V1 图（``build_decision_graph_v1``）—— Phase 2 基线：
    START → technical → sentiment → (bull || bear) → portfolio → END

V2 图（``build_decision_graph_v2``）—— Phase 4 增强：
    START → prepare_research → technical → sentiment
          → (bull_round_1 || bear_round_1)
          → (bull_round_2 || bear_round_2)   if max_debate_rounds >= 2
          → ...
          → portfolio_manager → END

策略层（``LLMMultiAgentStrategy``）按 ``prompt_version`` 选图。
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from qtrade.core.config import get_settings
from qtrade.core.exceptions import LLMError
from qtrade.core.logging import get_logger
from qtrade.core.time import utcnow
from qtrade.llm.budget import LLMBudgetDecision, check_budget
from qtrade.persistence import session_scope
from qtrade.persistence.models import AgentDecision, DecisionAction
from qtrade.strategies.base import Signal, Strategy, StrategyContext
from qtrade.strategies.llm.agents import (
    bear_researcher_node,
    bull_researcher_node,
    portfolio_manager_node,
    prepare_research_node,
    sentiment_analyst_node,
    technical_analyst_node,
)
from qtrade.strategies.llm.decision_schema import normalize_decision, safe_hold
from qtrade.strategies.llm.state import AgentState

log = get_logger(__name__)


def _import_langgraph():
    try:
        from langgraph.graph import END, START, StateGraph
        return StateGraph, START, END
    except ImportError as e:
        raise LLMError("langgraph 未安装：pip install langgraph") from e


def build_decision_graph_v1():
    """V1：单轮辩论。"""
    StateGraph, START, END = _import_langgraph()
    g = StateGraph(AgentState)
    g.add_node("technical_analyst", technical_analyst_node)
    g.add_node("sentiment_analyst", sentiment_analyst_node)
    g.add_node("bull_researcher", bull_researcher_node)
    g.add_node("bear_researcher", bear_researcher_node)
    g.add_node("portfolio_manager", portfolio_manager_node)

    g.add_edge(START, "technical_analyst")
    g.add_edge("technical_analyst", "sentiment_analyst")
    g.add_edge("sentiment_analyst", "bull_researcher")
    g.add_edge("sentiment_analyst", "bear_researcher")
    g.add_edge("bull_researcher", "portfolio_manager")
    g.add_edge("bear_researcher", "portfolio_manager")
    g.add_edge("portfolio_manager", END)
    return g.compile()


def build_decision_graph_v2(max_debate_rounds: int = 2):
    """V2：多轮辩论。每轮 bull / bear 并行，前一轮的论点会作为下一轮输入。

    通过为每一轮注册独立的节点名（``bull_round_1``、``bull_round_2``…）实现循环展开，
    避免 LangGraph 的 cyclic state 复杂度。
    """
    StateGraph, START, END = _import_langgraph()
    g = StateGraph(AgentState)
    g.add_node("prepare_research", prepare_research_node)
    g.add_node("technical_analyst", technical_analyst_node)
    g.add_node("sentiment_analyst", sentiment_analyst_node)
    g.add_node("portfolio_manager", portfolio_manager_node)

    g.add_edge(START, "prepare_research")
    g.add_edge("prepare_research", "technical_analyst")
    g.add_edge("technical_analyst", "sentiment_analyst")

    # 每轮 bull / bear 用独立节点名，闭包捕获 round_num
    prev = "sentiment_analyst"
    for r in range(1, max_debate_rounds + 1):
        bull_name = f"bull_round_{r}"
        bear_name = f"bear_round_{r}"

        # 包装：在调用前把 current_round 设进 state
        def _make_bull(round_num: int):
            def fn(state):
                state2 = {**state, "current_round": round_num,
                          "max_debate_rounds": max_debate_rounds}
                return bull_researcher_node(state2)
            return fn

        def _make_bear(round_num: int):
            def fn(state):
                state2 = {**state, "current_round": round_num,
                          "max_debate_rounds": max_debate_rounds}
                return bear_researcher_node(state2)
            return fn

        g.add_node(bull_name, _make_bull(r))
        g.add_node(bear_name, _make_bear(r))
        g.add_edge(prev, bull_name)
        g.add_edge(prev, bear_name)

        # 下一轮的"输入"是本轮 bull 和 bear 都跑完
        # 用一个 join 节点（passthrough）把两个并行分支汇起来再继续
        join_name = f"join_round_{r}"
        g.add_node(join_name, lambda state: {})    # passthrough
        g.add_edge(bull_name, join_name)
        g.add_edge(bear_name, join_name)
        prev = join_name

    g.add_edge(prev, "portfolio_manager")
    g.add_edge("portfolio_manager", END)
    return g.compile()


def build_decision_graph(version: str = "v1", *, max_debate_rounds: int = 2):
    """对外统一入口。"""
    if version == "v1":
        return build_decision_graph_v1()
    if version == "v2":
        return build_decision_graph_v2(max_debate_rounds=max_debate_rounds)
    raise ValueError(f"unknown graph version: {version}")


# ---------------- 策略适配 ----------------


def _market_summary_from_history(df: pd.DataFrame, lookback: int = 60) -> dict[str, Any]:
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


def _first_trace_error(trace: list) -> str | None:
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        output = entry.get("output")
        if isinstance(output, dict) and output.get("_error"):
            agent = entry.get("agent") or "agent"
            return f"llm_agent_error[{agent}]: {output['_error']}"
    return None


class LLMMultiAgentStrategy(Strategy):
    id = "llm_multi_agent_v1"

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        super().__init__(params=params, universe=universe)
        llm_cfg = get_settings().llm
        # 阈值从 0.6 降到 0.45：实测 5 只票 V1/V2 决策信心普遍在 0.40-0.55，
        # 0.6 太严格导致几乎不下单。0.45 让"温和但有方向"的决策能触发。
        self.decision_threshold = float(self.params.get("decision_threshold", 0.45))
        self.position_pct = float(self.params.get("position_pct", 0.2))
        # 关键：策略层 clamp LLM target_pct，避免被风控拒后白浪费一次决策。
        # 默认值与 risk.yaml 里的 max_pct_per_symbol 对齐（0.30）。
        self.max_position_pct = float(self.params.get("max_position_pct", 0.30))
        self.allow_limit_orders = bool(self.params.get("allow_limit_orders", False))
        self.lookback = int(self.params.get("lookback", 60))
        self.prompt_version = str(self.params.get("prompt_version", "v1"))
        self.debate_rounds = int(self.params.get("debate_rounds", 1))
        self.daily_max_decisions = int(self.params.get("daily_max_decisions", llm_cfg.daily_max_decisions or 0))
        self.daily_max_tokens = int(self.params.get("daily_max_tokens", llm_cfg.daily_max_tokens or 0))
        self.daily_max_cost_usd = float(self.params.get("daily_max_cost_usd", llm_cfg.daily_max_cost_usd or 0.0))
        self._graph = None
        self._news_provider = None
        # 由 LiveRunner.start() 注入；standalone（如 qtrade decide）保持 None
        self.run_id: int | None = None

    def on_start(self) -> None:
        self._graph = build_decision_graph(self.prompt_version, max_debate_rounds=self.debate_rounds)
        from qtrade.data.loader import get_loader
        self._news_provider = get_loader()

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        if self._graph is None:
            self.on_start()

        df = ctx.history
        if len(df) < 30:
            return []

        budget = check_budget(
            max_decisions=self.daily_max_decisions,
            max_tokens=self.daily_max_tokens,
            max_cost_usd=self.daily_max_cost_usd,
        )
        if not budget.allowed:
            self._persist_budget_block(ctx, budget)
            log.warning("{}: {}", ctx.symbol, budget.reason)
            return []

        market_summary = _market_summary_from_history(df, self.lookback)
        news = self._news_provider.get_news(ctx.symbol, limit=10) if self._news_provider else []
        decision_memory = self._memory_context(ctx.symbol)

        state: AgentState = {
            "symbol": ctx.symbol,
            "symbol_name": ctx.symbol,
            "as_of_date": ctx.ts.strftime("%Y-%m-%d"),
            "market_summary": market_summary,
            "news_items": [
                {"title": n.title, "source": n.source, "published_at": n.published_at.isoformat()}
                for n in news
            ],
            "fundamentals": {},
            "decision_memory": decision_memory,
            "research_data": {},
            "prompt_version": self.prompt_version,
            "max_debate_rounds": self.debate_rounds,
            "trace": [],
            "errors": [],
            "bull_history": [],
            "bear_history": [],
            "debate_round": 0,
        }

        from qtrade.llm.cost import get_cost_tracker
        tracker = get_cost_tracker()
        tokens_before = tracker.total_prompt_tokens + tracker.total_completion_tokens
        cost_before = tracker.total_cost_usd

        try:
            final_state: AgentState = self._graph.invoke(state)
        except Exception as e:
            log.exception("decision graph failed: {}", e)
            tokens_used = (tracker.total_prompt_tokens + tracker.total_completion_tokens) - tokens_before
            cost_used = tracker.total_cost_usd - cost_before
            reason = f"llm_graph_error: {type(e).__name__}: {e}"
            self._persist_policy_block(
                ctx,
                reason=reason,
                tokens_used=tokens_used,
                cost_used=cost_used,
                trace=[],
                original_decision={},
            )
            return []

        tokens_used = (tracker.total_prompt_tokens + tracker.total_completion_tokens) - tokens_before
        cost_used = tracker.total_cost_usd - cost_before

        decision = final_state.get("decision", {})
        structured = normalize_decision(decision)
        decision = structured.to_legacy_dict()
        policy_block_reason = self._policy_block_reason(final_state, decision)
        original_decision: dict | None = None
        if policy_block_reason:
            original_decision = dict(decision)
            decision = self._blocked_hold_decision(policy_block_reason, original=original_decision)
        action = decision.get("action", "HOLD")
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        target_pct = float(decision.get("target_position_pct", 0.0) or 0.0)

        if policy_block_reason is None and confidence < self.decision_threshold:
            policy_block_reason = (
                f"low_confidence: {confidence:.2f} < threshold {self.decision_threshold:.2f}"
            )
            original_decision = dict(decision)
            decision = self._blocked_hold_decision(policy_block_reason, original=original_decision)
            action = "HOLD"
            confidence = 0.0
            target_pct = 0.0

        decision_id = self._persist_decision(
            ctx, final_state, decision,
            tokens_used=tokens_used, cost_used=cost_used,
            block_reason=policy_block_reason,
            original_decision=original_decision,
        )

        if policy_block_reason is not None:
            return []

        if decision.get("order_type") == "LIMIT" and not self.allow_limit_orders:
            log.info(
                "{}: LIMIT decision persisted but not auto-submitted; persistent limit manager not enabled",
                ctx.symbol,
            )
            return []

        pos = ctx.account.positions.get(ctx.symbol)
        price = Decimal(str(df["close"].iloc[-1]))
        total_assets = ctx.account.total_assets

        # ----- 关键升级：HOLD 但 target_pct 与当前持仓差距大时，按差额调仓 -----
        # V2 prompts 鼓励 LLM 用 HOLD + target_pct 表达"温和调仓"（如"建议持有 30%"），
        # 我们这里把这种"分级仓位"翻译成 BUY/SELL 信号。
        # 容差 5% 防止反复微调引发 churn。
        REBALANCE_TOLERANCE = 0.05
        if action == "HOLD":
            current_pct = float(pos.market_value / total_assets) if total_assets > 0 else 0.0
            if abs(target_pct - current_pct) < REBALANCE_TOLERANCE:
                return []
            # 差距大 → 把 HOLD 翻译成实际调仓动作
            action = "BUY" if target_pct > current_pct else "SELL"

        # 关键升级 2：把 LLM target_pct clamp 到策略允许的最大单标的仓位。
        # LLM 不知道风控的硬上限，经常给 0.5 甚至 1.0；不 clamp 的话会被风控拒，浪费决策。
        effective_pct = min(target_pct or self.position_pct, self.max_position_pct)
        if effective_pct < (target_pct or 0):
            log.info(
                "{}: LLM target_pct {:.0%} clamped to {:.0%}（风控上限）",
                ctx.symbol, target_pct, effective_pct,
            )

        target_value = total_assets * Decimal(str(effective_pct))
        target_shares = int(target_value / price // 100) * 100

        if action == "BUY" and target_shares > pos.quantity:
            qty = target_shares - pos.quantity
            return [Signal(
                symbol=ctx.symbol, action="BUY", quantity=qty,
                target_pct=target_pct, confidence=confidence,
                reason=decision.get("reasoning", "")[:300],
                metadata=self._signal_metadata(decision_id, decision),
            )]
        if action == "SELL" and pos.available > 0:
            qty = pos.available if target_pct == 0 else max(0, pos.quantity - target_shares)
            qty = min(qty, pos.available) // 100 * 100
            if qty >= 100:
                return [Signal(
                    symbol=ctx.symbol, action="SELL", quantity=qty,
                    target_pct=target_pct, confidence=confidence,
                    reason=decision.get("reasoning", "")[:300],
                    metadata=self._signal_metadata(decision_id, decision),
                )]
        return []

    def _persist_decision(
        self,
        ctx: StrategyContext,
        final_state: AgentState,
        decision: dict,
        *,
        tokens_used: int = 0,
        cost_used: float = 0.0,
        block_reason: str | None = None,
        original_decision: dict | None = None,
    ) -> int | None:
        try:
            structured = normalize_decision(decision)
            decision = structured.to_legacy_dict()
            with session_scope() as s:
                row = AgentDecision(
                    run_id=self.run_id,                           # LiveRunner 注入；standalone 时为 None
                    strategy_id=self.id,
                    symbol=ctx.symbol,
                    action=DecisionAction(decision.get("action", "HOLD")),
                    confidence=float(decision.get("confidence", 0.0) or 0.0),
                    target_position_pct=float(decision.get("target_position_pct", 0.0) or 0.0),
                    reasoning=decision.get("reasoning", "")[:5000],
                    trace_json={
                        "decision_schema_version": structured.schema_version,
                        "prompt_version": self.prompt_version,
                        "debate_rounds": self.debate_rounds,
                        "decision": decision,
                        "original_decision": original_decision,
                        "policy": {
                            "status": "blocked" if block_reason else "passed",
                            "reason": block_reason,
                        },
                        "trace": final_state.get("trace", []),
                    },
                    total_tokens=int(tokens_used),
                    total_cost_usd=Decimal(str(cost_used)),
                    risk_passed=block_reason is None,
                    risk_reject_reason=block_reason[:5000] if block_reason else None,
                    ts=utcnow(),
                )
                s.add(row)
                s.flush()
                return row.id
        except Exception as e:
            log.warning("persist decision failed: {}", e)
            return None

    def _persist_policy_block(
        self,
        ctx: StrategyContext,
        *,
        reason: str,
        tokens_used: int,
        cost_used: float,
        trace: list,
        original_decision: dict,
    ) -> int | None:
        return self._persist_decision(
            ctx,
            {
                "trace": trace,
            },
            self._blocked_hold_decision(reason, original=original_decision),
            tokens_used=tokens_used,
            cost_used=cost_used,
            block_reason=reason,
            original_decision=original_decision,
        )

    def _persist_budget_block(self, ctx: StrategyContext, budget: LLMBudgetDecision) -> int | None:
        """Persist a HOLD audit row when the daily LLM budget blocks a decision."""
        reason = budget.reason or "LLM daily budget exceeded"
        try:
            with session_scope() as s:
                row = AgentDecision(
                    run_id=self.run_id,
                    strategy_id=self.id,
                    symbol=ctx.symbol,
                    action=DecisionAction.HOLD,
                    confidence=0.0,
                    target_position_pct=0.0,
                    reasoning=f"AI 决策预算保护触发，保守 HOLD：{reason}",
                    trace_json={
                        "prompt_version": self.prompt_version,
                        "debate_rounds": self.debate_rounds,
                        "decision": {
                            "action": "HOLD",
                            "target_position_pct": 0.0,
                            "confidence": 0.0,
                            "reasoning": reason,
                            "order_type": "MARKET",
                        },
                        "budget": {
                            "reason": reason,
                            "daily_max_decisions": self.daily_max_decisions,
                            "daily_max_tokens": self.daily_max_tokens,
                            "daily_max_cost_usd": self.daily_max_cost_usd,
                            "usage": {
                                "day": budget.usage.day.isoformat(),
                                "decision_count": budget.usage.decision_count,
                                "total_tokens": budget.usage.total_tokens,
                                "total_cost_usd": budget.usage.total_cost_usd,
                            },
                        },
                        "trace": [],
                    },
                    total_tokens=0,
                    total_cost_usd=Decimal("0"),
                    risk_passed=False,
                    risk_reject_reason=reason[:5000],
                    ts=utcnow(),
                )
                s.add(row)
                s.flush()
                return row.id
        except Exception as e:
            log.warning("persist budget block decision failed: {}", e)
            return None

    @staticmethod
    def _memory_context(symbol: str) -> dict[str, Any]:
        try:
            from qtrade.runtime.decision_memory import build_decision_memory_context
            return build_decision_memory_context(symbol=symbol, limit=5)
        except Exception as e:
            log.debug("decision memory unavailable for {}: {}", symbol, e)
            return {}

    @staticmethod
    def _blocked_hold_decision(reason: str, *, original: dict | None = None) -> dict[str, Any]:
        decision = safe_hold(f"策略保护触发，保守 HOLD：{reason}").to_legacy_dict()
        if original:
            decision["blocked_original_action"] = original.get("action")
            decision["blocked_original_confidence"] = original.get("confidence")
            decision["blocked_original_target_pct"] = original.get("target_position_pct")
        return decision

    @staticmethod
    def _policy_block_reason(final_state: AgentState, decision: dict) -> str | None:
        trace_error = _first_trace_error(final_state.get("trace", []))
        if trace_error:
            return trace_error
        reasoning = str(decision.get("reasoning") or "")
        if reasoning.startswith("决策失败"):
            return reasoning
        if reasoning.startswith("决策 schema 校验失败"):
            return reasoning
        return None

    @staticmethod
    def _signal_metadata(decision_id: int | None, decision: dict) -> dict[str, Any]:
        return {
            "decision_id": decision_id,
            "decision_schema": decision,
            "order_type": decision.get("order_type"),
            "limit_price": decision.get("limit_price"),
            "time_in_force": decision.get("time_in_force"),
        }
