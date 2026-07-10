"""LLM 多 Agent 决策图测试（用 mock 客户端，不需要真实 API key）。

覆盖：
- 5 个 Agent 节点都能被调到，最终落到 portfolio_manager
- trace 字段正确累积（多空两个并行节点不会互相覆盖）
- LLM 全部失败时，portfolio_manager 应返回安全的 HOLD 兜底
- 决策落 SQLite agent_decisions 表，token / cost 是本次差值不是全局总数
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from qtrade.core.exceptions import LLMError
from qtrade.core.time import utcnow
from qtrade.execution.account import Account
from qtrade.llm.client import LLMResponse
from qtrade.persistence import init_db, session_scope
from qtrade.persistence.models import AgentDecision, DecisionAction
from qtrade.strategies.base import StrategyContext
from qtrade.strategies.llm.graph import LLMMultiAgentStrategy

SYM = "000001.SZ"


# -------------------- Mock client --------------------


class _MockLLMClient:
    """按 prompt 关键字返回固定 JSON 输出的假 client。

    用于驱动 LangGraph 全图，不真的调外部 API。
    """

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[dict] = []
        self.deep_model = "deepseek-reasoner"
        self.quick_model = "deepseek-chat"

    def chat(self, *, prompt: str, system: str | None = None,
             model: str | None = None, json_schema: dict | None = None,
             temperature: float | None = None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "model": model})    # 存全文不截断，方便测试匹配

        # 模拟真实 LLMClient：调用时记账到全局 cost tracker，否则 graph 里算不出 tokens_used 差值
        from qtrade.llm.cost import get_cost_tracker
        get_cost_tracker().record(
            model=model or "mock",
            prompt_tokens=120,
            completion_tokens=40,
        )

        # 根据 prompt 关键字路由
        if "技术分析师" in prompt:
            data = self.responses.get("technical", {
                "trend": "bullish", "signals": ["MA5 上穿 MA20"], "confidence": 0.7,
                "reasoning": "近期短期均线上穿长期均线",
            })
        elif "舆情分析师" in prompt:
            data = self.responses.get("sentiment", {
                "sentiment": "neutral", "key_news": [], "confidence": 0.4,
                "reasoning": "无显著新闻",
            })
        elif "多头研究员" in prompt:
            data = self.responses.get("bull", {
                "stance": "bull", "arguments": ["趋势向上", "成交量放大"],
                "confidence": 0.7, "rebuttals": [],
            })
        elif "空头研究员" in prompt:
            data = self.responses.get("bear", {
                "stance": "bear", "arguments": ["估值偏高"], "confidence": 0.4,
                "rebuttals": [],
            })
        elif "组合经理" in prompt:
            data = self.responses.get("portfolio", {
                "action": "BUY", "target_position_pct": 0.2,
                "confidence": 0.7, "reasoning": "技术面 + 多头胜出空头",
                "risks": ["短期波动"],
            })
        else:
            data = {"_error": "unrecognized agent"}

        text = json.dumps(data, ensure_ascii=False)
        return LLMResponse(
            text=text, parsed=data,
            model=model or "mock",
            prompt_tokens=120, completion_tokens=40,
            cost_usd=0.0001, cached=False,
        )


class _FailingLLMClient:
    """模拟 LLM API / JSON 修复失败后抛 LLMError。"""

    deep_model = "deepseek-reasoner"
    quick_model = "deepseek-chat"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, *, prompt: str, system: str | None = None,
             model: str | None = None, json_schema: dict | None = None,
             temperature: float | None = None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "model": model})
        raise LLMError("mock json parse failed")


@pytest.fixture
def mock_llm(monkeypatch):
    client = _MockLLMClient()
    # 同时打补丁两个调用入口
    from qtrade.strategies.llm import agents as agents_mod
    monkeypatch.setattr(agents_mod, "get_llm", lambda: client)
    # 重置全局成本追踪器
    from qtrade.llm import cost as cost_mod
    cost_mod.get_cost_tracker.cache_clear()
    return client


def _ctx() -> StrategyContext:
    n = 80
    # freq="D" 而非 "B"：freq="B" + (end+periods) 在某些 pandas 版本下会返回 != n 项
    dates = pd.date_range(end=date.today() - timedelta(days=1), periods=n, freq="D")
    prices = np.linspace(8.0, 11.0, n)
    df = pd.DataFrame({
        "date": dates,
        "open": prices, "high": prices * 1.01, "low": prices * 0.99,
        "close": prices, "volume": np.full(n, 1e7), "amount": prices * 1e7,
        "symbol": SYM, "adjust": "qfq",
    })
    return StrategyContext(
        ts=datetime.now(),
        symbol=SYM,
        history=df,
        account=Account(initial_cash=Decimal("100000")),
    )


# -------------------- Tests --------------------


def test_graph_invokes_all_5_agents(mock_llm, monkeypatch):
    """全图跑通：5 个 Agent 都被调用，最终决策能产生信号。"""
    init_db()
    # 屏蔽 news loader（避免真的去拉新闻）
    strat = LLMMultiAgentStrategy(universe=[SYM])
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    signals = strat.on_bar(_ctx())

    # 5 个 Agent 各被调一次
    assert len(mock_llm.calls) == 5
    prompts_seen = [c["prompt"] for c in mock_llm.calls]
    assert any("技术分析师" in p for p in prompts_seen)
    assert any("舆情分析师" in p for p in prompts_seen)
    assert any("多头研究员" in p for p in prompts_seen)
    assert any("空头研究员" in p for p in prompts_seen)
    assert any("组合经理" in p for p in prompts_seen)

    # mock 返回 BUY 决策 + 信心 0.7（高于阈值 0.6），应产 BUY 信号
    assert len(signals) == 1
    assert signals[0].action == "BUY"
    assert signals[0].quantity > 0
    assert signals[0].symbol == SYM


def test_decision_persisted_with_correct_tokens(mock_llm, monkeypatch):
    """决策落库；token/cost 是【本次】消耗，不是全局累积。"""
    init_db()
    strat = LLMMultiAgentStrategy(universe=[SYM])
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    # 跑两次，第二次的 token 不应该是第一次的 2 倍（应该是各次的本次差）
    strat.on_bar(_ctx())
    strat.on_bar(_ctx())

    with session_scope() as s:
        rows = s.scalars(select(AgentDecision).order_by(AgentDecision.ts)).all()
        assert len(rows) == 2
        # mock 每次 5 个 agent × 160 tokens = 800
        assert rows[0].total_tokens == 800
        assert rows[1].total_tokens == 800     # ← 关键：每次独立算，不是 1600

        # trace_json 在 Phase 4 升级为 dict（含 prompt_version + debate_rounds + trace）
        # trace 内部应有 5 个 agent 条目
        assert isinstance(rows[0].trace_json, dict)
        assert rows[0].trace_json.get("prompt_version") in {"v1", "v2"}
        assert len(rows[0].trace_json.get("trace", [])) == 5


def test_daily_decision_budget_blocks_llm_before_call(mock_llm, monkeypatch):
    """达到每日决策次数上限后，不再调用 LLM，并落 HOLD 审计记录。"""
    init_db()
    strat = LLMMultiAgentStrategy(
        universe=[SYM],
        params={
            "daily_max_decisions": 1,
            "daily_max_tokens": 0,
            "daily_max_cost_usd": 0,
        },
    )
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    first = strat.on_bar(_ctx())
    second = strat.on_bar(_ctx())

    assert len(first) == 1
    assert second == []
    assert len(mock_llm.calls) == 5
    with session_scope() as s:
        rows = s.scalars(select(AgentDecision).order_by(AgentDecision.ts)).all()
        assert len(rows) == 2
        assert rows[0].total_tokens == 800
        assert rows[1].total_tokens == 0
        assert rows[1].action.value == "HOLD"
        assert rows[1].risk_passed is False
        assert "daily decision budget exceeded" in rows[1].risk_reject_reason
        assert rows[1].trace_json["budget"]["usage"]["decision_count"] == 1


def test_daily_token_budget_blocks_llm_before_call(mock_llm, monkeypatch):
    """达到每日 token 上限后，不调用 LLM。"""
    init_db()
    now = utcnow()
    with session_scope() as s:
        s.add(
            AgentDecision(
                strategy_id="llm_multi_agent_v1",
                symbol=SYM,
                action=DecisionAction.HOLD,
                confidence=0,
                target_position_pct=0,
                reasoning="prior usage",
                trace_json={},
                total_tokens=200,
                total_cost_usd=Decimal("0"),
                ts=now,
            )
        )

    strat = LLMMultiAgentStrategy(
        universe=[SYM],
        params={
            "daily_max_decisions": 0,
            "daily_max_tokens": 100,
            "daily_max_cost_usd": 0,
        },
    )
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    assert strat.on_bar(_ctx()) == []
    assert mock_llm.calls == []
    with session_scope() as s:
        latest = s.scalars(select(AgentDecision).order_by(AgentDecision.ts.desc())).first()
        assert latest.total_tokens == 0
        assert latest.risk_passed is False
        assert "daily token budget exceeded" in latest.risk_reject_reason


def test_low_confidence_no_signal(mock_llm, monkeypatch):
    """组合经理输出低 confidence → 不应产生信号。"""
    init_db()
    mock_llm.responses["portfolio"] = {
        "action": "BUY", "target_position_pct": 0.2, "confidence": 0.3,
        "reasoning": "拿不准", "risks": [],
    }
    strat = LLMMultiAgentStrategy(universe=[SYM])
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    signals = strat.on_bar(_ctx())
    assert signals == []
    with session_scope() as s:
        row = s.scalars(select(AgentDecision)).one()
        assert row.action.value == "HOLD"
        assert row.risk_passed is False
        assert "low_confidence" in row.risk_reject_reason
        assert row.trace_json["original_decision"]["action"] == "BUY"


def test_hold_action_no_signal(mock_llm, monkeypatch):
    """组合经理决定 HOLD → 不下单。"""
    init_db()
    mock_llm.responses["portfolio"] = {
        "action": "HOLD", "target_position_pct": 0.0, "confidence": 0.9,
        "reasoning": "等下一个信号", "risks": [],
    }
    strat = LLMMultiAgentStrategy(universe=[SYM])
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    assert strat.on_bar(_ctx()) == []


def test_portfolio_manager_falls_back_to_hold_on_invalid_output(mock_llm, monkeypatch):
    """组合经理返回非法 action → 兜底成 HOLD，不抛。"""
    init_db()
    mock_llm.responses["portfolio"] = {
        "action": "BUY_EVERYTHING",        # 非法值
        "confidence": 0.99,
    }
    strat = LLMMultiAgentStrategy(universe=[SYM])
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    signals = strat.on_bar(_ctx())
    # 兜底为 HOLD，无信号
    assert signals == []
    # 但决策仍要落库以便审计
    with session_scope() as s:
        rows = s.scalars(select(AgentDecision)).all()
        assert len(rows) == 1
        assert rows[0].action.value == "HOLD"
        assert rows[0].risk_passed is False
        assert "schema" in rows[0].risk_reject_reason


def test_llm_api_failure_persists_blocked_hold(monkeypatch):
    """LLM API / JSON 失败应落 HOLD 审计，而不是静默无信号。"""
    init_db()
    client = _FailingLLMClient()
    from qtrade.strategies.llm import agents as agents_mod
    monkeypatch.setattr(agents_mod, "get_llm", lambda: client)
    strat = LLMMultiAgentStrategy(universe=[SYM])
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    assert strat.on_bar(_ctx()) == []

    with session_scope() as s:
        row = s.scalars(select(AgentDecision)).one()
        assert row.action.value == "HOLD"
        assert row.total_tokens == 0
        assert row.risk_passed is False
        assert "llm_agent_error" in row.risk_reject_reason
        assert "mock json parse failed" in row.risk_reject_reason


def test_graph_exception_persists_blocked_hold(mock_llm, monkeypatch):
    """LangGraph 执行异常应落 HOLD 审计。"""
    init_db()

    class _BrokenGraph:
        def invoke(self, state):
            raise RuntimeError("graph exploded")

    strat = LLMMultiAgentStrategy(universe=[SYM])
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)
    strat._graph = _BrokenGraph()

    assert strat.on_bar(_ctx()) == []

    with session_scope() as s:
        row = s.scalars(select(AgentDecision)).one()
        assert row.action.value == "HOLD"
        assert row.risk_passed is False
        assert "llm_graph_error" in row.risk_reject_reason
        assert "graph exploded" in row.risk_reject_reason


def test_limit_decision_is_persisted_but_not_auto_submitted_by_default(mock_llm, monkeypatch):
    """自动策略默认不执行 LIMIT，避免缺少持久化挂单管理时误成交。"""
    init_db()
    mock_llm.responses["portfolio"] = {
        "action": "BUY",
        "target_position_pct": 0.2,
        "confidence": 0.8,
        "reasoning": "等待回调到限价再买入",
        "risks": ["回调不触发"],
        "execution": {"order_type": "LIMIT", "limit_price": 10.2, "time_in_force": "DAY"},
    }
    strat = LLMMultiAgentStrategy(universe=[SYM])
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    signals = strat.on_bar(_ctx())

    assert signals == []
    with session_scope() as s:
        row = s.scalars(select(AgentDecision)).one()
        decision = row.trace_json["decision"]
        assert decision["order_type"] == "LIMIT"
        assert decision["limit_price"] == 10.2


def test_limit_decision_can_emit_limit_signal_when_enabled(mock_llm, monkeypatch):
    """LiveRunner 会启用该开关；启用后 LIMIT 决策会转成带元数据的 Signal。"""
    init_db()
    mock_llm.responses["portfolio"] = {
        "action": "BUY",
        "target_position_pct": 0.2,
        "confidence": 0.8,
        "reasoning": "等待回调到限价再买入",
        "risks": ["回调不触发"],
        "execution": {"order_type": "LIMIT", "limit_price": 10.2, "time_in_force": "DAY"},
    }
    strat = LLMMultiAgentStrategy(universe=[SYM], params={"allow_limit_orders": True})
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)

    signals = strat.on_bar(_ctx())

    assert len(signals) == 1
    assert signals[0].metadata["order_type"] == "LIMIT"
    assert signals[0].metadata["limit_price"] == 10.2
