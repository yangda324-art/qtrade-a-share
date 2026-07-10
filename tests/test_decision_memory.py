"""AI 决策记忆测试。"""
from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select

from qtrade.persistence import init_db, session_scope
from qtrade.persistence.models import (
    AgentDecision,
    DecisionAction,
    DecisionOutcome,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
)
from qtrade.runtime.decision_memory import build_decision_memory_context, record_decision_outcome

SYM = "000001.SZ"


def _seed_decision_with_order(action: DecisionAction = DecisionAction.BUY) -> int:
    init_db()
    with session_scope() as s:
        decision = AgentDecision(
            run_id=None,
            strategy_id="llm_multi_agent_v1",
            symbol=SYM,
            action=action,
            confidence=Decimal("0.7000"),
            target_position_pct=Decimal("0.2000"),
            reasoning="趋势向上，计划试仓",
            trace_json={},
            total_tokens=100,
            total_cost_usd=Decimal("0.001"),
        )
        s.add(decision)
        s.flush()
        order = OrderRecord(
            run_id=None,
            strategy_id="llm_multi_agent_v1",
            client_order_id="COID-test",
            broker_order_id="PB-test",
            symbol=SYM,
            side=OrderSide.BUY if action != DecisionAction.SELL else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1000,
            status=OrderStatus.FILLED,
            filled_quantity=1000,
            avg_fill_price=Decimal("10.0000"),
            decision_id=decision.id,
        )
        s.add(order)
        s.flush()
        return decision.id


def test_record_decision_outcome_from_linked_order():
    decision_id = _seed_decision_with_order()

    outcome = record_decision_outcome(
        decision_id=decision_id,
        exit_price=11.0,
        horizon_days=5,
        price_path=[10.0, 9.5, 10.8, 11.0],
    )

    assert outcome.return_pct == Decimal("0.100000")
    assert outcome.max_drawdown_pct == Decimal("0.050000")
    assert outcome.success is True

    with session_scope() as s:
        rows = s.scalars(select(DecisionOutcome)).all()
        assert len(rows) == 1


def test_memory_context_includes_outcome_summary():
    decision_id = _seed_decision_with_order()
    record_decision_outcome(decision_id=decision_id, exit_price=9.5, horizon_days=3)

    ctx = build_decision_memory_context(symbol=SYM, limit=5)

    assert ctx["schema_version"] == "decision_memory_v1"
    assert ctx["symbol"] == SYM
    assert len(ctx["samples"]) == 1
    sample = ctx["samples"][0]
    assert sample["decision_id"] == decision_id
    assert sample["outcome"]["return_pct"] == -0.05
    assert sample["outcome"]["success"] is False
