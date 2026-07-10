"""AI 决策记忆与复盘。

Memory 不是聊天历史，而是交易系统能反复检索的结构化经验：
某次 Agent 为什么决策、之后表现如何、同类决策是否经常失败。
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import desc, select

from qtrade.persistence import init_db, session_scope
from qtrade.persistence.models import (
    AgentDecision,
    DecisionAction,
    DecisionOutcome,
    OrderRecord,
    OrderStatus,
)

MEMORY_SCHEMA_VERSION = "decision_memory_v1"


def record_decision_outcome(
    *,
    decision_id: int,
    exit_price: float | Decimal,
    horizon_days: int,
    entry_price: float | Decimal | None = None,
    price_path: list[float | Decimal] | None = None,
    metrics: dict[str, Any] | None = None,
) -> DecisionOutcome:
    """记录或更新一条决策复盘结果。"""
    init_db()
    with session_scope() as s:
        decision = s.get(AgentDecision, decision_id)
        if decision is None:
            raise ValueError(f"decision not found: {decision_id}")

        entry = Decimal(str(entry_price)) if entry_price is not None else _entry_price_from_orders(s, decision_id)
        exit_px = Decimal(str(exit_price))
        ret = _decision_return_pct(decision.action, entry, exit_px)
        drawdown = _max_drawdown_pct(decision.action, entry, price_path or [])
        success = _is_success(decision.action, ret)

        existing = s.scalars(
            select(DecisionOutcome).where(
                DecisionOutcome.decision_id == decision_id,
                DecisionOutcome.horizon_days == horizon_days,
            )
        ).first()

        payload = {
            "run_id": decision.run_id,
            "strategy_id": decision.strategy_id,
            "symbol": decision.symbol,
            "action": decision.action,
            "horizon_days": horizon_days,
            "entry_price": entry,
            "exit_price": exit_px,
            "return_pct": ret,
            "max_drawdown_pct": drawdown,
            "success": success,
            "metrics_json": metrics or {},
        }
        if existing:
            for key, value in payload.items():
                setattr(existing, key, value)
            s.flush()
            return s.get(DecisionOutcome, existing.id)

        row = DecisionOutcome(decision_id=decision_id, **payload)
        s.add(row)
        s.flush()
        return s.get(DecisionOutcome, row.id)


def build_decision_memory_context(*, symbol: str, limit: int = 5) -> dict[str, Any]:
    """构造给 Agent 使用的短上下文记忆。"""
    init_db()
    with session_scope() as s:
        decisions = s.scalars(
            select(AgentDecision)
            .where(AgentDecision.symbol == symbol)
            .order_by(desc(AgentDecision.ts))
            .limit(limit)
        ).all()

        samples: list[dict[str, Any]] = []
        for d in decisions:
            outcome = s.scalars(
                select(DecisionOutcome)
                .where(DecisionOutcome.decision_id == d.id)
                .order_by(desc(DecisionOutcome.horizon_days))
            ).first()
            samples.append({
                "decision_id": d.id,
                "ts": d.ts.isoformat() if d.ts else None,
                "action": d.action.value,
                "confidence": float(d.confidence),
                "target_position_pct": float(d.target_position_pct or 0),
                "reasoning": (d.reasoning or "")[:300],
                "outcome": _outcome_summary(outcome),
            })

        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "symbol": symbol,
            "samples": samples,
        }


def _entry_price_from_orders(s, decision_id: int) -> Decimal | None:
    order = s.scalars(
        select(OrderRecord)
        .where(
            OrderRecord.decision_id == decision_id,
            OrderRecord.status == OrderStatus.FILLED,
            OrderRecord.avg_fill_price.is_not(None),
        )
        .order_by(OrderRecord.created_at.asc())
    ).first()
    if order is None or order.avg_fill_price is None:
        return None
    return Decimal(str(order.avg_fill_price))


def _decision_return_pct(
    action: DecisionAction,
    entry_price: Decimal | None,
    exit_price: Decimal,
) -> Decimal | None:
    if entry_price is None or entry_price <= 0:
        return None
    raw = (exit_price / entry_price) - Decimal("1")
    if action == DecisionAction.SELL:
        raw = -raw
    if action == DecisionAction.HOLD:
        raw = Decimal("0")
    return raw.quantize(Decimal("0.000001"))


def _max_drawdown_pct(
    action: DecisionAction,
    entry_price: Decimal | None,
    price_path: list[float | Decimal],
) -> Decimal | None:
    if entry_price is None or entry_price <= 0 or not price_path:
        return None
    prices = [Decimal(str(p)) for p in price_path if Decimal(str(p)) > 0]
    if not prices:
        return None
    if action == DecisionAction.BUY:
        worst = min((p / entry_price) - Decimal("1") for p in prices)
    elif action == DecisionAction.SELL:
        worst = min((entry_price / p) - Decimal("1") for p in prices)
    else:
        return Decimal("0.000000")
    return max(Decimal("0"), -worst).quantize(Decimal("0.000001"))


def _is_success(action: DecisionAction, return_pct: Decimal | None) -> bool | None:
    if return_pct is None:
        return None
    if action == DecisionAction.HOLD:
        return None
    return return_pct > 0


def _outcome_summary(outcome: DecisionOutcome | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    return {
        "horizon_days": outcome.horizon_days,
        "return_pct": float(outcome.return_pct) if outcome.return_pct is not None else None,
        "max_drawdown_pct": (
            float(outcome.max_drawdown_pct) if outcome.max_drawdown_pct is not None else None
        ),
        "success": outcome.success,
        "evaluated_at": outcome.evaluated_at.isoformat() if outcome.evaluated_at else None,
    }
