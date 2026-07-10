"""LLM daily budget guard.

The guard is intentionally deterministic and database-backed. It does not trust the
LLM path to self-throttle; strategy code must ask this module before invoking agents.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func, select

from qtrade.core.time import MARKET_TZ, market_today
from qtrade.persistence import session_scope
from qtrade.persistence.models import AgentDecision


@dataclass(frozen=True)
class LLMUsage:
    decision_count: int
    total_tokens: int
    total_cost_usd: float
    day: date


@dataclass(frozen=True)
class LLMBudgetDecision:
    allowed: bool
    reason: str | None
    usage: LLMUsage


def daily_usage(*, day: date | None = None) -> LLMUsage:
    """Return global LLM decision usage for an A-share market day."""
    d = day or market_today()
    start_utc = _market_day_boundary_utc(d, end=False)
    end_utc = _market_day_boundary_utc(d, end=True)
    with session_scope() as s:
        row = s.execute(
            select(
                func.count(AgentDecision.id),
                func.coalesce(func.sum(AgentDecision.total_tokens), 0),
                func.coalesce(func.sum(AgentDecision.total_cost_usd), Decimal("0")),
            ).where(
                AgentDecision.ts >= start_utc,
                AgentDecision.ts < end_utc,
            )
        ).one()
    return LLMUsage(
        decision_count=int(row[0] or 0),
        total_tokens=int(row[1] or 0),
        total_cost_usd=float(row[2] or 0.0),
        day=d,
    )


def check_budget(
    *,
    max_decisions: int | None,
    max_tokens: int | None,
    max_cost_usd: float | None,
    day: date | None = None,
) -> LLMBudgetDecision:
    """Check whether another LLM decision is allowed today.

    A non-positive limit disables that specific guard.
    """
    usage = daily_usage(day=day)
    if max_decisions and max_decisions > 0 and usage.decision_count >= max_decisions:
        return LLMBudgetDecision(
            allowed=False,
            reason=f"LLM daily decision budget exceeded: {usage.decision_count}/{max_decisions}",
            usage=usage,
        )
    if max_tokens and max_tokens > 0 and usage.total_tokens >= max_tokens:
        return LLMBudgetDecision(
            allowed=False,
            reason=f"LLM daily token budget exceeded: {usage.total_tokens}/{max_tokens}",
            usage=usage,
        )
    if max_cost_usd and max_cost_usd > 0 and usage.total_cost_usd >= max_cost_usd:
        return LLMBudgetDecision(
            allowed=False,
            reason=f"LLM daily cost budget exceeded: ${usage.total_cost_usd:.4f}/${max_cost_usd:.4f}",
            usage=usage,
        )
    return LLMBudgetDecision(allowed=True, reason=None, usage=usage)


def _market_day_boundary_utc(day: date, *, end: bool) -> datetime:
    if end:
        # Use the next local midnight as an exclusive upper bound.
        local = datetime.combine(day + timedelta(days=1), time.min, tzinfo=MARKET_TZ)
    else:
        local = datetime.combine(day, time.min, tzinfo=MARKET_TZ)
    return local.astimezone(timezone.utc).replace(tzinfo=None)
