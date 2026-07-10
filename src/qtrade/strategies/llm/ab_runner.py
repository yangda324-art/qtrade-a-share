"""Prompt A/B 测试 runner。

用途：在**同一份输入**（同一只票、同一时刻）上跑多个 prompt 版本，对比：
- 决策一致性（action/confidence 是否一致）
- 信心分布（哪个版本更敢做决策）
- token 消耗 + 成本
- 决策理由（人类可读对比）

返回结构化结果，配合 CLI 命令展示。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from qtrade.core.logging import get_logger
from qtrade.execution.account import Account
from qtrade.llm.cost import get_cost_tracker
from qtrade.strategies.base import StrategyContext
from qtrade.strategies.llm.graph import LLMMultiAgentStrategy

log = get_logger(__name__)


@dataclass
class ABRunResult:
    version: str
    debate_rounds: int
    elapsed_s: float
    tokens_used: int
    cost_usd: float
    decision: dict[str, Any]
    trace: list[dict] = field(default_factory=list)
    error: str | None = None


@dataclass
class ABComparison:
    symbol: str
    as_of: datetime
    runs: list[ABRunResult]

    def consistency(self) -> dict:
        """简单一致性指标：所有版本的 action 是否相同 / 平均信心 / 信心方差。"""
        if not self.runs:
            return {}
        actions = [r.decision.get("action", "?") for r in self.runs if not r.error]
        confs = [float(r.decision.get("confidence", 0) or 0) for r in self.runs if not r.error]
        action_agreement = len(set(actions)) == 1
        return {
            "all_agree_on_action": action_agreement,
            "actions": actions,
            "avg_confidence": sum(confs) / len(confs) if confs else 0.0,
            "confidence_spread": (max(confs) - min(confs)) if confs else 0.0,
        }


def run_ab(
    symbol: str,
    history: pd.DataFrame,
    versions: list[str],
    *,
    debate_rounds_map: dict[str, int] | None = None,
    initial_cash: float = 100_000.0,
    persist: bool = False,
) -> ABComparison:
    """对同一份 history 跑多个 prompt 版本。

    ``debate_rounds_map``：{"v2": 2}，未指定的版本按各自默认。
    """
    debate_rounds_map = debate_rounds_map or {}
    ctx = StrategyContext(
        ts=datetime.now(),
        symbol=symbol,
        history=history,
        account=Account(initial_cash=Decimal(str(initial_cash))),
    )

    runs: list[ABRunResult] = []
    for version in versions:
        debate_rounds = debate_rounds_map.get(version, 2 if version == "v2" else 1)
        strat = LLMMultiAgentStrategy(
            universe=[symbol],
            params={"prompt_version": version, "debate_rounds": debate_rounds},
        )
        if not persist:
            strat._persist_decision = lambda *a, **kw: None     # type: ignore

        # 屏蔽 news 拉取，A/B 时所有版本看到的输入要一样
        strat._news_provider = None

        tracker = get_cost_tracker()
        tokens_before = tracker.total_prompt_tokens + tracker.total_completion_tokens
        cost_before = tracker.total_cost_usd

        t0 = datetime.now()
        try:
            strat.on_start()
            # 直接 invoke 拿到 final_state，绕过 on_bar 的下单逻辑
            graph = strat._graph
            from qtrade.strategies.llm.graph import _market_summary_from_history
            state = {
                "symbol": symbol, "symbol_name": symbol,
                "as_of_date": ctx.ts.strftime("%Y-%m-%d"),
                "market_summary": _market_summary_from_history(history, strat.lookback),
                "news_items": [],
                "research_data": {},
                "prompt_version": version,
                "max_debate_rounds": debate_rounds,
                "trace": [], "errors": [],
                "bull_history": [], "bear_history": [],
            }
            final_state = graph.invoke(state)
            elapsed = (datetime.now() - t0).total_seconds()
            tokens = (tracker.total_prompt_tokens + tracker.total_completion_tokens) - tokens_before
            cost = tracker.total_cost_usd - cost_before
            runs.append(ABRunResult(
                version=version,
                debate_rounds=debate_rounds,
                elapsed_s=elapsed,
                tokens_used=tokens,
                cost_usd=cost,
                decision=final_state.get("decision", {}),
                trace=final_state.get("trace", []),
            ))
        except Exception as e:
            log.exception("AB run {} failed", version)
            runs.append(ABRunResult(
                version=version, debate_rounds=debate_rounds,
                elapsed_s=(datetime.now() - t0).total_seconds(),
                tokens_used=0, cost_usd=0.0,
                decision={}, error=str(e),
            ))

    return ABComparison(symbol=symbol, as_of=ctx.ts, runs=runs)
