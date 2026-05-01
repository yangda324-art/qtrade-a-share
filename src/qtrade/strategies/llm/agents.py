"""5 个 Agent 节点的实现：技术 / 舆情 / 多头 / 空头 / 组合经理。

每个节点都是一个 ``(state) -> partial_state`` 的纯函数。
LangGraph 自己管 state 合并和路由。
"""
from __future__ import annotations

import json
from typing import Any

from qtrade.core.exceptions import LLMError
from qtrade.core.logging import get_logger
from qtrade.llm.client import LLMClient, get_llm
from qtrade.strategies.llm import prompts
from qtrade.strategies.llm.state import AgentState

log = get_logger(__name__)


def _safe_invoke(client: LLMClient, *, system: str, prompt: str, model: str, json_schema: dict | None) -> dict:
    """调 LLM + 容错：失败时返回带 ``_error`` 字段的 dict 而不是抛。"""
    try:
        resp = client.chat(system=system, prompt=prompt, model=model, json_schema=json_schema)
        if not isinstance(resp.parsed, dict):
            return {"_error": f"non-dict output: {resp.text[:200]}"}
        return resp.parsed
    except LLMError as e:
        log.warning("llm error in agent: {}", e)
        return {"_error": str(e)}


def _trace_entry(agent: str, output: Any) -> list[dict]:
    """返回**仅本节点**新增的 trace 条目（一个 list，配合 state.trace 的 add reducer 累加）。"""
    return [{"agent": agent, "output": output}]


# -------------------- 节点实现 --------------------


def technical_analyst_node(state: AgentState) -> dict:
    client = get_llm()
    prompt = prompts.TECHNICAL_ANALYST.format(
        symbol=state.get("symbol", ""),
        symbol_name=state.get("symbol_name", ""),
        as_of_date=state.get("as_of_date", ""),
        market_summary=json.dumps(state.get("market_summary", {}), ensure_ascii=False),
    )
    out = _safe_invoke(client, system=prompts.SYSTEM_BASE, prompt=prompt,
                       model=client.quick_model, json_schema={})
    return {
        "technical_view": out,
        "trace": _trace_entry("technical_analyst", out),
    }


def sentiment_analyst_node(state: AgentState) -> dict:
    client = get_llm()
    news = state.get("news_items", [])
    prompt = prompts.SENTIMENT_ANALYST.format(
        symbol=state.get("symbol", ""),
        symbol_name=state.get("symbol_name", ""),
        n_news=len(news),
        news_items=json.dumps(news[:10], ensure_ascii=False),
    )
    out = _safe_invoke(client, system=prompts.SYSTEM_BASE, prompt=prompt,
                       model=client.quick_model, json_schema={})
    return {
        "sentiment_view": out,
        "trace": _trace_entry("sentiment_analyst", out),
    }


def bull_researcher_node(state: AgentState) -> dict:
    client = get_llm()
    prompt = prompts.BULL_RESEARCHER.format(
        technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
        sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
        bear_arguments=json.dumps(state.get("bear_view", {}).get("arguments", []), ensure_ascii=False),
    )
    out = _safe_invoke(client, system=prompts.SYSTEM_BASE, prompt=prompt,
                       model=client.deep_model, json_schema={})
    return {
        "bull_view": out,
        "trace": _trace_entry("bull_researcher", out),
    }


def bear_researcher_node(state: AgentState) -> dict:
    client = get_llm()
    prompt = prompts.BEAR_RESEARCHER.format(
        technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
        sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
        bull_arguments=json.dumps(state.get("bull_view", {}).get("arguments", []), ensure_ascii=False),
    )
    out = _safe_invoke(client, system=prompts.SYSTEM_BASE, prompt=prompt,
                       model=client.deep_model, json_schema={})
    return {
        "bear_view": out,
        "trace": _trace_entry("bear_researcher", out),
    }


def portfolio_manager_node(state: AgentState) -> dict:
    client = get_llm()
    prompt = prompts.PORTFOLIO_MANAGER.format(
        symbol=state.get("symbol", ""),
        symbol_name=state.get("symbol_name", ""),
        technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
        sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
        bull_view=json.dumps(state.get("bull_view", {}), ensure_ascii=False),
        bear_view=json.dumps(state.get("bear_view", {}), ensure_ascii=False),
    )
    out = _safe_invoke(client, system=prompts.SYSTEM_BASE, prompt=prompt,
                       model=client.deep_model, json_schema={})
    # 兜底：解析失败给出保守 HOLD
    if "_error" in out or out.get("action") not in {"BUY", "SELL", "HOLD"}:
        out = {
            "action": "HOLD",
            "target_position_pct": 0.0,
            "confidence": 0.0,
            "reasoning": f"决策失败，保守 HOLD：{out.get('_error', 'invalid output')}",
            "risks": ["LLM 决策不可用"],
        }
    return {
        "decision": out,
        "trace": _trace_entry("portfolio_manager", out),
    }
