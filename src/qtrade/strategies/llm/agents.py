"""Agent 节点实现（V1 兼容 + V2 增强）。

每个节点都是一个 ``(state) -> partial_state`` 的纯函数。
- V1 版本（``technical_analyst_node`` 等）：保留原签名，向后兼容
- V2 增强：``..._node_v2`` 走新 prompt + 接收 research_data + 多轮辩论上下文
- 路由层（graph.py）按 state["prompt_version"] 决定调哪个

注意：mock 测试通过 monkeypatch ``get_llm`` 注入假 client，所以这里所有 LLM 调用
都通过 ``get_llm()`` 获取，不要用模块级缓存。
"""
from __future__ import annotations

import json
from typing import Any

from qtrade.core.exceptions import LLMError
from qtrade.core.logging import get_logger
from qtrade.llm.client import LLMClient, get_llm
from qtrade.strategies.llm.decision_schema import normalize_decision
from qtrade.strategies.llm.prompt_registry import get_prompt_version
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


def _trace_entry(agent: str, output: Any, *, round_num: int | None = None) -> list[dict]:
    """单条 trace 条目（配合 state.trace 的 add reducer 累加）。"""
    e: dict[str, Any] = {"agent": agent, "output": output}
    if round_num is not None:
        e["round"] = round_num
    return [e]


def _resolve_prompts(state: AgentState):
    """根据 state.prompt_version 拿 PromptVersion 对象。默认 v1（向后兼容）。"""
    version = state.get("prompt_version", "v1")
    return get_prompt_version(version)


# ==================================================================
#  V1 节点（保留，agents 测试和老代码路径都依赖它们的签名）
# ==================================================================


def technical_analyst_node(state: AgentState) -> dict:
    client = get_llm()
    pv = _resolve_prompts(state)
    if pv.version == "v2":
        prompt = pv.technical_analyst.format(
            symbol=state.get("symbol", ""),
            symbol_name=state.get("symbol_name", ""),
            as_of_date=state.get("as_of_date", ""),
            market_summary=json.dumps(state.get("market_summary", {}), ensure_ascii=False),
            multi_period_returns=json.dumps(state.get("research_data", {}).get("multi_period_returns", {}), ensure_ascii=False),
            volatility=json.dumps(state.get("research_data", {}).get("volatility", {}), ensure_ascii=False),
            vs_csi300=json.dumps(state.get("research_data", {}).get("vs_csi300", {}), ensure_ascii=False),
        )
    else:
        prompt = pv.technical_analyst.format(
            symbol=state.get("symbol", ""),
            symbol_name=state.get("symbol_name", ""),
            as_of_date=state.get("as_of_date", ""),
            market_summary=json.dumps(state.get("market_summary", {}), ensure_ascii=False),
        )
    out = _safe_invoke(client, system=pv.system, prompt=prompt,
                       model=client.quick_model, json_schema={})
    return {"technical_view": out, "trace": _trace_entry("technical_analyst", out)}


def sentiment_analyst_node(state: AgentState) -> dict:
    client = get_llm()
    pv = _resolve_prompts(state)
    news = state.get("news_items", [])
    if pv.version == "v2":
        prompt = pv.sentiment_analyst.format(
            symbol=state.get("symbol", ""),
            symbol_name=state.get("symbol_name", ""),
            n_news=len(news),
            news_items=json.dumps(news[:10], ensure_ascii=False),
            fundamentals=json.dumps(state.get("research_data", {}).get("fundamentals", {}), ensure_ascii=False),
        )
    else:
        prompt = pv.sentiment_analyst.format(
            symbol=state.get("symbol", ""),
            symbol_name=state.get("symbol_name", ""),
            n_news=len(news),
            news_items=json.dumps(news[:10], ensure_ascii=False),
        )
    out = _safe_invoke(client, system=pv.system, prompt=prompt,
                       model=client.quick_model, json_schema={})
    return {"sentiment_view": out, "trace": _trace_entry("sentiment_analyst", out)}


def bull_researcher_node(state: AgentState) -> dict:
    client = get_llm()
    pv = _resolve_prompts(state)
    cur_round = state.get("current_round", 1)

    if pv.version == "v2":
        prompt = pv.bull_researcher.format(
            current_round=cur_round,
            max_rounds=state.get("max_debate_rounds", 1),
            technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
            sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
            research_data=json.dumps(state.get("research_data", {}), ensure_ascii=False),
            bear_history=json.dumps(state.get("bear_history", []), ensure_ascii=False),
        )
    else:
        prompt = pv.bull_researcher.format(
            technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
            sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
            bear_arguments=json.dumps(state.get("bear_view", {}).get("arguments", []), ensure_ascii=False),
        )
    out = _safe_invoke(client, system=pv.system, prompt=prompt,
                       model=client.deep_model, json_schema={})
    # V2 同时维护 history（操作 add 累加）和 view（最后一轮）
    update: dict = {
        "bull_view": out,
        "trace": _trace_entry("bull_researcher", out, round_num=cur_round),
    }
    if pv.version == "v2":
        update["bull_history"] = [out]
    return update


def bear_researcher_node(state: AgentState) -> dict:
    client = get_llm()
    pv = _resolve_prompts(state)
    cur_round = state.get("current_round", 1)

    if pv.version == "v2":
        prompt = pv.bear_researcher.format(
            current_round=cur_round,
            max_rounds=state.get("max_debate_rounds", 1),
            technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
            sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
            research_data=json.dumps(state.get("research_data", {}), ensure_ascii=False),
            bull_history=json.dumps(state.get("bull_history", []), ensure_ascii=False),
        )
    else:
        prompt = pv.bear_researcher.format(
            technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
            sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
            bull_arguments=json.dumps(state.get("bull_view", {}).get("arguments", []), ensure_ascii=False),
        )
    out = _safe_invoke(client, system=pv.system, prompt=prompt,
                       model=client.deep_model, json_schema={})
    update: dict = {
        "bear_view": out,
        "trace": _trace_entry("bear_researcher", out, round_num=cur_round),
    }
    if pv.version == "v2":
        update["bear_history"] = [out]
    return update


def portfolio_manager_node(state: AgentState) -> dict:
    client = get_llm()
    pv = _resolve_prompts(state)
    if pv.version == "v2":
        prompt = pv.portfolio_manager.format(
            symbol=state.get("symbol", ""),
            symbol_name=state.get("symbol_name", ""),
            technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
            sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
            research_data=json.dumps(state.get("research_data", {}), ensure_ascii=False),
            decision_memory=json.dumps(state.get("decision_memory", {}), ensure_ascii=False),
            bull_history=json.dumps(state.get("bull_history", []), ensure_ascii=False),
            bear_history=json.dumps(state.get("bear_history", []), ensure_ascii=False),
        )
    else:
        prompt = pv.portfolio_manager.format(
            symbol=state.get("symbol", ""),
            symbol_name=state.get("symbol_name", ""),
            technical_view=json.dumps(state.get("technical_view", {}), ensure_ascii=False),
            sentiment_view=json.dumps(state.get("sentiment_view", {}), ensure_ascii=False),
            decision_memory=json.dumps(state.get("decision_memory", {}), ensure_ascii=False),
            bull_view=json.dumps(state.get("bull_view", {}), ensure_ascii=False),
            bear_view=json.dumps(state.get("bear_view", {}), ensure_ascii=False),
        )
    out = _safe_invoke(client, system=pv.system, prompt=prompt,
                       model=client.deep_model, json_schema={})
    out = normalize_decision(out).to_legacy_dict()
    return {"decision": out, "trace": _trace_entry("portfolio_manager", out)}


# ==================================================================
#  Phase 4 新增：研究数据准备节点（无 LLM 调用，纯数据）
# ==================================================================


def prepare_research_node(state: AgentState) -> dict:
    """拉齐 research_data 给后续 Agent 用。失败容忍。"""
    from qtrade.strategies.llm.tools import prepare_research_data
    sym = state.get("symbol")
    if not sym:
        return {"research_data": {}}
    try:
        data = prepare_research_data(sym)
    except Exception as e:
        log.warning("prepare_research_data failed: {}", e)
        data = {}
    return {"research_data": data, "trace": _trace_entry("prepare_research", data)}
