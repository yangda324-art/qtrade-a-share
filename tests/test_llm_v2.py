"""V2 prompts + 多轮辩论 + A/B runner 测试（mock client，无网络无 API key）。"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from qtrade.execution.account import Account
from qtrade.llm.client import LLMResponse
from qtrade.persistence import init_db
from qtrade.strategies.base import StrategyContext
from qtrade.strategies.llm.graph import LLMMultiAgentStrategy

SYM = "000001.SZ"


class _MockLLMClient:
    """根据 prompt 关键字 + 轮次返回固定 JSON；同时正确记账到全局 cost tracker。"""

    def __init__(self):
        self.calls = []
        self.deep_model = "deepseek-reasoner"
        self.quick_model = "deepseek-chat"

    def chat(self, *, prompt, system=None, model=None, json_schema=None, temperature=None):
        self.calls.append({"prompt": prompt, "model": model})
        from qtrade.llm.cost import get_cost_tracker
        get_cost_tracker().record(model=model or "mock", prompt_tokens=100, completion_tokens=30)

        if "技术分析师" in prompt:
            data = {"trend": "bullish", "signals": ["MA5↑MA20"], "confidence": 0.7,
                    "reasoning": "近期趋势向上，60 日涨幅 +12%"}
        elif "舆情分析师" in prompt:
            data = {"sentiment": "neutral", "key_news": [], "confidence": 0.4,
                    "reasoning": "无显著新闻"}
        elif "多头研究员" in prompt:
            data = {"stance": "bull", "arguments": ["趋势向上", "成交量放大", "相对指数 +5%"],
                    "confidence": 0.7, "rebuttals": ["针对空头估值偏高的反驳：PE 合理"]}
        elif "空头研究员" in prompt:
            data = {"stance": "bear", "arguments": ["估值偏高", "行业逆风"],
                    "confidence": 0.4, "rebuttals": []}
        elif "组合经理" in prompt:
            data = {"action": "BUY", "target_position_pct": 0.2, "confidence": 0.7,
                    "reasoning": "多空都看到 PE 偏高但多头有趋势支撑",
                    "risks": ["短期波动"]}
        else:
            data = {"_error": "unrecognized agent"}

        return LLMResponse(text=json.dumps(data, ensure_ascii=False), parsed=data,
                           model=model or "mock", prompt_tokens=100, completion_tokens=30,
                           cost_usd=0.0001, cached=False)


@pytest.fixture
def mock_llm(monkeypatch):
    client = _MockLLMClient()
    from qtrade.strategies.llm import agents as agents_mod
    monkeypatch.setattr(agents_mod, "get_llm", lambda: client)
    from qtrade.llm import cost as cost_mod
    cost_mod.get_cost_tracker.cache_clear()
    return client


def _ctx() -> StrategyContext:
    n = 80
    dates = pd.date_range(end=date.today() - timedelta(days=1), periods=n, freq="D")
    prices = np.linspace(8.0, 11.0, n)
    df = pd.DataFrame({
        "date": dates,
        "open": prices, "high": prices * 1.01, "low": prices * 0.99,
        "close": prices, "volume": np.full(n, 1e7), "amount": prices * 1e7,
        "symbol": SYM, "adjust": "qfq",
    })
    return StrategyContext(ts=datetime.now(), symbol=SYM, history=df,
                           account=Account(initial_cash=Decimal("100000")))


# -------------------- V2 多轮辩论 --------------------


def test_v2_two_round_debate_calls_bull_bear_twice(mock_llm, monkeypatch):
    """V2 + debate_rounds=2 → bull/bear 各被调 2 次（=2 轮）+ tech/sent/portfolio 各 1 次 = 7 调用。"""
    init_db()
    strat = LLMMultiAgentStrategy(universe=[SYM], params={
        "prompt_version": "v2", "debate_rounds": 2,
    })
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)
    # 屏蔽 prepare_research 的真实数据拉取（它会走 akshare，沙箱里会失败）
    from qtrade.strategies.llm import agents as agents_mod
    monkeypatch.setattr(agents_mod, "prepare_research_node",
                        lambda state: {"research_data": {"multi_period_returns": {"60d": 0.12}}})

    strat.on_bar(_ctx())

    bull_calls = sum(1 for c in mock_llm.calls if "多头研究员" in c["prompt"])
    bear_calls = sum(1 for c in mock_llm.calls if "空头研究员" in c["prompt"])
    assert bull_calls == 2, f"expected 2 bull calls, got {bull_calls}"
    assert bear_calls == 2, f"expected 2 bear calls, got {bear_calls}"
    # tech / sent / portfolio 各 1 次
    assert sum(1 for c in mock_llm.calls if "技术分析师" in c["prompt"]) == 1
    assert sum(1 for c in mock_llm.calls if "舆情分析师" in c["prompt"]) == 1
    assert sum(1 for c in mock_llm.calls if "组合经理" in c["prompt"]) == 1


def test_v2_bull_round_2_sees_bear_round_1(mock_llm, monkeypatch):
    """第 2 轮多头的 prompt 应该包含第 1 轮空头的 arguments。"""
    init_db()
    strat = LLMMultiAgentStrategy(universe=[SYM], params={
        "prompt_version": "v2", "debate_rounds": 2,
    })
    strat.on_start()
    monkeypatch.setattr(strat, "_news_provider", None)
    from qtrade.strategies.llm import agents as agents_mod
    monkeypatch.setattr(agents_mod, "prepare_research_node",
                        lambda state: {"research_data": {}})

    strat.on_bar(_ctx())

    # 找出多头第 2 轮的 prompt（最后一个出现的多头 prompt）
    bull_prompts = [c["prompt"] for c in mock_llm.calls if "多头研究员" in c["prompt"]]
    assert len(bull_prompts) == 2
    round_2 = bull_prompts[1]
    # 应该包含空头第一轮的 arguments（mock 写死了"估值偏高"）
    assert "估值偏高" in round_2 or "bear_history" in round_2


# -------------------- V2 prompt 版本注册 --------------------


def test_prompt_versions_registered():
    from qtrade.strategies.llm.prompt_registry import get_prompt_version, list_versions
    versions = list_versions()
    assert "v1" in versions and "v2" in versions
    v2 = get_prompt_version("v2")
    assert "research_data" in v2.bull_researcher.lower() or "research_data" in v2.bull_researcher
    assert "current_round" in v2.bull_researcher


# -------------------- A/B runner --------------------


def test_ab_runner_compares_two_versions(mock_llm, monkeypatch):
    """run_ab 跑 v1 + v2 各一次，应返回 2 个结果。"""
    init_db()
    from qtrade.strategies.llm import agents as agents_mod
    from qtrade.strategies.llm.ab_runner import run_ab
    monkeypatch.setattr(agents_mod, "prepare_research_node",
                        lambda state: {"research_data": {}})

    n = 80
    dates = pd.date_range(end=date.today() - timedelta(days=1), periods=n, freq="D")
    prices = np.linspace(8.0, 11.0, n)
    df = pd.DataFrame({
        "date": dates,
        "open": prices, "high": prices * 1.01, "low": prices * 0.99,
        "close": prices, "volume": np.full(n, 1e7), "amount": prices * 1e7,
        "symbol": SYM, "adjust": "qfq",
    })

    cmp = run_ab(SYM, df, versions=["v1", "v2"], debate_rounds_map={"v2": 2})
    assert len(cmp.runs) == 2
    assert cmp.runs[0].version == "v1"
    assert cmp.runs[1].version == "v2"
    assert all(r.error is None for r in cmp.runs), [r.error for r in cmp.runs]
    # mock 返回都是 BUY，应该 action 一致
    cons = cmp.consistency()
    assert cons["all_agree_on_action"] is True
    # V2 调用更多 LLM（含多轮辩论），tokens 应大于 V1
    assert cmp.runs[1].tokens_used > cmp.runs[0].tokens_used
