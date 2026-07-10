"""V1 Prompts —— Phase 2 的基线版本（保留作为 A/B 对照组）。"""
from __future__ import annotations

from qtrade.strategies.llm.prompt_registry import PromptVersion, register
from qtrade.strategies.llm.prompts import (
    BEAR_RESEARCHER,
    BULL_RESEARCHER,
    PORTFOLIO_MANAGER,
    SENTIMENT_ANALYST,
    SYSTEM_BASE,
    TECHNICAL_ANALYST,
)

V1 = PromptVersion(
    version="v1",
    description="Phase 2 基线：单轮辩论、仅看行情摘要 + 新闻列表，无基本面/相对指数等扩展数据",
    system=SYSTEM_BASE,
    technical_analyst=TECHNICAL_ANALYST,
    sentiment_analyst=SENTIMENT_ANALYST,
    bull_researcher=BULL_RESEARCHER,
    bear_researcher=BEAR_RESEARCHER,
    portfolio_manager=PORTFOLIO_MANAGER,
)

register(V1)
