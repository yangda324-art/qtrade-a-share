"""Prompt 版本注册表。

为什么需要版本化：
- LLM 决策质量很大程度上取决于 prompt 设计
- 想验证"换 prompt 是不是真的让模型决策更好"，需要 A/B 对比
- prompt 版本应该可追溯（落库的 AgentDecision.trace 里要记录用了哪个版本）

约定：
- 每个版本是一组完整的 5 个 prompts（technical / sentiment / bull / bear / portfolio）
- 同一版本内 prompts 共用 ``SYSTEM`` 文案
- 版本号用 v1 / v2 / v3 命名，描述写在 ``description`` 里

调用：
    from qtrade.strategies.llm.prompt_registry import get_prompt_version
    pv = get_prompt_version("v2")
    technical_prompt = pv.technical_analyst.format(...)
"""
from __future__ import annotations

from dataclasses import dataclass

from qtrade.core.exceptions import QtradeError


@dataclass(frozen=True)
class PromptVersion:
    """一组 prompts 的快照。"""
    version: str
    description: str
    system: str
    technical_analyst: str
    sentiment_analyst: str
    bull_researcher: str
    bear_researcher: str
    portfolio_manager: str


_REGISTRY: dict[str, PromptVersion] = {}


def register(pv: PromptVersion) -> None:
    if pv.version in _REGISTRY:
        raise QtradeError(f"prompt version {pv.version} 已注册")
    _REGISTRY[pv.version] = pv


def get_prompt_version(version: str = "v2") -> PromptVersion:
    if version not in _REGISTRY:
        raise QtradeError(f"未知 prompt 版本：{version}（已注册：{list(_REGISTRY)}）")
    return _REGISTRY[version]


def list_versions() -> list[str]:
    return sorted(_REGISTRY.keys())


# ------------------ 自动加载所有内置版本 ------------------
# import 这两个模块就会触发 register()
from qtrade.strategies.llm import prompts_v1, prompts_v2  # noqa: E402,F401
