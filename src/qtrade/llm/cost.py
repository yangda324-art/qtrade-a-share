"""token 计费追踪。

简化的费率表（USD per 1M tokens）。这些价格会变，仅作粗略估算用，
真实账单以官方为准。

DeepSeek 价格特别低：``deepseek-chat`` 约 $0.27/1M input、$1.10/1M output（2024 末水平）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock

PRICE_TABLE_USD_PER_1M = {
    "deepseek-chat":     {"input": 0.27,  "output": 1.10},
    "deepseek-reasoner": {"input": 0.55,  "output": 2.19},
    # OpenAI 兼容模型样例
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4o":            {"input": 2.50,  "output": 10.0},
    # Anthropic 样例
    "claude-3-5-sonnet": {"input": 3.00,  "output": 15.0},
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICE_TABLE_USD_PER_1M.get(model)
    if not p:
        return 0.0
    return prompt_tokens / 1_000_000 * p["input"] + completion_tokens / 1_000_000 * p["output"]


@dataclass
class CostTracker:
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    by_model: dict[str, dict] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, *, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        with self._lock:
            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_usd += cost
            entry = self.by_model.setdefault(
                model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
            )
            entry["calls"] += 1
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            entry["cost_usd"] += cost
        return cost

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "by_model": self.by_model,
        }


@lru_cache(maxsize=1)
def get_cost_tracker() -> CostTracker:
    return CostTracker()
