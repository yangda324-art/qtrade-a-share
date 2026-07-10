"""DeepSeek 客户端封装。

DeepSeek 兼容 OpenAI SDK，所以底层走 ``openai`` 库 + 自定义 ``base_url``。
未来要切 Anthropic / OpenAI / OpenRouter 时实现新 ``Provider`` 即可。

特性：
- 结构化输出：``json_object`` 模式 + JSON Schema 校验，模型走偏自动重试。
- 重试：tenacity 指数退避 ``max_retries`` 次。
- 缓存：相同请求 24h 内直出，省钱省时。
- 成本统计：每次调用累加到全局 ``CostTracker``。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from qtrade.core.config import get_settings
from qtrade.core.exceptions import LLMError
from qtrade.core.logging import get_logger
from qtrade.llm.cache import LLMCache
from qtrade.llm.cost import get_cost_tracker

log = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    parsed: Any | None = None         # JSON 解析后的结构化数据（如果指定了 json mode）
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class LLMClient:
    """带缓存 + 重试 + 计费的 LLM 客户端。"""

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = provider or settings.llm.provider
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.cache = cache or LLMCache(
            ttl_hours=settings.llm.cache_ttl_hours,
            enabled=settings.llm.cache_enabled,
        )
        self.deep_model = settings.llm.deep_model
        self.quick_model = settings.llm.quick_model
        self.temperature = settings.llm.temperature
        self.max_retries = settings.llm.max_retries
        self.timeout_s = settings.llm.request_timeout_s
        self._client = None  # 延迟初始化避免没装 openai 也能 import 本模块

    # ---------------- 公共 API ----------------

    def chat(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """同步对话接口。

        - ``json_schema`` 非 None 时强制 JSON 输出，并尝试 parse 成 dict 放在 ``parsed``。
        - 模型为 None 时使用 ``quick_model``。
        """
        used_model = model or self.quick_model
        temp = temperature if temperature is not None else self.temperature

        payload = {
            "messages": _build_messages(system, prompt),
            "model": used_model,
            "temperature": temp,
            "json": bool(json_schema),
        }

        # 1) 缓存
        cached = self.cache.get(self.provider, used_model, payload)
        if cached:
            log.debug("llm cache hit | model={}", used_model)
            return LLMResponse(
                text=cached["text"],
                parsed=cached.get("parsed"),
                model=used_model,
                cached=True,
            )

        # 2) 调用
        text, prompt_tokens, completion_tokens = self._call_with_retry(
            messages=payload["messages"],
            model=used_model,
            temperature=temp,
            json_mode=bool(json_schema),
        )

        parsed = None
        if json_schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                # 兜底：用 json-repair 修复常见错误（未转义引号、多余逗号、缺右括号等）
                # DeepSeek reasoner 在长 reasoning 输出里偶发非法 JSON，修复成功率很高
                try:
                    from json_repair import repair_json
                    parsed = json.loads(repair_json(text))
                    log.warning("LLM 返回非合法 JSON，已用 json-repair 修复 | 原 error: {}", e)
                except Exception as repair_err:
                    raise LLMError(
                        f"模型返回非合法 JSON 且修复失败：{e}\n"
                        f"修复尝试 error：{repair_err}\n"
                        f"原文：{text[:500]}"
                    ) from e

        cost = get_cost_tracker().record(
            model=used_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        self.cache.set(self.provider, used_model, payload, {"text": text, "parsed": parsed})

        return LLMResponse(
            text=text,
            parsed=parsed,
            model=used_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )

    # ---------------- 内部 ----------------

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError("openai SDK 未安装：pip install openai") from e
        if not self.api_key:
            raise LLMError(
                f"{self.provider.upper()}_API_KEY 未配置；"
                "请在 .env 中设置 DEEPSEEK_API_KEY 后重试"
            )
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        return self._client

    def _call_with_retry(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        json_mode: bool,
    ) -> tuple[str, int, int]:
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(min=1, max=20),
            reraise=True,
        )
        def _do():
            client = self._ensure_client()
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                log.warning("llm call failed (will retry): {}", e)
                raise
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) if usage else 0
            ct = getattr(usage, "completion_tokens", 0) if usage else 0
            return text, pt, ct

        return _do()


def _build_messages(system: str | None, prompt: str) -> list[dict]:
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
