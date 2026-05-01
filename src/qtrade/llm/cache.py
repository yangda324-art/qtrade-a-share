"""LLM 结果缓存。

价值：
- 同样的 prompt 重复调用直接命中本地，省钱省时（pre-prompt + 行情 + 风控规则不变时常常命中）
- 回测/复现实验时可固定 LLM 输出，避免重跑结果漂移

实现：
- 简单的本地 JSON-on-disk，按 (provider, model, prompt_hash) 路由
- 支持 TTL（小时）；过期自动忽略
- 不需要并发安全（单进程足够），并发场景再考虑 SQLite 包一层
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qtrade.core.config import get_settings


def _hash_key(provider: str, model: str, payload: dict) -> str:
    s = json.dumps({"p": provider, "m": model, "x": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class LLMCache:
    ttl_hours: int = 24
    enabled: bool = True

    def __post_init__(self) -> None:
        settings = get_settings()
        self.dir: Path = settings.paths.cache_dir / "llm"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key[:2]}/{key}.json"

    def get(self, provider: str, model: str, payload: dict) -> Any | None:
        if not self.enabled:
            return None
        key = _hash_key(provider, model, payload)
        path = self._path(key)
        if not path.exists():
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if time.time() - obj.get("ts", 0) > self.ttl_hours * 3600:
            return None
        return obj.get("value")

    def set(self, provider: str, model: str, payload: dict, value: Any) -> None:
        if not self.enabled:
            return
        key = _hash_key(provider, model, payload)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ts": time.time(), "value": value}, ensure_ascii=False),
            encoding="utf-8",
        )
