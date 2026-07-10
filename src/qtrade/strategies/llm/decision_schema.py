"""结构化交易决策 schema。

LLM 可以参与研究和推理，但进入交易链路前必须被收敛成稳定、可校验、
可审计的数据结构。这个模块负责把不稳定的自然语言/半结构化 JSON
规范化成 ``StructuredDecision``。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

Action = Literal["BUY", "SELL", "HOLD"]
DecisionOrderType = Literal["MARKET", "LIMIT"]
TimeInForce = Literal["DAY", "GTC"]

SCHEMA_VERSION = "decision_schema_v1"


class DecisionExecution(BaseModel):
    """交易执行意图。"""

    order_type: DecisionOrderType = "MARKET"
    limit_price: float | None = None
    time_in_force: TimeInForce = "DAY"

    @field_validator("order_type", "time_in_force", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.upper() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _limit_requires_price(self):
        if self.order_type == "LIMIT" and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("LIMIT decision requires positive limit_price")
        if self.order_type == "MARKET":
            self.limit_price = None
        return self


class StructuredDecision(BaseModel):
    """进入策略层和审计层的标准决策。"""

    schema_version: str = SCHEMA_VERSION
    action: Action
    target_position_pct: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    execution: DecisionExecution = Field(default_factory=DecisionExecution)
    horizon_days: int = Field(default=5, ge=1, le=252)
    max_loss_pct: float | None = Field(default=None, ge=0.0, le=1.0)
    take_profit_pct: float | None = Field(default=None, ge=0.0, le=5.0)

    @field_validator("action", mode="before")
    @classmethod
    def _action_upper(cls, v):
        return v.upper() if isinstance(v, str) else v

    @field_validator("risks", "evidence", mode="before")
    @classmethod
    def _list_of_strings(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    summary = item.get("summary") or item.get("text") or item.get("title")
                    if summary:
                        out.append(str(summary))
                else:
                    out.append(str(item))
            return out
        return [str(v)]

    def to_legacy_dict(self) -> dict[str, Any]:
        """兼容旧策略代码的扁平 dict。"""
        data = self.model_dump(mode="json")
        data.update({
            "order_type": self.execution.order_type,
            "limit_price": self.execution.limit_price,
            "time_in_force": self.execution.time_in_force,
        })
        return data


def _clamp_float(value: Any, *, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, num))


def normalize_decision(raw: dict[str, Any] | None, *, fallback_reason: str = "invalid decision") -> StructuredDecision:
    """把 LLM 原始输出规范化成 ``StructuredDecision``。

    设计取向：交易系统里，坏输出不能抛到执行路径；坏输出必须降级为 HOLD，
    并保留原因用于审计。
    """
    raw = dict(raw or {})
    if "_error" in raw:
        return safe_hold(f"决策失败，保守 HOLD：{raw.get('_error')}")

    execution_raw = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
    if "order_type" in raw:
        execution_raw = {**execution_raw, "order_type": raw.get("order_type")}
    if "limit_price" in raw:
        execution_raw = {**execution_raw, "limit_price": raw.get("limit_price")}
    if "time_in_force" in raw:
        execution_raw = {**execution_raw, "time_in_force": raw.get("time_in_force")}

    payload = {
        "action": raw.get("action", "HOLD"),
        "target_position_pct": _clamp_float(raw.get("target_position_pct"), default=0.0),
        "confidence": _clamp_float(raw.get("confidence"), default=0.0),
        "reasoning": str(raw.get("reasoning") or fallback_reason)[:5000],
        "risks": raw.get("risks") or [],
        "evidence": raw.get("evidence") or [],
        "execution": execution_raw or {"order_type": "MARKET"},
        "horizon_days": int(_clamp_float(raw.get("horizon_days"), default=5, lo=1, hi=252)),
        "max_loss_pct": raw.get("max_loss_pct"),
        "take_profit_pct": raw.get("take_profit_pct"),
    }
    try:
        return StructuredDecision.model_validate(payload)
    except (ValidationError, ValueError) as e:
        return safe_hold(f"决策 schema 校验失败，保守 HOLD：{e}")


def safe_hold(reason: str) -> StructuredDecision:
    """生成安全 HOLD 决策。"""
    return StructuredDecision(
        action="HOLD",
        target_position_pct=0.0,
        confidence=0.0,
        reasoning=reason[:5000],
        risks=["LLM 决策不可用或 schema 非法"],
        execution=DecisionExecution(order_type="MARKET"),
    )
