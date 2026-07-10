"""Preflight checks for LLM paper trading.

The checks here are deterministic and local. They do not call an LLM, do not
fetch live quotes, and do not submit orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import desc, select

from qtrade.core.config import get_settings
from qtrade.core.time import utcnow
from qtrade.data.schema import normalize_symbol
from qtrade.llm.budget import check_budget
from qtrade.persistence import session_scope
from qtrade.persistence.models import TickLog

LLM_PREFLIGHT_PROMPT_VERSIONS = {"v1", "v2"}
LLMPreflightStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class LLMPreflightItem:
    name: str
    status: LLMPreflightStatus
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMPreflightReport:
    items: list[LLMPreflightItem]

    @property
    def failures(self) -> list[LLMPreflightItem]:
        return [item for item in self.items if item.status == "FAIL"]

    @property
    def warnings(self) -> list[LLMPreflightItem]:
        return [item for item in self.items if item.status == "WARN"]

    @property
    def passed(self) -> list[LLMPreflightItem]:
        return [item for item in self.items if item.status == "PASS"]

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)

    @property
    def overall_status(self) -> LLMPreflightStatus:
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"


def run_llm_paper_preflight(
    *,
    symbols: list[str],
    prompt_version: str = "v2",
    debate_rounds: int = 2,
    threshold: float = 0.45,
    decision_interval_minutes: int = 15,
    position_pct: float = 0.2,
    max_position_pct: float = 0.3,
) -> LLMPreflightReport:
    """Check whether an LLM paper run is safe to start."""
    items: list[LLMPreflightItem] = []
    items.append(_symbols_item(symbols))
    items.append(_provider_key_item())
    items.extend(
        _parameter_items(
            prompt_version=prompt_version,
            debate_rounds=debate_rounds,
            threshold=threshold,
            decision_interval_minutes=decision_interval_minutes,
            position_pct=position_pct,
            max_position_pct=max_position_pct,
        )
    )
    items.append(_budget_item())
    items.append(_recent_market_item())
    items.append(
        LLMPreflightItem(
            "执行边界",
            "PASS",
            "LLM 只输出交易意图；订单仍需经过行情新鲜度、资金、仓位、T+1 和风控检查",
        )
    )
    return LLMPreflightReport(items=items)


def _symbols_item(symbols: list[str]) -> LLMPreflightItem:
    normalized: list[str] = []
    invalid: list[str] = []
    for raw in symbols:
        try:
            normalized.append(normalize_symbol(raw))
        except Exception:
            invalid.append(raw)
    normalized = list(dict.fromkeys(normalized))
    if not normalized:
        return LLMPreflightItem("标的", "FAIL", "没有有效标的", data={"invalid": invalid})
    if invalid:
        return LLMPreflightItem(
            "标的",
            "WARN",
            f"有效 {','.join(normalized)}；无效 {','.join(invalid)}",
            data={"symbols": normalized, "invalid": invalid},
        )
    return LLMPreflightItem("标的", "PASS", ",".join(normalized), data={"symbols": normalized})


def _provider_key_item() -> LLMPreflightItem:
    settings = get_settings()
    provider = settings.llm.provider
    if provider == "deepseek" and not settings.deepseek_api_key:
        return LLMPreflightItem(
            "LLM Key",
            "FAIL",
            "缺少 DEEPSEEK_API_KEY；不会启动 LLM 模拟盘",
            data={"provider": provider},
        )
    return LLMPreflightItem("LLM Key", "PASS", f"provider={provider}", data={"provider": provider})


def _parameter_items(
    *,
    prompt_version: str,
    debate_rounds: int,
    threshold: float,
    decision_interval_minutes: int,
    position_pct: float,
    max_position_pct: float,
) -> list[LLMPreflightItem]:
    items: list[LLMPreflightItem] = []
    if prompt_version not in LLM_PREFLIGHT_PROMPT_VERSIONS:
        items.append(LLMPreflightItem("Prompt", "FAIL", f"prompt_version 必须是 {sorted(LLM_PREFLIGHT_PROMPT_VERSIONS)}"))
    else:
        items.append(LLMPreflightItem("Prompt", "PASS", prompt_version))

    if not 1 <= debate_rounds <= 4:
        items.append(LLMPreflightItem("辩论轮数", "FAIL", "debate_rounds 必须在 1-4 之间"))
    else:
        items.append(LLMPreflightItem("辩论轮数", "PASS", str(debate_rounds)))

    if not 0 <= threshold <= 1:
        items.append(LLMPreflightItem("信心阈值", "FAIL", "threshold 必须在 0-1 之间"))
    else:
        status: LLMPreflightStatus = "WARN" if threshold < 0.35 else "PASS"
        detail = f"{threshold:.2f}" if status == "PASS" else f"{threshold:.2f}，偏低，可能增加交易频率"
        items.append(LLMPreflightItem("信心阈值", status, detail))

    if decision_interval_minutes < 0:
        items.append(LLMPreflightItem("决策间隔", "FAIL", "decision_interval_minutes 不能为负数"))
    elif decision_interval_minutes == 0:
        items.append(LLMPreflightItem("决策间隔", "WARN", "0 分钟；LLM 可能每个 tick 都尝试决策，成本较高"))
    else:
        items.append(LLMPreflightItem("决策间隔", "PASS", f"{decision_interval_minutes} 分钟"))

    if not 0 < position_pct <= 1:
        items.append(LLMPreflightItem("目标仓位", "FAIL", "position_pct 必须在 0-1 之间"))
    elif position_pct > max_position_pct:
        items.append(LLMPreflightItem("目标仓位", "WARN", f"{position_pct:.0%} 高于仓位上限 {max_position_pct:.0%}，策略会被裁剪"))
    else:
        items.append(LLMPreflightItem("目标仓位", "PASS", f"{position_pct:.0%}"))

    if not 0 < max_position_pct <= 1:
        items.append(LLMPreflightItem("仓位上限", "FAIL", "max_position_pct 必须在 0-1 之间"))
    else:
        items.append(LLMPreflightItem("仓位上限", "PASS", f"{max_position_pct:.0%}"))
    return items


def _budget_item() -> LLMPreflightItem:
    settings = get_settings()
    try:
        decision = check_budget(
            max_decisions=settings.llm.daily_max_decisions,
            max_tokens=settings.llm.daily_max_tokens,
            max_cost_usd=settings.llm.daily_max_cost_usd,
        )
    except Exception as exc:
        return LLMPreflightItem("AI 预算", "WARN", f"无法读取预算使用情况：{exc}")
    usage = decision.usage
    detail = (
        f"{usage.day} | decisions={usage.decision_count}/{_limit_text(settings.llm.daily_max_decisions)} "
        f"| tokens={usage.total_tokens}/{_limit_text(settings.llm.daily_max_tokens)} "
        f"| cost=${usage.total_cost_usd:.4f}/{_limit_text(settings.llm.daily_max_cost_usd, money=True)}"
    )
    if not decision.allowed:
        return LLMPreflightItem("AI 预算", "FAIL", decision.reason or detail, data={"usage": usage.__dict__})

    near = _near_budget_labels(
        usage.decision_count,
        settings.llm.daily_max_decisions,
        usage.total_tokens,
        settings.llm.daily_max_tokens,
        usage.total_cost_usd,
        settings.llm.daily_max_cost_usd,
    )
    if near:
        return LLMPreflightItem("AI 预算", "WARN", detail + f" | 接近上限：{','.join(near)}", data={"usage": usage.__dict__})
    return LLMPreflightItem("AI 预算", "PASS", detail, data={"usage": usage.__dict__})


def _recent_market_item() -> LLMPreflightItem:
    try:
        with session_scope() as s:
            tick = s.scalars(select(TickLog).order_by(desc(TickLog.ts))).first()
    except Exception as exc:
        return LLMPreflightItem("最近行情", "WARN", f"无法读取 tick 日志：{exc}")
    if tick is None:
        return LLMPreflightItem("最近行情", "WARN", "暂无 tick 日志；启动后会由模拟盘刷新")

    age_seconds = max((utcnow() - tick.ts).total_seconds(), 0.0)
    if age_seconds > 600:
        return LLMPreflightItem("最近行情", "WARN", f"最近 tick 已 {age_seconds / 60:.1f} 分钟前；启动后需重新确认行情")
    if tick.quotes_count <= 0:
        return LLMPreflightItem("最近行情", "WARN", "最近 tick 无行情")
    if tick.tradeable_quotes <= 0:
        return LLMPreflightItem("最近行情", "WARN", "最近 tick 行情不可用于实时模拟")
    if tick.fallback_quotes > 0 or tick.tradeable_quotes < tick.quotes_count:
        return LLMPreflightItem(
            "最近行情",
            "WARN",
            f"最近 tick 部分行情不可模拟：{tick.tradeable_quotes}/{tick.quotes_count} 可用，fallback={tick.fallback_quotes}",
        )
    return LLMPreflightItem("最近行情", "PASS", f"最近 tick 可模拟行情 {tick.tradeable_quotes}/{tick.quotes_count}")


def _near_budget_labels(
    decisions: int,
    max_decisions: int | None,
    tokens: int,
    max_tokens: int | None,
    cost: float,
    max_cost: float | None,
) -> list[str]:
    labels: list[str] = []
    if _near_ratio(decisions, max_decisions):
        labels.append("决策")
    if _near_ratio(tokens, max_tokens):
        labels.append("tokens")
    if _near_ratio(cost, max_cost):
        labels.append("cost")
    return labels


def _near_ratio(used: float, limit: int | float | None) -> bool:
    return bool(limit and float(limit) > 0 and used / float(limit) >= 0.8)


def _limit_text(value: int | float | None, *, money: bool = False) -> str:
    if value is None or float(value) <= 0:
        return "不限"
    if money:
        return f"${float(value):.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return f"{value:,}"
