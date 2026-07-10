"""结构化 LLM 决策 schema 测试。"""
from __future__ import annotations

from qtrade.strategies.llm.decision_schema import normalize_decision


def test_normalize_legacy_decision_defaults_to_market():
    decision = normalize_decision({
        "action": "buy",
        "target_position_pct": 0.2,
        "confidence": 0.7,
        "reasoning": "趋势和多头论点一致",
        "risks": "短期波动",
    })

    out = decision.to_legacy_dict()
    assert out["action"] == "BUY"
    assert out["order_type"] == "MARKET"
    assert out["limit_price"] is None
    assert out["risks"] == ["短期波动"]


def test_normalize_limit_decision_keeps_execution_intent():
    decision = normalize_decision({
        "action": "BUY",
        "target_position_pct": 0.15,
        "confidence": 0.8,
        "reasoning": "只在回调到支撑位时买入",
        "execution": {
            "order_type": "limit",
            "limit_price": 10.45,
            "time_in_force": "day",
        },
        "horizon_days": 10,
        "max_loss_pct": 0.03,
    })

    out = decision.to_legacy_dict()
    assert out["order_type"] == "LIMIT"
    assert out["limit_price"] == 10.45
    assert out["time_in_force"] == "DAY"
    assert out["horizon_days"] == 10
    assert out["max_loss_pct"] == 0.03


def test_invalid_decision_falls_back_to_hold():
    decision = normalize_decision({
        "action": "BUY_EVERYTHING",
        "target_position_pct": 8,
        "confidence": 2,
        "reasoning": "all in",
    })

    assert decision.action == "HOLD"
    assert decision.confidence == 0.0
    assert "schema 校验失败" in decision.reasoning
