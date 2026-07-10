"""LiveRunner 集成测试：用 fake adapter + fake strategy 验证全链路。

测试设计原则：
- **不耦合具体策略**——用一个"按需返回 BUY/SELL/HOLD 信号"的可控 strategy
- **不耦合真实数据源**——用 fake adapter 返回固定快照
- 重点验证：行情→策略→风控→OMS→Broker→SQLite 的数据流是否正确
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from qtrade.core import config as cfg
from qtrade.core.time import MARKET_TZ
from qtrade.data import cache as bar_cache
from qtrade.data.adapters.base import DataAdapter, Quote
from qtrade.data.loader import DataLoader
from qtrade.data.schema import AdjustMode
from qtrade.execution.account import Account
from qtrade.execution.broker.paper import PaperBroker
from qtrade.persistence import init_db, session_scope
from qtrade.persistence.models import (
    AccountSnapshot,
    AgentDecision,
    DecisionAction,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    TickLog,
)
from qtrade.risk.engine import RiskEngine
from qtrade.risk.policies import RiskPolicies
from qtrade.runtime.live_runner import LiveRunner
from qtrade.runtime.state import StateManager
from qtrade.strategies.base import Signal, Strategy, StrategyContext

SYM = "000001.SZ"


def _market_dt_to_utc_naive(year: int, month: int, day: int, hour: int, minute: int):
    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TZ).astimezone(timezone.utc).replace(tzinfo=None)


# -------------------- Fakes --------------------


def _flat_history(n: int = 30) -> pd.DataFrame:
    """平价历史数据，仅用作 strategy ctx.history 的占位。

    注意用 freq="D"（自然日）而不是 "B"（业务日）：
    某些 pandas 版本下 `freq="B"` + `end + periods` 会返回 != periods 的项，
    导致和价格数组长度对不齐。策略只看 price 序列不在乎是不是周末，无影响。
    """
    dates = pd.date_range(end=date.today() - timedelta(days=1), periods=n, freq="D")
    prices = np.full(n, 10.0)
    return pd.DataFrame({
        "date": dates,
        "open": prices, "high": prices, "low": prices,
        "close": prices, "volume": np.full(n, 1e7), "amount": prices * 1e7,
        "symbol": SYM, "adjust": "qfq",
    })


class _FakeAdapter(DataAdapter):
    name = "fake"

    def __init__(
        self,
        current_price: float = 11.0,
        *,
        source: str = "fake",
        quote_age_seconds: int = 0,
    ) -> None:
        self.current_price = current_price
        self.source = source
        self.quote_age_seconds = quote_age_seconds
        self.quote_calls = 0

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        return _flat_history()

    def fetch_quotes(self, symbols):
        from qtrade.core.time import utcnow
        self.quote_calls += 1
        now = utcnow()
        fetched_at = now - timedelta(seconds=self.quote_age_seconds)
        return {
            sym: Quote(
                symbol=sym, price=self.current_price,
                open=self.current_price, high=self.current_price * 1.01,
                low=self.current_price * 0.99, prev_close=self.current_price,
                volume=1e6, amount=self.current_price * 1e6,
                ts=fetched_at,
                fetched_at=fetched_at,
                source=self.source,
            )
            for sym in symbols
        }


class _ScriptedStrategy(Strategy):
    """按预设脚本依次返回信号；调试 / 测试 LiveRunner 专用。"""
    id = "scripted_test"

    def __init__(self, signals_per_call: list[list[Signal]], universe: list[str]) -> None:
        super().__init__(universe=universe)
        self._scripted = list(signals_per_call)
        self.calls = 0

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        if self.calls >= len(self._scripted):
            self.calls += 1
            return []
        out = self._scripted[self.calls]
        self.calls += 1
        return out


def _buy(qty: int = 1000) -> Signal:
    return Signal(symbol=SYM, action="BUY", quantity=qty, confidence=0.8, reason="scripted buy")


def _buy_with_reference(qty: int = 1000, reference_price: float = 10.0) -> Signal:
    return Signal(
        symbol=SYM,
        action="BUY",
        quantity=qty,
        confidence=0.8,
        reason="scripted buy with reference",
        metadata={"reference_price": reference_price},
    )


def _sell(qty: int = 1000) -> Signal:
    return Signal(symbol=SYM, action="SELL", quantity=qty, confidence=0.8, reason="scripted sell")


def _limit_buy(
    qty: int = 1000,
    limit_price: float = 10.5,
    reference_price: float | None = None,
) -> Signal:
    metadata = {"order_type": "LIMIT", "limit_price": limit_price}
    if reference_price is not None:
        metadata["reference_price"] = reference_price
    return Signal(
        symbol=SYM,
        action="BUY",
        quantity=qty,
        confidence=0.8,
        reason="scripted limit buy",
        metadata=metadata,
    )


# -------------------- Helper --------------------


def _build_runner(
    strategy: Strategy,
    *,
    current_price: float = 11.0,
    source: str = "fake",
    decision_interval_seconds: int = 0,
    quote_age_seconds: int = 0,
    max_price_deviation_bps: int | None = None,
) -> tuple[LiveRunner, _FakeAdapter]:
    init_db()
    bar_cache.save_bars(_flat_history(), SYM)

    adapter = _FakeAdapter(
        current_price=current_price,
        source=source,
        quote_age_seconds=quote_age_seconds,
    )
    loader = DataLoader(adapters=[adapter])
    state = StateManager(strategy_id=strategy.id, mode="paper")
    risk = RiskEngine(policies=RiskPolicies())
    broker = PaperBroker(account=Account())

    return LiveRunner(
        strategy=strategy,
        state=state,
        risk=risk,
        broker=broker,
        loader=loader,
        decision_interval_seconds=decision_interval_seconds,
        max_price_deviation_bps=max_price_deviation_bps,
    ), adapter


# -------------------- Tests --------------------


def test_initial_state_is_clean():
    strat = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    runner, _ = _build_runner(strat)
    session = runner.start(initial_cash=100_000.0)
    assert session.account.cash == Decimal("100000")
    assert session.account.total_assets == Decimal("100000")
    assert len(list(session.account.positions)) == 0


def test_buy_signal_flows_to_broker_and_db():
    """注入一个 BUY 信号，验证：策略 → 风控 → OMS → Broker → SQLite 全链路。"""
    strat = _ScriptedStrategy(signals_per_call=[[_buy(1000)]], universe=[SYM])
    runner, adapter = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)

    report = runner.tick()

    assert adapter.quote_calls == 1
    assert report.quotes_count == 1
    assert report.signals_generated == 1
    assert report.orders_placed == 1
    assert report.risk_rejected == 0

    pos = session.account.positions.get(SYM)
    assert pos.quantity == 1000
    assert pos.avg_cost > 0
    assert session.account.cash < Decimal("100000")    # 现金确实扣了

    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).all()
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.FILLED
        assert orders[0].filled_quantity == 1000

        snaps = s.scalars(select(AccountSnapshot).where(AccountSnapshot.run_id == session.run_id)).all()
        assert len(snaps) >= 1
        tick = s.scalars(select(TickLog).where(TickLog.run_id == session.run_id)).one()
        assert tick.tradeable_quotes == 1
        assert tick.signals_generated == 1
        assert tick.orders_placed == 1
        assert tick.per_symbol_json[0]["strategy_executed"] is True


def test_signal_decision_id_is_linked_to_order():
    """LLM 策略产出的 decision_id 应贯穿到订单表，保证事后审计能从订单回看决策。"""
    strat = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)

    with session_scope() as s:
        decision = AgentDecision(
            run_id=session.run_id,
            strategy_id=strat.id,
            symbol=SYM,
            action=DecisionAction.BUY,
            confidence=Decimal("0.8000"),
            target_position_pct=Decimal("0.2000"),
            reasoning="scripted decision",
            trace_json={},
        )
        s.add(decision)
        s.flush()
        decision_id = decision.id

    strat._scripted = [[Signal(
        symbol=SYM,
        action="BUY",
        quantity=1000,
        confidence=0.8,
        reason="scripted buy with decision",
        metadata={"decision_id": decision_id},
    )]]
    strat.calls = 0

    report = runner.tick()

    assert report.orders_placed == 1
    with session_scope() as s:
        order = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).one()
        assert order.decision_id == decision_id


def test_risk_blocks_oversized_order():
    """200000 元单笔触发 max_order_value 上限（默认 50000），应被风控拒。"""
    strat = _ScriptedStrategy(signals_per_call=[[_buy(20000)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=1_000_000.0)

    report = runner.tick()

    assert report.signals_generated == 1
    assert report.orders_placed == 0
    assert report.risk_rejected == 1
    assert session.account.positions.get(SYM).quantity == 0
    assert "超过" in report.errors[0]
    with session_scope() as s:
        tick = s.scalars(select(TickLog).where(TickLog.run_id == session.run_id)).one()
        assert "超过" in tick.errors_json[0]
        assert "超过" in tick.per_symbol_json[0]["order_rejections"][0]["reason"]


def test_blocked_signal_updates_agent_decision_execution_reason():
    """AI 决策产出的信号被风控拦截时，AgentDecision 应记录执行失败原因。"""
    strat = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=1_000_000.0)

    with session_scope() as s:
        decision = AgentDecision(
            run_id=session.run_id,
            strategy_id=strat.id,
            symbol=SYM,
            action=DecisionAction.BUY,
            confidence=Decimal("0.8000"),
            target_position_pct=Decimal("0.5000"),
            reasoning="scripted decision oversized",
            trace_json={},
        )
        s.add(decision)
        s.flush()
        decision_id = decision.id

    strat._scripted = [[Signal(
        symbol=SYM,
        action="BUY",
        quantity=20000,
        confidence=0.8,
        reason="scripted oversized buy",
        metadata={"decision_id": decision_id},
    )]]
    strat.calls = 0

    report = runner.tick()

    assert report.risk_rejected == 1
    with session_scope() as s:
        row = s.get(AgentDecision, decision_id)
        assert row.risk_passed is False
        assert "超过" in row.risk_reject_reason
        assert row.trace_json["execution"]["status"] == "risk"
        assert "超过" in row.trace_json["execution"]["reason"]


def test_skip_if_traded_today_blocks_repeat():
    """同一个 symbol 一天内只下一单，第二个 tick 即使有信号也跳过。"""
    strat = _ScriptedStrategy(
        signals_per_call=[[_buy(1000)], [_buy(500)], [_buy(500)]],
        universe=[SYM],
    )
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)

    runner.tick()
    qty_after_1st = session.account.positions.get(SYM).quantity

    runner.tick()
    runner.tick()
    assert session.account.positions.get(SYM).quantity == qty_after_1st
    # strategy 因为 skip_if_traded_today，第 2/3 次 tick 根本没被调到
    assert strat.calls == 1


def test_decision_interval_skips_strategy_but_keeps_quote_updates():
    """秒级行情可以持续刷新，但策略决策能按间隔降频。"""
    strat = _ScriptedStrategy(signals_per_call=[[], []], universe=[SYM])
    runner, adapter = _build_runner(strat, current_price=11.0, decision_interval_seconds=60)
    runner.start(initial_cash=100_000.0)

    first = runner.tick()
    adapter.current_price = 12.0
    second = runner.tick()

    assert first.strategy_skipped == 0
    assert second.strategy_skipped == 1
    assert adapter.quote_calls == 2
    assert strat.calls == 1
    assert runner.broker.last_prices[SYM] == 12.0


def test_state_restored_after_crash():
    """模拟进程崩溃：runner 没 stop，重新创建 runner 应恢复同一个 run + 账户状态。"""
    strat = _ScriptedStrategy(signals_per_call=[[_buy(1000)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)
    runner.tick()

    cash_before = session.account.cash
    qty_before = session.account.positions.get(SYM).quantity
    run_id_before = session.run_id

    # 不调 stop()，直接抛掉 runner 模拟崩溃
    del runner

    # 重启
    strat2 = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    strat2.id = "scripted_test"  # 必须用同一个 strategy_id 才能命中 resume
    runner2, _ = _build_runner(strat2, current_price=11.0)
    sess2 = runner2.start(initial_cash=100_000.0)

    assert sess2.run_id == run_id_before, "应恢复同一未结束的 run"
    assert sess2.account.cash == cash_before
    assert sess2.account.positions.get(SYM).quantity == qty_before
    assert sess2.account.positions.get(SYM).available == 0
    assert sess2.account.positions.get(SYM).today_buy == qty_before


def test_state_restore_unlocks_t1_when_snapshot_is_from_previous_day():
    """重启时如果最新快照已是昨日，应恢复为可卖，不能永久卡在 T+1。"""
    strat = _ScriptedStrategy(signals_per_call=[[_buy(1000)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)
    runner.tick()

    with session_scope() as s:
        for snap in s.scalars(select(AccountSnapshot).where(AccountSnapshot.run_id == session.run_id)).all():
            snap.ts = snap.ts - timedelta(days=1)

    del runner

    strat2 = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    strat2.id = "scripted_test"
    runner2, _ = _build_runner(strat2, current_price=11.0)
    sess2 = runner2.start(initial_cash=100_000.0)

    pos = sess2.account.positions.get(SYM)
    assert pos.quantity == 1000
    assert pos.available == 1000
    assert pos.today_buy == 0


def test_state_restore_uses_market_date_for_t1(monkeypatch):
    """UTC 日期已到前一天但上海仍是同一交易日时，不能提前解锁 T+1。"""
    market_day = date(2026, 5, 10)
    same_market_day_utc = _market_dt_to_utc_naive(2026, 5, 10, 0, 30)
    monkeypatch.setattr("qtrade.runtime.state.market_today", lambda: market_day)

    strat = _ScriptedStrategy(signals_per_call=[[_buy(1000)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)
    runner.tick()

    with session_scope() as s:
        for snap in s.scalars(select(AccountSnapshot).where(AccountSnapshot.run_id == session.run_id)).all():
            snap.ts = same_market_day_utc

    del runner

    strat2 = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    strat2.id = "scripted_test"
    runner2, _ = _build_runner(strat2, current_price=11.0)
    sess2 = runner2.start(initial_cash=100_000.0)

    pos = sess2.account.positions.get(SYM)
    assert pos.quantity == 1000
    assert pos.available == 0
    assert pos.today_buy == 1000


def test_state_restore_reconciles_open_limit_buy_freeze():
    """若限价挂单已入库但 snapshot 未记录冻结资金，重启时应修正冻结金额。"""
    strat = _ScriptedStrategy(signals_per_call=[[_limit_buy(limit_price=10.5)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)
    runner.tick()

    with session_scope() as s:
        snap = s.scalars(
            select(AccountSnapshot)
            .where(AccountSnapshot.run_id == session.run_id)
            .order_by(AccountSnapshot.ts.desc())
        ).first()
        snap.cash = Decimal("100000")
        snap.frozen_cash = Decimal("0")

    del runner

    strat2 = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    strat2.id = "scripted_test"
    runner2, _ = _build_runner(strat2, current_price=11.0)
    sess2 = runner2.start(initial_cash=100_000.0)

    assert sess2.account.frozen_cash > Decimal("0")
    assert sess2.account.cash < Decimal("100000")
    assert sess2.account.total_assets == Decimal("100000.0000")


def test_state_restore_reconciles_missing_frozen_without_double_charging_cash():
    """若 cash 已扣但 frozen 丢失，恢复时只补 frozen，不应二次扣现金。"""
    strat = _ScriptedStrategy(signals_per_call=[[_limit_buy(limit_price=10.5)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)
    runner.tick()
    cash_after_freeze = session.account.cash

    with session_scope() as s:
        snap = s.scalars(
            select(AccountSnapshot)
            .where(AccountSnapshot.run_id == session.run_id)
            .order_by(AccountSnapshot.ts.desc())
        ).first()
        snap.cash = cash_after_freeze
        snap.frozen_cash = Decimal("0")

    del runner

    strat2 = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    strat2.id = "scripted_test"
    runner2, _ = _build_runner(strat2, current_price=11.0)
    sess2 = runner2.start(initial_cash=100_000.0)

    assert sess2.account.cash == cash_after_freeze
    assert sess2.account.frozen_cash > Decimal("0")
    assert sess2.account.total_assets == Decimal("100000.0000")


def test_t1_unlock_in_pre_market():
    """昨日买入今日不可卖；pre_market 跑过后 today_buy 进入 available。"""
    strat = _ScriptedStrategy(signals_per_call=[[_buy(1000)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)

    runner.tick()
    pos = session.account.positions.get(SYM)
    assert pos.quantity == 1000
    assert pos.available == 0     # T+1 锁仓
    assert pos.today_buy == 1000

    runner.pre_market()           # 模拟跨日
    assert pos.available == 1000
    assert pos.today_buy == 0


def test_limit_signal_persists_then_matches_on_later_tick():
    """自动策略 LIMIT 信号应持久化挂单，并在后续价格触发时成交写库。"""
    strat = _ScriptedStrategy(signals_per_call=[[_limit_buy(limit_price=10.5)], []], universe=[SYM])
    runner, adapter = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)

    first = runner.tick()

    assert first.orders_placed == 1
    assert first.limit_orders_submitted == 1
    assert first.limit_orders_matched == 0
    assert session.account.positions.get(SYM).quantity == 0
    assert session.account.frozen_cash > 0

    with session_scope() as s:
        order = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).one()
        assert order.order_type == OrderType.LIMIT
        assert order.status == OrderStatus.SUBMITTED
        assert float(order.limit_price) == 10.5

    adapter.current_price = 10.0
    second = runner.tick()

    assert second.limit_orders_matched == 1
    assert session.account.positions.get(SYM).quantity == 1000
    assert session.account.frozen_cash == Decimal("0.0000")

    with session_scope() as s:
        order = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).one()
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1000
        assert float(order.avg_fill_price) == 10.5


def test_cached_quote_does_not_trigger_strategy_or_order():
    """缓存收盘价只能展示/估值，不能触发自动策略和下单。"""
    strat = _ScriptedStrategy(signals_per_call=[[_limit_buy(limit_price=10.5)], []], universe=[SYM])
    runner, adapter = _build_runner(strat, current_price=11.0, source="cache_daily_close")
    session = runner.start(initial_cash=100_000.0)

    first = runner.tick()
    adapter.current_price = 10.0
    second = runner.tick()

    assert first.signals_generated == 0
    assert second.limit_orders_matched == 0
    assert strat.calls == 0
    assert session.account.positions.get(SYM).quantity == 0
    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).all()
        assert orders == []
        ticks = s.scalars(select(TickLog).where(TickLog.run_id == session.run_id)).all()
        assert ticks[-1].fallback_quotes == 1
        assert ticks[-1].per_symbol_json[0]["skip_reason"] == "fallback_quote"


def test_stale_quote_does_not_trigger_strategy_or_order():
    """过期实时行情不能触发实时模拟盘策略或成交。"""
    strat = _ScriptedStrategy(signals_per_call=[[_buy(1000)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0, quote_age_seconds=120)
    session = runner.start(initial_cash=100_000.0)

    report = runner.tick()

    assert report.signals_generated == 0
    assert report.orders_placed == 0
    assert strat.calls == 0
    assert session.account.positions.get(SYM).quantity == 0
    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).all()
        assert orders == []
        tick = s.scalars(select(TickLog).where(TickLog.run_id == session.run_id)).one()
        assert tick.tradeable_quotes == 0
        assert tick.fallback_quotes == 0
        assert tick.per_symbol_json[0]["skip_reason"] == "stale_quote"


def test_market_signal_price_deviation_rejects_auto_order():
    """自动策略市价信号的参考价和提交价偏离过大时，不应下单。"""
    strat = _ScriptedStrategy(
        signals_per_call=[[_buy_with_reference(reference_price=10.0)]],
        universe=[SYM],
    )
    runner, _ = _build_runner(strat, current_price=10.1, max_price_deviation_bps=50)
    session = runner.start(initial_cash=100_000.0)

    report = runner.tick()

    assert report.signals_generated == 1
    assert report.orders_placed == 0
    assert report.risk_rejected == 1
    assert session.account.positions.get(SYM).quantity == 0
    assert "price_deviation" in report.errors[0]
    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).all()
        assert orders == []
        tick = s.scalars(select(TickLog).where(TickLog.run_id == session.run_id)).one()
        assert "price_deviation" in tick.errors_json[0]
        assert tick.per_symbol_json[0]["order_rejections"][0]["kind"] == "risk"
        assert "price_deviation" in tick.per_symbol_json[0]["order_rejections"][0]["reason"]


def test_marketable_limit_signal_price_deviation_rejects_auto_order():
    """会立即成交的自动限价单也必须通过参考价偏离保护。"""
    strat = _ScriptedStrategy(
        signals_per_call=[[_limit_buy(limit_price=10.5, reference_price=11.0)]],
        universe=[SYM],
    )
    runner, _ = _build_runner(strat, current_price=10.0, max_price_deviation_bps=50)
    session = runner.start(initial_cash=100_000.0)

    report = runner.tick()

    assert report.signals_generated == 1
    assert report.orders_placed == 0
    assert report.risk_rejected == 1
    assert "price_deviation" in report.errors[0]
    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).all()
        assert orders == []
        tick = s.scalars(select(TickLog).where(TickLog.run_id == session.run_id)).one()
        assert "price_deviation" in tick.errors_json[0]
        assert tick.per_symbol_json[0]["order_rejections"][0]["kind"] == "risk"


def test_pending_limit_signal_allows_price_deviation():
    """不立即成交的限价单可以挂单，因为限价本身保护成交价。"""
    strat = _ScriptedStrategy(
        signals_per_call=[[_limit_buy(limit_price=10.5, reference_price=10.0)]],
        universe=[SYM],
    )
    runner, _ = _build_runner(strat, current_price=11.0, max_price_deviation_bps=50)
    session = runner.start(initial_cash=100_000.0)

    report = runner.tick()

    assert report.orders_placed == 1
    assert report.limit_orders_submitted == 1
    with session_scope() as s:
        order = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).one()
        assert order.status == OrderStatus.SUBMITTED


def test_live_runner_price_deviation_threshold_comes_from_settings(monkeypatch):
    monkeypatch.setenv("QTRADE_TRADING__REALTIME__MAX_PRICE_DEVIATION_BPS", "25")
    cfg.get_settings.cache_clear()

    strat = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=10.0)

    assert runner.max_price_deviation_bps == 25


def test_restart_with_open_limit_order_does_not_duplicate_order():
    """重启后已有未成交限价单时，不应再次对同标的生成新挂单。"""
    strat = _ScriptedStrategy(signals_per_call=[[_limit_buy(limit_price=10.5)]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)
    runner.tick()

    del runner

    strat2 = _ScriptedStrategy(signals_per_call=[[_limit_buy(limit_price=10.5)]], universe=[SYM])
    strat2.id = "scripted_test"
    runner2, _ = _build_runner(strat2, current_price=11.0)
    runner2.start(initial_cash=100_000.0)
    report = runner2.tick()

    assert report.signals_generated == 0
    assert report.limit_orders_submitted == 0
    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == session.run_id)).all()
        assert len(orders) == 1


def test_limit_order_write_failure_rolls_back_buy_freeze():
    """订单入库失败时，已经冻结的买入资金必须回滚。"""
    strat = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)

    manager = runner._limit_orders
    assert manager is not None
    before_cash = session.account.cash
    before_frozen = session.account.frozen_cash

    def fail_write(**kwargs):
        raise RuntimeError("db write failed")

    manager._write_submitted_order = fail_write
    with pytest.raises(RuntimeError):
        manager.submit_limit_order(
            run=session,
            symbol=SYM,
            side=OrderSide.BUY,
            quantity=1000,
            limit_price=Decimal("10.5"),
        )

    assert session.account.cash == before_cash
    assert session.account.frozen_cash == before_frozen


def test_limit_order_write_failure_rolls_back_sell_lock():
    """订单入库失败时，已经锁定的可卖股数必须回滚。"""
    strat = _ScriptedStrategy(signals_per_call=[[]], universe=[SYM])
    runner, _ = _build_runner(strat, current_price=11.0)
    session = runner.start(initial_cash=100_000.0)
    pos = session.account.positions.get(SYM)
    pos.quantity = 1000
    pos.available = 1000
    pos.avg_cost = Decimal("10")
    pos.market_price = Decimal("11")

    manager = runner._limit_orders
    assert manager is not None

    def fail_write(**kwargs):
        raise RuntimeError("db write failed")

    manager._write_submitted_order = fail_write
    with pytest.raises(RuntimeError):
        manager.submit_limit_order(
            run=session,
            symbol=SYM,
            side=OrderSide.SELL,
            quantity=600,
            limit_price=Decimal("11.5"),
        )

    assert pos.available == 1000
