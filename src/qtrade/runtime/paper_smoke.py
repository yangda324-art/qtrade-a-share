"""Offline smoke checks for the paper-trading runtime path.

This is not a backtest. It uses deterministic fake quotes and scripted signals
to exercise the same paper runtime components used by live paper trading:
LiveRunner, PaperBroker, OMS, risk checks, snapshots, orders, and tick logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd
from sqlalchemy import select

from qtrade.core.time import utcnow
from qtrade.data.adapters.base import DataAdapter, Quote
from qtrade.data.schema import AdjustMode, normalize_symbol
from qtrade.strategies.base import Signal, Strategy, StrategyContext

PaperSmokeStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class PaperSmokeItem:
    name: str
    status: PaperSmokeStatus
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperSmokeReport:
    symbol: str
    items: list[PaperSmokeItem]
    db_path: str
    isolated: bool

    @property
    def failures(self) -> list[PaperSmokeItem]:
        return [item for item in self.items if item.status == "FAIL"]

    @property
    def warnings(self) -> list[PaperSmokeItem]:
        return [item for item in self.items if item.status == "WARN"]

    @property
    def passed(self) -> list[PaperSmokeItem]:
        return [item for item in self.items if item.status == "PASS"]

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)

    @property
    def overall_status(self) -> PaperSmokeStatus:
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"


class _FakeAdapter(DataAdapter):
    name = "paper_smoke_fake"

    def __init__(
        self,
        *,
        current_price: float = 10.0,
        source: str = "paper_smoke_fake",
        quote_age_seconds: int = 0,
    ) -> None:
        self.current_price = current_price
        self.source = source
        self.quote_age_seconds = quote_age_seconds

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: AdjustMode = "qfq",
    ) -> pd.DataFrame:
        dates = pd.date_range(end=date.today() - timedelta(days=1), periods=30, freq="D")
        prices = [10.0] * len(dates)
        return pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [10_000_000.0] * len(dates),
                "amount": [100_000_000.0] * len(dates),
                "symbol": symbol,
                "adjust": adjust,
            }
        )

    def fetch_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        now = utcnow()
        fetched_at = now - timedelta(seconds=self.quote_age_seconds)
        return {
            sym: Quote(
                symbol=sym,
                price=self.current_price,
                open=self.current_price,
                high=self.current_price * 1.01,
                low=self.current_price * 0.99,
                prev_close=self.current_price,
                volume=1_000_000.0,
                amount=self.current_price * 1_000_000.0,
                ts=fetched_at,
                fetched_at=fetched_at,
                source=self.source,
            )
            for sym in symbols
        }


class _ScriptedStrategy(Strategy):
    id = "paper_smoke_scripted_v1"

    def __init__(
        self,
        *,
        strategy_id: str,
        universe: list[str],
        signals_per_call: list[list[Signal]],
    ) -> None:
        super().__init__(universe=universe)
        self.id = strategy_id
        self._signals_per_call = signals_per_call
        self.calls = 0

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        if self.calls >= len(self._signals_per_call):
            self.calls += 1
            return []
        signals = self._signals_per_call[self.calls]
        self.calls += 1
        return signals


def run_paper_smoke_check(
    *,
    symbol: str = "000001",
    initial_cash: float = 100_000.0,
    isolated: bool = True,
) -> PaperSmokeReport:
    """Run deterministic paper runtime smoke checks."""
    from qtrade.core.config import get_settings
    from qtrade.persistence import init_db

    sym = normalize_symbol(symbol)
    init_db()
    items = [
        _market_order_item(sym, initial_cash=initial_cash),
        _limit_order_item(sym, initial_cash=initial_cash),
        _stale_quote_item(sym, initial_cash=initial_cash),
    ]
    return PaperSmokeReport(
        symbol=sym,
        items=items,
        db_path=str(get_settings().paths.db_path),
        isolated=isolated,
    )


def _market_order_item(symbol: str, *, initial_cash: float) -> PaperSmokeItem:
    try:
        runner, adapter = _build_runner(
            strategy_id="paper_smoke_market_v1",
            symbol=symbol,
            signals_per_call=[[_market_buy(symbol)]],
            current_price=10.0,
        )
        session = runner.start(initial_cash=initial_cash)
        try:
            report = runner.tick()
            _assert_market_order_result(session.run_id, report)
        finally:
            runner.stop(note="paper-smoke market")
        return PaperSmokeItem(
            "市价单链路",
            "PASS",
            f"quotes={report.quotes_count} signals={report.signals_generated} orders={report.orders_placed}",
            data={"run_id": session.run_id, "quote_source": adapter.source},
        )
    except Exception as exc:
        return PaperSmokeItem("市价单链路", "FAIL", str(exc))


def _limit_order_item(symbol: str, *, initial_cash: float) -> PaperSmokeItem:
    try:
        runner, adapter = _build_runner(
            strategy_id="paper_smoke_limit_v1",
            symbol=symbol,
            signals_per_call=[[_limit_buy(symbol)], []],
            current_price=11.0,
        )
        session = runner.start(initial_cash=initial_cash)
        try:
            first = runner.tick()
            if first.limit_orders_submitted != 1 or first.limit_orders_matched != 0:
                raise AssertionError(
                    f"expected one submitted limit order, got submitted={first.limit_orders_submitted} "
                    f"matched={first.limit_orders_matched}"
                )
            adapter.current_price = 10.0
            second = runner.tick()
            _assert_limit_order_result(session.run_id, second)
        finally:
            runner.stop(note="paper-smoke limit")
        return PaperSmokeItem(
            "限价单挂单/成交",
            "PASS",
            f"submitted={first.limit_orders_submitted} matched={second.limit_orders_matched}",
            data={"run_id": session.run_id},
        )
    except Exception as exc:
        return PaperSmokeItem("限价单挂单/成交", "FAIL", str(exc))


def _stale_quote_item(symbol: str, *, initial_cash: float) -> PaperSmokeItem:
    try:
        runner, _adapter = _build_runner(
            strategy_id="paper_smoke_stale_quote_v1",
            symbol=symbol,
            signals_per_call=[[_market_buy(symbol)]],
            current_price=10.0,
            quote_age_seconds=120,
        )
        session = runner.start(initial_cash=initial_cash)
        try:
            report = runner.tick()
            _assert_stale_quote_result(session.run_id, report)
        finally:
            runner.stop(note="paper-smoke stale quote")
        return PaperSmokeItem(
            "过期行情保护",
            "PASS",
            f"skipped={report.strategy_skipped} orders={report.orders_placed}",
            data={"run_id": session.run_id},
        )
    except Exception as exc:
        return PaperSmokeItem("过期行情保护", "FAIL", str(exc))


def _build_runner(
    *,
    strategy_id: str,
    symbol: str,
    signals_per_call: list[list[Signal]],
    current_price: float,
    quote_age_seconds: int = 0,
):
    from qtrade.data.loader import DataLoader
    from qtrade.execution.account import Account
    from qtrade.execution.broker.paper import PaperBroker
    from qtrade.risk.engine import RiskEngine
    from qtrade.risk.policies import RiskPolicies
    from qtrade.runtime.live_runner import LiveRunner
    from qtrade.runtime.state import StateManager

    adapter = _FakeAdapter(current_price=current_price, quote_age_seconds=quote_age_seconds)
    loader = DataLoader(adapters=[adapter])
    strategy = _ScriptedStrategy(
        strategy_id=strategy_id,
        universe=[symbol],
        signals_per_call=signals_per_call,
    )
    runner = LiveRunner(
        strategy=strategy,
        state=StateManager(strategy_id=strategy.id, mode="paper_smoke"),
        risk=RiskEngine(policies=RiskPolicies()),
        broker=PaperBroker(account=Account()),
        loader=loader,
    )
    return runner, adapter


def _market_buy(symbol: str) -> Signal:
    return Signal(symbol=symbol, action="BUY", quantity=1000, confidence=0.9, reason="paper smoke market buy")


def _limit_buy(symbol: str) -> Signal:
    return Signal(
        symbol=symbol,
        action="BUY",
        quantity=1000,
        confidence=0.9,
        reason="paper smoke limit buy",
        metadata={"order_type": "LIMIT", "limit_price": 10.5},
    )


def _assert_market_order_result(run_id: int, report) -> None:
    from qtrade.persistence import session_scope
    from qtrade.persistence.models import AccountSnapshot, OrderRecord, OrderStatus, TickLog

    if report.quotes_count != 1 or report.signals_generated != 1 or report.orders_placed != 1:
        raise AssertionError(
            f"unexpected market report: quotes={report.quotes_count} "
            f"signals={report.signals_generated} orders={report.orders_placed}"
        )
    if report.risk_rejected != 0 or report.errors:
        raise AssertionError(f"market order was rejected/errors={report.errors}")
    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == run_id)).all()
        if len(orders) != 1 or orders[0].status != OrderStatus.FILLED:
            raise AssertionError("market order was not filled and persisted")
        ticks = s.scalars(select(TickLog).where(TickLog.run_id == run_id)).all()
        if len(ticks) != 1 or ticks[0].orders_placed != 1:
            raise AssertionError("market tick log missing order placement")
        snapshots = s.scalars(select(AccountSnapshot).where(AccountSnapshot.run_id == run_id)).all()
        if not snapshots:
            raise AssertionError("account snapshot missing")


def _assert_limit_order_result(run_id: int, report) -> None:
    from qtrade.persistence import session_scope
    from qtrade.persistence.models import OrderRecord, OrderStatus

    if report.limit_orders_matched != 1 or report.errors:
        raise AssertionError(f"limit order did not match cleanly: matched={report.limit_orders_matched} errors={report.errors}")
    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == run_id)).all()
        if len(orders) != 1:
            raise AssertionError(f"expected one limit order, got {len(orders)}")
        order = orders[0]
        if order.status != OrderStatus.FILLED or int(order.filled_quantity or 0) != 1000:
            raise AssertionError("limit order was not filled after price crossed")


def _assert_stale_quote_result(run_id: int, report) -> None:
    from qtrade.persistence import session_scope
    from qtrade.persistence.models import OrderRecord, TickLog

    if report.signals_generated != 0 or report.orders_placed != 0:
        raise AssertionError("stale quote triggered strategy/order unexpectedly")
    with session_scope() as s:
        orders = s.scalars(select(OrderRecord).where(OrderRecord.run_id == run_id)).all()
        if orders:
            raise AssertionError("stale quote persisted unexpected order")
        tick = s.scalars(select(TickLog).where(TickLog.run_id == run_id)).one()
        if tick.tradeable_quotes != 0 or tick.per_symbol_json[0].get("skip_reason") != "stale_quote":
            raise AssertionError("stale quote was not logged as stale_quote")
