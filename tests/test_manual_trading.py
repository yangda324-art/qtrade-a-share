"""手动模拟交易服务测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select

from qtrade.core import config as cfg
from qtrade.core.exceptions import DataError, RiskRejected
from qtrade.core.time import MARKET_TZ, utcnow
from qtrade.data.adapters.base import Quote
from qtrade.persistence import session_scope
from qtrade.persistence.models import AccountSnapshot, OrderRecord, OrderStatus, TradeFill
from qtrade.runtime.manual_trading import MANUAL_STRATEGY_ID, ManualTradingService

SYM = "000001.SZ"


def _market_dt_to_utc_naive(year: int, month: int, day: int, hour: int, minute: int):
    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TZ).astimezone(timezone.utc).replace(tzinfo=None)


class _QuoteLoader:
    def __init__(
        self,
        *,
        price: float = 11.0,
        empty: bool = False,
        source: str = "test",
        quote_age_seconds: int = 0,
    ) -> None:
        self.price = price
        self.empty = empty
        self.source = source
        self.quote_age_seconds = quote_age_seconds
        self.calls: list[list[str]] = []

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        self.calls.append(symbols)
        if self.empty:
            return {}
        now = utcnow()
        fetched_at = now - timedelta(seconds=self.quote_age_seconds)
        return {
            sym: Quote(
                symbol=sym,
                price=self.price,
                open=self.price,
                high=self.price * 1.01,
                low=self.price * 0.99,
                prev_close=self.price,
                volume=1_000_000,
                amount=self.price * 1_000_000,
                ts=fetched_at,
                fetched_at=fetched_at,
                source=self.source,
            )
            for sym in symbols
        }

    def set_price(self, price: float) -> None:
        self.price = price


def test_manual_market_buy_writes_order_and_snapshot():
    service = ManualTradingService(loader=_QuoteLoader(price=11.0), initial_cash=100_000.0)

    result = service.place_market_order(symbol="000001", side="buy", quantity=1000)

    assert result.ok
    assert result.symbol == SYM
    assert result.side == "BUY"
    assert result.filled_quantity == 1000
    assert result.avg_fill_price is not None
    assert result.account["cash"] < 100_000

    with session_scope() as s:
        orders = s.scalars(
            select(OrderRecord).where(OrderRecord.strategy_id == MANUAL_STRATEGY_ID)
        ).all()
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.FILLED
        assert orders[0].run_id == result.run_id

        snaps = s.scalars(
            select(AccountSnapshot).where(AccountSnapshot.run_id == result.run_id)
        ).all()
        assert len(snaps) == 1
        assert snaps[0].cash < Decimal("100000")


def test_manual_sell_without_available_shares_is_rejected():
    service = ManualTradingService(loader=_QuoteLoader(price=11.0), initial_cash=100_000.0)

    with pytest.raises(RiskRejected) as exc:
        service.place_market_order(symbol="000001", side="SELL", quantity=1000)

    assert exc.value.rule == "risk_check"
    assert "可卖" in exc.value.detail


def test_manual_position_unlocks_after_overnight_snapshot():
    service = ManualTradingService(loader=_QuoteLoader(price=11.0), initial_cash=100_000.0)
    buy = service.place_market_order(symbol="000001", side="BUY", quantity=1000)

    with session_scope() as s:
        snaps = s.scalars(
            select(AccountSnapshot).where(AccountSnapshot.run_id == buy.run_id)
        ).all()
        for snap in snaps:
            snap.ts = snap.ts - timedelta(days=1)

    service_next_day = ManualTradingService(loader=_QuoteLoader(price=11.5), initial_cash=100_000.0)
    sell = service_next_day.place_market_order(symbol="000001", side="SELL", quantity=1000)

    assert sell.ok
    assert sell.side == "SELL"
    assert sell.filled_quantity == 1000


def test_manual_t1_does_not_unlock_when_only_utc_date_changed(monkeypatch):
    market_day = datetime(2026, 5, 10).date()
    same_market_day_utc = _market_dt_to_utc_naive(2026, 5, 10, 0, 30)
    monkeypatch.setattr("qtrade.runtime.state.market_today", lambda: market_day)
    monkeypatch.setattr("qtrade.runtime.manual_trading.market_today", lambda: market_day)

    service = ManualTradingService(loader=_QuoteLoader(price=11.0), initial_cash=100_000.0)
    buy = service.place_market_order(symbol="000001", side="BUY", quantity=1000)

    with session_scope() as s:
        snaps = s.scalars(
            select(AccountSnapshot).where(AccountSnapshot.run_id == buy.run_id)
        ).all()
        for snap in snaps:
            snap.ts = same_market_day_utc

    service_same_market_day = ManualTradingService(loader=_QuoteLoader(price=11.5), initial_cash=100_000.0)
    with pytest.raises(RiskRejected) as exc:
        service_same_market_day.place_market_order(symbol="000001", side="SELL", quantity=1000)

    assert "可卖" in exc.value.detail


def test_manual_quote_missing_raises_data_error():
    service = ManualTradingService(loader=_QuoteLoader(empty=True))

    with pytest.raises(DataError):
        service.get_quote("000001")


def test_manual_limit_buy_pending_freezes_cash():
    service = ManualTradingService(loader=_QuoteLoader(price=11.0), initial_cash=100_000.0)

    result = service.place_limit_order(symbol="000001", side="BUY", quantity=1000, limit_price=10.5)

    assert result.pending
    assert result.order_type == "LIMIT"
    assert result.limit_price == 10.5

    with session_scope() as s:
        order = s.get(OrderRecord, result.order_id)
        assert order.status == OrderStatus.SUBMITTED
        assert float(order.limit_price) == 10.5

        snap = s.scalars(
            select(AccountSnapshot)
            .where(AccountSnapshot.run_id == result.run_id)
            .order_by(AccountSnapshot.ts.desc())
        ).first()
        assert snap.frozen_cash > Decimal("0")
        assert snap.cash < Decimal("100000")


def test_manual_limit_buy_matches_after_price_crosses():
    loader = _QuoteLoader(price=11.0)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0)
    pending = service.place_limit_order(symbol="000001", side="BUY", quantity=1000, limit_price=10.5)

    loader.set_price(10.0)
    matched = service.match_open_limit_orders()

    assert len(matched) == 1
    assert matched[0].order_id == pending.order_id
    assert matched[0].ok
    assert matched[0].avg_fill_price == 10.5

    with session_scope() as s:
        order = s.get(OrderRecord, pending.order_id)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1000
        fills = s.scalars(select(TradeFill).where(TradeFill.order_id == pending.order_id)).all()
        assert len(fills) == 1

        snap = s.scalars(
            select(AccountSnapshot)
            .where(AccountSnapshot.run_id == pending.run_id)
            .order_by(AccountSnapshot.ts.desc())
        ).first()
        assert snap.frozen_cash == Decimal("0.0000")


def test_manual_limit_order_does_not_match_cached_quote():
    loader = _QuoteLoader(price=9.0, source="cache_daily_close")
    service = ManualTradingService(loader=loader, initial_cash=100_000.0)

    pending = service.place_limit_order(symbol="000001", side="BUY", quantity=1000, limit_price=10.5)
    matched = service.match_open_limit_orders()

    assert matched == []
    with session_scope() as s:
        order = s.get(OrderRecord, pending.order_id)
        assert order.status == OrderStatus.SUBMITTED


def test_manual_market_order_rejects_cached_quote():
    loader = _QuoteLoader(price=9.0, source="cache_daily_close")
    service = ManualTradingService(loader=loader, initial_cash=100_000.0)

    with pytest.raises(DataError, match="缓存价不能用于市价模拟成交"):
        service.place_market_order(symbol="000001", side="BUY", quantity=1000)


def test_manual_market_order_rejects_stale_quote():
    loader = _QuoteLoader(price=9.0, quote_age_seconds=120)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0)

    with pytest.raises(DataError, match="行情已过期"):
        service.place_market_order(symbol="000001", side="BUY", quantity=1000)


def test_manual_market_order_rejects_price_deviation_from_displayed_quote():
    loader = _QuoteLoader(price=10.10)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0, max_price_deviation_bps=50)

    with pytest.raises(DataError, match="价格偏离过大"):
        service.place_market_order(
            symbol="000001",
            side="BUY",
            quantity=1000,
            expected_price=10.00,
        )


def test_manual_price_deviation_threshold_comes_from_settings(monkeypatch):
    monkeypatch.setenv("QTRADE_TRADING__REALTIME__MAX_PRICE_DEVIATION_BPS", "25")
    cfg.get_settings.cache_clear()

    loader = _QuoteLoader(price=10.03)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0)

    assert service.max_price_deviation_bps == 25
    with pytest.raises(DataError, match="价格偏离过大"):
        service.place_market_order(
            symbol="000001",
            side="BUY",
            quantity=1000,
            expected_price=10.00,
        )


def test_manual_limit_immediate_fill_rejects_price_deviation_from_displayed_quote():
    loader = _QuoteLoader(price=10.00)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0, max_price_deviation_bps=50)

    with pytest.raises(DataError, match="价格偏离过大"):
        service.place_limit_order(
            symbol="000001",
            side="BUY",
            quantity=1000,
            limit_price=10.50,
            expected_price=11.00,
        )


def test_manual_pending_limit_order_allows_display_price_deviation():
    loader = _QuoteLoader(price=11.00)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0, max_price_deviation_bps=50)

    result = service.place_limit_order(
        symbol="000001",
        side="BUY",
        quantity=1000,
        limit_price=10.50,
        expected_price=10.00,
    )

    assert result.pending


def test_manual_limit_order_does_not_match_stale_quote():
    loader = _QuoteLoader(price=9.0, quote_age_seconds=120)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0)

    pending = service.place_limit_order(symbol="000001", side="BUY", quantity=1000, limit_price=10.5)
    matched = service.match_open_limit_orders()

    assert matched == []
    with session_scope() as s:
        order = s.get(OrderRecord, pending.order_id)
        assert order.status == OrderStatus.SUBMITTED


def test_manual_cancel_limit_buy_releases_frozen_cash():
    service = ManualTradingService(loader=_QuoteLoader(price=11.0), initial_cash=100_000.0)
    pending = service.place_limit_order(symbol="000001", side="BUY", quantity=1000, limit_price=10.5)

    cancelled = service.cancel_order(pending.order_id)

    assert cancelled.status == OrderStatus.CANCELLED.value
    with session_scope() as s:
        order = s.get(OrderRecord, pending.order_id)
        assert order.status == OrderStatus.CANCELLED
        snap = s.scalars(
            select(AccountSnapshot)
            .where(AccountSnapshot.run_id == pending.run_id)
            .order_by(AccountSnapshot.ts.desc())
        ).first()
        assert snap.frozen_cash == Decimal("0.0000")
        assert snap.cash == Decimal("100000.0000")


def test_manual_limit_buy_write_failure_rolls_back_freeze():
    service = ManualTradingService(loader=_QuoteLoader(price=11.0), initial_cash=100_000.0)
    original_write = service._write_order
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("db write failed")
        return original_write(**kwargs)

    service._write_order = fail_once
    result = service.place_limit_order(symbol="000001", side="BUY", quantity=1000, limit_price=10.5)

    assert result.status == OrderStatus.REJECTED.value
    assert result.account["cash"] == 100000.0
    assert result.account["frozen_cash"] == 0.0


def test_manual_limit_sell_write_failure_rolls_back_available_lock():
    loader = _QuoteLoader(price=11.0)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0)
    buy = service.place_market_order(symbol="000001", side="BUY", quantity=1000)

    with session_scope() as s:
        snaps = s.scalars(
            select(AccountSnapshot).where(AccountSnapshot.run_id == buy.run_id)
        ).all()
        for snap in snaps:
            snap.ts = snap.ts - timedelta(days=1)

    service_next_day = ManualTradingService(loader=loader, initial_cash=100_000.0)
    original_write = service_next_day._write_order
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("db write failed")
        return original_write(**kwargs)

    service_next_day._write_order = fail_once
    result = service_next_day.place_limit_order(
        symbol="000001",
        side="SELL",
        quantity=600,
        limit_price=12.0,
    )

    assert result.status == OrderStatus.REJECTED.value
    assert result.account["positions"][SYM]["available"] == 1000


def test_manual_limit_sell_reserves_then_fills_after_t1():
    loader = _QuoteLoader(price=11.0)
    service = ManualTradingService(loader=loader, initial_cash=100_000.0)
    buy = service.place_market_order(symbol="000001", side="BUY", quantity=1000)

    with session_scope() as s:
        snaps = s.scalars(
            select(AccountSnapshot).where(AccountSnapshot.run_id == buy.run_id)
        ).all()
        for snap in snaps:
            snap.ts = snap.ts - timedelta(days=1)

    service_next_day = ManualTradingService(loader=loader, initial_cash=100_000.0)
    pending = service_next_day.place_limit_order(
        symbol="000001",
        side="SELL",
        quantity=1000,
        limit_price=12.0,
    )

    assert pending.pending
    loader.set_price(12.5)
    matched = service_next_day.match_open_limit_orders()

    assert len(matched) == 1
    assert matched[0].ok
    assert matched[0].side == "SELL"
