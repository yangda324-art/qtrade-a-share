"""手动模拟交易服务。

UI 手动买卖不能直接改数据库或持仓，必须走和自动策略相同的链路：

实时行情 -> RiskEngine -> OMS -> Broker -> SQLite snapshot

当前实现绑定 ``PaperBroker``，但 UI 只依赖这个服务层。以后接真实券商时，服务层可以
注入 RealBroker，UI 不需要重写。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import desc, select

from qtrade.core.config import get_settings
from qtrade.core.exceptions import DataError, QtradeError, RiskRejected
from qtrade.core.time import market_date_from_utc_naive, market_today, utcnow
from qtrade.data.adapters.base import Quote
from qtrade.data.loader import DataLoader, get_loader
from qtrade.data.schema import normalize_symbol
from qtrade.execution.broker.paper import PaperBroker
from qtrade.execution.oms import OMS
from qtrade.persistence import init_db, session_scope
from qtrade.persistence.models import (
    AccountSnapshot,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeFill,
)
from qtrade.risk.engine import RiskEngine
from qtrade.risk.policies import RiskPolicies, load_risk_policies
from qtrade.runtime.state import RunSession, StateManager

MANUAL_STRATEGY_ID = "manual_paper"


@dataclass(frozen=True)
class ManualOrderResult:
    run_id: int
    run_uid: str
    order_id: int
    status: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    avg_fill_price: float | None
    reject_reason: str | None
    quote_price: float
    account: dict
    order_type: str = OrderType.MARKET.value
    limit_price: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == OrderStatus.FILLED.value

    @property
    def pending(self) -> bool:
        return self.status == OrderStatus.SUBMITTED.value


class ManualTradingService:
    """手动模拟交易入口。"""

    def __init__(
        self,
        *,
        initial_cash: float = 100_000.0,
        loader: DataLoader | None = None,
        policies: RiskPolicies | None = None,
        state: StateManager | None = None,
        max_quote_age_seconds: int | None = None,
        max_price_deviation_bps: int | None = None,
    ) -> None:
        realtime_cfg = get_settings().trading.realtime
        self.initial_cash = initial_cash
        self.loader = loader or get_loader()
        self.policies = policies or load_risk_policies()
        self.state = state or StateManager(strategy_id=MANUAL_STRATEGY_ID, mode="paper")
        self.risk = RiskEngine(policies=self.policies)
        self.max_quote_age_seconds = (
            max_quote_age_seconds
            if max_quote_age_seconds is not None
            else realtime_cfg.max_quote_age_seconds
        )
        self.max_price_deviation_bps = (
            max_price_deviation_bps
            if max_price_deviation_bps is not None
            else realtime_cfg.max_price_deviation_bps
        )

    def get_quote(self, symbol: str) -> Quote:
        """获取单只股票最新快照。"""
        sym = normalize_symbol(symbol)
        quotes = self.loader.get_quotes([sym])
        quote = quotes.get(sym)
        if quote is None or quote.price <= 0:
            raise DataError(f"无法获取 {sym} 最新行情")
        return quote

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        expected_price: float | Decimal | None = None,
    ) -> ManualOrderResult:
        """提交一笔手动市价单。"""
        return self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            expected_price=expected_price,
        )

    def place_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: float | Decimal,
        expected_price: float | Decimal | None = None,
    ) -> ManualOrderResult:
        """提交一笔手动限价单。"""
        return self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            expected_price=expected_price,
        )

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str | OrderType = OrderType.MARKET,
        limit_price: float | Decimal | None = None,
        expected_price: float | Decimal | None = None,
    ) -> ManualOrderResult:
        """提交手动模拟订单。"""
        if quantity <= 0:
            raise QtradeError("quantity must be positive")

        sym = normalize_symbol(symbol)
        side_enum = OrderSide(side.upper())
        type_enum = order_type if isinstance(order_type, OrderType) else OrderType(order_type.upper())

        if type_enum == OrderType.LIMIT:
            return self._place_limit_order(
                symbol=sym,
                side=side_enum,
                quantity=quantity,
                limit_price=self._normalize_limit_price(limit_price),
                expected_price=expected_price,
            )
        return self._place_market_order(
            symbol=sym,
            side=side_enum,
            quantity=quantity,
            expected_price=expected_price,
        )

    def _place_market_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        expected_price: float | Decimal | None,
    ) -> ManualOrderResult:
        init_db()
        session = self.state.load_or_create(
            initial_cash=self.initial_cash,
            config={"source": "manual_ui"},
        )
        self._settle_overnight_t1(session)
        broker = PaperBroker(account=session.account)

        quote = self.get_quote(symbol)
        reject_reason = quote.realtime_reject_reason(max_age_seconds=self.max_quote_age_seconds)
        if reject_reason is not None:
            raise DataError(self._quote_reject_message(symbol, reject_reason, order_type=OrderType.MARKET))
        self._check_expected_price(symbol=symbol, quote=quote, expected_price=expected_price)
        broker.update_market({symbol: quote.price})

        oms = OMS(
            broker=broker,
            strategy_id=MANUAL_STRATEGY_ID,
            run_id=session.run_id,
            risk_check=lambda ctx: self._risk_adapter(ctx, session, broker),
        )
        order = oms.place_order(
            symbol=symbol,
            side=side,
            quantity=int(quantity),
            order_type=OrderType.MARKET,
            salt=f"manual-{uuid.uuid4().hex[:12]}",
        )

        self.state.snapshot(session)
        return self._result_from_order(order, quote=quote, run=session)

    def _place_limit_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: Decimal,
        expected_price: float | Decimal | None,
    ) -> ManualOrderResult:
        init_db()
        session = self.state.load_or_create(
            initial_cash=self.initial_cash,
            config={"source": "manual_ui"},
        )
        self._settle_overnight_t1(session)
        broker = PaperBroker(account=session.account)

        quote = self.get_quote(symbol)
        quote_reject_reason = quote.realtime_reject_reason(max_age_seconds=self.max_quote_age_seconds)
        if quote_reject_reason is None:
            broker.update_market({symbol: quote.price})

        decision = self.risk.check_pre_trade(
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            price=float(limit_price),
            account=session.account,
        )
        if not decision.passed:
            raise RiskRejected("risk_check", decision.detail or "rejected")

        marketable = self._is_marketable_limit(side=side, limit_price=limit_price, quote=quote)
        if marketable and quote_reject_reason is None:
            self._check_expected_price(symbol=symbol, quote=quote, expected_price=expected_price)
            client_order_id = self._new_client_order_id()
            res = broker.execute_order_at_price(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=limit_price,
            )
            order = self._write_order(
                run=session,
                client_order_id=client_order_id,
                broker_order_id=res.broker_order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                limit_price=limit_price,
                status=res.status,
                filled_quantity=res.filled_quantity,
                avg_fill_price=res.avg_fill_price,
                commission=res.commission,
                stamp_tax=res.stamp_tax,
                reject_reason=res.reject_reason,
            )
            self.state.snapshot(session)
            return self._result_from_order(order, quote=quote, run=session)

        reserved_cash: Decimal | None = None
        try:
            broker_order_id = f"PB-{uuid.uuid4().hex[:12]}"
            client_order_id = self._new_client_order_id()
            reserved_cash = self._reserve_limit_order(
                run=session,
                broker=broker,
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
            )
            order = self._write_order(
                run=session,
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.SUBMITTED,
            )
            self.state.snapshot(session)
            return self._result_from_order(order, quote=quote, run=session)
        except Exception as e:
            if reserved_cash is not None:
                self._release_limit_reserve(
                    run=session,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    reserved_cash=reserved_cash,
                )
            order = self._write_order(
                run=session,
                client_order_id=self._new_client_order_id(),
                broker_order_id=f"PB-{uuid.uuid4().hex[:12]}",
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.REJECTED,
                reject_reason=str(e),
            )
            return self._result_from_order(order, quote=quote, run=session)

    def match_open_limit_orders(self) -> list[ManualOrderResult]:
        """用最新实时行情撮合当前手动模拟盘的未成交限价单。

        只用真实行情源触发成交；如果行情是 ``cache_daily_close``，只展示价格，不制造假成交。
        """
        init_db()
        pending = self._pending_limit_orders()
        if not pending:
            return []

        session = self.state.load_or_create(
            initial_cash=self.initial_cash,
            config={"source": "manual_ui"},
        )
        self._settle_overnight_t1(session)

        symbols = sorted({o["symbol"] for o in pending})
        quotes = self.loader.get_quotes(symbols)
        live_quotes = {
            sym: q for sym, q in quotes.items()
            if q.is_realtime_tradeable(max_age_seconds=self.max_quote_age_seconds)
        }
        if not live_quotes:
            return []

        broker = PaperBroker(account=session.account)
        broker.update_market({sym: quote.price for sym, quote in live_quotes.items()})

        results: list[ManualOrderResult] = []
        for row in pending:
            quote = live_quotes.get(row["symbol"])
            if quote is None:
                continue
            limit_price = Decimal(str(row["limit_price"]))
            side = OrderSide(row["side"])
            if not self._is_marketable_limit(side=side, limit_price=limit_price, quote=quote):
                continue

            if side == OrderSide.BUY:
                session.account.unfreeze(
                    self._buy_reserve(
                        broker=broker,
                        symbol=row["symbol"],
                        quantity=row["quantity"],
                        price=limit_price,
                    )
                )
            else:
                pos = session.account.positions.get(row["symbol"])
                pos.available = min(pos.quantity, pos.available + row["quantity"])

            res = broker.execute_order_at_price(
                broker_order_id=row["broker_order_id"],
                client_order_id=row["client_order_id"],
                symbol=row["symbol"],
                side=side,
                quantity=row["quantity"],
                price=limit_price,
            )
            order = self._update_order_from_broker_result(order_id=row["id"], result=res)
            results.append(self._result_from_order(order, quote=quote, run=session))

        if results:
            self.state.snapshot(session)
        return results

    def cancel_order(self, order_id: int) -> ManualOrderResult:
        """撤销一笔手动限价挂单。"""
        init_db()
        row = self._get_open_limit_order(order_id)
        if row is None:
            raise QtradeError("只能撤销未成交的手动限价挂单")

        session = self.state.load_or_create(
            initial_cash=self.initial_cash,
            config={"source": "manual_ui"},
        )
        broker = PaperBroker(account=session.account)

        side = OrderSide(row["side"])
        limit_price = Decimal(str(row["limit_price"]))
        if side == OrderSide.BUY:
            session.account.unfreeze(
                self._buy_reserve(
                    broker=broker,
                    symbol=row["symbol"],
                    quantity=row["quantity"],
                    price=limit_price,
                )
            )
        else:
            pos = session.account.positions.get(row["symbol"])
            pos.available = min(pos.quantity, pos.available + row["quantity"])

        with session_scope() as s:
            order = s.get(OrderRecord, order_id)
            order.status = OrderStatus.CANCELLED
            order.reject_reason = "manual cancel"
            s.flush()
            saved = s.get(OrderRecord, order_id)
        self.state.snapshot(session)
        quote = self._quote_from_limit(row)
        return self._result_from_order(saved, quote=quote, run=session)

    def _result_from_order(
        self,
        order: OrderRecord,
        *,
        quote: Quote,
        run: RunSession,
    ) -> ManualOrderResult:
        return ManualOrderResult(
            run_id=run.run_id,
            run_uid=run.run_uid,
            order_id=order.id,
            status=order.status.value,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            avg_fill_price=float(order.avg_fill_price) if order.avg_fill_price is not None else None,
            reject_reason=order.reject_reason,
            quote_price=quote.price,
            account=run.account.to_dict(),
            order_type=order.order_type.value,
            limit_price=float(order.limit_price) if order.limit_price is not None else None,
        )

    def _risk_adapter(
        self,
        ctx: dict,
        session: RunSession,
        broker: PaperBroker,
    ) -> tuple[bool, str | None]:
        price = float(broker.last_prices.get(ctx["symbol"], Decimal("0")) or 0)
        decision = self.risk.check_pre_trade(
            symbol=ctx["symbol"],
            side=ctx["side"],
            quantity=int(ctx["quantity"]),
            price=price,
            account=session.account,
        )
        return decision.passed, decision.detail

    def _settle_overnight_t1(self, session: RunSession) -> None:
        """如果最近快照不是今天，释放上一交易日买入的可卖数量。"""
        with session_scope() as s:
            snap = s.scalars(
                select(AccountSnapshot)
                .where(AccountSnapshot.run_id == session.run_id)
                .order_by(desc(AccountSnapshot.ts))
            ).first()
        if snap is None or market_date_from_utc_naive(snap.ts) >= market_today():
            return

        changed = False
        for pos in session.account.positions:
            if pos.quantity > pos.available:
                pos.available = pos.quantity
                pos.today_buy = 0
                changed = True
        if changed:
            self.state.snapshot(session)

    def _reserve_limit_order(
        self,
        *,
        run: RunSession,
        broker: PaperBroker,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: Decimal,
    ) -> Decimal:
        if side == OrderSide.BUY:
            reserved_cash = self._buy_reserve(
                broker=broker,
                symbol=symbol,
                quantity=quantity,
                price=limit_price,
            )
            run.account.freeze(reserved_cash)
            return reserved_cash

        pos = run.account.positions.get(symbol)
        if pos.available < quantity:
            raise QtradeError(f"insufficient available shares ({pos.available} < {quantity})")
        pos.available -= quantity
        return Decimal("0")

    def _release_limit_reserve(
        self,
        *,
        run: RunSession,
        symbol: str,
        side: OrderSide,
        quantity: int,
        reserved_cash: Decimal,
    ) -> None:
        if side == OrderSide.BUY:
            run.account.unfreeze(reserved_cash)
            return

        pos = run.account.positions.get(symbol)
        pos.available = min(pos.quantity, pos.available + quantity)

    def _write_order(
        self,
        *,
        run: RunSession,
        client_order_id: str,
        broker_order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        limit_price: Decimal | None = None,
        status: OrderStatus,
        filled_quantity: int = 0,
        avg_fill_price: Decimal | None = None,
        commission: Decimal = Decimal("0"),
        stamp_tax: Decimal = Decimal("0"),
        reject_reason: str | None = None,
    ) -> OrderRecord:
        with session_scope() as s:
            order = OrderRecord(
                run_id=run.run_id,
                strategy_id=MANUAL_STRATEGY_ID,
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                status=status,
                filled_quantity=filled_quantity,
                avg_fill_price=avg_fill_price,
                commission=commission,
                stamp_tax=stamp_tax,
                reject_reason=reject_reason,
            )
            s.add(order)
            s.flush()
            order_id = order.id
            if status == OrderStatus.FILLED and filled_quantity > 0:
                s.add(
                    TradeFill(
                        order_id=order_id,
                        quantity=filled_quantity,
                        price=avg_fill_price or Decimal("0"),
                        commission=commission,
                        stamp_tax=stamp_tax,
                        filled_at=utcnow(),
                    )
                )
            s.flush()
            return s.get(OrderRecord, order_id)

    def _update_order_from_broker_result(self, *, order_id: int, result) -> OrderRecord:
        with session_scope() as s:
            order = s.get(OrderRecord, order_id)
            order.status = result.status
            order.filled_quantity = result.filled_quantity
            order.avg_fill_price = result.avg_fill_price
            order.commission = result.commission
            order.stamp_tax = result.stamp_tax
            order.reject_reason = result.reject_reason
            if result.status == OrderStatus.FILLED and result.filled_quantity > 0:
                s.add(
                    TradeFill(
                        order_id=order_id,
                        quantity=result.filled_quantity,
                        price=result.avg_fill_price or Decimal("0"),
                        commission=result.commission,
                        stamp_tax=result.stamp_tax,
                        filled_at=utcnow(),
                    )
                )
            s.flush()
            return s.get(OrderRecord, order_id)

    def _pending_limit_orders(self) -> list[dict[str, Any]]:
        with session_scope() as s:
            rows = s.scalars(
                select(OrderRecord).where(
                    OrderRecord.strategy_id == MANUAL_STRATEGY_ID,
                    OrderRecord.order_type == OrderType.LIMIT,
                    OrderRecord.status == OrderStatus.SUBMITTED,
                )
            ).all()
            return [self._order_row(o) for o in rows]

    def _get_open_limit_order(self, order_id: int) -> dict[str, Any] | None:
        with session_scope() as s:
            order = s.get(OrderRecord, order_id)
            if (
                order is None
                or order.strategy_id != MANUAL_STRATEGY_ID
                or order.order_type != OrderType.LIMIT
                or order.status != OrderStatus.SUBMITTED
            ):
                return None
            return self._order_row(order)

    @staticmethod
    def _order_row(order: OrderRecord) -> dict[str, Any]:
        return {
            "id": order.id,
            "run_id": order.run_id,
            "client_order_id": order.client_order_id,
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "limit_price": float(order.limit_price or 0),
        }

    @staticmethod
    def _price_deviation_bps(*, expected_price: Decimal, quote_price: Decimal) -> Decimal:
        return abs((quote_price / expected_price) - Decimal("1")) * Decimal("10000")

    def _check_expected_price(
        self,
        *,
        symbol: str,
        quote: Quote,
        expected_price: float | Decimal | None,
    ) -> None:
        if expected_price is None:
            return
        expected = Decimal(str(expected_price))
        if expected <= 0:
            raise DataError(f"{symbol} 参考价无效，不能用于实时模拟成交")
        actual = Decimal(str(quote.price))
        deviation = self._price_deviation_bps(expected_price=expected, quote_price=actual)
        limit = Decimal(str(self.max_price_deviation_bps))
        if deviation > limit:
            raise DataError(
                f"{symbol} 价格偏离过大：最新价 {float(actual):.3f} 相对界面参考价 "
                f"{float(expected):.3f} 偏离 {float(deviation):.1f}bp，超过 {self.max_price_deviation_bps}bp；"
                "请刷新行情后重试"
            )

    @staticmethod
    def _quote_reject_message(symbol: str, reason: str, *, order_type: OrderType) -> str:
        action = "市价模拟成交" if order_type == OrderType.MARKET else "实时模拟成交"
        if reason == "fallback_quote":
            return f"{symbol} 实时行情不可用，缓存价不能用于{action}"
        if reason == "stale_quote":
            return f"{symbol} 行情已过期，不能用于{action}"
        if reason == "invalid_quote":
            return f"{symbol} 行情价格无效，不能用于{action}"
        return f"{symbol} 行情状态不明确（{reason}），不能用于{action}"

    @staticmethod
    def _normalize_limit_price(limit_price: float | Decimal | None) -> Decimal:
        if limit_price is None:
            raise QtradeError("LIMIT order missing limit_price")
        price = Decimal(str(limit_price))
        if price <= 0:
            raise QtradeError("limit_price must be positive")
        return price.quantize(Decimal("0.0001"))

    @staticmethod
    def _is_marketable_limit(*, side: OrderSide, limit_price: Decimal, quote: Quote) -> bool:
        price = Decimal(str(quote.price))
        if side == OrderSide.BUY:
            return price <= limit_price
        return price >= limit_price

    @staticmethod
    def _buy_reserve(
        *,
        broker: PaperBroker,
        symbol: str,
        quantity: int,
        price: Decimal,
    ) -> Decimal:
        fees = broker.fee_model.calc(
            symbol=symbol,
            side=OrderSide.BUY.value,
            quantity=quantity,
            price=float(price),
        )
        return price * Decimal(quantity) + fees.total

    @staticmethod
    def _new_client_order_id() -> str:
        return f"COID-{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _quote_from_limit(row: dict[str, Any]) -> Quote:
        price = float(row["limit_price"])
        return Quote(
            symbol=row["symbol"],
            price=price,
            open=price,
            high=price,
            low=price,
            prev_close=price,
            volume=0,
            amount=0,
            ts=utcnow(),
            source="manual_order",
        )
