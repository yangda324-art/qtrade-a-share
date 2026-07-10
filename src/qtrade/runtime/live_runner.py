"""LiveRunner：实时盯盘核心循环。

每次 ``tick()`` 做的事：
1. 拉所有 universe 标的的最新快照（akshare）
2. 把价格喂给 ``PaperBroker`` → 触发挂单撮合 + 更新持仓 mark price
3. 拉历史日线 + 拼上今日实时价 → 给策略生成信号
4. 把每个信号过风控 → 经 OMS 下单 → 落 SQLite
5. 调用 ``StateManager.snapshot()`` 把账户 + 持仓写一条快照

设计取舍：
- 我们的策略是日级的（双均线、LLM Agent 等都看日线）。盘中频繁 tick **不应**让策略
  反复"下又撤"。处理方式：每个 tick 把当日实时价当成"今天还没收盘的最后一根 K 线"
  追加到历史末尾，策略按这条新 K 线做判断。
- 当日已经在某个 symbol 上下过单了，再来的 tick 就跳过——避免追单/撤单循环。
  这是简化做法；真正灵活的策略要自己管理"目标仓位"语义。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal

import pandas as pd

from qtrade.core.config import get_settings
from qtrade.core.exceptions import RiskRejected
from qtrade.core.logging import get_logger
from qtrade.core.time import market_today, utcnow
from qtrade.data.adapters.base import Quote
from qtrade.data.loader import DataLoader, get_loader
from qtrade.execution.broker.paper import PaperBroker
from qtrade.execution.oms import OMS
from qtrade.persistence import session_scope
from qtrade.persistence.models import AgentDecision, OrderSide, OrderStatus, OrderType, TickLog
from qtrade.risk.engine import RiskEngine
from qtrade.runtime.limit_order_manager import PersistentLimitOrderManager
from qtrade.runtime.state import RunSession, StateManager
from qtrade.strategies.base import Signal, Strategy, StrategyContext

log = get_logger(__name__)


@dataclass
class TickReport:
    """单次 tick 执行结果，给 scheduler / UI 看。"""

    ts: datetime
    quotes_count: int
    signals_generated: int
    orders_placed: int
    risk_rejected: int
    strategy_skipped: int = 0
    limit_orders_submitted: int = 0
    limit_orders_matched: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SubmitResult:
    """单个策略信号的提交结果，用于 tick 审计。"""

    kind: str
    reason: str | None = None


@dataclass
class LiveRunner:
    strategy: Strategy
    state: StateManager
    risk: RiskEngine
    broker: PaperBroker
    loader: DataLoader = field(default_factory=get_loader)
    history_lookback_days: int = 90       # 拉多少天历史日线供策略看
    skip_if_traded_today: bool = True
    decision_interval_seconds: int = 0    # >0 时限制每个标的策略决策频率；行情和撮合仍每 tick 执行
    max_quote_age_seconds: int | None = None
    max_price_deviation_bps: int | None = None

    # 内部
    _session: RunSession | None = None
    _oms: OMS | None = None
    _limit_orders: PersistentLimitOrderManager | None = None
    _today_traded: set[str] = field(default_factory=set)
    _today_date: date | None = None
    _last_decision_at: dict[str, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        realtime_cfg = get_settings().trading.realtime
        if self.max_quote_age_seconds is None:
            self.max_quote_age_seconds = realtime_cfg.max_quote_age_seconds
        if self.max_price_deviation_bps is None:
            self.max_price_deviation_bps = realtime_cfg.max_price_deviation_bps

    # ---------------- 生命周期 ----------------

    def start(self, *, initial_cash: float = 100_000.0) -> RunSession:
        self._session = self.state.load_or_create(initial_cash=initial_cash)
        self.broker.account = self._session.account     # 把恢复出的账户接给 broker

        self._oms = OMS(
            broker=self.broker,
            strategy_id=self.strategy.id,
            run_id=self._session.run_id,
            risk_check=self._risk_adapter,
        )
        self._limit_orders = PersistentLimitOrderManager(
            broker=self.broker,
            strategy_id=self.strategy.id,
            run_id=self._session.run_id,
            risk_check=self._risk_adapter,
        )
        # 让策略知道当前 run_id（用于 LLM agent_decisions 落库归属）
        if hasattr(self.strategy, "run_id"):
            self.strategy.run_id = self._session.run_id
        if hasattr(self.strategy, "allow_limit_orders") and "allow_limit_orders" not in getattr(self.strategy, "params", {}):
            self.strategy.allow_limit_orders = True
        self.strategy.on_start()
        log.info("LiveRunner started | strategy={} run_id={}",
                 self.strategy.id, self._session.run_id)
        return self._session

    def stop(self, note: str | None = None) -> None:
        if self._session is None:
            return
        try:
            self.strategy.on_finish()
        except Exception as e:
            log.warning("strategy on_finish error: {}", e)
        self.state.snapshot(self._session)
        self.state.end(self._session, note=note)
        log.info("LiveRunner stopped | run_id={}", self._session.run_id)

    # ---------------- 盘前 / 盘后 ----------------

    def pre_market(self) -> None:
        """盘前 09:00 调用：T+1 解锁、风控日重置、清今日交易记忆。"""
        if not self._session:
            return
        self._session.account.positions.settle_t1_all()
        self.broker.reset_daily()
        self.risk.reset_daily()
        self._today_traded.clear()
        self._last_decision_at.clear()
        self._today_date = market_today()
        self.state.snapshot(self._session)
        log.info("pre-market settled | T+1 unlocked + daily counters reset")

    def post_market(self) -> None:
        """盘后 15:30 调用：写日终 snapshot + 给账户级风控做检查。"""
        if not self._session:
            return
        ts = utcnow()
        decision = self.risk.check_account(self._session.account, ts=ts)
        if not decision.passed:
            log.warning("ACCOUNT RISK TRIGGERED: {}", decision.detail)
        self._session.account.snapshot(ts)
        self.state.snapshot(self._session)
        log.info(
            "post-market | total_assets={:.2f} cash={:.2f} mv={:.2f} realized={:.2f}",
            float(self._session.account.total_assets),
            float(self._session.account.cash),
            float(self._session.account.market_value),
            float(self._session.account.realized_pnl),
        )

    # ---------------- 盘中 tick ----------------

    def tick(self, *, quotes: dict | None = None) -> TickReport:
        """跑一个 tick。

        ``quotes`` 可由上层（如 ``MultiStrategyRunner``）预拉后注入，
        避免多策略并行时同一个 tick 内重复拉行情。
        """
        if not self._session or not self._oms:
            raise RuntimeError("LiveRunner not started; call start() first")

        ts = utcnow()
        report = TickReport(ts=ts, quotes_count=0, signals_generated=0, orders_placed=0, risk_rejected=0)
        symbols = list(self.strategy.universe)

        # 1) 拉快照 + 推给 broker（除非上层已预拉）
        if quotes is None:
            try:
                quotes = self.loader.get_quotes(symbols)
            except Exception as e:
                log.exception("quote fetch failed: {}", e)
                report.errors.append(f"quotes: {e}")
                report.strategy_skipped = len(symbols)
                self._write_tick_log(report, symbols=symbols, quotes={}, per_symbol=self._per_symbol_status(symbols, {}))
                return report
        # 只取本策略 universe 内的 quotes
        quotes = {sym: q for sym, q in quotes.items() if sym in symbols}
        per_symbol = self._per_symbol_status(symbols, quotes)
        report.quotes_count = len(quotes)
        if not quotes:
            log.warning("empty quotes for {}, skipping tick", self.strategy.id)
            report.strategy_skipped = len(symbols)
            self._write_tick_log(report, symbols=symbols, quotes=quotes, per_symbol=per_symbol)
            return report
        live_prices = {
            sym: q.price for sym, q in quotes.items()
            if q.is_realtime_tradeable(max_age_seconds=self.max_quote_age_seconds, now=ts)
        }
        if live_prices:
            self.broker.update_market(live_prices)
        try:
            matched = self._limit_orders.match_open_orders(run=self._session, prices=live_prices) if self._limit_orders else []
            report.limit_orders_matched = len(matched)
            for order in matched:
                if order.status.value == "FILLED":
                    self._today_traded.add(order.symbol)
        except Exception as e:
            log.exception("limit order match failed: {}", e)
            report.errors.append(f"limit_match: {e}")
        open_limit_symbols = self._limit_orders.open_symbols() if self._limit_orders else set()

        # 2) 跑策略 & 下单
        for sym in symbols:
            if self.skip_if_traded_today and sym in self._today_traded:
                per_symbol[sym]["skip_reason"] = "traded_today"
                report.strategy_skipped += 1
                continue
            if sym in open_limit_symbols:
                per_symbol[sym]["skip_reason"] = "open_limit_order"
                report.strategy_skipped += 1
                continue
            quote = quotes.get(sym)
            if quote is None or quote.price <= 0:
                per_symbol[sym]["skip_reason"] = "no_quote" if quote is None else "invalid_price"
                report.strategy_skipped += 1
                continue
            reject_reason = quote.realtime_reject_reason(
                max_age_seconds=self.max_quote_age_seconds,
                now=ts,
            )
            if reject_reason is not None:
                log.info(
                    "skip strategy on rejected realtime quote | strategy={} symbol={} source={} reason={}",
                    self.strategy.id, sym, quote.source, reject_reason,
                )
                per_symbol[sym]["skip_reason"] = reject_reason
                report.strategy_skipped += 1
                continue
            if not self._decision_due(sym, ts):
                per_symbol[sym]["skip_reason"] = "decision_interval"
                report.strategy_skipped += 1
                continue

            try:
                self._last_decision_at[sym] = ts
                per_symbol[sym]["strategy_executed"] = True
                signals = self._run_strategy_with_intraday_price(sym, quote, ts)
            except Exception as e:
                log.exception("strategy error on {}: {}", sym, e)
                report.errors.append(f"strategy[{sym}]: {e}")
                per_symbol[sym]["skip_reason"] = "strategy_error"
                continue

            for sig in signals:
                report.signals_generated += 1
                result = self._submit(sig, ts, quote=quote)
                if result.kind == "placed":
                    report.orders_placed += 1
                    self._today_traded.add(sym)
                elif result.kind == "limit":
                    report.orders_placed += 1
                    report.limit_orders_submitted += 1
                    self._today_traded.add(sym)
                elif result.kind == "limit_filled":
                    report.orders_placed += 1
                    report.limit_orders_submitted += 1
                    report.limit_orders_matched += 1
                    self._today_traded.add(sym)
                elif result.kind == "risk":
                    report.risk_rejected += 1
                    self._record_signal_issue(report, per_symbol[sym], sig, result, issue_key="order_rejections")
                elif result.kind in {"skipped", "error"}:
                    self._record_signal_issue(report, per_symbol[sym], sig, result, issue_key="order_skips")

        # 3) 写 snapshot
        self.state.snapshot(self._session)
        self._write_tick_log(report, symbols=symbols, quotes=quotes, per_symbol=per_symbol)
        return report

    # ---------------- 内部 ----------------

    def _per_symbol_status(self, symbols: list[str], quotes: dict[str, Quote]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for sym in symbols:
            quote = quotes.get(sym)
            out[sym] = {
                "symbol": sym,
                "price": float(quote.price) if quote is not None else None,
                "source": quote.source if quote is not None else None,
                "tradeable": bool(quote and quote.is_tradeable),
                "realtime_tradeable": bool(
                    quote and quote.is_realtime_tradeable(
                        max_age_seconds=self.max_quote_age_seconds,
                    )
                ),
                "age_seconds": quote.age_seconds() if quote is not None else None,
                "strategy_executed": False,
                "skip_reason": None if quote is not None else "no_quote",
            }
        return out

    def _write_tick_log(
        self,
        report: TickReport,
        *,
        symbols: list[str],
        quotes: dict[str, Quote],
        per_symbol: dict[str, dict],
    ) -> None:
        if self._session is None:
            return
        try:
            with session_scope() as s:
                s.add(
                    TickLog(
                        run_id=self._session.run_id,
                        strategy_id=self.strategy.id,
                        ts=report.ts,
                        quotes_count=report.quotes_count,
                        tradeable_quotes=sum(
                            1 for q in quotes.values()
                            if q.is_realtime_tradeable(max_age_seconds=self.max_quote_age_seconds, now=report.ts)
                        ),
                        fallback_quotes=sum(1 for q in quotes.values() if q.is_cache_fallback),
                        signals_generated=report.signals_generated,
                        strategy_skipped=report.strategy_skipped,
                        orders_placed=report.orders_placed,
                        risk_rejected=report.risk_rejected,
                        limit_orders_submitted=report.limit_orders_submitted,
                        limit_orders_matched=report.limit_orders_matched,
                        symbols_json=list(symbols),
                        per_symbol_json=[per_symbol[sym] for sym in symbols if sym in per_symbol],
                        errors_json=list(report.errors),
                    )
                )
        except Exception as e:
            log.warning("tick log write failed: {}", e)

    def _run_strategy_with_intraday_price(
        self, sym: str, quote: Quote, ts: datetime
    ) -> list[Signal]:
        """构造策略上下文：历史日线末尾追加"今日还在变的"虚拟 K 线。"""
        df = self.loader.get_daily_bars(
            sym,
            start=(ts.date() - timedelta(days=self.history_lookback_days * 2)).isoformat(),
            end=ts.date().isoformat(),
        )
        # 追加今日（如最后一行已是今天就替换）
        today_row = pd.DataFrame([{
            "date": pd.Timestamp(ts.date()),
            "open": quote.open,
            "high": quote.high,
            "low": quote.low,
            "close": quote.price,
            "volume": quote.volume,
            "amount": quote.amount,
            "symbol": sym,
            "adjust": "qfq",
        }])
        if not df.empty and pd.Timestamp(ts.date()) in df["date"].values:
            df = df[df["date"] != pd.Timestamp(ts.date())]
        df = pd.concat([df, today_row], ignore_index=True)

        ctx = StrategyContext(ts=ts, symbol=sym, history=df, account=self._session.account)
        signals = self.strategy.on_bar(ctx)
        for sig in signals:
            sig.metadata.setdefault("reference_price", quote.price)
            sig.metadata.setdefault("reference_ts", ts.isoformat())
        return signals

    def _decision_due(self, sym: str, ts: datetime) -> bool:
        if self.decision_interval_seconds <= 0:
            return True
        last = self._last_decision_at.get(sym)
        if last is None:
            return True
        return (ts - last).total_seconds() >= self.decision_interval_seconds

    def _submit(self, sig: Signal, ts: datetime, *, quote: Quote | None = None) -> SubmitResult:
        if sig.action not in {"BUY", "SELL"}:
            return SubmitResult("skipped", f"unsupported_action:{sig.action}")
        side = OrderSide(sig.action)
        decision_id = sig.metadata.get("decision_id")
        if not isinstance(decision_id, int):
            decision_id = None
        try:
            order_type = self._order_type(sig)
            if order_type == OrderType.LIMIT:
                if self._limit_orders is None or self._session is None:
                    raise RuntimeError("limit order manager not initialized")
                limit_price = self._limit_price(sig)
                if limit_price is None:
                    raise ValueError("LIMIT signal missing limit_price")
                match_price = (
                    Decimal(str(quote.price))
                    if quote is not None and quote.is_realtime_tradeable(
                        max_age_seconds=self.max_quote_age_seconds,
                        now=ts,
                    )
                    else None
                )
                if match_price is not None and self._is_marketable_limit(
                    side=side,
                    limit_price=limit_price,
                    price=match_price,
                ):
                    deviation_reason = self._price_deviation_reject_reason(sig, quote)
                    if deviation_reason is not None:
                        log.warning(
                            "LIMIT ORDER SKIPPED | {} {} {} @ {} | {}",
                            sig.action, sig.quantity, sig.symbol, limit_price, deviation_reason,
                        )
                        return SubmitResult("risk", deviation_reason)
                order = self._limit_orders.submit_limit_order(
                    run=self._session,
                    symbol=sig.symbol,
                    side=side,
                    quantity=sig.quantity,
                    limit_price=limit_price,
                    decision_id=decision_id,
                    ts=ts,
                    match_price=match_price,
                )
                log.info("LIMIT ORDER SUBMITTED | {} {} {} @ {} | reason={}",
                         sig.action, sig.quantity, sig.symbol, limit_price, sig.reason[:80])
                if order.status == OrderStatus.FILLED:
                    return SubmitResult("limit_filled")
                return SubmitResult("limit")

            reject_reason = (
                "missing_quote"
                if quote is None
                else quote.realtime_reject_reason(
                    max_age_seconds=self.max_quote_age_seconds,
                    now=ts,
                )
            )
            if reject_reason is not None:
                log.warning(
                    "MARKET ORDER SKIPPED | {} {} {} | quote rejected: {}",
                    sig.action, sig.quantity, sig.symbol, reject_reason,
                )
                return SubmitResult("skipped", f"quote_rejected:{reject_reason}")
            deviation_reason = self._price_deviation_reject_reason(sig, quote)
            if deviation_reason is not None:
                log.warning(
                    "MARKET ORDER SKIPPED | {} {} {} | {}",
                    sig.action, sig.quantity, sig.symbol, deviation_reason,
                )
                return SubmitResult("risk", deviation_reason)

            self._oms.place_order(
                symbol=sig.symbol,
                side=side,
                quantity=sig.quantity,
                order_type=OrderType.MARKET,
                decision_id=decision_id,
                ts=ts,
                salt=str(ts.minute),
            )
            log.info("ORDER PLACED | {} {} {} | reason={}",
                     sig.action, sig.quantity, sig.symbol, sig.reason[:80])
            return SubmitResult("placed")
        except RiskRejected as e:
            log.warning("ORDER REJECTED | {} {} {} | {}",
                        sig.action, sig.quantity, sig.symbol, e.detail)
            return SubmitResult("risk", e.detail)
        except Exception as e:
            log.exception("OMS error: {}", e)
            return SubmitResult("error", str(e))

    def _record_signal_issue(
        self,
        report: TickReport,
        symbol_status: dict,
        sig: Signal,
        result: SubmitResult,
        *,
        issue_key: str,
    ) -> None:
        if not result.reason:
            return
        issue = {
            "kind": result.kind,
            "action": sig.action,
            "quantity": sig.quantity,
            "reason": result.reason,
        }
        symbol_status.setdefault(issue_key, []).append(issue)
        report.errors.append(f"{result.kind}[{sig.symbol} {sig.action} x{sig.quantity}]: {result.reason}")
        self._mark_decision_blocked(sig, result)

    def _mark_decision_blocked(self, sig: Signal, result: SubmitResult) -> None:
        decision_id = sig.metadata.get("decision_id")
        if not isinstance(decision_id, int) or not result.reason:
            return
        try:
            with session_scope() as s:
                decision = s.get(AgentDecision, decision_id)
                if decision is None:
                    return
                decision.risk_passed = False
                decision.risk_reject_reason = result.reason[:5000]
                trace = dict(decision.trace_json or {})
                trace["execution"] = {
                    "status": result.kind,
                    "reason": result.reason,
                    "signal": {
                        "action": sig.action,
                        "quantity": sig.quantity,
                        "target_pct": sig.target_pct,
                    },
                }
                decision.trace_json = trace
        except Exception as e:
            log.warning("mark decision blocked failed | decision_id={} | {}", decision_id, e)

    def _order_type(self, sig: Signal) -> OrderType:
        raw = str(sig.metadata.get("order_type") or OrderType.MARKET.value).upper()
        try:
            return OrderType(raw)
        except ValueError:
            return OrderType.MARKET

    def _limit_price(self, sig: Signal):
        order_type = self._order_type(sig)
        if order_type != OrderType.LIMIT:
            return None
        raw = sig.metadata.get("limit_price")
        if raw is None:
            return None
        return Decimal(str(raw))

    @staticmethod
    def _is_marketable_limit(*, side: OrderSide, limit_price: Decimal, price: Decimal) -> bool:
        if side == OrderSide.BUY:
            return price <= limit_price
        return price >= limit_price

    @staticmethod
    def _price_deviation_bps(*, reference_price: Decimal, quote_price: Decimal) -> Decimal:
        return abs((quote_price / reference_price) - Decimal("1")) * Decimal("10000")

    def _price_deviation_reject_reason(self, sig: Signal, quote: Quote | None) -> str | None:
        if quote is None:
            return None
        raw = sig.metadata.get("reference_price")
        if raw is None:
            return None
        reference = Decimal(str(raw))
        if reference <= 0:
            return f"reference_price invalid: {raw}"
        actual = Decimal(str(quote.price))
        deviation = self._price_deviation_bps(reference_price=reference, quote_price=actual)
        limit = Decimal(str(self.max_price_deviation_bps))
        if deviation <= limit:
            return None
        return (
            f"price_deviation {float(deviation):.1f}bp exceeds "
            f"{self.max_price_deviation_bps}bp "
            f"(reference={float(reference):.3f}, quote={float(actual):.3f})"
        )

    def _risk_adapter(self, ctx: dict) -> tuple[bool, str | None]:
        d = self.risk.check_pre_trade(
            symbol=ctx["symbol"],
            side=ctx["side"],
            quantity=int(ctx["quantity"]),
            price=float(ctx.get("limit_price") or 0) or float(self.broker.last_prices.get(ctx["symbol"], 1) or 1),
            account=self._session.account,
        )
        return d.passed, d.detail
