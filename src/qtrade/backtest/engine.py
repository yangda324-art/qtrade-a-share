"""事件驱动回测引擎。

最重要的设计点：**防 look-ahead**。

实现：
- 所有标的的日线对齐到统一的交易日序列。
- 对每个交易日 ``t``：
    1. **盘前**：``positions.settle_t1_all()`` 把昨日买入解锁、reset_daily 计数
    2. **每个标的**：``ctx.history`` **只切到 t（含）**——策略看不到未来
    3. 策略产 ``Signal``，**用 t 的收盘价 + 1 bp 滑点撮合**（保守口径）
    4. **盘后**：``account.snapshot(t)``，并跑 ``risk.check_account``
- 单日撮合用收盘价是日级回测的标准做法；如果想严格点，
  可以改成下一日开盘价（避免"用收盘价决定+用收盘价成交"）

后续优化点（已留接口）：
- 滑点模型可换：``slippage_bps`` 或注入自定义函数
- 多股票组合：当前简化为按信号顺序处理；做多策略组合时可加 portfolio 层
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from qtrade.backtest.metrics import Metrics, compute_metrics
from qtrade.core.logging import get_logger
from qtrade.data.loader import DataLoader, get_loader
from qtrade.execution.account import Account
from qtrade.execution.broker.paper import PaperBroker
from qtrade.execution.fees import AStockFeeModel
from qtrade.execution.oms import OMS
from qtrade.persistence import init_db
from qtrade.persistence.models import OrderSide, OrderStatus, OrderType
from qtrade.risk.engine import RiskEngine
from qtrade.risk.policies import RiskPolicies, load_risk_policies
from qtrade.strategies.base import Signal, Strategy, StrategyContext

log = get_logger(__name__)


@dataclass
class BacktestResult:
    metrics: Metrics
    equity_curve: list[tuple[datetime, Decimal]]
    trades: list[dict]
    final_account: dict
    trade_count: int | None = None


@dataclass
class BacktestEngine:
    strategy: Strategy
    start: date
    end: date
    initial_cash: float = 100_000.0
    risk_policies: RiskPolicies | None = None
    loader: DataLoader = field(default_factory=get_loader)
    slippage_bps: float = 1.0
    persist: bool = False              # 是否把订单写库（默认关，避免回测污染审计表）
    record_trades: bool = True         # 批量回测可关闭完整成交明细，降低内存占用

    def run(self) -> BacktestResult:
        # ----- 准备数据 -----
        symbols = list(self.strategy.universe)
        if not symbols:
            raise ValueError("strategy.universe 为空")

        bars: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.loader.get_daily_bars(sym, start=self.start, end=self.end)
            if df.empty:
                log.warning("no bars for {}, skipping", sym)
                continue
            df = df.sort_values("date").reset_index(drop=True)
            df["date"] = pd.to_datetime(df["date"])
            bars[sym] = df

        if not bars:
            raise RuntimeError("no usable data")

        # 统一交易日序列（任何一个标的有数据即视为交易日）
        all_dates = sorted({d for df in bars.values() for d in df["date"]})

        # ----- 准备账户/Broker/OMS/风控 -----
        account = Account(initial_cash=Decimal(str(self.initial_cash)))
        broker = PaperBroker(
            account=account,
            fee_model=AStockFeeModel(),
            slippage_bps=self.slippage_bps,
        )
        risk = RiskEngine(policies=self.risk_policies or load_risk_policies())

        if self.persist:
            init_db()
        oms = (
            OMS(
                broker=broker,
                strategy_id=self.strategy.id,
                risk_check=lambda _ctx: (True, None),  # 已在 _execute_signal 用真实成交价做过风控
            )
            if self.persist
            else None
        )

        trades: list[dict] = []
        trade_pnls: list[float] = []
        trade_count = 0

        self.strategy.on_start()
        try:
            for t in all_dates:
                # 1) 盘前：T+1 解锁、风控日重置
                account.positions.settle_t1_all()
                broker.reset_daily()
                risk.reset_daily(t.date() if hasattr(t, "date") else t)

                # 2) 把 t 的收盘价推给 broker（更新 last_prices 和持仓 mark）
                close_today = {}
                for sym, df in bars.items():
                    row = df[df["date"] == t]
                    if not row.empty:
                        close_today[sym] = float(row.iloc[0]["close"])
                broker.update_market(close_today)

                # 3) 每个标的跑策略
                for sym, df in bars.items():
                    hist = df[df["date"] <= t].copy()
                    if hist.empty:
                        continue
                    ctx = StrategyContext(
                        ts=t.to_pydatetime() if hasattr(t, "to_pydatetime") else datetime.combine(t, datetime.min.time()),
                        symbol=sym,
                        history=hist,
                        account=account,
                    )
                    try:
                        signals = self.strategy.on_bar(ctx)
                    except Exception as e:
                        log.exception("strategy error on {} @ {}: {}", sym, t, e)
                        continue

                    for sig in signals:
                        filled = self._execute_signal(
                            sig,
                            broker,
                            risk,
                            account,
                            t,
                            trades,
                            trade_pnls,
                            oms,
                            trade_index=trade_count,
                            record_trade=self.record_trades,
                        )
                        if filled:
                            trade_count += 1

                # 4) 盘后：账户级风控 + 权益快照
                ts_dt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else datetime.combine(t, datetime.min.time())
                acct_decision = risk.check_account(account, ts=ts_dt)
                if not acct_decision.passed:
                    log.warning("circuit broken: {}", acct_decision.detail)
                account.snapshot(ts_dt)
        finally:
            self.strategy.on_finish()

        metrics = compute_metrics(account.equity_curve, trade_pnls=trade_pnls)
        return BacktestResult(
            metrics=metrics,
            equity_curve=list(account.equity_curve),
            trades=trades,
            final_account=account.to_dict(),
            trade_count=trade_count,
        )

    # ------------------------------------------------------------------

    def _execute_signal(
        self,
        sig: Signal,
        broker: PaperBroker,
        risk: RiskEngine,
        account: Account,
        ts,
        trades: list[dict],
        trade_pnls: list[float],
        oms: OMS | None = None,
        *,
        trade_index: int = 0,
        record_trade: bool = True,
    ) -> bool:
        if sig.action not in {"BUY", "SELL"}:
            return False
        side = OrderSide(sig.action)
        price_now = float(broker.last_prices.get(sig.symbol, Decimal("0")))
        if price_now <= 0:
            return False

        decision = risk.check_pre_trade(
            symbol=sig.symbol,
            side=sig.action,
            quantity=sig.quantity,
            price=price_now,
            account=account,
            today=ts.date() if hasattr(ts, "date") else ts,
        )
        if not decision.passed:
            log.info("risk reject | {} {} {}@{} | {}",
                     sig.action, sig.quantity, sig.symbol, price_now, decision.detail)
            return False

        cash_before = account.cash
        realized_before = account.realized_pnl
        try:
            if oms is not None:
                order = oms.place_order(
                    symbol=sig.symbol,
                    side=side,
                    quantity=sig.quantity,
                    order_type=OrderType.MARKET,
                    decision_id=sig.metadata.get("decision_id")
                    if isinstance(sig.metadata.get("decision_id"), int)
                    else None,
                    ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    salt=f"bt-{trade_index}",
                )
                status = order.status
                filled_quantity = order.filled_quantity
                avg_fill_price = order.avg_fill_price
                commission = order.commission
                stamp_tax = order.stamp_tax
            else:
                res = broker.submit_order(
                    client_order_id=f"BT-{sig.symbol}-{ts}",
                    symbol=sig.symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=sig.quantity,
                )
                status = res.status
                filled_quantity = res.filled_quantity
                avg_fill_price = res.avg_fill_price
                commission = res.commission
                stamp_tax = res.stamp_tax
        except Exception as e:
            log.exception("broker error: {}", e)
            return False

        if status == OrderStatus.FILLED:
            pnl_delta = float(account.realized_pnl - realized_before) if sig.action == "SELL" else 0.0
            if sig.action == "SELL":
                trade_pnls.append(pnl_delta)
            if record_trade:
                trades.append(
                    {
                        "ts": str(ts),
                        "symbol": sig.symbol,
                        "side": sig.action,
                        "quantity": filled_quantity,
                        "price": float(avg_fill_price or 0),
                        "commission": float(commission),
                        "stamp_tax": float(stamp_tax),
                        "reason": sig.reason,
                        "cash_before": float(cash_before),
                        "cash_after": float(account.cash),
                        "realized_pnl_delta": pnl_delta,
                    }
                )
            return True
        return False

    def _risk_check_adapter(self, ctx: dict, risk: RiskEngine, account: Account) -> tuple[bool, str | None]:
        d = risk.check_pre_trade(
            symbol=ctx["symbol"],
            side=ctx["side"],
            quantity=int(ctx["quantity"]),
            price=float(ctx.get("limit_price") or 0) or 1.0,
            account=account,
        )
        return d.passed, d.detail
