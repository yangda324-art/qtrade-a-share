"""模拟券商（用于回测和 paper trading）。

行为：
- 接到 MARKET 单 → 立刻按"当前 tick price + 滑点"成交。
- 接到 LIMIT 单 → 价格满足才成交（回测引擎在每根 K 线推时检查）。
- 拒单条件：可用现金不足、可卖股数不足、价格异常等。

这个实现会被 ``BacktestEngine`` 反复调用，**必须无副作用、无 IO**——所有持久化由上层做。
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

from qtrade.execution.account import Account
from qtrade.execution.broker.base import Broker, BrokerOrderResult
from qtrade.execution.fees import AStockFeeModel, FeeModel
from qtrade.persistence.models import OrderSide, OrderStatus, OrderType


@dataclass
class _PendingLimit:
    """挂在簿子上等待成交的限价单。"""

    broker_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    limit_price: Decimal


@dataclass
class PaperBroker(Broker):
    name: str = "paper"
    account: Account = field(default_factory=Account)
    fee_model: FeeModel = field(default_factory=AStockFeeModel)
    slippage_bps: float = 2.0       # 单边 2 bps 滑点（万 2）
    last_prices: dict[str, Decimal] = field(default_factory=dict)
    _pending_limits: dict[str, _PendingLimit] = field(default_factory=dict)
    _filled_today: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ---------------- 价格推送 ----------------

    def update_market(self, prices: dict[str, float]) -> None:
        """回测引擎或行情线在每个 tick 调用一次，刷新最新价并尝试撮合限价单。"""
        for sym, p in prices.items():
            self.last_prices[sym] = Decimal(str(p))
        self.account.positions.update_prices(prices)
        self._try_match_limit_orders()

    # ---------------- 接口实现 ----------------

    def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        limit_price: Decimal | None = None,
    ) -> BrokerOrderResult:
        broker_order_id = f"PB-{uuid.uuid4().hex[:12]}"

        if order_type == OrderType.MARKET:
            ref_price = self.last_prices.get(symbol)
            if ref_price is None:
                return BrokerOrderResult(
                    broker_order_id=broker_order_id,
                    status=OrderStatus.REJECTED,
                    reject_reason=f"no market price for {symbol}",
                )
            return self._fill(
                broker_order_id=broker_order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=self._apply_slippage(ref_price, side),
            )

        # LIMIT：先校验资金/持仓充足再挂单
        if limit_price is None:
            return BrokerOrderResult(
                broker_order_id=broker_order_id,
                status=OrderStatus.REJECTED,
                reject_reason="LIMIT order missing limit_price",
            )
        if side == OrderSide.BUY:
            need = limit_price * Decimal(quantity) * Decimal("1.005")  # 预留 0.5% 给手续费
            if need > self.account.cash:
                return BrokerOrderResult(
                    broker_order_id=broker_order_id,
                    status=OrderStatus.REJECTED,
                    reject_reason=f"insufficient cash (need ~{need:.2f}, have {self.account.cash:.2f})",
                )
            self.account.freeze(need)
        else:
            pos = self.account.positions.get(symbol)
            if pos.available < quantity:
                return BrokerOrderResult(
                    broker_order_id=broker_order_id,
                    status=OrderStatus.REJECTED,
                    reject_reason=f"insufficient available shares ({pos.available} < {quantity})",
                )

        self._pending_limits[broker_order_id] = _PendingLimit(
            broker_order_id=broker_order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
        )
        return BrokerOrderResult(
            broker_order_id=broker_order_id, status=OrderStatus.SUBMITTED
        )

    def cancel_order(self, broker_order_id: str) -> bool:
        order = self._pending_limits.pop(broker_order_id, None)
        if not order:
            return False
        if order.side == OrderSide.BUY:
            # 解冻资金
            need = order.limit_price * Decimal(order.quantity) * Decimal("1.005")
            self.account.unfreeze(need)
        return True

    def get_account_balance(self) -> dict:
        return self.account.to_dict()

    def get_positions(self) -> dict[str, dict]:
        return self.account.positions.to_dict()

    def execute_order_at_price(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Decimal,
        broker_order_id: str | None = None,
    ) -> BrokerOrderResult:
        """按指定价格立即成交。

        用于持久化限价单被外部行情触发后的撮合；此时挂单状态在数据库里，
        不依赖 ``_pending_limits`` 这份进程内簿子。
        """
        return self._fill(
            broker_order_id=broker_order_id or f"PB-{uuid.uuid4().hex[:12]}",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )

    # ---------------- 内部 ----------------

    def _apply_slippage(self, price: Decimal, side: OrderSide) -> Decimal:
        """市价单滑点：买入向上，卖出向下。"""
        bps = Decimal(str(self.slippage_bps)) / Decimal("10000")
        return (price * (Decimal("1") + bps)) if side == OrderSide.BUY else (price * (Decimal("1") - bps))

    def _try_match_limit_orders(self) -> list[BrokerOrderResult]:
        """根据最新价撮合挂单。返回成交结果列表（OMS 会查 callback 拿）。"""
        results: list[BrokerOrderResult] = []
        for boid, order in list(self._pending_limits.items()):
            mkt = self.last_prices.get(order.symbol)
            if mkt is None:
                continue
            hit = (order.side == OrderSide.BUY and mkt <= order.limit_price) or (
                order.side == OrderSide.SELL and mkt >= order.limit_price
            )
            if not hit:
                continue
            del self._pending_limits[boid]

            if order.side == OrderSide.BUY:
                # 解冻原冻结金额，再按真实价扣
                frozen = order.limit_price * Decimal(order.quantity) * Decimal("1.005")
                self.account.unfreeze(frozen)

            res = self._fill(
                broker_order_id=boid,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.limit_price,
            )
            results.append(res)
        return results

    def _fill(
        self,
        *,
        broker_order_id: str,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Decimal,
    ) -> BrokerOrderResult:
        fees = self.fee_model.calc(symbol=symbol, side=side.value, quantity=quantity, price=float(price))
        notional = price * Decimal(quantity)

        if side == OrderSide.BUY:
            total_cost = notional + fees.total
            if total_cost > self.account.cash:
                return BrokerOrderResult(
                    broker_order_id=broker_order_id,
                    status=OrderStatus.REJECTED,
                    reject_reason=f"insufficient cash on fill (need {total_cost:.2f})",
                )
            self.account.cash -= total_cost
            # 把买入手续费并入持仓成本：浮动盈亏直接反映真实"平仓即亏"
            self.account.positions.get(symbol).add_buy(quantity, price, fees=fees.total)
            # 双重保险：成交后立即把 market_price 同步到成交价 + last_prices 记录，
            # 避免新建持仓时市值快照为 0
            self.account.positions.get(symbol).update_market_price(price)
            self.last_prices[symbol] = price
        else:
            pos = self.account.positions.get(symbol)
            if pos.available < quantity:
                return BrokerOrderResult(
                    broker_order_id=broker_order_id,
                    status=OrderStatus.REJECTED,
                    reject_reason=f"insufficient shares on fill ({pos.available} < {quantity})",
                )
            realized_before = pos.realized_pnl
            pos.add_sell(quantity, price)
            realized_delta = pos.realized_pnl - realized_before - fees.total
            proceeds = notional - fees.total
            self.account.settle_sell(proceeds=proceeds, realized_delta=realized_delta)

        self._filled_today[symbol] += quantity
        return BrokerOrderResult(
            broker_order_id=broker_order_id,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            avg_fill_price=price,
            commission=fees.commission + fees.transfer_fee,
            stamp_tax=fees.stamp_tax,
        )

    def reset_daily(self) -> None:
        self._filled_today.clear()
