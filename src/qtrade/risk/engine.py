"""风控引擎主体。

3 层检查：
- ``check_pre_trade(order_intent)``     下单前：金额/数量/黑名单/单标的占比/T+0
- ``check_in_trade(positions, prices)`` 持仓中：止损/跟踪止损/止盈，返回需要平仓的标的
- ``check_account(account, history)``   账户级：最大回撤、单日亏损、熔断

设计原则：
- 检查方法都返回 ``RiskDecision``——结构化，方便日志和告警有据可查。
- 风控**不修改**账户或下单——它只回答"是否通过"和"为什么"。具体动作由调用方决定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from qtrade.execution.account import Account
from qtrade.execution.position import Position, PositionBook
from qtrade.risk.policies import RiskPolicies


@dataclass
class RiskDecision:
    passed: bool
    rule: str | None = None      # 触发规则名（拒绝时填）
    detail: str | None = None
    actions: list[dict] = field(default_factory=list)   # 推荐动作（如止损平仓）


@dataclass
class RiskEngine:
    policies: RiskPolicies
    _orders_today: dict[date, int] = field(default_factory=dict)
    _peak_equity: Decimal = Decimal("0")
    _open_equity: Decimal | None = None    # 当日开盘权益（用于 daily_loss_limit）
    _today_date: date | None = None
    _circuit_broken: bool = False

    # ------------------------------------------------------------------
    # 下单前检查
    # ------------------------------------------------------------------

    def check_pre_trade(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        symbol_name: str | None = None,
        account: Account,
        today: date | None = None,
    ) -> RiskDecision:
        if self._circuit_broken:
            return RiskDecision(False, "circuit_breaker", "账户级熔断已触发，禁止任何下单")

        today = today or datetime.utcnow().date()
        notional = price * quantity

        # 黑名单
        if symbol_name and self.policies.is_blacklisted(symbol_name):
            return RiskDecision(False, "blacklist", f"{symbol_name} 命中黑名单")

        # 单笔金额
        if notional > self.policies.order.max_order_value:
            return RiskDecision(
                False,
                "max_order_value",
                f"单笔 {notional:.2f} 超过上限 {self.policies.order.max_order_value:.2f}",
            )
        if notional < self.policies.order.min_order_value:
            return RiskDecision(
                False,
                "min_order_value",
                f"单笔 {notional:.2f} 低于最小 {self.policies.order.min_order_value:.2f}",
            )

        # 整手
        if quantity % 100 != 0:
            return RiskDecision(False, "lot_size", "A 股买入必须为 100 股整数倍")

        # 当日下单次数
        cnt = self._orders_today.get(today, 0)
        if cnt >= self.policies.order.max_orders_per_day:
            return RiskDecision(
                False,
                "max_orders_per_day",
                f"当日下单 {cnt} 次已达上限 {self.policies.order.max_orders_per_day}",
            )

        # 现金保留
        if side == "BUY":
            cash_after = account.cash - Decimal(str(notional))
            min_reserve = account.total_assets * Decimal(str(self.policies.account.min_cash_reserve_pct))
            if cash_after < min_reserve:
                return RiskDecision(
                    False,
                    "min_cash_reserve",
                    f"成交后现金 {cash_after:.2f} 低于 {min_reserve:.2f}（{self.policies.account.min_cash_reserve_pct:.0%}）",
                )

            # 单标的最大仓位
            pos = account.positions.get(symbol)
            new_value = pos.market_value + Decimal(str(notional))
            max_value = account.total_assets * Decimal(str(self.policies.position.max_pct_per_symbol))
            if new_value > max_value:
                return RiskDecision(
                    False,
                    "max_pct_per_symbol",
                    f"{symbol} 仓位将达 {new_value:.2f}，超过 {max_value:.2f}（{self.policies.position.max_pct_per_symbol:.0%}）",
                )

            # 同时持仓数量
            holding = sum(1 for p in account.positions if p.quantity > 0)
            if symbol not in account.positions and holding >= self.policies.position.max_symbols:
                return RiskDecision(
                    False,
                    "max_symbols",
                    f"已持仓 {holding} 只，达到上限 {self.policies.position.max_symbols}",
                )

        # T+0 禁止：当日买入今日不能卖
        if side == "SELL" and self.policies.order.forbid_t0:
            pos = account.positions.get(symbol)
            if quantity > pos.available:
                return RiskDecision(
                    False,
                    "t1_lock",
                    f"{symbol} 可卖 {pos.available} < 申报 {quantity}（T+1 限制）",
                )

        # 全部通过
        self._orders_today[today] = cnt + 1
        return RiskDecision(True)

    # ------------------------------------------------------------------
    # 持仓中检查（止损/止盈/跟踪止损）
    # ------------------------------------------------------------------

    def check_in_trade(self, positions: PositionBook) -> RiskDecision:
        actions: list[dict] = []
        for p in positions:
            if p.quantity <= 0 or p.avg_cost <= 0:
                continue
            ret = (p.market_price - p.avg_cost) / p.avg_cost
            if ret <= -self.policies.position.stop_loss_pct:
                actions.append(
                    {
                        "type": "stop_loss",
                        "symbol": p.symbol,
                        "quantity": p.available,
                        "reason": f"止损：浮亏 {ret:.2%} 触发 {self.policies.position.stop_loss_pct:.0%}",
                    }
                )
            elif ret >= self.policies.position.take_profit_pct:
                actions.append(
                    {
                        "type": "take_profit",
                        "symbol": p.symbol,
                        "quantity": p.available,
                        "reason": f"止盈：浮盈 {ret:.2%} 达到 {self.policies.position.take_profit_pct:.0%}",
                    }
                )
        return RiskDecision(passed=True, actions=actions)

    # ------------------------------------------------------------------
    # 账户级熔断
    # ------------------------------------------------------------------

    def check_account(self, account: Account, *, ts: datetime) -> RiskDecision:
        equity = account.total_assets

        # 当日开盘锚点
        today = ts.date()
        if self._today_date != today:
            self._today_date = today
            self._open_equity = equity

        # 历史峰值
        if equity > self._peak_equity:
            self._peak_equity = equity

        # 最大回撤
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if float(dd) > self.policies.account.max_drawdown_pct:
                self._circuit_broken = True
                return RiskDecision(
                    False,
                    "max_drawdown",
                    f"账户回撤 {float(dd):.2%} 超过 {self.policies.account.max_drawdown_pct:.0%}，熔断",
                )

        # 当日亏损
        if self._open_equity and self._open_equity > 0:
            day_ret = (equity - self._open_equity) / self._open_equity
            if float(-day_ret) > self.policies.account.daily_loss_limit_pct:
                self._circuit_broken = True
                return RiskDecision(
                    False,
                    "daily_loss_limit",
                    f"当日亏损 {float(-day_ret):.2%} 超过 {self.policies.account.daily_loss_limit_pct:.0%}，熔断",
                )

        return RiskDecision(True)

    def reset_daily(self, today: date | None = None) -> None:
        """每日盘前调用，清零当日计数器。熔断状态需要手动 ``resume()``。"""
        today = today or datetime.utcnow().date()
        self._orders_today.pop(today, None)
        self._open_equity = None

    def resume(self) -> None:
        """人工解除熔断。注意：实盘下应该非常慎重。"""
        self._circuit_broken = False
