"""Factor threshold strategy.

This strategy converts a researched factor into a simple long-only trading rule:
buy when the latest factor value enters a historical quantile band, and sell
when it exits that band. It is meant for backtesting researched factors before
any paper/live trading use.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np

from qtrade.backtest.factors import compute_factor_series
from qtrade.core.logging import get_logger
from qtrade.strategies.base import Signal, Strategy, StrategyContext

log = get_logger(__name__)


class FactorThresholdStrategy(Strategy):
    id = "factor_threshold_v1"

    def __init__(self, *, params: dict | None = None, universe: list[str] | None = None) -> None:
        super().__init__(params=params, universe=universe)
        self.factor_name = str(self.params.get("factor_name", "ret_20"))
        self.windows = [int(v) for v in self.params.get("windows", [5, 20, 60])]
        self.lookback = int(self.params.get("lookback", 120))
        self.entry_quantile = float(self.params.get("entry_quantile", 0.8))
        self.exit_quantile = float(self.params.get("exit_quantile", 0.5))
        self.direction = str(self.params.get("direction", "high"))
        self.position_pct = float(self.params.get("position_pct", 0.3))
        self.include_interactions = bool(self.params.get("include_interactions", False))
        self.max_interactions = int(self.params.get("max_interactions", 50))
        self._invalid_factor_logged = False
        if self.lookback < 5:
            raise ValueError("lookback must be >= 5")
        if not 0 < self.entry_quantile < 1:
            raise ValueError("entry_quantile must be between 0 and 1")
        if not 0 < self.exit_quantile < 1:
            raise ValueError("exit_quantile must be between 0 and 1")
        if self.direction not in {"high", "low"}:
            raise ValueError("direction must be high or low")
        if not 0 < self.position_pct <= 1:
            raise ValueError("position_pct must be in (0, 1]")

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        factor = self._factor_series(ctx)
        factor = factor.replace([np.inf, -np.inf], np.nan).dropna()
        if len(factor) < self.lookback:
            return []

        window = factor.tail(self.lookback)
        latest = float(window.iloc[-1])
        percentile = float((window <= latest).mean())
        pos = ctx.account.positions.get(ctx.symbol)

        if pos.quantity == 0 and self._should_enter(percentile):
            shares = self._target_shares(ctx)
            if shares >= 100:
                return [
                    Signal(
                        symbol=ctx.symbol,
                        action="BUY",
                        quantity=shares,
                        confidence=0.6,
                        reason=self._reason("entry", latest, percentile),
                        metadata={
                            "factor_name": self.factor_name,
                            "factor_value": latest,
                            "factor_percentile": percentile,
                            "direction": self.direction,
                        },
                    )
                ]
        if pos.available > 0 and self._should_exit(percentile):
            return [
                Signal(
                    symbol=ctx.symbol,
                    action="SELL",
                    quantity=pos.available,
                    confidence=0.6,
                    reason=self._reason("exit", latest, percentile),
                    metadata={
                        "factor_name": self.factor_name,
                        "factor_value": latest,
                        "factor_percentile": percentile,
                        "direction": self.direction,
                    },
                )
            ]
        return []

    def _factor_series(self, ctx: StrategyContext):
        try:
            return compute_factor_series(
                ctx.history,
                factor_name=self.factor_name,
                windows=self.windows,
                include_interactions=self.include_interactions,
                max_interactions=self.max_interactions,
            )
        except ValueError as exc:
            if not self._invalid_factor_logged:
                log.warning("factor strategy disabled by invalid factor: {}", exc)
                self._invalid_factor_logged = True
            return ctx.history["close"].iloc[0:0]

    def _target_shares(self, ctx: StrategyContext) -> int:
        price = Decimal(str(ctx.history["close"].iloc[-1]))
        if price <= 0:
            return 0
        target_value = ctx.account.total_assets * Decimal(str(self.position_pct))
        return int(target_value / price // 100) * 100

    def _should_enter(self, percentile: float) -> bool:
        if self.direction == "high":
            return percentile >= self.entry_quantile
        return percentile <= 1 - self.entry_quantile

    def _should_exit(self, percentile: float) -> bool:
        if self.direction == "high":
            return percentile <= self.exit_quantile
        return percentile >= 1 - self.exit_quantile

    def _reason(self, phase: str, value: float, percentile: float) -> str:
        action = "入场" if phase == "entry" else "退出"
        return (
            f"因子{action}: {self.factor_name}={value:.6f}, "
            f"历史分位={percentile:.1%}, 方向={self.direction}"
        )
