"""Cross-sectional factor portfolio backtests.

This module is research-only. It ranks a universe by one factor on each
rebalance date, holds the selected names for the next return period, and
applies simple turnover-based costs. It does not submit orders or touch runtime
state.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtrade.backtest.data_cache import ensure_in_memory_bar_loader
from qtrade.backtest.factors import compute_factor_series
from qtrade.backtest.metrics import Metrics, compute_metrics
from qtrade.core.time import utcnow
from qtrade.data.loader import DataLoader, get_loader
from qtrade.risk.ashare_rules import ashare_price_limit_pct


@dataclass(frozen=True)
class FactorPortfolioDay:
    date: date
    selected_symbols: list[str]
    blocked_buy_symbols: list[str]
    blocked_sell_symbols: list[str]
    gross_return: float
    cost_return: float
    net_return: float
    turnover: float
    holdings: int
    eligible_count: int
    equity: float

    def to_row(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "selected_symbols": ",".join(self.selected_symbols),
            "blocked_buy_symbols": ",".join(self.blocked_buy_symbols),
            "blocked_sell_symbols": ",".join(self.blocked_sell_symbols),
            "gross_return": self.gross_return,
            "cost_return": self.cost_return,
            "net_return": self.net_return,
            "turnover": self.turnover,
            "holdings": self.holdings,
            "eligible_count": self.eligible_count,
            "equity": self.equity,
        }


@dataclass(frozen=True)
class FactorPortfolioReport:
    symbols: list[str]
    factor: str
    start: date
    end: date
    windows: list[int]
    direction: str
    rebalance_days: int
    select_count: int | None
    select_quantile: float
    fee_bps: float
    slippage_bps: float
    respect_tradeability: bool
    min_volume: float
    price_limit_pct: float | None
    limit_buffer_bps: float
    initial_equity: float
    metrics: Metrics
    days: list[FactorPortfolioDay]
    generated_at: datetime = field(default_factory=utcnow)

    @property
    def rebalance_count(self) -> int:
        return sum(1 for day in self.days if day.turnover > 1e-12)

    @property
    def average_turnover(self) -> float:
        rows = [day.turnover for day in self.days if day.turnover > 1e-12]
        return float(np.mean(rows)) if rows else 0.0

    @property
    def average_holdings(self) -> float:
        rows = [day.holdings for day in self.days]
        return float(np.mean(rows)) if rows else 0.0

    @property
    def total_cost_return(self) -> float:
        return sum(day.cost_return for day in self.days)

    @property
    def blocked_buy_count(self) -> int:
        return sum(len(day.blocked_buy_symbols) for day in self.days)

    @property
    def blocked_sell_count(self) -> int:
        return sum(len(day.blocked_sell_symbols) for day in self.days)


def run_factor_portfolio_backtest(
    *,
    symbols: list[str],
    factor_name: str,
    start: date,
    end: date,
    windows: list[int] | None = None,
    direction: str = "high",
    rebalance_days: int = 5,
    select_count: int | None = None,
    select_quantile: float = 0.2,
    initial_equity: float = 100_000.0,
    fee_bps: float = 3.0,
    slippage_bps: float = 1.0,
    respect_tradeability: bool = True,
    min_volume: float = 1.0,
    price_limit_pct: float | None = None,
    limit_buffer_bps: float = 5.0,
    include_interactions: bool = False,
    max_interactions: int = 50,
    loader: DataLoader | None = None,
) -> FactorPortfolioReport:
    if not symbols:
        raise ValueError("至少需要一个标的")
    if start >= end:
        raise ValueError("start 必须早于 end")
    if not factor_name:
        raise ValueError("factor_name 不能为空")
    if direction not in {"high", "low"}:
        raise ValueError("direction 只能是 high 或 low")
    if rebalance_days <= 0:
        raise ValueError("rebalance_days 必须为正数")
    if select_count is not None and select_count <= 0:
        raise ValueError("select_count 必须为正数")
    if not 0 < select_quantile <= 1:
        raise ValueError("select_quantile 必须在 0-1 之间")
    if initial_equity <= 0:
        raise ValueError("initial_equity 必须为正数")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("fee_bps 和 slippage_bps 不能为负数")
    if min_volume < 0:
        raise ValueError("min_volume 不能为负数")
    if price_limit_pct is not None and not 0 < price_limit_pct < 1:
        raise ValueError("price_limit_pct 必须在 0-1 之间")
    if limit_buffer_bps < 0:
        raise ValueError("limit_buffer_bps 不能为负数")

    windows = sorted(set(windows or [5, 20, 60]))
    if any(window <= 1 for window in windows):
        raise ValueError("window 必须大于 1")

    loader = ensure_in_memory_bar_loader(loader or get_loader())
    sample = _load_portfolio_factor_sample(
        symbols=symbols,
        start=start,
        end=end,
        windows=windows,
        factor_name=factor_name,
        include_interactions=include_interactions,
        max_interactions=max_interactions,
        respect_tradeability=respect_tradeability,
        min_volume=min_volume,
        price_limit_pct=price_limit_pct,
        limit_buffer_bps=limit_buffer_bps,
        loader=loader,
    )
    if sample.empty:
        raise ValueError("没有可用因子样本，无法进行组合回测")

    days = _simulate_factor_portfolio(
        sample=sample,
        direction=direction,
        rebalance_days=rebalance_days,
        select_count=select_count,
        select_quantile=select_quantile,
        initial_equity=initial_equity,
        total_cost_bps=fee_bps + slippage_bps,
        respect_tradeability=respect_tradeability,
    )
    if not days:
        raise ValueError("没有生成可用组合权益曲线")

    first_day = days[0].date - timedelta(days=1)
    equity_curve = [(datetime.combine(first_day, datetime.min.time()), Decimal(str(initial_equity)))]
    equity_curve.extend(
        (datetime.combine(day.date, datetime.min.time()), Decimal(str(day.equity)))
        for day in days
    )
    metrics = compute_metrics(equity_curve)
    return FactorPortfolioReport(
        symbols=list(symbols),
        factor=factor_name,
        start=start,
        end=end,
        windows=windows,
        direction=direction,
        rebalance_days=rebalance_days,
        select_count=select_count,
        select_quantile=select_quantile,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        respect_tradeability=respect_tradeability,
        min_volume=min_volume,
        price_limit_pct=price_limit_pct,
        limit_buffer_bps=limit_buffer_bps,
        initial_equity=initial_equity,
        metrics=metrics,
        days=days,
    )


def factor_portfolio_report_to_markdown(report: FactorPortfolioReport) -> str:
    m = report.metrics
    lines = [
        "# 因子组合回测报告",
        "",
        f"- 生成时间: {report.generated_at.isoformat()}",
        f"- 标的: {','.join(report.symbols)}",
        f"- 因子: {report.factor}",
        f"- 区间: {report.start.isoformat()} → {report.end.isoformat()}",
        f"- 窗口: {report.windows}",
        f"- 方向: {report.direction}",
        f"- 调仓间隔: {report.rebalance_days} 个交易样本日",
        f"- 选股: {_selection_desc(report)}",
        f"- 交易成本: fee {report.fee_bps:.2f}bp + slippage {report.slippage_bps:.2f}bp",
        f"- 可交易性约束: {'on' if report.respect_tradeability else 'off'}",
        f"- 停牌/无量阈值: volume <= {report.min_volume:g}",
        f"- 涨跌停近似: {_price_limit_desc(report)} ± {report.limit_buffer_bps:.1f}bp",
        f"- 总收益: {m.total_return:.2%}",
        f"- 年化收益: {m.annual_return:.2%}",
        f"- Sharpe: {m.sharpe:.2f}",
        f"- 最大回撤: {m.max_drawdown:.2%}",
        f"- 调仓次数: {report.rebalance_count}",
        f"- 平均换手: {report.average_turnover:.2%}",
        f"- 平均持仓数: {report.average_holdings:.2f}",
        f"- 拦截买入次数: {report.blocked_buy_count}",
        f"- 拦截卖出次数: {report.blocked_sell_count}",
        "",
        "| 日期 | 持仓数 | 可选数 | 换手 | 成本收益 | 组合收益 | 权益 | 持仓 | 买入拦截 | 卖出拦截 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for day in _rebalance_rows(report):
        lines.append(
            "| "
            f"{day.date.isoformat()} | "
            f"{day.holdings} | "
            f"{day.eligible_count} | "
            f"{day.turnover:.2%} | "
            f"{day.cost_return:.4%} | "
            f"{day.net_return:.4%} | "
            f"{day.equity:,.2f} | "
            f"{','.join(day.selected_symbols)} | "
            f"{','.join(day.blocked_buy_symbols)} | "
            f"{','.join(day.blocked_sell_symbols)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_factor_portfolio_markdown(report: FactorPortfolioReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(factor_portfolio_report_to_markdown(report), encoding="utf-8")
    return out


def write_factor_portfolio_csv(report: FactorPortfolioReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [day.to_row() for day in report.days]
    fieldnames = list(rows[0]) if rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _load_portfolio_factor_sample(
    *,
    symbols: list[str],
    start: date,
    end: date,
    windows: list[int],
    factor_name: str,
    include_interactions: bool,
    max_interactions: int,
    respect_tradeability: bool,
    min_volume: float,
    price_limit_pct: float | None,
    limit_buffer_bps: float,
    loader: DataLoader,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = loader.get_daily_bars(symbol, start=start, end=end)
        if df.empty:
            continue
        work = df.copy().sort_values("date").reset_index(drop=True)
        work["date"] = pd.to_datetime(work["date"]).dt.date
        close = work["close"].astype(float)
        prev_close = close.shift(1)
        volume = work["volume"].astype(float)
        amount = work["amount"].astype(float) if "amount" in work else pd.Series(0.0, index=work.index)
        bar_return = close / prev_close - 1
        suspended = _suspended_mask(volume=volume, amount=amount, min_volume=min_volume)
        symbol_limit_pct = price_limit_pct
        if symbol_limit_pct is None:
            symbol_limit_pct = ashare_price_limit_pct(symbol)
        limit_up, limit_down = _price_limit_masks(
            bar_return=bar_return,
            price_limit_pct=symbol_limit_pct,
            limit_buffer_bps=limit_buffer_bps,
        )
        if respect_tradeability:
            buyable = ~(suspended | limit_up)
            sellable = ~(suspended | limit_down)
        else:
            buyable = pd.Series(True, index=work.index)
            sellable = pd.Series(True, index=work.index)
        factor = compute_factor_series(
            work,
            factor_name=factor_name,
            windows=windows,
            include_interactions=include_interactions,
            max_interactions=max_interactions,
        ).reset_index(drop=True)
        frame = pd.DataFrame(
            {
                "date": work["date"],
                "symbol": work["symbol"].astype(str),
                "factor": factor,
                "next_return": close.shift(-1) / close - 1,
                "buyable": buyable,
                "sellable": sellable,
                "is_suspended": suspended,
                "is_limit_up": limit_up,
                "is_limit_down": limit_down,
            }
        )
        frames.append(frame)
    if not frames:
        raise ValueError("没有可用历史 K 线，无法进行组合回测")
    sample = pd.concat(frames, ignore_index=True)
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
    return sample.sort_values(["date", "symbol"]).reset_index(drop=True)


def _simulate_factor_portfolio(
    *,
    sample: pd.DataFrame,
    direction: str,
    rebalance_days: int,
    select_count: int | None,
    select_quantile: float,
    initial_equity: float,
    total_cost_bps: float,
    respect_tradeability: bool,
) -> list[FactorPortfolioDay]:
    dates = sorted(sample["date"].dropna().unique())
    current_weights: dict[str, float] = {}
    equity = initial_equity
    out: list[FactorPortfolioDay] = []
    for idx, day in enumerate(dates):
        day_sample = sample[sample["date"] == day]
        eligible_count = int(day_sample["buyable"].sum()) if "buyable" in day_sample else len(day_sample)
        blocked_buy_symbols: list[str] = []
        blocked_sell_symbols: list[str] = []
        cost_return = 0.0
        turnover = 0.0
        if idx % rebalance_days == 0:
            raw_target_weights = _select_equal_weights(
                day_sample,
                direction=direction,
                select_count=select_count,
                select_quantile=select_quantile,
                require_buyable=False,
            )
            target_weights = _tradeable_target_weights(
                day_sample=day_sample,
                current_weights=current_weights,
                raw_target_weights=raw_target_weights,
                direction=direction,
                select_count=select_count,
                select_quantile=select_quantile,
                respect_tradeability=respect_tradeability,
            )
            blocked_buy_symbols = sorted(set(raw_target_weights) - set(target_weights))
            blocked_sell_symbols = sorted(
                symbol
                for symbol in current_weights
                if symbol in target_weights and symbol not in raw_target_weights
            )
            turnover = _portfolio_turnover(current_weights, target_weights)
            cost_return = turnover * total_cost_bps / 10_000
            current_weights = target_weights

        returns = dict(zip(day_sample["symbol"].astype(str), day_sample["next_return"].astype(float), strict=False))
        gross_return = sum(weight * _finite_return(returns.get(symbol, 0.0)) for symbol, weight in current_weights.items())
        net_return = gross_return - cost_return
        equity *= 1 + net_return
        out.append(
            FactorPortfolioDay(
                date=day,
                selected_symbols=sorted(current_weights),
                blocked_buy_symbols=blocked_buy_symbols,
                blocked_sell_symbols=blocked_sell_symbols,
                gross_return=gross_return,
                cost_return=cost_return,
                net_return=net_return,
                turnover=turnover,
                holdings=len(current_weights),
                eligible_count=eligible_count,
                equity=equity,
            )
        )
    return out


def _select_equal_weights(
    sample: pd.DataFrame,
    *,
    direction: str,
    select_count: int | None,
    select_quantile: float,
    require_buyable: bool = True,
) -> dict[str, float]:
    if sample.empty:
        return {}
    if require_buyable and "buyable" in sample:
        sample = sample[sample["buyable"].astype(bool)]
        if sample.empty:
            return {}
    ascending = direction == "low"
    ranked = sample.sort_values(["factor", "symbol"], ascending=[ascending, True])
    n = len(ranked)
    count = select_count if select_count is not None else max(1, math.ceil(n * select_quantile))
    count = min(max(count, 1), n)
    selected = ranked.head(count)["symbol"].astype(str).tolist()
    weight = 1.0 / len(selected)
    return {symbol: weight for symbol in selected}


def _tradeable_target_weights(
    *,
    day_sample: pd.DataFrame,
    current_weights: dict[str, float],
    raw_target_weights: dict[str, float],
    direction: str,
    select_count: int | None,
    select_quantile: float,
    respect_tradeability: bool,
) -> dict[str, float]:
    if not respect_tradeability:
        return raw_target_weights

    sellable_by_symbol = _bool_by_symbol(day_sample, "sellable")
    blocked_keeps = {
        symbol: weight
        for symbol, weight in current_weights.items()
        if not sellable_by_symbol.get(symbol, False)
    }
    available_weight = max(0.0, 1.0 - sum(blocked_keeps.values()))
    if available_weight <= 1e-12:
        return blocked_keeps
    buyable_target = _select_equal_weights(
        day_sample,
        direction=direction,
        select_count=select_count,
        select_quantile=select_quantile,
        require_buyable=True,
    )
    buyable_target = {symbol: weight for symbol, weight in buyable_target.items() if symbol not in blocked_keeps}
    if not buyable_target:
        return blocked_keeps

    scaled = {
        symbol: weight * available_weight
        for symbol, weight in buyable_target.items()
        if weight * available_weight > 1e-12
    }
    return {**blocked_keeps, **scaled}


def _bool_by_symbol(sample: pd.DataFrame, column: str) -> dict[str, bool]:
    if column not in sample:
        return {}
    return dict(zip(sample["symbol"].astype(str), sample[column].astype(bool), strict=False))


def _suspended_mask(*, volume: pd.Series, amount: pd.Series, min_volume: float) -> pd.Series:
    return (volume <= min_volume) | (amount <= 0)


def _price_limit_masks(
    *,
    bar_return: pd.Series,
    price_limit_pct: float | None,
    limit_buffer_bps: float,
) -> tuple[pd.Series, pd.Series]:
    if price_limit_pct is None:
        return pd.Series(False, index=bar_return.index), pd.Series(False, index=bar_return.index)
    buffer = limit_buffer_bps / 10_000
    limit_up = bar_return >= price_limit_pct - buffer
    limit_down = bar_return <= -price_limit_pct + buffer
    return limit_up.fillna(False), limit_down.fillna(False)


def _portfolio_turnover(old: dict[str, float], new: dict[str, float]) -> float:
    symbols = set(old) | set(new)
    return sum(abs(new.get(symbol, 0.0) - old.get(symbol, 0.0)) for symbol in symbols)


def _finite_return(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _selection_desc(report: FactorPortfolioReport) -> str:
    if report.select_count is not None:
        return f"Top {report.select_count}"
    return f"Top {report.select_quantile:.0%}"


def _price_limit_desc(report: FactorPortfolioReport) -> str:
    if report.price_limit_pct is None:
        return "board-aware auto"
    return f"{report.price_limit_pct:.2%}"


def _rebalance_rows(report: FactorPortfolioReport) -> list[FactorPortfolioDay]:
    rows = [day for day in report.days if day.turnover > 1e-12]
    return rows or report.days[:1]
