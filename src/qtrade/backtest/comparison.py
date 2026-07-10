"""Backtest comparison reports for multiple strategies."""
from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

from qtrade.backtest.data_cache import ensure_in_memory_bar_loader
from qtrade.backtest.engine import BacktestEngine, BacktestResult
from qtrade.backtest.metrics import Metrics
from qtrade.core.time import utcnow
from qtrade.data.loader import DataLoader, get_loader
from qtrade.strategies.base import Strategy
from qtrade.strategies.technical import STRATEGY_REGISTRY

ProgressCallback = Callable[[int, int, dict[str, Any]], None]


@dataclass(frozen=True)
class BacktestSummary:
    strategy_name: str
    strategy_id: str
    symbols: list[str]
    start: date
    end: date
    metrics: Metrics
    trades: list[dict] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    trade_count: int | None = None

    @property
    def n_trades(self) -> int:
        return self.trade_count if self.trade_count is not None else len(self.trades)

    def to_row(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "strategy": self.strategy_name,
            "strategy_id": self.strategy_id,
            "params": _format_params(self.params),
            "symbols": ",".join(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "total_return": m.total_return,
            "annual_return": m.annual_return,
            "annual_volatility": m.annual_volatility,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "calmar": m.calmar,
            "max_drawdown": m.max_drawdown,
            "max_drawdown_duration_days": m.max_drawdown_duration_days,
            "n_trades": m.n_trades,
            "win_rate": m.win_rate,
            "profit_loss_ratio": m.profit_loss_ratio,
            "initial_equity": m.initial_equity,
            "final_equity": m.final_equity,
        }


@dataclass(frozen=True)
class BacktestFailure:
    strategy_name: str
    symbols: list[str]
    start: date
    end: date
    params: dict[str, Any] = field(default_factory=dict)
    error_type: str = ""
    error: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "strategy_id": "",
            "params": _format_params(self.params),
            "symbols": ",".join(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "total_return": "",
            "annual_return": "",
            "annual_volatility": "",
            "sharpe": "",
            "sortino": "",
            "calmar": "",
            "max_drawdown": "",
            "max_drawdown_duration_days": "",
            "n_trades": "",
            "win_rate": "",
            "profit_loss_ratio": "",
            "initial_equity": "",
            "final_equity": "",
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass(frozen=True)
class BacktestComparison:
    rows: list[BacktestSummary]
    failures: list[BacktestFailure] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utcnow)

    @property
    def attempted_count(self) -> int:
        return len(self.rows) + len(self.failures)

    @property
    def success_rate(self) -> float:
        if self.attempted_count == 0:
            return 0.0
        return len(self.rows) / self.attempted_count

    @property
    def failure_rate(self) -> float:
        if self.attempted_count == 0:
            return 0.0
        return len(self.failures) / self.attempted_count

    @property
    def best_by_sharpe(self) -> BacktestSummary | None:
        if not self.rows:
            return None
        return max(self.rows, key=lambda row: row.metrics.sharpe)

    @property
    def best_by_return(self) -> BacktestSummary | None:
        if not self.rows:
            return None
        return max(self.rows, key=lambda row: row.metrics.total_return)


@dataclass(frozen=True)
class ParameterSensitivity:
    param_name: str
    top_count: int
    dominant_value: str
    dominant_count: int
    dominant_share: float
    top_unique_count: int
    full_unique_count: int
    boundary_side: str = ""

    @property
    def is_boundary(self) -> bool:
        return bool(self.boundary_side)


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    selected_params: dict[str, Any]
    train: BacktestSummary
    test: BacktestSummary

    def to_row(self) -> dict[str, Any]:
        train = self.train.metrics
        test = self.test.metrics
        return {
            "fold": self.fold,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "selected_params": _format_params(self.selected_params),
            "train_total_return": train.total_return,
            "train_sharpe": train.sharpe,
            "train_max_drawdown": train.max_drawdown,
            "train_n_trades": train.n_trades,
            "test_total_return": test.total_return,
            "test_annual_return": test.annual_return,
            "test_sharpe": test.sharpe,
            "sharpe_decay": train.sharpe - test.sharpe,
            "test_sortino": test.sortino,
            "test_calmar": test.calmar,
            "test_max_drawdown": test.max_drawdown,
            "test_n_trades": test.n_trades,
            "test_win_rate": test.win_rate,
            "test_final_equity": test.final_equity,
        }


@dataclass(frozen=True)
class WalkForwardReport:
    strategy_name: str
    symbols: list[str]
    start: date
    end: date
    train_days: int
    test_days: int
    step_days: int
    sort_by: str
    folds: list[WalkForwardFold]
    generated_at: datetime = field(default_factory=utcnow)

    @property
    def compounded_test_return(self) -> float:
        total = 1.0
        for fold in self.folds:
            total *= 1 + fold.test.metrics.total_return
        return total - 1

    @property
    def average_test_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        return sum(fold.test.metrics.sharpe for fold in self.folds) / len(self.folds)

    @property
    def average_train_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        return sum(fold.train.metrics.sharpe for fold in self.folds) / len(self.folds)

    @property
    def average_sharpe_decay(self) -> float:
        return self.average_train_sharpe - self.average_test_sharpe

    @property
    def test_sharpe_retention(self) -> float:
        train = self.average_train_sharpe
        if abs(train) <= 1e-12:
            return 0.0
        return self.average_test_sharpe / train

    @property
    def average_test_max_drawdown(self) -> float:
        if not self.folds:
            return 0.0
        return sum(fold.test.metrics.max_drawdown for fold in self.folds) / len(self.folds)

    @property
    def profitable_test_fold_count(self) -> int:
        return sum(1 for fold in self.folds if fold.test.metrics.total_return > 0)

    @property
    def positive_test_return_rate(self) -> float:
        if not self.folds:
            return 0.0
        return self.profitable_test_fold_count / len(self.folds)


def build_backtest_strategy(
    name: str,
    *,
    symbols: list[str],
    params: dict | None = None,
    allow_llm: bool = False,
) -> Strategy:
    """Build a strategy for backtesting."""
    if name == "factor":
        from qtrade.strategies.factor import FactorThresholdStrategy

        return FactorThresholdStrategy(universe=symbols, params=params or {})
    if name in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[name](universe=symbols, params=params or {})
    if name == "llm" and allow_llm:
        from qtrade.strategies.llm.graph import LLMMultiAgentStrategy

        return LLMMultiAgentStrategy(universe=symbols, params=params or {})
    if name == "llm":
        raise ValueError("LLM 回测默认关闭，避免产生 API 成本；请用单策略 backtest 明确运行")
    raise ValueError(f"未知策略：{name}（可用：{list(STRATEGY_REGISTRY)}）")


def run_backtest_comparison(
    *,
    strategies: list[str],
    symbols: list[str],
    start: date,
    end: date,
    initial_cash: float = 100_000.0,
    loader: DataLoader | None = None,
    slippage_bps: float = 1.0,
    keep_trades: bool = False,
    progress_callback: ProgressCallback | None = None,
    continue_on_error: bool = False,
    skip_task_keys: set[str] | None = None,
) -> BacktestComparison:
    if not strategies:
        raise ValueError("至少需要一个策略")
    loader = ensure_in_memory_bar_loader(loader or get_loader())
    rows: list[BacktestSummary] = []
    failures: list[BacktestFailure] = []
    skip_task_keys = skip_task_keys or set()
    total = len(strategies)
    for done, name in enumerate(strategies, start=1):
        task_key = backtest_task_key(strategy_name=name, symbols=symbols, start=start, end=end)
        if task_key in skip_task_keys:
            _notify_progress(progress_callback, done, total, {"strategy": name, "skipped": True})
            continue
        try:
            strategy = build_backtest_strategy(name, symbols=symbols)
            result = BacktestEngine(
                strategy=strategy,
                start=start,
                end=end,
                initial_cash=initial_cash,
                loader=loader,
                slippage_bps=slippage_bps,
                record_trades=keep_trades,
            ).run()
        except Exception as exc:
            if not continue_on_error:
                raise
            failure = _failure_from_exception(
                strategy_name=name,
                symbols=symbols,
                start=start,
                end=end,
                params=None,
                exc=exc,
            )
            failures.append(failure)
            _notify_progress(progress_callback, done, total, {"strategy": name, "failure": failure})
            continue
        summary = _summary_from_result(name, symbols, start, end, strategy, result, keep_trades=keep_trades)
        rows.append(summary)
        _notify_progress(progress_callback, done, total, {"strategy": name, "summary": summary})
    return BacktestComparison(rows=rows, failures=failures)


def run_backtest_param_grid(
    *,
    strategy_name: str,
    symbols: list[str],
    start: date,
    end: date,
    param_grid: dict[str, list[Any]],
    initial_cash: float = 100_000.0,
    loader: DataLoader | None = None,
    slippage_bps: float = 1.0,
    keep_trades: bool = False,
    progress_callback: ProgressCallback | None = None,
    continue_on_error: bool = False,
    skip_task_keys: set[str] | None = None,
) -> BacktestComparison:
    if strategy_name == "llm":
        raise ValueError("LLM 不支持参数网格回测，避免产生 API 成本")
    if not param_grid:
        raise ValueError("参数网格不能为空")
    loader = ensure_in_memory_bar_loader(loader or get_loader())
    param_sets = expand_param_grid(param_grid)
    total = len(param_sets)
    rows: list[BacktestSummary] = []
    failures: list[BacktestFailure] = []
    skip_task_keys = skip_task_keys or set()
    for done, params in enumerate(param_sets, start=1):
        task_key = backtest_task_key(strategy_name=strategy_name, symbols=symbols, start=start, end=end, params=params)
        if task_key in skip_task_keys:
            _notify_progress(
                progress_callback,
                done,
                total,
                {"strategy": strategy_name, "params": params, "skipped": True},
            )
            continue
        try:
            strategy = build_backtest_strategy(strategy_name, symbols=symbols, params=params)
            result = BacktestEngine(
                strategy=strategy,
                start=start,
                end=end,
                initial_cash=initial_cash,
                loader=loader,
                slippage_bps=slippage_bps,
                record_trades=keep_trades,
            ).run()
        except Exception as exc:
            if not continue_on_error:
                raise
            failure = _failure_from_exception(
                strategy_name=strategy_name,
                symbols=symbols,
                start=start,
                end=end,
                params=params,
                exc=exc,
            )
            failures.append(failure)
            _notify_progress(
                progress_callback,
                done,
                total,
                {"strategy": strategy_name, "params": params, "failure": failure},
            )
            continue
        summary = _summary_from_result(
            strategy_name,
            symbols,
            start,
            end,
            strategy,
            result,
            params=params,
            keep_trades=keep_trades,
        )
        rows.append(summary)
        _notify_progress(
            progress_callback,
            done,
            total,
            {"strategy": strategy_name, "params": params, "summary": summary},
        )
    return BacktestComparison(rows=rows, failures=failures)


def expand_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    for key, values in param_grid.items():
        if not key:
            raise ValueError("参数名不能为空")
        if not values:
            raise ValueError(f"参数 {key} 的候选值不能为空")
    keys = list(param_grid)
    return [dict(zip(keys, values, strict=True)) for values in product(*(param_grid[key] for key in keys))]


def backtest_task_key(
    *,
    strategy_name: str,
    symbols: list[str],
    start: date,
    end: date,
    params: dict[str, Any] | None = None,
) -> str:
    """Stable identity for one backtest task, used by checkpoint resume."""
    return json.dumps(
        {
            "strategy": strategy_name,
            "symbols": list(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "params": params or {},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sort_backtest_comparison(
    comparison: BacktestComparison,
    *,
    sort_by: str = "sharpe",
    descending: bool = True,
) -> BacktestComparison:
    key_func = _SORT_KEYS.get(sort_by)
    if key_func is None:
        raise ValueError(f"未知排序指标：{sort_by}（可用：{list(_SORT_KEYS)}）")
    return BacktestComparison(
        rows=sorted(comparison.rows, key=key_func, reverse=descending),
        failures=comparison.failures,
        generated_at=comparison.generated_at,
    )


def limit_backtest_comparison(comparison: BacktestComparison, *, top: int | None) -> BacktestComparison:
    if top is None or top <= 0:
        return comparison
    return BacktestComparison(
        rows=comparison.rows[:top],
        failures=comparison.failures,
        generated_at=comparison.generated_at,
    )


def compute_parameter_sensitivity(
    comparison: BacktestComparison,
    *,
    top_n: int = 10,
    sort_by: str = "sharpe",
    descending: bool = True,
) -> list[ParameterSensitivity]:
    """Summarize whether top results concentrate in specific parameter values."""
    if top_n <= 0 or not comparison.rows:
        return []
    key_func = _SORT_KEYS.get(sort_by)
    if key_func is None:
        raise ValueError(f"未知排序指标：{sort_by}（可用：{list(_SORT_KEYS)}）")
    rows_with_params = [row for row in comparison.rows if row.params]
    if not rows_with_params:
        return []
    ranked = sorted(rows_with_params, key=key_func, reverse=descending)
    top_rows = ranked[: min(top_n, len(ranked))]
    param_names = sorted({key for row in rows_with_params for key in row.params})
    out: list[ParameterSensitivity] = []
    for param_name in param_names:
        full_values = [row.params[param_name] for row in rows_with_params if param_name in row.params]
        top_values = [row.params[param_name] for row in top_rows if param_name in row.params]
        if not top_values:
            continue
        counts = Counter(_format_param_value(value) for value in top_values)
        dominant_value, dominant_count = counts.most_common(1)[0]
        out.append(
            ParameterSensitivity(
                param_name=param_name,
                top_count=len(top_values),
                dominant_value=dominant_value,
                dominant_count=dominant_count,
                dominant_share=dominant_count / len(top_values),
                top_unique_count=len(counts),
                full_unique_count=len({_format_param_value(value) for value in full_values}),
                boundary_side=_boundary_side(dominant_value, full_values),
            )
        )
    return out


def run_walk_forward_param_grid(
    *,
    strategy_name: str,
    symbols: list[str],
    start: date,
    end: date,
    param_grid: dict[str, list[Any]],
    train_days: int,
    test_days: int,
    step_days: int | None = None,
    sort_by: str = "sharpe",
    descending: bool = True,
    initial_cash: float = 100_000.0,
    loader: DataLoader | None = None,
    slippage_bps: float = 1.0,
    keep_trades: bool = False,
    progress_callback: ProgressCallback | None = None,
    continue_on_error: bool = False,
) -> WalkForwardReport:
    if start >= end:
        raise ValueError("start 必须早于 end")
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days 和 test_days 必须为正数")
    step_days = step_days or test_days
    if step_days <= 0:
        raise ValueError("step_days 必须为正数")
    # Validate sort key early, before potentially expensive backtests.
    if sort_by not in _SORT_KEYS:
        raise ValueError(f"未知排序指标：{sort_by}（可用：{list(_SORT_KEYS)}）")
    loader = ensure_in_memory_bar_loader(loader or get_loader())
    windows = _walk_forward_windows(start=start, end=end, train_days=train_days, test_days=test_days, step_days=step_days)
    param_count = len(expand_param_grid(param_grid))
    total_runs = len(windows) * (param_count + 1)
    completed_runs = 0
    folds: list[WalkForwardFold] = []

    def forward_progress(stage: str, fold_no: int) -> ProgressCallback:
        def callback(_done: int, _total: int, payload: dict[str, Any]) -> None:
            nonlocal completed_runs
            completed_runs += 1
            _notify_progress(
                progress_callback,
                completed_runs,
                total_runs,
                {"fold": fold_no, "stage": stage, **payload},
            )

        return callback

    for fold_no, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        train_comparison = run_backtest_param_grid(
            strategy_name=strategy_name,
            symbols=symbols,
            start=train_start,
            end=train_end,
            param_grid=param_grid,
            initial_cash=initial_cash,
            loader=loader,
            slippage_bps=slippage_bps,
            keep_trades=keep_trades,
            progress_callback=forward_progress("train", fold_no),
            continue_on_error=continue_on_error,
        )
        ranked_train = sort_backtest_comparison(train_comparison, sort_by=sort_by, descending=descending)
        if not ranked_train.rows:
            continue
        best_train = ranked_train.rows[0]
        test_comparison = run_backtest_param_grid(
            strategy_name=strategy_name,
            symbols=symbols,
            start=test_start,
            end=test_end,
            param_grid={key: [value] for key, value in best_train.params.items()},
            initial_cash=initial_cash,
            loader=loader,
            slippage_bps=slippage_bps,
            keep_trades=keep_trades,
            progress_callback=forward_progress("test", fold_no),
            continue_on_error=continue_on_error,
        )
        if not test_comparison.rows:
            continue
        folds.append(
            WalkForwardFold(
                fold=fold_no,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                selected_params=dict(best_train.params),
                train=best_train,
                test=test_comparison.rows[0],
            )
        )
    if not folds:
        raise ValueError("没有生成可用 walk-forward 折，请检查时间区间和窗口长度")
    return WalkForwardReport(
        strategy_name=strategy_name,
        symbols=list(symbols),
        start=start,
        end=end,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        sort_by=sort_by,
        folds=folds,
    )


def walk_forward_report_to_markdown(report: WalkForwardReport) -> str:
    lines = [
        "# Walk-forward 验证报告",
        "",
        f"- 生成时间: {report.generated_at.isoformat()}",
        f"- 策略: {report.strategy_name}",
        f"- 标的: {','.join(report.symbols)}",
        f"- 区间: {report.start.isoformat()} → {report.end.isoformat()}",
        f"- 训练/验证/步长: {report.train_days}/{report.test_days}/{report.step_days} 天",
        f"- 训练择优指标: {report.sort_by}",
        f"- 折数: {len(report.folds)}",
        f"- 验证期复合收益: {report.compounded_test_return:.2%}",
        f"- 平均训练 Sharpe: {report.average_train_sharpe:.2f}",
        f"- 平均验证 Sharpe: {report.average_test_sharpe:.2f}",
        f"- 平均 Sharpe 衰减: {report.average_sharpe_decay:.2f}",
        f"- 验证/训练 Sharpe 保留率: {report.test_sharpe_retention:.2%}",
        f"- 验证正收益折数: {report.profitable_test_fold_count}/{len(report.folds)} ({report.positive_test_return_rate:.2%})",
        "",
        "| 折 | 训练区间 | 验证区间 | 参数 | 训练 Sharpe | 验证收益 | 验证 Sharpe | Sharpe 衰减 | 验证最大回撤 | 验证交易 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in report.folds:
        lines.append(
            "| "
            f"{fold.fold} | "
            f"{fold.train_start.isoformat()} → {fold.train_end.isoformat()} | "
            f"{fold.test_start.isoformat()} → {fold.test_end.isoformat()} | "
            f"{_format_params(fold.selected_params)} | "
            f"{fold.train.metrics.sharpe:.2f} | "
            f"{fold.test.metrics.total_return:.2%} | "
            f"{fold.test.metrics.sharpe:.2f} | "
            f"{fold.train.metrics.sharpe - fold.test.metrics.sharpe:.2f} | "
            f"{fold.test.metrics.max_drawdown:.2%} | "
            f"{fold.test.metrics.n_trades} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_walk_forward_markdown(report: WalkForwardReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(walk_forward_report_to_markdown(report), encoding="utf-8")
    return out


def write_walk_forward_csv(report: WalkForwardReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [fold.to_row() for fold in report.folds]
    fieldnames = list(rows[0]) if rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _summary_from_result(
    name: str,
    symbols: list[str],
    start: date,
    end: date,
    strategy: Strategy,
    result: BacktestResult,
    *,
    params: dict[str, Any] | None = None,
    keep_trades: bool = False,
) -> BacktestSummary:
    trade_count = result.trade_count if result.trade_count is not None else len(result.trades)
    return BacktestSummary(
        strategy_name=name,
        strategy_id=strategy.id,
        symbols=list(symbols),
        start=start,
        end=end,
        metrics=result.metrics,
        trades=list(result.trades) if keep_trades else [],
        params=dict(params or {}),
        trade_count=trade_count,
    )


def _failure_from_exception(
    *,
    strategy_name: str,
    symbols: list[str],
    start: date,
    end: date,
    params: dict[str, Any] | None,
    exc: Exception,
) -> BacktestFailure:
    return BacktestFailure(
        strategy_name=strategy_name,
        symbols=list(symbols),
        start=start,
        end=end,
        params=dict(params or {}),
        error_type=exc.__class__.__name__,
        error=str(exc),
    )


def _notify_progress(
    progress_callback: ProgressCallback | None,
    done: int,
    total: int,
    payload: dict[str, Any],
) -> None:
    if progress_callback is None:
        return
    progress_callback(done, total, payload)


def comparison_to_markdown(
    comparison: BacktestComparison,
    *,
    sensitivity_source: BacktestComparison | None = None,
    sensitivity_sort_by: str = "sharpe",
    sensitivity_top_n: int = 10,
) -> str:
    best_sharpe = comparison.best_by_sharpe
    best_return = comparison.best_by_return
    has_params = any(row.params for row in comparison.rows)
    lines = [
        "# 回测策略对比报告",
        "",
        f"- 生成时间: {comparison.generated_at.isoformat()}",
        f"- 成功结果: {len(comparison.rows)}",
        f"- 失败组合: {len(comparison.failures)}",
        f"- 失败率: {comparison.failure_rate:.2%}",
    ]
    if best_sharpe:
        lines.append(f"- Sharpe 最优: {best_sharpe.strategy_name} ({best_sharpe.metrics.sharpe:.2f})")
    if best_return:
        lines.append(f"- 总收益最优: {best_return.strategy_name} ({best_return.metrics.total_return:.2%})")
    lines.append("")
    if has_params:
        lines.extend(
            [
                "| 策略 | 参数 | 总收益 | 年化 | 波动 | Sharpe | Sortino | Calmar | 最大回撤 | 交易 | 胜率 | 终值 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| 策略 | 总收益 | 年化 | 波动 | Sharpe | Sortino | Calmar | 最大回撤 | 交易 | 胜率 | 终值 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in comparison.rows:
        m = row.metrics
        cells = [row.strategy_name]
        if has_params:
            cells.append(_format_params(row.params))
        cells.extend(
            [
                f"{m.total_return:.2%}",
                f"{m.annual_return:.2%}",
                f"{m.annual_volatility:.2%}",
                f"{m.sharpe:.2f}",
                f"{m.sortino:.2f}",
                f"{m.calmar:.2f}",
                f"{m.max_drawdown:.2%}",
                f"{m.n_trades}",
                f"{m.win_rate:.2%}",
                f"{m.final_equity:,.2f}",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    sensitivity_rows = compute_parameter_sensitivity(
        sensitivity_source or comparison,
        top_n=sensitivity_top_n,
        sort_by=sensitivity_sort_by,
    )
    if sensitivity_rows:
        top_count = sensitivity_rows[0].top_count
        lines.extend(
            [
                "",
                "## 参数敏感性",
                "",
                f"- 统计口径: 按 {sensitivity_sort_by} 前 {top_count} 条成功结果统计。",
                "- 边界值提示: Top 主导值落在全网格数值最小/最大值，建议扩展参数范围继续验证。",
                "",
                "| 参数 | Top 主导值 | Top 占比 | Top 唯一值 | 全样本唯一值 | 边界 |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sensitivity_rows:
            boundary = row.boundary_side or "-"
            lines.append(
                "| "
                f"{row.param_name} | "
                f"{row.dominant_value} | "
                f"{row.dominant_share:.2%} | "
                f"{row.top_unique_count} | "
                f"{row.full_unique_count} | "
                f"{boundary} |"
            )
    if comparison.failures:
        lines.extend(
            [
                "",
                "## 失败组合",
                "",
                "| 策略 | 参数 | 错误类型 | 错误 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for failure in comparison.failures:
            lines.append(
                "| "
                f"{failure.strategy_name} | "
                f"{_format_params(failure.params)} | "
                f"{failure.error_type} | "
                f"{failure.error} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_comparison_markdown(
    comparison: BacktestComparison,
    path: str | Path,
    *,
    sensitivity_source: BacktestComparison | None = None,
    sensitivity_sort_by: str = "sharpe",
    sensitivity_top_n: int = 10,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        comparison_to_markdown(
            comparison,
            sensitivity_source=sensitivity_source,
            sensitivity_sort_by=sensitivity_sort_by,
            sensitivity_top_n=sensitivity_top_n,
        ),
        encoding="utf-8",
    )
    return out


def write_comparison_csv(comparison: BacktestComparison, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.to_row() for row in comparison.rows]
    fieldnames = list(rows[0]) if rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return ""
    return json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _format_param_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _boundary_side(dominant_value: str, values: list[Any]) -> str:
    numeric_values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if len(numeric_values) != len(values) or not numeric_values:
        return ""
    numeric_dominant = next(
        (float(value) for value in numeric_values if _format_param_value(value) == dominant_value),
        None,
    )
    if numeric_dominant is None:
        return ""
    min_value = min(float(value) for value in numeric_values)
    max_value = max(float(value) for value in numeric_values)
    if numeric_dominant == min_value:
        return "min"
    if numeric_dominant == max_value:
        return "max"
    return ""


def _walk_forward_windows(
    *,
    start: date,
    end: date,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[tuple[date, date, date, date]]:
    windows: list[tuple[date, date, date, date]] = []
    train_start = start
    while True:
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        if test_start > end:
            break
        test_end = min(test_start + timedelta(days=test_days - 1), end)
        windows.append((train_start, train_end, test_start, test_end))
        if test_end >= end:
            break
        train_start = train_start + timedelta(days=step_days)
    return windows


_SORT_KEYS: dict[str, Callable[[BacktestSummary], float]] = {
    "sharpe": lambda row: row.metrics.sharpe,
    "sortino": lambda row: row.metrics.sortino,
    "calmar": lambda row: row.metrics.calmar,
    "total_return": lambda row: row.metrics.total_return,
    "annual_return": lambda row: row.metrics.annual_return,
    "final_equity": lambda row: row.metrics.final_equity,
    "max_drawdown": lambda row: row.metrics.max_drawdown,
}
