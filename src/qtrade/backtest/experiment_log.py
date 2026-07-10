"""Append-only experiment records for backtest research runs."""
from __future__ import annotations

import csv
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from qtrade.backtest.comparison import (
    BacktestComparison,
    BacktestSummary,
    WalkForwardReport,
    compute_parameter_sensitivity,
)
from qtrade.core.time import utcnow


@dataclass(frozen=True)
class ExperimentCandidate:
    record_type: str
    generated_at: str
    strategy: str
    symbols: list[str]
    start: str
    end: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    failure_rate: float | None
    source_path: str
    source_line: int
    note: str = ""

    def metric(self, name: str) -> float | None:
        value = self.failure_rate if name == "failure_rate" else self.metrics.get(name)
        if isinstance(value, bool) or value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class ValidationCommand:
    candidate: ExperimentCandidate
    command: str
    validation_type: str


@dataclass(frozen=True)
class PlanCommand:
    index: int
    command: str
    argv: list[str]
    source_line: int


@dataclass(frozen=True)
class PlanRunResult:
    index: int
    command: str
    status: str
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""


def append_backtest_grid_experiment(
    path: str | Path,
    *,
    kind: str,
    strategy_name: str,
    symbols: list[str],
    start: date,
    end: date,
    param_grid: dict[str, list[Any]],
    comparison: BacktestComparison,
    sort_by: str,
    descending: bool,
    top: int,
    csv_output: str | Path | None = None,
    markdown_output: str | Path | None = None,
    checkpoint_csv: str | Path | None = None,
    sensitivity_top_n: int = 10,
) -> Path:
    record = {
        "schema_version": 1,
        "record_type": kind,
        "generated_at": utcnow().isoformat(),
        "strategy": strategy_name,
        "symbols": list(symbols),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "param_grid": _jsonable(param_grid),
        "sort_by": sort_by,
        "descending": descending,
        "display_top": top,
        "success_count": len(comparison.rows),
        "failure_count": len(comparison.failures),
        "failure_rate": comparison.failure_rate,
        "best_by_sharpe": _summary_payload(comparison.best_by_sharpe),
        "best_by_return": _summary_payload(comparison.best_by_return),
        "top_results": [_summary_payload(row) for row in comparison.rows[: max(sensitivity_top_n, 0)]],
        "parameter_sensitivity": [
            {
                "param_name": row.param_name,
                "top_count": row.top_count,
                "dominant_value": row.dominant_value,
                "dominant_count": row.dominant_count,
                "dominant_share": row.dominant_share,
                "top_unique_count": row.top_unique_count,
                "full_unique_count": row.full_unique_count,
                "boundary_side": row.boundary_side,
            }
            for row in compute_parameter_sensitivity(
                comparison,
                top_n=sensitivity_top_n,
                sort_by=sort_by,
                descending=descending,
            )
        ],
        "outputs": _outputs_payload(
            csv_output=csv_output,
            markdown_output=markdown_output,
            checkpoint_csv=checkpoint_csv,
        ),
    }
    return _append_jsonl(path, record)


def append_walk_forward_experiment(
    path: str | Path,
    *,
    kind: str,
    strategy_name: str,
    symbols: list[str],
    start: date,
    end: date,
    param_grid: dict[str, list[Any]],
    report: WalkForwardReport,
    sort_by: str,
    descending: bool,
    csv_output: str | Path | None = None,
    markdown_output: str | Path | None = None,
    checkpoint_csv: str | Path | None = None,
) -> Path:
    record = {
        "schema_version": 1,
        "record_type": kind,
        "generated_at": utcnow().isoformat(),
        "strategy": strategy_name,
        "symbols": list(symbols),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "param_grid": _jsonable(param_grid),
        "train_days": report.train_days,
        "test_days": report.test_days,
        "step_days": report.step_days,
        "sort_by": sort_by,
        "descending": descending,
        "fold_count": len(report.folds),
        "compounded_test_return": report.compounded_test_return,
        "average_train_sharpe": report.average_train_sharpe,
        "average_test_sharpe": report.average_test_sharpe,
        "average_sharpe_decay": report.average_sharpe_decay,
        "test_sharpe_retention": report.test_sharpe_retention,
        "positive_test_return_rate": report.positive_test_return_rate,
        "profitable_test_fold_count": report.profitable_test_fold_count,
        "folds": [
            {
                "fold": fold.fold,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "selected_params": _jsonable(fold.selected_params),
                "train": _summary_payload(fold.train),
                "test": _summary_payload(fold.test),
            }
            for fold in report.folds
        ],
        "outputs": _outputs_payload(
            csv_output=csv_output,
            markdown_output=markdown_output,
            checkpoint_csv=checkpoint_csv,
        ),
    }
    return _append_jsonl(path, record)


def read_experiment_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"实验记录不存在：{source}")
        with source.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_no} 不是合法 JSONL：{exc}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{source}:{line_no} 实验记录必须是 JSON object")
                payload["_source_path"] = str(source)
                payload["_source_line"] = line_no
                records.append(payload)
    return records


def experiment_candidates_from_records(
    records: list[dict[str, Any]],
    *,
    record_types: set[str] | None = None,
) -> list[ExperimentCandidate]:
    candidates: list[ExperimentCandidate] = []
    for record in records:
        record_type = str(record.get("record_type", ""))
        if record_types and record_type not in record_types:
            continue
        if record_type in {"backtest_grid", "factor_grid"}:
            candidates.extend(_grid_candidates(record))
        elif record_type == "factor_walk_forward":
            candidate = _walk_forward_candidate(record)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def sort_experiment_candidates(
    candidates: list[ExperimentCandidate],
    *,
    sort_by: str = "sharpe",
    descending: bool = True,
) -> list[ExperimentCandidate]:
    if sort_by not in _EXPERIMENT_SORT_KEYS:
        raise ValueError(f"未知排序指标：{sort_by}（可用：{sorted(_EXPERIMENT_SORT_KEYS)}）")
    sortable = [candidate for candidate in candidates if candidate.metric(sort_by) is not None]
    unsortable = [candidate for candidate in candidates if candidate.metric(sort_by) is None]
    sortable.sort(key=lambda candidate: candidate.metric(sort_by) or 0.0, reverse=descending)
    return [*sortable, *unsortable]


def limit_experiment_candidates(candidates: list[ExperimentCandidate], *, top: int | None) -> list[ExperimentCandidate]:
    if top is None or top <= 0:
        return candidates
    return candidates[:top]


def write_experiment_candidates_csv(candidates: list[ExperimentCandidate], path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_type",
        "generated_at",
        "strategy",
        "symbols",
        "start",
        "end",
        "params",
        "total_return",
        "annual_return",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "n_trades",
        "win_rate",
        "final_equity",
        "test_sharpe_retention",
        "failure_rate",
        "note",
        "source_path",
        "source_line",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = {
                "record_type": candidate.record_type,
                "generated_at": candidate.generated_at,
                "strategy": candidate.strategy,
                "symbols": ",".join(candidate.symbols),
                "start": candidate.start,
                "end": candidate.end,
                "params": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
                "failure_rate": candidate.failure_rate,
                "note": candidate.note,
                "source_path": candidate.source_path,
                "source_line": candidate.source_line,
            }
            for key in fields:
                if key not in row:
                    row[key] = candidate.metrics.get(key)
            writer.writerow(row)
    return out


def build_validation_commands(
    candidates: list[ExperimentCandidate],
    *,
    command_prefix: str = "qtrade",
    validation_start: str | None = None,
    validation_end: str | None = None,
    train_days: int = 365,
    test_days: int = 90,
    step_days: int | None = None,
    next_experiment_log: str | Path | None = None,
) -> list[ValidationCommand]:
    commands: list[ValidationCommand] = []
    for candidate in candidates:
        command = _validation_command_for_candidate(
            candidate,
            command_prefix=command_prefix,
            validation_start=validation_start,
            validation_end=validation_end,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            next_experiment_log=next_experiment_log,
        )
        if command is not None:
            commands.append(command)
    return commands


def write_validation_plan_markdown(commands: list[ValidationCommand], path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 候选策略下一轮验证计划",
        "",
        "> 这里只生成命令，不自动执行，不下单。",
        "",
    ]
    if not commands:
        lines.extend(["没有可生成的验证命令。", ""])
    for index, plan in enumerate(commands, start=1):
        candidate = plan.candidate
        lines.extend(
            [
                f"## {index}. {candidate.record_type} / {candidate.strategy}",
                "",
                f"- 标的: {', '.join(candidate.symbols) or '-'}",
                f"- 原区间: {candidate.start} -> {candidate.end}",
                f"- 参数: `{json.dumps(candidate.params, ensure_ascii=False, sort_keys=True)}`",
                f"- 验证类型: {plan.validation_type}",
                "",
                "```bash",
                plan.command,
                "```",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def read_validation_plan_commands(path: str | Path) -> list[PlanCommand]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"验证计划不存在：{source}")
    commands: list[PlanCommand] = []
    in_code_block = False
    for line_no, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block or not line or line.startswith("#"):
            continue
        argv = shlex.split(line)
        if not argv:
            continue
        _validate_plan_argv(argv, source_line=line_no)
        commands.append(
            PlanCommand(
                index=len(commands) + 1,
                command=line,
                argv=argv,
                source_line=line_no,
            )
        )
    return commands


def run_validation_plan_commands(
    commands: list[PlanCommand],
    *,
    execute: bool = False,
    max_commands: int = 5,
    timeout_seconds: int = 600,
    cwd: str | Path | None = None,
) -> list[PlanRunResult]:
    if max_commands < 0:
        raise ValueError("max_commands 不能为负数")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须大于 0")
    selected = commands if max_commands == 0 else commands[:max_commands]
    results: list[PlanRunResult] = []
    for command in selected:
        if not execute:
            results.append(
                PlanRunResult(
                    index=command.index,
                    command=command.command,
                    status="dry_run",
                )
            )
            continue
        try:
            completed = subprocess.run(
                command.argv,
                cwd=str(Path(cwd).expanduser().resolve()) if cwd is not None else None,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            results.append(
                PlanRunResult(
                    index=command.index,
                    command=command.command,
                    status="timeout",
                    stdout_tail=_tail_text(exc.stdout),
                    stderr_tail=_tail_text(exc.stderr),
                    error=f"超过 {timeout_seconds}s",
                )
            )
            continue
        except OSError as exc:
            results.append(
                PlanRunResult(
                    index=command.index,
                    command=command.command,
                    status="failed",
                    error=str(exc),
                )
            )
            continue
        results.append(
            PlanRunResult(
                index=command.index,
                command=command.command,
                status="success" if completed.returncode == 0 else "failed",
                returncode=completed.returncode,
                stdout_tail=_tail_text(completed.stdout),
                stderr_tail=_tail_text(completed.stderr),
            )
        )
    return results


def write_plan_run_summary_json(
    path: str | Path,
    *,
    plan_path: str | Path,
    results: list[PlanRunResult],
    execute: bool,
    total_commands: int,
    max_commands: int,
    timeout_seconds: int,
    cwd: str | Path | None = None,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    payload = {
        "schema_version": 1,
        "record_type": "validation_plan_run",
        "generated_at": utcnow().isoformat(),
        "plan_path": _path_payload(plan_path),
        "execute": execute,
        "cwd": _path_payload(cwd) if cwd is not None else str(Path.cwd()),
        "total_commands": total_commands,
        "processed_commands": len(results),
        "skipped_commands": max(total_commands - len(results), 0),
        "max_commands": max_commands,
        "timeout_seconds": timeout_seconds,
        "status_counts": status_counts,
        "results": [
            {
                "index": result.index,
                "command": result.command,
                "status": result.status,
                "returncode": result.returncode,
                "stdout_tail": result.stdout_tail,
                "stderr_tail": result.stderr_tail,
                "error": result.error,
            }
            for result in results
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
        f.write("\n")
    return out


def _summary_payload(row: BacktestSummary | None) -> dict[str, Any] | None:
    if row is None:
        return None
    m = row.metrics
    return {
        "strategy": row.strategy_name,
        "strategy_id": row.strategy_id,
        "symbols": list(row.symbols),
        "start": row.start.isoformat(),
        "end": row.end.isoformat(),
        "params": _jsonable(row.params),
        "metrics": {
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
        },
    }


def _outputs_payload(
    *,
    csv_output: str | Path | None,
    markdown_output: str | Path | None,
    checkpoint_csv: str | Path | None,
) -> dict[str, str | None]:
    return {
        "csv_output": _path_payload(csv_output),
        "markdown_output": _path_payload(markdown_output),
        "checkpoint_csv": _path_payload(checkpoint_csv),
    }


def _path_payload(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, date):
        return value.isoformat()
    return value


def _grid_candidates(record: dict[str, Any]) -> list[ExperimentCandidate]:
    rows = _unique_summary_payloads(record)
    return [
        _candidate_from_summary(record, row, note=f"rank={index}")
        for index, row in enumerate(rows, start=1)
        if row is not None
    ]


def _unique_summary_payloads(record: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key in ("best_by_sharpe", "best_by_return"):
        value = record.get(key)
        if isinstance(value, dict):
            token = _summary_token(value)
            if token not in seen:
                seen.add(token)
                rows.append(value)
    top_results = record.get("top_results", [])
    if isinstance(top_results, list):
        for value in top_results:
            if not isinstance(value, dict):
                continue
            token = _summary_token(value)
            if token in seen:
                continue
            seen.add(token)
            rows.append(value)
    return rows


def _summary_token(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "strategy_id": row.get("strategy_id"),
            "params": row.get("params", {}),
            "start": row.get("start"),
            "end": row.get("end"),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _candidate_from_summary(record: dict[str, Any], row: dict[str, Any], *, note: str = "") -> ExperimentCandidate:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    params = row.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return ExperimentCandidate(
        record_type=str(record.get("record_type", "")),
        generated_at=str(record.get("generated_at", "")),
        strategy=str(row.get("strategy") or record.get("strategy") or ""),
        symbols=_string_list(row.get("symbols") or record.get("symbols")),
        start=str(row.get("start") or record.get("start") or ""),
        end=str(row.get("end") or record.get("end") or ""),
        params=params,
        metrics=metrics,
        failure_rate=_float_or_none(record.get("failure_rate")),
        source_path=str(record.get("_source_path", "")),
        source_line=int(record.get("_source_line", 0) or 0),
        note=note,
    )


def _walk_forward_candidate(record: dict[str, Any]) -> ExperimentCandidate | None:
    folds = record.get("folds", [])
    if not isinstance(folds, list):
        folds = []
    selected_params = [
        fold.get("selected_params", {})
        for fold in folds
        if isinstance(fold, dict) and isinstance(fold.get("selected_params", {}), dict)
    ]
    params: dict[str, Any]
    unique_params = {_params_token(params) for params in selected_params}
    if len(unique_params) == 1 and selected_params:
        params = selected_params[0]
    else:
        params = {"fold_params": selected_params}

    test_metrics = [
        fold.get("test", {}).get("metrics", {})
        for fold in folds
        if isinstance(fold, dict) and isinstance(fold.get("test", {}).get("metrics", {}), dict)
    ]
    metrics = {
        "total_return": record.get("compounded_test_return"),
        "sharpe": record.get("average_test_sharpe"),
        "average_test_sharpe": record.get("average_test_sharpe"),
        "average_train_sharpe": record.get("average_train_sharpe"),
        "sharpe_decay": record.get("average_sharpe_decay"),
        "test_sharpe_retention": record.get("test_sharpe_retention"),
        "win_rate": record.get("positive_test_return_rate"),
        "positive_test_return_rate": record.get("positive_test_return_rate"),
        "n_trades": sum(int(metric.get("n_trades", 0) or 0) for metric in test_metrics),
        "max_drawdown": _average_numeric(metric.get("max_drawdown") for metric in test_metrics),
    }
    if not folds and metrics["total_return"] is None and metrics["sharpe"] is None:
        return None
    return ExperimentCandidate(
        record_type=str(record.get("record_type", "")),
        generated_at=str(record.get("generated_at", "")),
        strategy=str(record.get("strategy", "")),
        symbols=_string_list(record.get("symbols")),
        start=str(record.get("start", "")),
        end=str(record.get("end", "")),
        params=params,
        metrics=metrics,
        failure_rate=None,
        source_path=str(record.get("_source_path", "")),
        source_line=int(record.get("_source_line", 0) or 0),
        note=f"folds={len(folds)}",
    )


def _params_token(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average_numeric(values: Any) -> float | None:
    nums = [_float_or_none(value) for value in values]
    nums = [value for value in nums if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _validation_command_for_candidate(
    candidate: ExperimentCandidate,
    *,
    command_prefix: str,
    validation_start: str | None,
    validation_end: str | None,
    train_days: int,
    test_days: int,
    step_days: int | None,
    next_experiment_log: str | Path | None,
) -> ValidationCommand | None:
    if candidate.record_type == "factor_grid":
        argv = _factor_walk_forward_argv(
            candidate,
            command_prefix=command_prefix,
            validation_start=validation_start,
            validation_end=validation_end,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            next_experiment_log=next_experiment_log,
        )
        return ValidationCommand(candidate, _shell_join(argv), "factor-walk-forward") if argv else None
    if candidate.record_type == "factor_walk_forward":
        argv = _factor_backtest_argv(
            candidate,
            command_prefix=command_prefix,
            validation_start=validation_start,
            validation_end=validation_end,
        )
        return ValidationCommand(candidate, _shell_join(argv), "factor-backtest") if argv else None
    if candidate.record_type == "backtest_grid":
        argv = _backtest_grid_argv(
            candidate,
            command_prefix=command_prefix,
            validation_start=validation_start,
            validation_end=validation_end,
            next_experiment_log=next_experiment_log,
        )
        return ValidationCommand(candidate, _shell_join(argv), "backtest-grid") if argv else None
    return None


def _factor_walk_forward_argv(
    candidate: ExperimentCandidate,
    *,
    command_prefix: str,
    validation_start: str | None,
    validation_end: str | None,
    train_days: int,
    test_days: int,
    step_days: int | None,
    next_experiment_log: str | Path | None,
) -> list[str]:
    params = candidate.params
    factor = _param_scalar(params, "factor_name")
    if factor is None:
        return []
    argv = [
        command_prefix,
        "factor-walk-forward",
        "--factor",
        str(factor),
    ]
    _append_symbol_args(argv, candidate.symbols)
    _append_date_args(argv, candidate, validation_start=validation_start, validation_end=validation_end)
    for window in _param_list(params, "windows"):
        argv.extend(["--window", str(window)])
    _append_optional_param(argv, params, "lookback", "--lookback")
    _append_optional_param(argv, params, "entry_quantile", "--entry-quantile")
    _append_optional_param(argv, params, "exit_quantile", "--exit-quantile")
    _append_optional_param(argv, params, "direction", "--direction")
    _append_optional_param(argv, params, "position_pct", "--position-pct")
    if params.get("include_interactions") is True:
        argv.append("--include-interactions")
        _append_optional_param(argv, params, "max_interactions", "--max-interactions")
    argv.extend(["--train-days", str(train_days), "--test-days", str(test_days)])
    if step_days is not None:
        argv.extend(["--step-days", str(step_days)])
    argv.extend(["--max-param-combinations", "1"])
    if next_experiment_log is not None:
        argv.extend(["--experiment-log", str(next_experiment_log)])
    return argv


def _factor_backtest_argv(
    candidate: ExperimentCandidate,
    *,
    command_prefix: str,
    validation_start: str | None,
    validation_end: str | None,
) -> list[str]:
    params = candidate.params
    if "fold_params" in params:
        return []
    factor = _param_scalar(params, "factor_name")
    if factor is None:
        return []
    argv = [
        command_prefix,
        "factor-backtest",
        "--factor",
        str(factor),
    ]
    _append_symbol_args(argv, candidate.symbols)
    _append_date_args(argv, candidate, validation_start=validation_start, validation_end=validation_end)
    for window in _param_list(params, "windows"):
        argv.extend(["--window", str(window)])
    _append_optional_param(argv, params, "lookback", "--lookback")
    _append_optional_param(argv, params, "entry_quantile", "--entry-quantile")
    _append_optional_param(argv, params, "exit_quantile", "--exit-quantile")
    _append_optional_param(argv, params, "direction", "--direction")
    _append_optional_param(argv, params, "position_pct", "--position-pct")
    if params.get("include_interactions") is True:
        argv.append("--include-interactions")
        _append_optional_param(argv, params, "max_interactions", "--max-interactions")
    return argv


def _backtest_grid_argv(
    candidate: ExperimentCandidate,
    *,
    command_prefix: str,
    validation_start: str | None,
    validation_end: str | None,
    next_experiment_log: str | Path | None,
) -> list[str]:
    argv = [
        command_prefix,
        "backtest-grid",
        "--strategy",
        candidate.strategy,
    ]
    _append_symbol_args(argv, candidate.symbols)
    _append_date_args(argv, candidate, validation_start=validation_start, validation_end=validation_end)
    for key, value in candidate.params.items():
        argv.extend(["--param", f"{key}={_param_value_for_grid(value)}"])
    argv.extend(["--max-param-combinations", "1"])
    if next_experiment_log is not None:
        argv.extend(["--experiment-log", str(next_experiment_log)])
    return argv


def _append_symbol_args(argv: list[str], symbols: list[str]) -> None:
    for symbol in symbols:
        argv.extend(["--symbol", symbol])


def _append_date_args(
    argv: list[str],
    candidate: ExperimentCandidate,
    *,
    validation_start: str | None,
    validation_end: str | None,
) -> None:
    argv.extend(["--start", validation_start or candidate.start])
    argv.extend(["--end", validation_end or candidate.end])


def _append_optional_param(argv: list[str], params: dict[str, Any], key: str, flag: str) -> None:
    value = _param_scalar(params, key)
    if value is not None:
        argv.extend([flag, str(value)])


def _param_scalar(params: dict[str, Any], key: str) -> Any:
    value = params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _param_list(params: dict[str, Any], key: str) -> list[Any]:
    value = params.get(key)
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _param_value_for_grid(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def _validate_plan_argv(argv: list[str], *, source_line: int) -> None:
    if len(argv) < 2:
        raise ValueError(f"第 {source_line} 行命令不完整")
    if argv[0] != "qtrade":
        raise ValueError(f"第 {source_line} 行不是 qtrade 命令：{argv[0]}")
    if argv[1] not in _ALLOWED_PLAN_SUBCOMMANDS:
        allowed = ", ".join(sorted(_ALLOWED_PLAN_SUBCOMMANDS))
        raise ValueError(f"第 {source_line} 行命令不在验证白名单内：{argv[1]}（允许：{allowed}）")


def _tail_text(value: str | bytes | None, *, max_chars: int = 2000) -> str:
    if value is None:
        return ""
    text = value.decode(errors="replace") if isinstance(value, bytes) else value
    return text[-max_chars:]


_ALLOWED_PLAN_SUBCOMMANDS = {"backtest-grid", "factor-walk-forward", "factor-backtest"}


_EXPERIMENT_SORT_KEYS = {
    "total_return",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "n_trades",
    "win_rate",
    "final_equity",
    "test_sharpe_retention",
    "failure_rate",
}
