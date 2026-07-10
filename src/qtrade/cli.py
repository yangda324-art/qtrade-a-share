"""qtrade 命令行入口（typer）。

子命令：
- ``init``       初始化数据库
- ``download``   下载历史数据到本地缓存
- ``backtest``   跑一个策略的回测，输出指标
- ``ui``         启动桌面 UI
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from qtrade.core.exceptions import DataError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="qtrade 量化平台")
console = Console()

_LLM_PROMPT_VERSIONS = {"v1", "v2"}


def _validate_common_run_options(
    *,
    interval_seconds: int | None,
    decision_interval_minutes: int | None,
    position_pct: float | None = None,
    max_position_pct: float | None = None,
) -> None:
    if interval_seconds is not None and not 1 <= interval_seconds <= 59:
        raise typer.BadParameter("interval_seconds 必须在 1-59 秒之间")
    if decision_interval_minutes is not None and decision_interval_minutes < 0:
        raise typer.BadParameter("decision_interval_minutes 不能为负数")
    if position_pct is not None and not 0 < position_pct <= 1:
        raise typer.BadParameter("position_pct 必须在 0-1 之间")
    if max_position_pct is not None and not 0 < max_position_pct <= 1:
        raise typer.BadParameter("max_position_pct 必须在 0-1 之间")


def _validate_llm_run_options(
    *,
    prompt_version: str,
    debate_rounds: int,
    threshold: float | None,
) -> None:
    if prompt_version not in _LLM_PROMPT_VERSIONS:
        raise typer.BadParameter(f"prompt_version 必须是 {sorted(_LLM_PROMPT_VERSIONS)}")
    if not 1 <= debate_rounds <= 4:
        raise typer.BadParameter("debate_rounds 必须在 1-4 之间")
    if threshold is not None and not 0 <= threshold <= 1:
        raise typer.BadParameter("threshold 必须在 0-1 之间")


def _require_deepseek_key_for_llm_run():
    from qtrade.core.config import get_settings

    settings = get_settings()
    if settings.llm.provider == "deepseek" and not settings.deepseek_api_key:
        console.print("[red]✗[/red] LLM 模拟实盘需要 DEEPSEEK_API_KEY")
        console.print("  在项目根目录 `.env` 写入：DEEPSEEK_API_KEY=sk-xxxxx")
        console.print("  或临时执行：export DEEPSEEK_API_KEY=sk-xxxxx")
        console.print("  先测试一次：qtrade decide 000001 --lookback-days 120")
        raise typer.Exit(1)
    return settings


def _print_llm_run_preflight(
    *,
    settings,
    prompt_version: str,
    debate_rounds: int,
    threshold: float | None,
    decision_interval_seconds: int,
    position_pct: float | None,
    max_position_pct: float | None,
) -> None:
    table = Table(title="LLM 模拟实盘启动前检查")
    table.add_column("项目", style="cyan")
    table.add_column("值")
    table.add_row("Provider", settings.llm.provider)
    table.add_row("Deep model", settings.llm.deep_model)
    table.add_row("Quick model", settings.llm.quick_model)
    table.add_row("Prompt", prompt_version)
    table.add_row("Debate rounds", str(debate_rounds))
    table.add_row("Decision threshold", f"{threshold if threshold is not None else 0.45:.2f}")
    table.add_row("Decision interval", f"{decision_interval_seconds // 60} 分钟")
    table.add_row("Position pct", f"{position_pct:.0%}" if position_pct is not None else "策略默认 20%")
    table.add_row("Max position pct", f"{max_position_pct:.0%}" if max_position_pct is not None else "策略默认 30%")
    table.add_row("Quote max age", f"{settings.trading.realtime.max_quote_age_seconds}s")
    table.add_row("Price deviation guard", f"{settings.trading.realtime.max_price_deviation_bps}bp")
    table.add_row("LLM daily decisions", _budget_value(settings.llm.daily_max_decisions))
    table.add_row("LLM daily tokens", _budget_value(settings.llm.daily_max_tokens))
    table.add_row("LLM daily cost", _budget_value(settings.llm.daily_max_cost_usd, prefix="$"))
    console.print(table)


def _budget_value(value: int | float | None, *, prefix: str = "") -> str:
    if value is None or value <= 0:
        return "不限制"
    if isinstance(value, float):
        return f"{prefix}{value:.4f}" if prefix else f"{value:.4f}"
    return f"{prefix}{value:,}"


@contextmanager
def _quiet_qtrade_logs(enabled: bool = True):
    """Temporarily hide qtrade runtime logs for commands with their own summary UI."""
    if not enabled:
        yield
        return
    from loguru import logger as _logger

    _logger.disable("qtrade")
    try:
        yield
    finally:
        _logger.enable("qtrade")


@app.command()
def init() -> None:
    """初始化数据库（创建所有表）。幂等。"""
    from qtrade.persistence import init_db
    init_db()
    console.print("[green]✓[/green] 数据库已初始化")


@app.command()
def download(
    symbols: list[str] = typer.Argument(..., help="标的列表，如 600519 000001"),
    start: str = typer.Option("2024-01-01", help="起始日 YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="终止日 YYYY-MM-DD，默认今天"),
) -> None:
    """下载日线到本地 Parquet 缓存。"""
    from qtrade.data.loader import get_loader
    loader = get_loader()
    end_d = date.fromisoformat(end) if end else date.today()
    for sym in symbols:
        df = loader.get_daily_bars(sym, start=start, end=end_d, force_refresh=True)
        console.print(f"[green]✓[/green] {sym}: {len(df)} 行")


@app.command()
def quote(
    symbols: list[str] = typer.Argument(..., help="标的列表，如 000001 600519"),
) -> None:
    """拉取 AkShare 最新快照，检查行情源状态。"""
    from qtrade.core.time import utc_naive_to_market_time
    from qtrade.data.loader import get_loader
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    quotes = get_loader().get_quotes(syms)

    table = Table(title="实时行情快照")
    table.add_column("标的", style="cyan")
    table.add_column("最新价", justify="right")
    table.add_column("涨跌幅", justify="right")
    table.add_column("今开", justify="right")
    table.add_column("最高", justify="right")
    table.add_column("最低", justify="right")
    table.add_column("成交量", justify="right")
    table.add_column("源")
    table.add_column("获取时间")
    table.add_column("年龄", justify="right")
    table.add_column("状态")

    for sym in syms:
        q = quotes.get(sym)
        if q is None:
            table.add_row(sym, "-", "-", "-", "-", "-", "-", "-", "-", "-", "[red]无行情[/red]")
            continue
        pct = ((q.price - q.prev_close) / q.prev_close) if q.prev_close else 0.0
        pct_style = "red" if pct > 0 else "green" if pct < 0 else "white"
        fetched = utc_naive_to_market_time(q.fetched_at) if q.fetched_at else None
        reject_reason = q.realtime_reject_reason()
        if reject_reason is None:
            status = "[green]可用于实时模拟[/green]"
        elif reject_reason == "fallback_quote":
            status = "[yellow]仅展示[/yellow]"
        elif reject_reason == "stale_quote":
            status = "[red]过期[/red]"
        else:
            status = f"[red]{reject_reason}[/red]"
        table.add_row(
            q.symbol,
            f"{q.price:.3f}",
            f"[{pct_style}]{pct:.2%}[/{pct_style}]",
            f"{q.open:.3f}",
            f"{q.high:.3f}",
            f"{q.low:.3f}",
            f"{q.volume:,.0f}",
            q.source,
            fetched.strftime("%H:%M:%S") if fetched else "-",
            f"{q.age_seconds():.1f}s",
            status,
        )

    console.print(table)


@app.command(name="data-check")
def data_check(
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s", help="检查标的，可重复指定"),
    periods: list[str] = typer.Option(
        ["daily", "5min", "15min"],
        "--period",
        "-p",
        help="检查/预热 K 线周期，可重复指定；支持 5min/15min/30min/60min/daily/weekly/monthly",
    ),
    start: str | None = typer.Option(None, help="起始日 YYYY-MM-DD；默认最近 30 天"),
    end: str | None = typer.Option(None, help="终止日 YYYY-MM-DD；默认今天"),
    adjust: str = typer.Option("qfq", help="复权模式：none/qfq/hfq"),
    skip_quotes: bool = typer.Option(False, "--skip-quotes", help="跳过实时行情快照检查"),
    force_refresh: bool = typer.Option(False, "--force-refresh", help="强制刷新 K 线缓存"),
    json_output: Path | None = typer.Option(None, "--json-output", help="写入 JSON 报告"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="写入 Markdown 报告"),
) -> None:
    """行情源健康检查/缓存预热；不启动交易、不写订单。"""
    from qtrade.data.health import (
        run_data_health_check,
        write_data_health_json,
        write_data_health_markdown,
    )

    if adjust not in {"none", "qfq", "hfq"}:
        raise typer.BadParameter("adjust 必须是 none/qfq/hfq")
    try:
        report = run_data_health_check(
            symbols=symbols,
            periods=periods,
            start=start,
            end=end,
            adjust=adjust,  # type: ignore[arg-type]
            include_quotes=not skip_quotes,
            force_refresh=force_refresh,
        )
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_data_health_report(report)
    if json_output:
        saved = write_data_health_json(report, json_output)
        console.print(f"[green]✓[/green] JSON 报告已写入：{saved}")
    if markdown_output:
        saved = write_data_health_markdown(report, markdown_output)
        console.print(f"[green]✓[/green] Markdown 报告已写入：{saved}")
    if report.has_failures:
        raise typer.Exit(1)


@app.command(name="paper-check")
def paper_check(
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s", help="验收标的，可重复指定"),
    periods: list[str] = typer.Option([], "--period", "-p", help="检查 K 线周期；默认全检查"),
    skip_bars: bool = typer.Option(False, "--skip-bars", help="跳过 K 线周期检查"),
    run_once: bool = typer.Option(
        False,
        "--run-once",
        help="显式跑一次 paper tick，会写入模拟盘验收 run 和 tick 日志",
    ),
    initial_cash: float = typer.Option(100_000.0, help="--run-once 使用的初始资金"),
    json_output: Path | None = typer.Option(None, "--json-output", help="写入 JSON 验收报告"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="写入 Markdown 验收报告"),
) -> None:
    """交易日模拟实盘验收：行情、K 线、模拟盘健康、绩效、AI 预算。"""
    from qtrade.persistence import init_db
    from qtrade.runtime.paper_check import (
        run_paper_check,
        write_report_json,
        write_report_markdown,
    )

    init_db()
    report = run_paper_check(
        symbols=symbols,
        periods=periods or None,
        include_bars=not skip_bars,
        run_once=run_once,
        initial_cash=initial_cash,
    )
    _print_paper_check_report(report)
    if json_output:
        saved = write_report_json(report, json_output)
        console.print(f"[green]✓[/green] JSON 报告已写入：{saved}")
    if markdown_output:
        saved = write_report_markdown(report, markdown_output)
        console.print(f"[green]✓[/green] Markdown 报告已写入：{saved}")
    if report.has_failures:
        raise typer.Exit(1)


@app.command(name="net-check")
def net_check(
    symbol: str = typer.Option("000001", "--symbol", "-s", help="诊断标的"),
    periods: list[str] = typer.Option(
        ["daily", "5min"],
        "--period",
        "-p",
        help="诊断 EastMoney K 线周期，可重复指定",
    ),
    timeout_s: float = typer.Option(5.0, "--timeout", help="单个 HTTP 请求超时秒数"),
    json_output: Path | None = typer.Option(None, "--json-output", help="写入 JSON 诊断报告"),
) -> None:
    """行情网络诊断；不写数据库、不写行情缓存。"""
    from qtrade.data.network_check import run_network_check, write_network_check_json

    try:
        report = run_network_check(symbol=symbol, periods=periods, timeout_s=timeout_s)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_network_check_report(report)
    if json_output:
        saved = write_network_check_json(report, json_output)
        console.print(f"[green]✓[/green] JSON 报告已写入：{saved}")
    if report.failures:
        raise typer.Exit(1)


@app.command(name="doctor")
def doctor(
    symbol: str = typer.Option("000001", "--symbol", "-s", help="验收标的"),
    periods: list[str] = typer.Option(
        ["daily", "5min", "15min"],
        "--period",
        "-p",
        help="验收 K 线周期，可重复指定",
    ),
    start: str | None = typer.Option(None, help="行情/K线检查起始日 YYYY-MM-DD；默认由 data-check 决定"),
    end: str | None = typer.Option(None, help="行情/K线检查终止日 YYYY-MM-DD；默认今天"),
    timeout_s: float = typer.Option(5.0, "--timeout", help="网络诊断单请求超时秒数"),
    skip_smoke: bool = typer.Option(False, "--skip-smoke", help="跳过临时隔离 PaperBroker smoke"),
    json_output: Path | None = typer.Option(None, "--json-output", help="写入 JSON doctor 报告"),
    verbose_logs: bool = typer.Option(False, "--verbose-logs", help="显示底层 qtrade 运行日志"),
) -> None:
    """一键基础验收：网络、行情、K线、数据库、UI、模拟盘链路。"""
    from qtrade.runtime.doctor import run_doctor, write_doctor_json

    try:
        with _quiet_qtrade_logs(enabled=not verbose_logs):
            report = run_doctor(
                symbol=symbol,
                periods=periods,
                timeout_s=timeout_s,
                start=start,
                end=end,
                run_smoke=not skip_smoke,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_doctor_report(report)
    if json_output:
        saved = write_doctor_json(report, json_output)
        console.print(f"[green]✓[/green] JSON 报告已写入：{saved}")
    if report.failures:
        raise typer.Exit(1)


def _print_data_health_report(report) -> None:
    status_style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    table = Table(title=f"行情源健康检查 | {','.join(report.symbols) or '-'}")
    table.add_column("状态", justify="center")
    table.add_column("检查项", style="cyan")
    table.add_column("结果")
    for item in report.items:
        style = status_style.get(item.status, "white")
        table.add_row(f"[{style}]{item.status}[/{style}]", item.name, item.detail)
    console.print(table)
    console.print(
        f"[dim]范围={report.start.isoformat()} -> {report.end.isoformat()} | "
        f"周期={','.join(report.periods)} | 复权={report.adjust} | "
        f"force_refresh={report.force_refresh}[/dim]"
    )
    summary = (
        f"PASS={len(report.passed)} | WARN={len(report.warnings)} | "
        f"FAIL={len(report.failures)} | overall={report.overall_status}"
    )
    if report.has_failures:
        console.print(f"[red]✗[/red] {summary}")
    elif report.has_warnings:
        console.print(f"[yellow]![/yellow] {summary}")
    else:
        console.print(f"[green]✓[/green] {summary}")


def _print_doctor_report(report) -> None:
    status_style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    table = Table(title=f"基础验收 Doctor | {report.symbol}")
    table.add_column("状态", justify="center")
    table.add_column("检查项", style="cyan")
    table.add_column("结果")
    for item in report.items:
        style = status_style.get(item.status, "white")
        table.add_row(f"[{style}]{item.status}[/{style}]", item.name, item.detail)
    console.print(table)
    summary = (
        f"PASS={len(report.passed)} | WARN={len(report.warnings)} | "
        f"FAIL={len(report.failures)} | overall={report.overall_status}"
    )
    if report.failures:
        console.print(f"[red]✗[/red] {summary}")
    elif report.warnings:
        console.print(f"[yellow]![/yellow] {summary}")
    else:
        console.print(f"[green]✓[/green] {summary}")


def _print_network_check_report(report) -> None:
    status_style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    table = Table(title=f"行情网络诊断 | {report.symbol}")
    table.add_column("状态", justify="center")
    table.add_column("检查项", style="cyan")
    table.add_column("分类")
    table.add_column("耗时", justify="right")
    table.add_column("结果")
    for item in report.items:
        style = status_style.get(item.status, "white")
        latency = f"{item.latency_ms:.0f}ms" if item.latency_ms is not None else "-"
        table.add_row(
            f"[{style}]{item.status}[/{style}]",
            item.name,
            item.category,
            latency,
            item.detail,
        )
    console.print(table)
    summary = (
        f"PASS={len(report.passed)} | WARN={len(report.warnings)} | "
        f"FAIL={len(report.failures)} | overall={report.overall_status}"
    )
    if report.failures:
        console.print(f"[red]✗[/red] {summary}")
    elif report.warnings:
        console.print(f"[yellow]![/yellow] {summary}")
    else:
        console.print(f"[green]✓[/green] {summary}")


@app.command(name="llm-check")
def llm_check(
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s", help="LLM 模拟盘标的，可重复指定"),
    prompt_version: str = typer.Option("v2", help="LLM prompt 版本（v1/v2）"),
    debate_rounds: int = typer.Option(2, help="LLM 辩论轮数"),
    threshold: float = typer.Option(0.45, help="LLM 决策阈值"),
    decision_interval_minutes: int = typer.Option(15, help="每个标的策略决策最小间隔"),
    position_pct: float = typer.Option(0.2, help="目标仓位比例"),
    max_position_pct: float = typer.Option(0.3, help="目标仓位硬裁剪上限"),
) -> None:
    """LLM 模拟盘启动前预检：不调用 LLM、不下单。"""
    from qtrade.runtime.llm_preflight import run_llm_paper_preflight

    report = run_llm_paper_preflight(
        symbols=symbols,
        prompt_version=prompt_version,
        debate_rounds=debate_rounds,
        threshold=threshold,
        decision_interval_minutes=decision_interval_minutes,
        position_pct=position_pct,
        max_position_pct=max_position_pct,
    )
    _print_preflight_report(report, title="LLM 模拟盘启动前检查")
    if report.has_failures:
        raise typer.Exit(1)


@app.command(name="paper-smoke")
def paper_smoke(
    symbol: str = typer.Option("000001", "--symbol", "-s", help="自检标的"),
    initial_cash: float = typer.Option(100_000.0, help="自检初始资金"),
    write_db: bool = typer.Option(
        False,
        "--write-db",
        help="写入当前项目数据库；默认使用临时隔离数据库",
    ),
) -> None:
    """非交易时段模拟实盘链路自检：不拉真实行情、不调用 LLM。"""
    if write_db:
        from qtrade.runtime.paper_smoke import run_paper_smoke_check

        report = run_paper_smoke_check(
            symbol=symbol,
            initial_cash=initial_cash,
            isolated=False,
        )
    else:
        with _isolated_runtime_paths() as _tmp:
            from qtrade.runtime.paper_smoke import run_paper_smoke_check

            report = run_paper_smoke_check(
                symbol=symbol,
                initial_cash=initial_cash,
                isolated=True,
            )
    _print_smoke_report(report)
    if report.has_failures:
        raise typer.Exit(1)


def _print_paper_check_report(report) -> None:
    status_style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    table = Table(title=f"模拟实盘验收 | {','.join(report.symbols) or '-'}")
    table.add_column("状态", justify="center")
    table.add_column("检查项", style="cyan")
    table.add_column("结果")
    for item in report.items:
        style = status_style.get(item.status, "white")
        table.add_row(f"[{style}]{item.status}[/{style}]", item.name, item.detail)
    console.print(table)

    summary = (
        f"PASS={len(report.passed)} | WARN={len(report.warnings)} | "
        f"FAIL={len(report.failures)} | overall={report.overall_status}"
    )
    if report.has_failures:
        console.print(f"[red]✗[/red] {summary}")
    elif report.has_warnings:
        console.print(f"[yellow]![/yellow] {summary}")
    else:
        console.print(f"[green]✓[/green] {summary}")


def _print_smoke_report(report) -> None:
    status_style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    scope = "临时隔离库" if report.isolated else "当前项目库"
    table = Table(title=f"模拟实盘链路自检 | {report.symbol} | {scope}")
    table.add_column("状态", justify="center")
    table.add_column("检查项", style="cyan")
    table.add_column("结果")
    for item in report.items:
        style = status_style.get(item.status, "white")
        table.add_row(f"[{style}]{item.status}[/{style}]", item.name, item.detail)
    console.print(table)
    console.print(f"[dim]db={report.db_path}[/dim]")
    summary = (
        f"PASS={len(report.passed)} | WARN={len(report.warnings)} | "
        f"FAIL={len(report.failures)} | overall={report.overall_status}"
    )
    if report.has_failures:
        console.print(f"[red]✗[/red] {summary}")
    elif report.warnings:
        console.print(f"[yellow]![/yellow] {summary}")
    else:
        console.print(f"[green]✓[/green] {summary}")


@contextmanager
def _isolated_runtime_paths():
    keys = {
        "QTRADE_CONFIG_DIR",
        "QTRADE_PATHS__DATA_DIR",
        "QTRADE_PATHS__LOG_DIR",
        "QTRADE_PATHS__CACHE_DIR",
        "QTRADE_PATHS__DB_PATH",
    }
    old_env = {key: os.environ.get(key) for key in keys}
    with tempfile.TemporaryDirectory(prefix="qtrade-paper-smoke-") as tmp:
        base = Path(tmp)
        data_dir = base / "data"
        log_dir = base / "logs"
        cache_dir = data_dir / "cache"
        cfg_dir = base / "cfg"
        for path in (data_dir, log_dir, cache_dir, cfg_dir):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["QTRADE_CONFIG_DIR"] = str(cfg_dir)
        os.environ["QTRADE_PATHS__DATA_DIR"] = str(data_dir)
        os.environ["QTRADE_PATHS__LOG_DIR"] = str(log_dir)
        os.environ["QTRADE_PATHS__CACHE_DIR"] = str(cache_dir)
        os.environ["QTRADE_PATHS__DB_PATH"] = str(data_dir / "qtrade.sqlite")
        _reconfigure_runtime_paths()
        try:
            yield base
        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
            _reconfigure_runtime_paths()


def _reconfigure_runtime_paths() -> None:
    from qtrade.core import config as cfg

    cfg.get_settings.cache_clear()
    try:
        from qtrade.persistence import db as dbmod
    except Exception:
        return
    dbmod.engine.dispose()
    dbmod.engine = dbmod._build_engine()
    dbmod.SessionLocal.configure(bind=dbmod.engine)


def _print_preflight_report(report, *, title: str) -> None:
    status_style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    table = Table(title=title)
    table.add_column("状态", justify="center")
    table.add_column("检查项", style="cyan")
    table.add_column("结果")
    for item in report.items:
        style = status_style.get(item.status, "white")
        table.add_row(f"[{style}]{item.status}[/{style}]", item.name, item.detail)
    console.print(table)
    summary = (
        f"PASS={len(report.passed)} | WARN={len(report.warnings)} | "
        f"FAIL={len(report.failures)} | overall={report.overall_status}"
    )
    if report.has_failures:
        console.print(f"[red]✗[/red] {summary}")
    elif report.warnings:
        console.print(f"[yellow]![/yellow] {summary}")
    else:
        console.print(f"[green]✓[/green] {summary}")


@app.command()
def backtest(
    strategy: str = typer.Option("ma_cross", help="策略类型 ma_cross | macd | bollinger | momentum | turtle | llm"),
    symbols: list[str] = typer.Option(["600519"], "--symbol", "-s"),
    start: str = typer.Option("2024-01-01"),
    end: str | None = typer.Option(None),
    initial_cash: float = typer.Option(100_000.0),
) -> None:
    """跑回测并打印关键指标。"""
    from qtrade.backtest.comparison import build_backtest_strategy
    from qtrade.backtest.engine import BacktestEngine
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    try:
        strat = build_backtest_strategy(strategy, symbols=syms, allow_llm=True)
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    engine = BacktestEngine(
        strategy=strat,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end) if end else date.today(),
        initial_cash=initial_cash,
    )
    try:
        result = engine.run()
    except DataError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_backtest(result)


@app.command(name="backtest-compare")
def backtest_compare(
    strategies: list[str] = typer.Option(
        ["ma_cross", "macd", "bollinger", "momentum", "turtle"],
        "--strategy",
        "-S",
        help="参与对比的技术策略，可重复指定",
    ),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    start: str = typer.Option("2024-01-01"),
    end: str | None = typer.Option(None),
    initial_cash: float = typer.Option(100_000.0),
    slippage_bps: float = typer.Option(1.0, help="回测滑点，单位 bp"),
    keep_trades: bool = typer.Option(False, "--keep-trades", help="资源保护：保留每个批量结果的完整成交明细"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast", help="单个策略失败时记录错误并继续"),
    checkpoint_csv: Path | None = typer.Option(None, "--checkpoint-csv", help="增量写入已完成结果 CSV，中断时保留"),
    resume_from_checkpoint: bool = typer.Option(
        False,
        "--resume-from-checkpoint",
        help="读取 --checkpoint-csv 中已成功完成的结果，跳过重复任务并继续追加",
    ),
    csv_output: Path | None = typer.Option(None, "--csv-output", help="导出 CSV 汇总"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="导出 Markdown 报告"),
) -> None:
    """多策略批量回测对比。默认只跑本地技术策略，不跑 LLM。"""
    from qtrade.backtest.comparison import (
        BacktestComparison,
        backtest_task_key,
        run_backtest_comparison,
        write_comparison_csv,
        write_comparison_markdown,
    )
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end) if end else date.today()
    try:
        candidate_keys = {
            backtest_task_key(strategy_name=name, symbols=syms, start=start_d, end=end_d) for name in strategies
        }
        resumed_rows, skip_task_keys = _load_backtest_checkpoint_resume(
            checkpoint_csv,
            candidate_keys=candidate_keys,
            enabled=resume_from_checkpoint,
        )
        with (
            _progress_reporter("多策略回测", total=len(strategies)) as progress_callback,
            _checkpoint_csv_writer(checkpoint_csv, append=resume_from_checkpoint) as checkpoint_callback,
        ):
            comparison = run_backtest_comparison(
                strategies=strategies,
                symbols=syms,
                start=start_d,
                end=end_d,
                initial_cash=initial_cash,
                slippage_bps=slippage_bps,
                keep_trades=keep_trades,
                progress_callback=_chain_progress_callbacks(progress_callback, checkpoint_callback),
                continue_on_error=continue_on_error,
                skip_task_keys=skip_task_keys,
            )
        if resumed_rows:
            comparison = BacktestComparison(
                rows=[*resumed_rows, *comparison.rows],
                failures=comparison.failures,
                generated_at=comparison.generated_at,
            )
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_resume_status(resume_from_checkpoint=resume_from_checkpoint, resumed_count=len(resumed_rows))
    _print_backtest_comparison(comparison)
    _print_backtest_failures(comparison)
    if checkpoint_csv:
        console.print(f"[green]✓[/green] Checkpoint CSV 已写入：{checkpoint_csv.expanduser().resolve()}")
    if csv_output:
        saved = write_comparison_csv(comparison, csv_output)
        console.print(f"[green]✓[/green] CSV 已写入：{saved}")
    if markdown_output:
        saved = write_comparison_markdown(comparison, markdown_output)
        console.print(f"[green]✓[/green] Markdown 已写入：{saved}")


@app.command(name="backtest-grid")
def backtest_grid(
    strategy: str = typer.Option("ma_cross", help="技术策略：ma_cross | macd | bollinger | momentum | turtle"),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    start: str = typer.Option("2024-01-01"),
    end: str | None = typer.Option(None),
    param_items: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="参数网格，格式 key=v1,v2；可重复指定。例如 -p fast_window=5,10 -p slow_window=20,60",
    ),
    initial_cash: float = typer.Option(100_000.0),
    slippage_bps: float = typer.Option(1.0, help="回测滑点，单位 bp"),
    sort_by: str = typer.Option("sharpe", help="排序指标：sharpe/sortino/calmar/total_return/max_drawdown 等"),
    descending: bool = typer.Option(True, "--descending/--ascending", help="排序方向"),
    top: int = typer.Option(20, help="只显示/导出前 N 条；0 表示全部"),
    max_symbols: int = typer.Option(50, help="资源保护：最多允许多少个标的"),
    max_param_combinations: int = typer.Option(500, help="资源保护：最多允许多少个参数组合"),
    keep_trades: bool = typer.Option(False, "--keep-trades", help="资源保护：保留每个批量结果的完整成交明细"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast", help="单个参数组合失败时记录错误并继续"),
    checkpoint_csv: Path | None = typer.Option(None, "--checkpoint-csv", help="增量写入已完成结果 CSV，中断时保留"),
    resume_from_checkpoint: bool = typer.Option(
        False,
        "--resume-from-checkpoint",
        help="读取 --checkpoint-csv 中已成功完成的结果，跳过重复参数并继续追加",
    ),
    csv_output: Path | None = typer.Option(None, "--csv-output", help="导出 CSV 汇总"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="导出 Markdown 报告"),
    experiment_log: Path | None = typer.Option(None, "--experiment-log", help="追加写入 JSONL 实验记录"),
) -> None:
    """单策略参数网格回测。默认只跑技术策略，不跑 LLM。"""
    from qtrade.backtest.comparison import (
        BacktestComparison,
        backtest_task_key,
        expand_param_grid,
        limit_backtest_comparison,
        run_backtest_param_grid,
        sort_backtest_comparison,
        write_comparison_csv,
        write_comparison_markdown,
    )
    from qtrade.backtest.experiment_log import append_backtest_grid_experiment
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    try:
        param_grid = _parse_param_grid(param_items) if param_items else _default_param_grid(strategy)
        param_count = _grid_size(param_grid)
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end) if end else date.today()
        candidate_keys = {
            backtest_task_key(strategy_name=strategy, symbols=syms, start=start_d, end=end_d, params=params)
            for params in expand_param_grid(param_grid)
        }
        resumed_rows, skip_task_keys = _load_backtest_checkpoint_resume(
            checkpoint_csv,
            candidate_keys=candidate_keys,
            enabled=resume_from_checkpoint,
        )
        _guard_symbol_count(syms, max_symbols=max_symbols)
        _guard_param_combinations(param_count, max_param_combinations=max_param_combinations)
        _print_workload_estimate(
            "参数网格回测",
            symbols=len(syms),
            param_combinations=param_count,
            backtest_runs=param_count,
        )
        with (
            _progress_reporter("参数网格回测", total=param_count) as progress_callback,
            _checkpoint_csv_writer(checkpoint_csv, append=resume_from_checkpoint) as checkpoint_callback,
        ):
            comparison = run_backtest_param_grid(
                strategy_name=strategy,
                symbols=syms,
                start=start_d,
                end=end_d,
                param_grid=param_grid,
                initial_cash=initial_cash,
                slippage_bps=slippage_bps,
                keep_trades=keep_trades,
                progress_callback=_chain_progress_callbacks(progress_callback, checkpoint_callback),
                continue_on_error=continue_on_error,
                skip_task_keys=skip_task_keys,
            )
        if resumed_rows:
            comparison = BacktestComparison(
                rows=[*resumed_rows, *comparison.rows],
                failures=comparison.failures,
                generated_at=comparison.generated_at,
            )
        comparison = sort_backtest_comparison(comparison, sort_by=sort_by, descending=descending)
        sensitivity_comparison = comparison
        comparison = limit_backtest_comparison(comparison, top=top)
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_resume_status(resume_from_checkpoint=resume_from_checkpoint, resumed_count=len(resumed_rows))
    _print_backtest_comparison(comparison)
    _print_parameter_sensitivity(sensitivity_comparison, sort_by=sort_by, descending=descending)
    _print_backtest_failures(comparison)
    console.print(f"[dim]参数网格：{param_grid} | 排序：{sort_by} {'desc' if descending else 'asc'}[/dim]")
    saved_csv = None
    saved_markdown = None
    if checkpoint_csv:
        console.print(f"[green]✓[/green] Checkpoint CSV 已写入：{checkpoint_csv.expanduser().resolve()}")
    if csv_output:
        saved_csv = write_comparison_csv(comparison, csv_output)
        console.print(f"[green]✓[/green] CSV 已写入：{saved_csv}")
    if markdown_output:
        saved_markdown = write_comparison_markdown(
            comparison,
            markdown_output,
            sensitivity_source=sensitivity_comparison,
            sensitivity_sort_by=sort_by,
        )
        console.print(f"[green]✓[/green] Markdown 已写入：{saved_markdown}")
    if experiment_log:
        saved_log = append_backtest_grid_experiment(
            experiment_log,
            kind="backtest_grid",
            strategy_name=strategy,
            symbols=syms,
            start=start_d,
            end=end_d,
            param_grid=param_grid,
            comparison=sensitivity_comparison,
            sort_by=sort_by,
            descending=descending,
            top=top,
            csv_output=saved_csv,
            markdown_output=saved_markdown,
            checkpoint_csv=checkpoint_csv,
        )
        console.print(f"[green]✓[/green] 实验记录已追加：{saved_log}")


@app.command(name="factor-mine")
def factor_mine(
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    start: str = typer.Option("2022-01-01"),
    end: str | None = typer.Option(None),
    windows: list[int] = typer.Option([5, 20, 60], "--window", "-w", help="价量因子窗口，可重复指定"),
    horizon_days: int = typer.Option(5, help="预测目标：未来 N 日收益"),
    train_ratio: float = typer.Option(0.7, help="样本内比例，默认 70%"),
    include_interactions: bool = typer.Option(False, "--include-interactions", help="加入基础因子两两乘积"),
    max_interactions: int = typer.Option(50, help="最多加入多少个交互因子"),
    factor_sort_by: str = typer.Option(
        "test_rank_ic_abs",
        "--factor-sort-by",
        help="排序指标：test_rank_ic_abs/test_ic_abs/test_spread_return_abs 等",
    ),
    top: int = typer.Option(30, help="只显示/导出前 N 个因子；0 表示全部"),
    csv_output: Path | None = typer.Option(None, "--csv-output", help="导出 CSV 因子报告"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="导出 Markdown 因子报告"),
) -> None:
    """基础因子挖掘：价量因子 + 样本内/样本外 IC 检验，不下单。"""
    from qtrade.backtest.factors import (
        limit_factor_report,
        run_factor_mining,
        sort_factor_report,
        write_factor_report_csv,
        write_factor_report_markdown,
    )
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    try:
        report = run_factor_mining(
            symbols=syms,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end) if end else date.today(),
            windows=windows,
            horizon_days=horizon_days,
            train_ratio=train_ratio,
            include_interactions=include_interactions,
            max_interactions=max_interactions,
        )
        report = sort_factor_report(report, sort_by=factor_sort_by)
        report = limit_factor_report(report, top=top)
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_factor_report(report)
    if csv_output:
        saved = write_factor_report_csv(report, csv_output)
        console.print(f"[green]✓[/green] CSV 已写入：{saved}")
    if markdown_output:
        saved = write_factor_report_markdown(report, markdown_output)
        console.print(f"[green]✓[/green] Markdown 已写入：{saved}")


@app.command(name="factor-layer")
def factor_layer(
    factor_name: str = typer.Option(..., "--factor", "-f", help="要验证的因子名，如 ret_20 或 ma_gap_60"),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    start: str = typer.Option("2022-01-01"),
    end: str | None = typer.Option(None),
    windows: list[int] = typer.Option([5, 20, 60], "--window", "-w", help="价量因子窗口，可重复指定"),
    horizon_days: int = typer.Option(5, help="预测目标：未来 N 日收益"),
    train_ratio: float = typer.Option(0.7, help="样本内比例，默认 70%"),
    groups: int = typer.Option(5, help="分层数量，默认 5 组"),
    include_interactions: bool = typer.Option(False, "--include-interactions", help="允许验证交互因子"),
    max_interactions: int = typer.Option(50, help="最多生成多少个交互因子"),
    csv_output: Path | None = typer.Option(None, "--csv-output", help="导出 CSV 分层报告"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="导出 Markdown 分层报告"),
) -> None:
    """因子分层验证：检查 Top/Bottom 未来收益、单调性和换手率，不下单。"""
    from qtrade.backtest.factors import (
        run_factor_layering,
        write_factor_layer_report_csv,
        write_factor_layer_report_markdown,
    )
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    try:
        report = run_factor_layering(
            symbols=syms,
            factor_name=factor_name,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end) if end else date.today(),
            windows=windows,
            horizon_days=horizon_days,
            train_ratio=train_ratio,
            groups=groups,
            include_interactions=include_interactions,
            max_interactions=max_interactions,
        )
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_factor_layer_report(report)
    if csv_output:
        saved = write_factor_layer_report_csv(report, csv_output)
        console.print(f"[green]✓[/green] CSV 已写入：{saved}")
    if markdown_output:
        saved = write_factor_layer_report_markdown(report, markdown_output)
        console.print(f"[green]✓[/green] Markdown 已写入：{saved}")


@app.command(name="factor-backtest")
def factor_backtest(
    factor_name: str = typer.Option(..., "--factor", "-f", help="要回测的因子名，如 ret_20 或 ma_gap_60"),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    start: str = typer.Option("2022-01-01"),
    end: str | None = typer.Option(None),
    windows: list[int] = typer.Option([5, 20, 60], "--window", "-w", help="价量因子窗口，可重复指定"),
    lookback: int = typer.Option(120, help="历史分位数窗口"),
    entry_quantile: float = typer.Option(0.8, help="入场分位阈值，direction=high 时进入高分位"),
    exit_quantile: float = typer.Option(0.5, help="退出分位阈值"),
    direction: str = typer.Option("high", help="high 表示因子越高越看多；low 表示因子越低越看多"),
    position_pct: float = typer.Option(0.3, help="每次买入目标仓位比例"),
    include_interactions: bool = typer.Option(False, "--include-interactions", help="允许使用交互因子"),
    max_interactions: int = typer.Option(50, help="最多生成多少个交互因子"),
    initial_cash: float = typer.Option(100_000.0),
    slippage_bps: float = typer.Option(1.0, help="回测滑点，单位 bp"),
) -> None:
    """把候选因子转成历史分位阈值策略并回测。"""
    from qtrade.backtest.engine import BacktestEngine
    from qtrade.data.schema import normalize_symbol
    from qtrade.strategies.factor import FactorThresholdStrategy

    syms = [normalize_symbol(s) for s in symbols]
    try:
        strategy = FactorThresholdStrategy(
            universe=syms,
            params={
                "factor_name": factor_name,
                "windows": windows,
                "lookback": lookback,
                "entry_quantile": entry_quantile,
                "exit_quantile": exit_quantile,
                "direction": direction,
                "position_pct": position_pct,
                "include_interactions": include_interactions,
                "max_interactions": max_interactions,
            },
        )
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        result = BacktestEngine(
            strategy=strategy,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end) if end else date.today(),
            initial_cash=initial_cash,
            slippage_bps=slippage_bps,
        ).run()
    except DataError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_backtest(result)
    console.print(
        f"[dim]factor={factor_name} | direction={direction} | "
        f"entry={entry_quantile:.0%} | exit={exit_quantile:.0%} | "
        f"lookback={lookback} | windows={windows}[/dim]"
    )


@app.command(name="factor-grid")
def factor_grid(
    factor_name: str = typer.Option(..., "--factor", "-f", help="要回测的因子名，如 ret_20 或 ma_gap_60"),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    start: str = typer.Option("2022-01-01"),
    end: str | None = typer.Option(None),
    windows: list[int] = typer.Option([5, 20, 60], "--window", "-w", help="价量因子窗口，可重复指定"),
    lookbacks: list[int] = typer.Option([60, 120], "--lookback", help="历史分位数窗口，可重复指定"),
    entry_quantiles: list[float] = typer.Option([0.7, 0.8], "--entry-quantile", help="入场分位阈值，可重复指定"),
    exit_quantiles: list[float] = typer.Option([0.4, 0.5], "--exit-quantile", help="退出分位阈值，可重复指定"),
    directions: list[str] = typer.Option(["high"], "--direction", help="high 或 low，可重复指定"),
    position_pcts: list[float] = typer.Option([0.2, 0.3], "--position-pct", help="目标仓位比例，可重复指定"),
    include_interactions: bool = typer.Option(False, "--include-interactions", help="允许使用交互因子"),
    max_interactions: int = typer.Option(50, help="最多生成多少个交互因子"),
    initial_cash: float = typer.Option(100_000.0),
    slippage_bps: float = typer.Option(1.0, help="回测滑点，单位 bp"),
    sort_by: str = typer.Option("sharpe", help="排序指标：sharpe/sortino/calmar/total_return/max_drawdown 等"),
    descending: bool = typer.Option(True, "--descending/--ascending", help="排序方向"),
    top: int = typer.Option(20, help="只显示/导出前 N 条；0 表示全部"),
    max_symbols: int = typer.Option(50, help="资源保护：最多允许多少个标的"),
    max_param_combinations: int = typer.Option(500, help="资源保护：最多允许多少个参数组合"),
    keep_trades: bool = typer.Option(False, "--keep-trades", help="资源保护：保留每个批量结果的完整成交明细"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast", help="单个参数组合失败时记录错误并继续"),
    checkpoint_csv: Path | None = typer.Option(None, "--checkpoint-csv", help="增量写入已完成结果 CSV，中断时保留"),
    resume_from_checkpoint: bool = typer.Option(
        False,
        "--resume-from-checkpoint",
        help="读取 --checkpoint-csv 中已成功完成的结果，跳过重复参数并继续追加",
    ),
    csv_output: Path | None = typer.Option(None, "--csv-output", help="导出 CSV 汇总"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="导出 Markdown 报告"),
    experiment_log: Path | None = typer.Option(None, "--experiment-log", help="追加写入 JSONL 实验记录"),
) -> None:
    """因子策略参数网格回测。"""
    from qtrade.backtest.comparison import (
        BacktestComparison,
        backtest_task_key,
        expand_param_grid,
        limit_backtest_comparison,
        run_backtest_param_grid,
        sort_backtest_comparison,
        write_comparison_csv,
        write_comparison_markdown,
    )
    from qtrade.backtest.experiment_log import append_backtest_grid_experiment
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    try:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end) if end else date.today()
        param_grid = _build_factor_strategy_param_grid(
            factor_name=factor_name,
            windows=windows,
            lookbacks=lookbacks,
            entry_quantiles=entry_quantiles,
            exit_quantiles=exit_quantiles,
            directions=directions,
            position_pcts=position_pcts,
            include_interactions=include_interactions,
            max_interactions=max_interactions,
        )
        param_count = _grid_size(param_grid)
        candidate_keys = {
            backtest_task_key(strategy_name="factor", symbols=syms, start=start_d, end=end_d, params=params)
            for params in expand_param_grid(param_grid)
        }
        resumed_rows, skip_task_keys = _load_backtest_checkpoint_resume(
            checkpoint_csv,
            candidate_keys=candidate_keys,
            enabled=resume_from_checkpoint,
        )
        _guard_symbol_count(syms, max_symbols=max_symbols)
        _guard_param_combinations(param_count, max_param_combinations=max_param_combinations)
        _print_workload_estimate(
            "因子参数网格",
            symbols=len(syms),
            param_combinations=param_count,
            backtest_runs=param_count,
        )
        with (
            _progress_reporter("因子参数网格", total=param_count) as progress_callback,
            _checkpoint_csv_writer(checkpoint_csv, append=resume_from_checkpoint) as checkpoint_callback,
        ):
            comparison = run_backtest_param_grid(
                strategy_name="factor",
                symbols=syms,
                start=start_d,
                end=end_d,
                param_grid=param_grid,
                initial_cash=initial_cash,
                slippage_bps=slippage_bps,
                keep_trades=keep_trades,
                progress_callback=_chain_progress_callbacks(progress_callback, checkpoint_callback),
                continue_on_error=continue_on_error,
                skip_task_keys=skip_task_keys,
            )
        if resumed_rows:
            comparison = BacktestComparison(
                rows=[*resumed_rows, *comparison.rows],
                failures=comparison.failures,
                generated_at=comparison.generated_at,
            )
        comparison = sort_backtest_comparison(comparison, sort_by=sort_by, descending=descending)
        sensitivity_comparison = comparison
        comparison = limit_backtest_comparison(comparison, top=top)
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_resume_status(resume_from_checkpoint=resume_from_checkpoint, resumed_count=len(resumed_rows))
    _print_backtest_comparison(comparison)
    _print_parameter_sensitivity(sensitivity_comparison, sort_by=sort_by, descending=descending)
    _print_backtest_failures(comparison)
    console.print(
        f"[dim]factor={factor_name} | 参数组合={_grid_size(param_grid)} | "
        f"排序：{sort_by} {'desc' if descending else 'asc'}[/dim]"
    )
    saved_csv = None
    saved_markdown = None
    if checkpoint_csv:
        console.print(f"[green]✓[/green] Checkpoint CSV 已写入：{checkpoint_csv.expanduser().resolve()}")
    if csv_output:
        saved_csv = write_comparison_csv(comparison, csv_output)
        console.print(f"[green]✓[/green] CSV 已写入：{saved_csv}")
    if markdown_output:
        saved_markdown = write_comparison_markdown(
            comparison,
            markdown_output,
            sensitivity_source=sensitivity_comparison,
            sensitivity_sort_by=sort_by,
        )
        console.print(f"[green]✓[/green] Markdown 已写入：{saved_markdown}")
    if experiment_log:
        saved_log = append_backtest_grid_experiment(
            experiment_log,
            kind="factor_grid",
            strategy_name="factor",
            symbols=syms,
            start=start_d,
            end=end_d,
            param_grid=param_grid,
            comparison=sensitivity_comparison,
            sort_by=sort_by,
            descending=descending,
            top=top,
            csv_output=saved_csv,
            markdown_output=saved_markdown,
            checkpoint_csv=checkpoint_csv,
        )
        console.print(f"[green]✓[/green] 实验记录已追加：{saved_log}")


@app.command(name="factor-walk-forward")
def factor_walk_forward(
    factor_name: str = typer.Option(..., "--factor", "-f", help="要验证的因子名，如 ret_20 或 ma_gap_60"),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    start: str = typer.Option("2022-01-01"),
    end: str | None = typer.Option(None),
    windows: list[int] = typer.Option([5, 20, 60], "--window", "-w", help="价量因子窗口，可重复指定"),
    lookbacks: list[int] = typer.Option([60, 120], "--lookback", help="历史分位数窗口，可重复指定"),
    entry_quantiles: list[float] = typer.Option([0.7, 0.8], "--entry-quantile", help="入场分位阈值，可重复指定"),
    exit_quantiles: list[float] = typer.Option([0.4, 0.5], "--exit-quantile", help="退出分位阈值，可重复指定"),
    directions: list[str] = typer.Option(["high"], "--direction", help="high 或 low，可重复指定"),
    position_pcts: list[float] = typer.Option([0.2, 0.3], "--position-pct", help="目标仓位比例，可重复指定"),
    include_interactions: bool = typer.Option(False, "--include-interactions", help="允许使用交互因子"),
    max_interactions: int = typer.Option(50, help="最多生成多少个交互因子"),
    train_days: int = typer.Option(365, help="每折训练期自然日长度"),
    test_days: int = typer.Option(90, help="每折验证期自然日长度"),
    step_days: int | None = typer.Option(None, help="每折滚动步长；默认等于 test-days"),
    initial_cash: float = typer.Option(100_000.0),
    slippage_bps: float = typer.Option(1.0, help="回测滑点，单位 bp"),
    sort_by: str = typer.Option("sharpe", help="训练期择优指标：sharpe/sortino/calmar/total_return 等"),
    descending: bool = typer.Option(True, "--descending/--ascending", help="训练期排序方向"),
    max_symbols: int = typer.Option(50, help="资源保护：最多允许多少个标的"),
    max_param_combinations: int = typer.Option(500, help="资源保护：最多允许多少个训练期参数组合"),
    max_backtest_runs: int = typer.Option(1000, help="资源保护：最多允许多少次内部回测"),
    keep_trades: bool = typer.Option(False, "--keep-trades", help="资源保护：保留每个批量结果的完整成交明细"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--fail-fast", help="单个内部回测失败时记录错误并继续"),
    checkpoint_csv: Path | None = typer.Option(None, "--checkpoint-csv", help="增量写入内部回测结果 CSV，中断时保留"),
    csv_output: Path | None = typer.Option(None, "--csv-output", help="导出 CSV walk-forward 报告"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="导出 Markdown walk-forward 报告"),
    experiment_log: Path | None = typer.Option(None, "--experiment-log", help="追加写入 JSONL 实验记录"),
) -> None:
    """因子策略 walk-forward 验证：每折训练择优，下一段验证，不下单。"""
    from qtrade.backtest.comparison import (
        run_walk_forward_param_grid,
        write_walk_forward_csv,
        write_walk_forward_markdown,
    )
    from qtrade.backtest.experiment_log import append_walk_forward_experiment
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    try:
        param_grid = _build_factor_strategy_param_grid(
            factor_name=factor_name,
            windows=windows,
            lookbacks=lookbacks,
            entry_quantiles=entry_quantiles,
            exit_quantiles=exit_quantiles,
            directions=directions,
            position_pcts=position_pcts,
            include_interactions=include_interactions,
            max_interactions=max_interactions,
        )
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end) if end else date.today()
        param_count = _grid_size(param_grid)
        fold_count = _estimate_walk_forward_folds(
            start=start_d,
            end=end_d,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
        )
        backtest_runs = fold_count * (param_count + 1)
        _guard_symbol_count(syms, max_symbols=max_symbols)
        _guard_param_combinations(param_count, max_param_combinations=max_param_combinations)
        _guard_backtest_runs(backtest_runs, max_backtest_runs=max_backtest_runs)
        _print_workload_estimate(
            "Walk-forward",
            symbols=len(syms),
            param_combinations=param_count,
            folds=fold_count,
            backtest_runs=backtest_runs,
        )
        with (
            _progress_reporter("Walk-forward", total=backtest_runs) as progress_callback,
            _checkpoint_csv_writer(checkpoint_csv) as checkpoint_callback,
        ):
            report = run_walk_forward_param_grid(
                strategy_name="factor",
                symbols=syms,
                start=start_d,
                end=end_d,
                param_grid=param_grid,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                sort_by=sort_by,
                descending=descending,
                initial_cash=initial_cash,
                slippage_bps=slippage_bps,
                keep_trades=keep_trades,
                progress_callback=_chain_progress_callbacks(progress_callback, checkpoint_callback),
                continue_on_error=continue_on_error,
            )
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_walk_forward_report(report)
    console.print(
        f"[dim]factor={factor_name} | 参数组合={_grid_size(param_grid)} | "
        f"训练择优：{sort_by} {'desc' if descending else 'asc'}[/dim]"
    )
    saved_csv = None
    saved_markdown = None
    if checkpoint_csv:
        console.print(f"[green]✓[/green] Checkpoint CSV 已写入：{checkpoint_csv.expanduser().resolve()}")
    if csv_output:
        saved_csv = write_walk_forward_csv(report, csv_output)
        console.print(f"[green]✓[/green] CSV 已写入：{saved_csv}")
    if markdown_output:
        saved_markdown = write_walk_forward_markdown(report, markdown_output)
        console.print(f"[green]✓[/green] Markdown 已写入：{saved_markdown}")
    if experiment_log:
        saved_log = append_walk_forward_experiment(
            experiment_log,
            kind="factor_walk_forward",
            strategy_name="factor",
            symbols=syms,
            start=start_d,
            end=end_d,
            param_grid=param_grid,
            report=report,
            sort_by=sort_by,
            descending=descending,
            csv_output=saved_csv,
            markdown_output=saved_markdown,
            checkpoint_csv=checkpoint_csv,
        )
        console.print(f"[green]✓[/green] 实验记录已追加：{saved_log}")


@app.command(name="experiment-leaderboard")
def experiment_leaderboard(
    experiment_logs: list[Path] = typer.Option(
        [],
        "--experiment-log",
        "-l",
        help="读取 JSONL 实验记录，可重复指定",
    ),
    record_types: list[str] = typer.Option(
        [],
        "--record-type",
        help="筛选记录类型：backtest_grid/factor_grid/factor_walk_forward，可重复指定",
    ),
    sort_by: str = typer.Option("sharpe", help="排序指标：sharpe/total_return/max_drawdown/win_rate 等"),
    descending: bool = typer.Option(True, "--descending/--ascending", help="排序方向"),
    top: int = typer.Option(20, help="展示前 N 个候选；0 表示全部"),
    csv_output: Path | None = typer.Option(None, "--csv-output", help="导出候选榜单 CSV"),
) -> None:
    """读取实验 JSONL，展示候选策略榜单。只读，不跑回测、不下单。"""
    from qtrade.backtest.experiment_log import (
        experiment_candidates_from_records,
        limit_experiment_candidates,
        read_experiment_records,
        sort_experiment_candidates,
        write_experiment_candidates_csv,
    )

    if not experiment_logs:
        raise typer.BadParameter("至少提供一个 --experiment-log")
    if top < 0:
        raise typer.BadParameter("top 不能为负数")
    try:
        records = read_experiment_records(experiment_logs)
        candidates = experiment_candidates_from_records(
            records,
            record_types=set(record_types) if record_types else None,
        )
        candidates = sort_experiment_candidates(candidates, sort_by=sort_by, descending=descending)
        display_candidates = limit_experiment_candidates(candidates, top=top)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_experiment_candidates(
        display_candidates,
        total_count=len(candidates),
        sort_by=sort_by,
        descending=descending,
    )
    if csv_output:
        saved = write_experiment_candidates_csv(display_candidates, csv_output)
        console.print(f"[green]✓[/green] CSV 已写入：{saved}")


@app.command(name="experiment-plan")
def experiment_plan(
    experiment_logs: list[Path] = typer.Option(
        [],
        "--experiment-log",
        "-l",
        help="读取 JSONL 实验记录，可重复指定",
    ),
    record_types: list[str] = typer.Option(
        [],
        "--record-type",
        help="筛选记录类型：backtest_grid/factor_grid/factor_walk_forward，可重复指定",
    ),
    sort_by: str = typer.Option("sharpe", help="候选排序指标：sharpe/total_return/max_drawdown/win_rate 等"),
    descending: bool = typer.Option(True, "--descending/--ascending", help="候选排序方向"),
    top: int = typer.Option(5, help="为前 N 个候选生成验证命令；0 表示全部"),
    validation_start: str | None = typer.Option(None, "--validation-start", help="覆盖下一轮验证起始日期"),
    validation_end: str | None = typer.Option(None, "--validation-end", help="覆盖下一轮验证结束日期"),
    train_days: int = typer.Option(365, help="生成 factor-walk-forward 命令时使用的训练期天数"),
    test_days: int = typer.Option(90, help="生成 factor-walk-forward 命令时使用的验证期天数"),
    step_days: int | None = typer.Option(None, help="生成 factor-walk-forward 命令时使用的滚动步长"),
    next_experiment_log: Path | None = typer.Option(None, "--next-experiment-log", help="追加到生成命令里的下一轮实验记录路径"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="导出 Markdown 验证计划"),
) -> None:
    """从候选榜单生成下一轮验证命令。只生成计划，不执行。"""
    from qtrade.backtest.experiment_log import (
        build_validation_commands,
        experiment_candidates_from_records,
        limit_experiment_candidates,
        read_experiment_records,
        sort_experiment_candidates,
        write_validation_plan_markdown,
    )

    if not experiment_logs:
        raise typer.BadParameter("至少提供一个 --experiment-log")
    if top < 0:
        raise typer.BadParameter("top 不能为负数")
    if train_days <= 0 or test_days <= 0:
        raise typer.BadParameter("train_days/test_days 必须大于 0")
    if step_days is not None and step_days <= 0:
        raise typer.BadParameter("step_days 必须大于 0")
    try:
        records = read_experiment_records(experiment_logs)
        candidates = experiment_candidates_from_records(
            records,
            record_types=set(record_types) if record_types else None,
        )
        candidates = sort_experiment_candidates(candidates, sort_by=sort_by, descending=descending)
        candidates = limit_experiment_candidates(candidates, top=top)
        commands = build_validation_commands(
            candidates,
            validation_start=validation_start,
            validation_end=validation_end,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            next_experiment_log=next_experiment_log,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_validation_commands(commands)
    if markdown_output:
        saved = write_validation_plan_markdown(commands, markdown_output)
        console.print(f"[green]✓[/green] Markdown 已写入：{saved}")


@app.command(name="experiment-run-plan")
def experiment_run_plan(
    plan_path: Path = typer.Argument(..., help="experiment-plan 生成的 Markdown 验证计划"),
    execute: bool = typer.Option(False, "--execute", help="实际执行计划；默认只做 dry-run 预览"),
    max_commands: int = typer.Option(5, help="最多执行/预览多少条命令；0 表示全部"),
    timeout_seconds: int = typer.Option(600, help="单条命令超时时间"),
    cwd: Path | None = typer.Option(None, "--cwd", help="执行命令的工作目录；默认当前目录"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="统一输出目录；写入 validation-run-summary.json"),
    json_output: Path | None = typer.Option(None, "--json-output", help="写入运行摘要 JSON；优先于 --output-dir"),
) -> None:
    """受控执行验证计划。默认 dry-run；只允许验证类 qtrade 命令。"""
    from qtrade.backtest.experiment_log import (
        read_validation_plan_commands,
        run_validation_plan_commands,
        write_plan_run_summary_json,
    )

    try:
        commands = read_validation_plan_commands(plan_path)
        results = run_validation_plan_commands(
            commands,
            execute=execute,
            max_commands=max_commands,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_plan_run_results(results, execute=execute, total_commands=len(commands))
    summary_path = json_output or (output_dir / "validation-run-summary.json" if output_dir else None)
    if summary_path:
        saved = write_plan_run_summary_json(
            summary_path,
            plan_path=plan_path,
            results=results,
            execute=execute,
            total_commands=len(commands),
            max_commands=max_commands,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
        )
        console.print(f"[green]✓[/green] JSON 摘要已写入：{saved}")
    if execute and any(result.status != "success" for result in results):
        raise typer.Exit(1)


@app.command(name="factor-portfolio")
def factor_portfolio(
    factor_name: str = typer.Option(..., "--factor", "-f", help="组合排序因子，如 ret_20 或 ma_gap_60"),
    symbols: list[str] = typer.Option(["000001", "600519", "300750"], "--symbol", "-s"),
    start: str = typer.Option("2022-01-01"),
    end: str | None = typer.Option(None),
    windows: list[int] = typer.Option([5, 20, 60], "--window", "-w", help="价量因子窗口，可重复指定"),
    direction: str = typer.Option("high", help="high 选因子值高的；low 选因子值低的"),
    rebalance_days: int = typer.Option(5, help="每 N 个交易样本日调仓"),
    select_count: int | None = typer.Option(None, "--select-count", help="每次固定选 Top N；设置后优先于 select-quantile"),
    select_quantile: float = typer.Option(0.2, "--select-quantile", help="每次选 Top 分位，默认 20%"),
    initial_equity: float = typer.Option(100_000.0),
    fee_bps: float = typer.Option(3.0, help="单边费率估算，单位 bp"),
    slippage_bps: float = typer.Option(1.0, help="滑点估算，单位 bp"),
    respect_tradeability: bool = typer.Option(True, "--respect-tradeability/--ignore-tradeability", help="过滤停牌、涨停买入、跌停卖出"),
    min_volume: float = typer.Option(1.0, help="volume <= 该值时按停牌/不可交易近似处理"),
    price_limit_pct: float | None = typer.Option(None, help="涨跌停幅度覆盖；默认按 A 股板块自动推断"),
    limit_buffer_bps: float = typer.Option(5.0, help="涨跌停判断容差，单位 bp"),
    include_interactions: bool = typer.Option(False, "--include-interactions", help="允许使用交互因子"),
    max_interactions: int = typer.Option(50, help="最多生成多少个交互因子"),
    max_symbols: int = typer.Option(200, help="资源保护：最多允许多少个组合标的"),
    csv_output: Path | None = typer.Option(None, "--csv-output", help="导出 CSV 组合日收益"),
    markdown_output: Path | None = typer.Option(None, "--markdown-output", help="导出 Markdown 组合报告"),
) -> None:
    """多标的横截面因子组合回测：排序选股、等权持有、按换手扣成本。"""
    from qtrade.backtest.factor_portfolio import (
        run_factor_portfolio_backtest,
        write_factor_portfolio_csv,
        write_factor_portfolio_markdown,
    )
    from qtrade.data.schema import normalize_symbol

    syms = [normalize_symbol(s) for s in symbols]
    try:
        _guard_symbol_count(syms, max_symbols=max_symbols)
        _print_workload_estimate("因子组合回测", symbols=len(syms))
        report = run_factor_portfolio_backtest(
            symbols=syms,
            factor_name=factor_name,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end) if end else date.today(),
            windows=windows,
            direction=direction,
            rebalance_days=rebalance_days,
            select_count=select_count,
            select_quantile=select_quantile,
            initial_equity=initial_equity,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            respect_tradeability=respect_tradeability,
            min_volume=min_volume,
            price_limit_pct=price_limit_pct,
            limit_buffer_bps=limit_buffer_bps,
            include_interactions=include_interactions,
            max_interactions=max_interactions,
        )
    except (ValueError, DataError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_factor_portfolio_report(report)
    if csv_output:
        saved = write_factor_portfolio_csv(report, csv_output)
        console.print(f"[green]✓[/green] CSV 已写入：{saved}")
    if markdown_output:
        saved = write_factor_portfolio_markdown(report, markdown_output)
        console.print(f"[green]✓[/green] Markdown 已写入：{saved}")


def _print_backtest(result) -> None:
    m = result.metrics
    table = Table(title="回测结果")
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="white")
    table.add_row("总收益", f"{m.total_return:.2%}")
    table.add_row("年化收益", f"{m.annual_return:.2%}")
    table.add_row("年化波动", f"{m.annual_volatility:.2%}")
    table.add_row("Sharpe", f"{m.sharpe:.2f}")
    table.add_row("Sortino", f"{m.sortino:.2f}")
    table.add_row("Calmar", f"{m.calmar:.2f}")
    table.add_row("最大回撤", f"{m.max_drawdown:.2%}")
    table.add_row("最大回撤天数", f"{m.max_drawdown_duration_days}")
    table.add_row("交易次数", f"{m.n_trades}")
    table.add_row("胜率", f"{m.win_rate:.2%}")
    table.add_row("盈亏比", f"{m.profit_loss_ratio:.2f}")
    table.add_row("初始权益", f"{m.initial_equity:,.2f}")
    table.add_row("最终权益", f"{m.final_equity:,.2f}")
    console.print(table)


def _print_backtest_comparison(comparison) -> None:
    table = Table(title="多策略回测对比")
    has_params = any(row.params for row in comparison.rows)
    table.add_column("策略", style="cyan")
    if has_params:
        table.add_column("参数", overflow="fold")
    table.add_column("总收益", justify="right")
    table.add_column("年化", justify="right")
    table.add_column("波动", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Calmar", justify="right")
    table.add_column("最大回撤", justify="right")
    table.add_column("交易", justify="right")
    table.add_column("胜率", justify="right")
    table.add_column("终值", justify="right")
    for row in comparison.rows:
        m = row.metrics
        ret_style = "red" if m.total_return > 0 else "green" if m.total_return < 0 else "white"
        cells = [
            row.strategy_name,
        ]
        if has_params:
            cells.append(_format_params_for_console(row.params))
        cells.extend([
            f"[{ret_style}]{m.total_return:.2%}[/{ret_style}]",
            f"{m.annual_return:.2%}",
            f"{m.annual_volatility:.2%}",
            f"{m.sharpe:.2f}",
            f"{m.sortino:.2f}",
            f"{m.calmar:.2f}",
            f"{m.max_drawdown:.2%}",
            str(m.n_trades),
            f"{m.win_rate:.2%}",
            f"{m.final_equity:,.2f}",
        ])
        table.add_row(*cells)
    console.print(table)
    console.print(
        f"[cyan]批量质量[/cyan]：成功 {len(comparison.rows)} / "
        f"失败 {len(getattr(comparison, 'failures', []))} / "
        f"失败率 {getattr(comparison, 'failure_rate', 0.0):.2%}"
    )
    best_sharpe = comparison.best_by_sharpe
    best_return = comparison.best_by_return
    if best_sharpe:
        console.print(f"[cyan]Sharpe 最优[/cyan]：{best_sharpe.strategy_name} ({best_sharpe.metrics.sharpe:.2f})")
    if best_return:
        console.print(f"[cyan]总收益最优[/cyan]：{best_return.strategy_name} ({best_return.metrics.total_return:.2%})")


def _print_backtest_failures(comparison) -> None:
    failures = getattr(comparison, "failures", [])
    if not failures:
        return
    table = Table(title=f"失败组合 | {len(failures)}")
    table.add_column("策略", style="cyan")
    table.add_column("参数", overflow="fold")
    table.add_column("错误类型")
    table.add_column("错误", overflow="fold")
    for failure in failures[:20]:
        table.add_row(
            failure.strategy_name,
            _format_params_for_console(failure.params),
            failure.error_type,
            failure.error,
        )
    console.print(table)
    if len(failures) > 20:
        console.print("[yellow]仅展示前 20 个失败组合；完整失败记录请看 --checkpoint-csv[/yellow]")


def _print_parameter_sensitivity(comparison, *, sort_by: str, descending: bool = True) -> None:
    from qtrade.backtest.comparison import compute_parameter_sensitivity

    rows = compute_parameter_sensitivity(comparison, top_n=10, sort_by=sort_by, descending=descending)
    if not rows:
        return
    table = Table(title=f"参数敏感性 | Top {rows[0].top_count} by {sort_by}")
    table.add_column("参数", style="cyan")
    table.add_column("Top 主导值", overflow="fold")
    table.add_column("Top 占比", justify="right")
    table.add_column("Top 唯一", justify="right")
    table.add_column("全量唯一", justify="right")
    table.add_column("边界")
    for row in rows:
        boundary = row.boundary_side or "-"
        boundary_style = "yellow" if row.is_boundary and row.dominant_share >= 0.5 else "white"
        table.add_row(
            row.param_name,
            row.dominant_value,
            f"{row.dominant_share:.2%}",
            str(row.top_unique_count),
            str(row.full_unique_count),
            f"[{boundary_style}]{boundary}[/{boundary_style}]",
        )
    console.print(table)
    if any(row.is_boundary and row.dominant_share >= 0.5 for row in rows):
        console.print("[yellow]提示[/yellow]：Top 结果集中在参数边界时，建议扩大网格或做 walk-forward 验证。")


def _print_resume_status(*, resume_from_checkpoint: bool, resumed_count: int) -> None:
    if not resume_from_checkpoint:
        return
    console.print(f"[green]✓[/green] 从 checkpoint 恢复已完成结果：{resumed_count} 条")


def _print_walk_forward_report(report) -> None:
    table = Table(title=f"Walk-forward 验证 | {report.strategy_name} | {','.join(report.symbols)}")
    table.add_column("折", justify="right")
    table.add_column("训练区间")
    table.add_column("验证区间")
    table.add_column("参数", overflow="fold")
    table.add_column("训练 Sharpe", justify="right")
    table.add_column("验证收益", justify="right")
    table.add_column("验证 Sharpe", justify="right")
    table.add_column("验证回撤", justify="right")
    table.add_column("验证交易", justify="right")
    for fold in report.folds:
        test_return = fold.test.metrics.total_return
        ret_style = "red" if test_return > 0 else "green" if test_return < 0 else "white"
        table.add_row(
            str(fold.fold),
            f"{fold.train_start.isoformat()} → {fold.train_end.isoformat()}",
            f"{fold.test_start.isoformat()} → {fold.test_end.isoformat()}",
            _format_params_for_console(fold.selected_params),
            f"{fold.train.metrics.sharpe:.2f}",
            f"[{ret_style}]{test_return:.2%}[/{ret_style}]",
            f"{fold.test.metrics.sharpe:.2f}",
            f"{fold.test.metrics.max_drawdown:.2%}",
            str(fold.test.metrics.n_trades),
        )
    console.print(table)
    console.print(
        f"[cyan]验证期复合收益[/cyan]：{report.compounded_test_return:.2%} | "
        f"[cyan]平均训练 Sharpe[/cyan]：{report.average_train_sharpe:.2f} | "
        f"[cyan]平均验证 Sharpe[/cyan]：{report.average_test_sharpe:.2f} | "
        f"[cyan]Sharpe 保留率[/cyan]：{report.test_sharpe_retention:.2%} | "
        f"[cyan]验证正收益折数[/cyan]：{report.profitable_test_fold_count}/{len(report.folds)}"
    )
    console.print(f"[cyan]平均验证回撤[/cyan]：{report.average_test_max_drawdown:.2%}")
    console.print(
        "[yellow]注意[/yellow]：walk-forward 能降低单次回测过拟合风险，"
        "但仍需进一步做跨标的、交易成本、容量和稳定性检查。"
    )


def _print_experiment_candidates(candidates, *, total_count: int, sort_by: str, descending: bool) -> None:
    order = "desc" if descending else "asc"
    table = Table(title=f"实验候选榜单 | {sort_by} {order}")
    table.add_column("#", justify="right")
    table.add_column("类型", style="cyan")
    table.add_column("生成时间")
    table.add_column("策略")
    table.add_column("标的", overflow="fold")
    table.add_column("区间")
    table.add_column("参数", overflow="fold")
    table.add_column("收益", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("回撤", justify="right")
    table.add_column("交易/折", justify="right")
    table.add_column("胜率", justify="right")
    table.add_column("来源")
    for index, candidate in enumerate(candidates, start=1):
        total_return = candidate.metric("total_return")
        ret_style = "red" if total_return and total_return > 0 else "green" if total_return and total_return < 0 else "white"
        n_trades = candidate.metric("n_trades")
        source_name = Path(candidate.source_path).name if candidate.source_path else "-"
        table.add_row(
            str(index),
            candidate.record_type,
            candidate.generated_at.replace("T", " ")[:19],
            candidate.strategy,
            ",".join(candidate.symbols),
            f"{candidate.start}→{candidate.end}",
            _format_params_for_console(candidate.params),
            f"[{ret_style}]{_format_optional_pct(total_return)}[/{ret_style}]",
            _format_optional_float(candidate.metric("sharpe")),
            _format_optional_pct(candidate.metric("max_drawdown")),
            str(int(n_trades)) if n_trades is not None else candidate.note or "-",
            _format_optional_pct(candidate.metric("win_rate")),
            f"{source_name}:{candidate.source_line}" if candidate.source_line else source_name,
        )
    console.print(table)
    console.print(f"[cyan]候选数量[/cyan]：展示 {len(candidates)} / 总计 {total_count}")
    if not candidates:
        console.print("[yellow]没有可展示的实验候选。确认 JSONL 里包含 backtest_grid、factor_grid 或 factor_walk_forward 记录。[/yellow]")


def _print_validation_commands(commands) -> None:
    table = Table(title="下一轮验证计划")
    table.add_column("#", justify="right")
    table.add_column("类型", style="cyan")
    table.add_column("验证")
    table.add_column("策略")
    table.add_column("参数", overflow="fold")
    table.add_column("命令", overflow="fold")
    for index, plan in enumerate(commands, start=1):
        candidate = plan.candidate
        table.add_row(
            str(index),
            candidate.record_type,
            plan.validation_type,
            candidate.strategy,
            _format_params_for_console(candidate.params),
            plan.command,
        )
    console.print(table)
    console.print(f"[cyan]验证命令[/cyan]：{len(commands)} 条")
    if not commands:
        console.print("[yellow]没有生成验证命令。可能候选记录缺少 factor_name，或 walk-forward 每折参数不一致。[/yellow]")


def _print_plan_run_results(results, *, execute: bool, total_commands: int) -> None:
    mode = "execute" if execute else "dry-run"
    table = Table(title=f"验证计划执行 | {mode}")
    table.add_column("#", justify="right")
    table.add_column("状态", style="cyan")
    table.add_column("返回码", justify="right")
    table.add_column("命令", overflow="fold")
    table.add_column("输出尾部", overflow="fold")
    for result in results:
        if result.status == "success":
            status = "[green]success[/green]"
        elif result.status == "dry_run":
            status = "[yellow]dry-run[/yellow]"
        else:
            status = f"[red]{result.status}[/red]"
        output_tail = result.stderr_tail or result.stdout_tail or result.error
        table.add_row(
            str(result.index),
            status,
            "" if result.returncode is None else str(result.returncode),
            result.command,
            output_tail.strip()[-300:],
        )
    console.print(table)
    console.print(f"[cyan]计划命令[/cyan]：处理 {len(results)} / 总计 {total_commands}")
    if not execute:
        console.print("[yellow]dry-run[/yellow]：未执行命令；确认无误后可加 --execute。")


def _format_optional_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _format_optional_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _print_factor_portfolio_report(report) -> None:
    m = report.metrics
    summary = Table(title=f"因子组合回测 | {report.factor} | {','.join(report.symbols)}")
    summary.add_column("指标", style="cyan")
    summary.add_column("数值", justify="right")
    summary.add_row("总收益", f"{m.total_return:.2%}")
    summary.add_row("年化收益", f"{m.annual_return:.2%}")
    summary.add_row("年化波动", f"{m.annual_volatility:.2%}")
    summary.add_row("Sharpe", f"{m.sharpe:.2f}")
    summary.add_row("Sortino", f"{m.sortino:.2f}")
    summary.add_row("Calmar", f"{m.calmar:.2f}")
    summary.add_row("最大回撤", f"{m.max_drawdown:.2%}")
    summary.add_row("调仓次数", str(report.rebalance_count))
    summary.add_row("平均换手", f"{report.average_turnover:.2%}")
    summary.add_row("平均持仓数", f"{report.average_holdings:.2f}")
    summary.add_row("买入拦截", str(report.blocked_buy_count))
    summary.add_row("卖出拦截", str(report.blocked_sell_count))
    summary.add_row("成本合计", f"{report.total_cost_return:.2%}")
    summary.add_row("最终权益", f"{m.final_equity:,.2f}")
    console.print(summary)

    rows = [day for day in report.days if day.turnover > 1e-12][-10:]
    if rows:
        detail = Table(title="最近调仓")
        detail.add_column("日期", style="cyan")
        detail.add_column("持仓", overflow="fold")
        detail.add_column("换手", justify="right")
        detail.add_column("净收益", justify="right")
        detail.add_column("权益", justify="right")
        detail.add_column("拦截", overflow="fold")
        for day in rows:
            ret_style = "red" if day.net_return > 0 else "green" if day.net_return < 0 else "white"
            blocked = []
            if day.blocked_buy_symbols:
                blocked.append("买:" + ",".join(day.blocked_buy_symbols))
            if day.blocked_sell_symbols:
                blocked.append("卖:" + ",".join(day.blocked_sell_symbols))
            detail.add_row(
                day.date.isoformat(),
                ",".join(day.selected_symbols),
                f"{day.turnover:.2%}",
                f"[{ret_style}]{day.net_return:.2%}[/{ret_style}]",
                f"{day.equity:,.2f}",
                " | ".join(blocked),
            )
        console.print(detail)
    console.print(
        "[yellow]注意[/yellow]：这是研究型组合回测，按下一期 close-to-close 收益近似，"
        "真实交易前还需要交易日历、停牌、涨跌停、容量和撮合模型。"
    )


def _print_factor_report(report) -> None:
    table = Table(title=f"因子挖掘 | {','.join(report.symbols)} | horizon={report.horizon_days}d")
    table.add_column("因子", style="cyan", overflow="fold")
    table.add_column("表达式", overflow="fold")
    table.add_column("Train RankIC", justify="right")
    table.add_column("Test RankIC", justify="right")
    table.add_column("Test IC", justify="right")
    table.add_column("Test T-B", justify="right")
    table.add_column("稳定", justify="center")
    table.add_column("Test N", justify="right")
    for row in report.rows:
        stable_style = "green" if row.stable_rank_ic_sign else "yellow"
        table.add_row(
            row.name,
            row.expression,
            f"{row.train.rank_ic:.4f}",
            f"{row.test.rank_ic:.4f}",
            f"{row.test.ic:.4f}",
            f"{row.test.spread_return:.4%}",
            f"[{stable_style}]{'yes' if row.stable_rank_ic_sign else 'no'}[/{stable_style}]",
            str(row.test.n_obs),
        )
    console.print(table)
    best = report.best_by_test_rank_ic_abs
    if best:
        console.print(
            f"[cyan]样本外 Rank IC 绝对值最高[/cyan]：{best.name} "
            f"({best.test.rank_ic:.4f}) | 表达式：{best.expression}"
        )
    console.print(
        "[yellow]注意[/yellow]：这是因子相关性筛选，不等于策略可盈利；"
        "后续还需要分层回测、交易成本、换手率和样本外稳定性验证。"
    )


def _print_factor_layer_report(report) -> None:
    summary = Table(title=f"因子分层验证 | {report.factor} | horizon={report.horizon_days}d")
    summary.add_column("样本", style="cyan")
    summary.add_column("Rank IC", justify="right")
    summary.add_column("Top", justify="right")
    summary.add_column("Bottom", justify="right")
    summary.add_column("Top-Bottom", justify="right")
    summary.add_column("单调性", justify="right")
    summary.add_column("Top换手", justify="right")
    summary.add_column("N", justify="right")
    for name, window in (("train", report.train), ("test", report.test)):
        summary.add_row(
            name,
            f"{window.rank_ic:.4f}",
            f"{window.top_return:.4%}",
            f"{window.bottom_return:.4%}",
            f"{window.spread_return:.4%}",
            f"{window.monotonicity:.4f}",
            f"{window.turnover:.2%}",
            str(window.n_obs),
        )
    console.print(summary)

    buckets = Table(title="分组平均未来收益")
    buckets.add_column("样本", style="cyan")
    buckets.add_column("分组", justify="right")
    buckets.add_column("N", justify="right")
    buckets.add_column("平均未来收益", justify="right")
    for name, window in (("train", report.train), ("test", report.test)):
        for bucket in window.buckets:
            buckets.add_row(
                name,
                str(bucket.bucket),
                str(bucket.n_obs),
                f"{bucket.avg_forward_return:.4%}",
            )
    console.print(buckets)
    stable_style = "green" if report.stable_spread_sign else "yellow"
    console.print(
        f"[cyan]表达式[/cyan]：{report.expression} | "
        f"[cyan]样本外 Top-Bottom[/cyan]：{report.test.spread_return:.4%} | "
        f"[{stable_style}]样本内/样本外价差同向：{'yes' if report.stable_spread_sign else 'no'}[/{stable_style}]"
    )
    console.print(
        "[yellow]注意[/yellow]：分层验证仍是研究信号，不是交易策略；"
        "下一步需要把入选因子转成组合规则，并纳入交易成本与风控。"
    )


def _parse_param_grid(items: list[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"参数格式错误：{item}；应为 key=v1,v2")
        key, raw_values = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"参数名不能为空：{item}")
        values = [_parse_param_value(value.strip()) for value in raw_values.split(",") if value.strip()]
        if not values:
            raise ValueError(f"参数 {key} 的候选值不能为空")
        grid[key] = values
    if not grid:
        raise ValueError("参数网格不能为空")
    return grid


def _parse_param_value(raw: str) -> int | float | bool | str:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _default_param_grid(strategy: str) -> dict[str, list[Any]]:
    grids: dict[str, dict[str, list[Any]]] = {
        "ma_cross": {
            "fast_window": [5, 10],
            "slow_window": [20, 60],
            "position_pct": [0.2, 0.3],
        },
        "macd": {
            "fast": [8, 12],
            "slow": [21, 26],
            "signal": [9],
            "position_pct": [0.2, 0.3],
        },
        "bollinger": {
            "window": [20, 30],
            "num_std": [2.0, 2.5],
            "position_pct": [0.2, 0.3],
        },
        "momentum": {
            "lookback": [20, 60],
            "entry_return": [0.03, 0.05],
            "exit_ma": [10, 20],
            "position_pct": [0.2, 0.3],
        },
        "turtle": {
            "entry_window": [20, 55],
            "exit_window": [10, 20],
            "atr_window": [14],
            "atr_risk": [0.005, 0.01],
            "max_position_pct": [0.2, 0.3],
        },
    }
    try:
        return grids[strategy]
    except KeyError as exc:
        raise ValueError(f"未知策略：{strategy}（可用：{list(grids)}）") from exc


def _format_params_for_console(params: dict[str, Any]) -> str:
    if not params:
        return ""
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _build_factor_strategy_param_grid(
    *,
    factor_name: str,
    windows: list[int],
    lookbacks: list[int],
    entry_quantiles: list[float],
    exit_quantiles: list[float],
    directions: list[str],
    position_pcts: list[float],
    include_interactions: bool,
    max_interactions: int,
) -> dict[str, list[Any]]:
    if not factor_name:
        raise ValueError("factor 不能为空")
    if any(window <= 1 for window in windows):
        raise ValueError("window 必须大于 1")
    if any(lookback < 5 for lookback in lookbacks):
        raise ValueError("lookback 必须 >= 5")
    if any(not 0 < q < 1 for q in entry_quantiles + exit_quantiles):
        raise ValueError("entry_quantile / exit_quantile 必须在 0-1 之间")
    if any(entry <= exit for entry in entry_quantiles for exit in exit_quantiles):
        raise ValueError("所有 entry_quantile 必须大于 exit_quantile，用于形成买卖滞后区间")
    invalid_directions = [direction for direction in directions if direction not in {"high", "low"}]
    if invalid_directions:
        raise ValueError(f"direction 只能是 high 或 low：{invalid_directions}")
    if any(not 0 < pct <= 1 for pct in position_pcts):
        raise ValueError("position_pct 必须在 0-1 之间")
    if max_interactions < 0:
        raise ValueError("max_interactions 不能为负数")
    return {
        "factor_name": [factor_name],
        "windows": [sorted(set(windows))],
        "lookback": sorted(set(lookbacks)),
        "entry_quantile": sorted(set(entry_quantiles)),
        "exit_quantile": sorted(set(exit_quantiles)),
        "direction": sorted(set(directions)),
        "position_pct": sorted(set(position_pcts)),
        "include_interactions": [include_interactions],
        "max_interactions": [max_interactions],
    }


def _grid_size(param_grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def _guard_symbol_count(symbols: list[str], *, max_symbols: int) -> None:
    if max_symbols <= 0:
        raise ValueError("max_symbols 必须为正数")
    if len(symbols) > max_symbols:
        raise ValueError(
            f"标的数量 {len(symbols)} 超过资源保护上限 {max_symbols}；"
            "请减少 --symbol，或确认机器可承受后调高 --max-symbols"
        )


def _guard_param_combinations(param_count: int, *, max_param_combinations: int) -> None:
    if max_param_combinations <= 0:
        raise ValueError("max_param_combinations 必须为正数")
    if param_count > max_param_combinations:
        raise ValueError(
            f"参数组合 {param_count} 超过资源保护上限 {max_param_combinations}；"
            "请缩小参数网格，或确认机器可承受后调高 --max-param-combinations"
        )


def _guard_backtest_runs(backtest_runs: int, *, max_backtest_runs: int) -> None:
    if max_backtest_runs <= 0:
        raise ValueError("max_backtest_runs 必须为正数")
    if backtest_runs > max_backtest_runs:
        raise ValueError(
            f"内部回测次数 {backtest_runs} 超过资源保护上限 {max_backtest_runs}；"
            "请缩小参数网格/缩短 walk-forward 折数，或确认机器可承受后调高 --max-backtest-runs"
        )


def _estimate_walk_forward_folds(
    *,
    start: date,
    end: date,
    train_days: int,
    test_days: int,
    step_days: int | None,
) -> int:
    if start >= end or train_days <= 0 or test_days <= 0:
        return 0
    step = step_days or test_days
    if step <= 0:
        return 0
    count = 0
    train_start = start
    while True:
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        if test_start > end:
            break
        test_end = min(test_start + timedelta(days=test_days - 1), end)
        count += 1
        if test_end >= end:
            break
        train_start = train_start + timedelta(days=step)
    return count


def _print_workload_estimate(
    label: str,
    *,
    symbols: int,
    param_combinations: int | None = None,
    folds: int | None = None,
    backtest_runs: int | None = None,
) -> None:
    parts = [f"标的={symbols}"]
    if param_combinations is not None:
        parts.append(f"参数组合={param_combinations}")
    if folds is not None:
        parts.append(f"折数={folds}")
    if backtest_runs is not None:
        parts.append(f"内部回测={backtest_runs}")
    console.print(f"[dim]资源预估 | {label} | {' | '.join(parts)}[/dim]")


@contextmanager
def _progress_reporter(label: str, *, total: int):
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    if total <= 0:
        yield None
        return
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        disable=not console.is_terminal,
    )
    with progress:
        task_id = progress.add_task(label, total=total)

        def callback(done: int, total_count: int, payload: dict[str, Any]) -> None:
            progress.update(
                task_id,
                completed=done,
                total=total_count,
                description=_progress_description(label, payload),
            )

        yield callback


def _progress_description(label: str, payload: dict[str, Any]) -> str:
    parts = [label]
    fold = payload.get("fold")
    if fold is not None:
        parts.append(f"fold={fold}")
    stage = payload.get("stage")
    if stage:
        parts.append(str(stage))
    strategy = payload.get("strategy")
    if strategy:
        parts.append(str(strategy))
    params = payload.get("params")
    if isinstance(params, dict) and params:
        parts.append(_format_params_for_console(params))
    return " | ".join(parts)


def _chain_progress_callbacks(*callbacks: Callable[[int, int, dict[str, Any]], None] | None):
    active = [callback for callback in callbacks if callback is not None]
    if not active:
        return None

    def callback(done: int, total: int, payload: dict[str, Any]) -> None:
        for item in active:
            item(done, total, payload)

    return callback


@contextmanager
def _checkpoint_csv_writer(path: Path | None, *, append: bool = False):
    if path is None:
        yield None
        return
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not out.exists() or out.stat().st_size == 0
    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8", newline="") as f:
        writer: csv.DictWriter | None = None

        def callback(done: int, total: int, payload: dict[str, Any]) -> None:
            nonlocal writer
            row = _checkpoint_row(done, total, payload)
            if row is None:
                return
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()

        yield callback


def _checkpoint_row(done: int, total: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    row = {
        "done": done,
        "total": total,
        "fold": payload.get("fold") or "",
        "stage": payload.get("stage") or "",
        "status": "ok",
        "error_type": "",
        "error": "",
    }
    summary = payload.get("summary")
    if summary is not None and hasattr(summary, "to_row"):
        row.update(summary.to_row())
        return row
    failure = payload.get("failure")
    if failure is not None and hasattr(failure, "to_row"):
        row["status"] = "failed"
        row.update(failure.to_row())
        return row
    return None


def _load_backtest_checkpoint_resume(
    path: Path | None,
    *,
    candidate_keys: set[str],
    enabled: bool,
) -> tuple[list[Any], set[str]]:
    if not enabled:
        return [], set()
    if path is None:
        raise ValueError("--resume-from-checkpoint 需要同时指定 --checkpoint-csv")
    checkpoint = path.expanduser().resolve()
    if not checkpoint.exists():
        return [], set()
    if not candidate_keys:
        return [], set()

    from qtrade.backtest.comparison import backtest_task_key

    summaries_by_key: dict[str, Any] = {}
    with checkpoint.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            if raw_row.get("status", "ok") != "ok":
                continue
            if raw_row.get("fold") or raw_row.get("stage"):
                continue
            summary = _checkpoint_summary_from_row(raw_row)
            if summary is None:
                continue
            key = backtest_task_key(
                strategy_name=summary.strategy_name,
                symbols=summary.symbols,
                start=summary.start,
                end=summary.end,
                params=summary.params,
            )
            if key in candidate_keys:
                summaries_by_key[key] = summary
    return list(summaries_by_key.values()), set(summaries_by_key)


def _checkpoint_summary_from_row(row: dict[str, str]):
    from qtrade.backtest.comparison import BacktestSummary
    from qtrade.backtest.metrics import Metrics

    try:
        params = _parse_checkpoint_params(row.get("params", ""))
        symbols = [symbol for symbol in row.get("symbols", "").split(",") if symbol]
        metrics = Metrics(
            total_return=float(row["total_return"]),
            annual_return=float(row["annual_return"]),
            annual_volatility=float(row["annual_volatility"]),
            sharpe=float(row["sharpe"]),
            sortino=float(row["sortino"]),
            calmar=float(row["calmar"]),
            max_drawdown=float(row["max_drawdown"]),
            max_drawdown_duration_days=int(float(row["max_drawdown_duration_days"])),
            win_rate=float(row["win_rate"]),
            profit_loss_ratio=float(row["profit_loss_ratio"]),
            n_trades=int(float(row["n_trades"])),
            final_equity=float(row["final_equity"]),
            initial_equity=float(row["initial_equity"]),
        )
        return BacktestSummary(
            strategy_name=row["strategy"],
            strategy_id=row.get("strategy_id", ""),
            symbols=symbols,
            start=date.fromisoformat(row["start"]),
            end=date.fromisoformat(row["end"]),
            metrics=metrics,
            trades=[],
            params=params,
            trade_count=metrics.n_trades,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _parse_checkpoint_params(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


@app.command()
def ui() -> None:
    """启动桌面应用。"""
    from qtrade.ui.main import run
    run()


@app.command()
def run(
    strategy: str = typer.Option("ma_cross", help="策略：ma_cross | llm"),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    initial_cash: float = typer.Option(100_000.0),
    interval_minutes: int = typer.Option(5, help="盘中 tick 间隔（分钟）"),
    interval_seconds: int | None = typer.Option(None, help="盘中 tick 间隔（秒，1-59；设置后优先于分钟）"),
    once: bool = typer.Option(False, "--once", help="只跑一次 tick 后退出（调试用）"),
    pre_market: bool = typer.Option(False, "--pre-market", help="只跑盘前任务后退出"),
    post_market: bool = typer.Option(False, "--post-market", help="只跑盘后任务后退出"),
    threshold: float = typer.Option(None, help="LLM 决策阈值（覆盖默认 0.45），仅 --strategy llm 有效"),
    prompt_version: str = typer.Option("v2", help="LLM prompt 版本（v1/v2），仅 --strategy llm 有效"),
    debate_rounds: int = typer.Option(2, help="LLM 辩论轮数，仅 --strategy llm 有效"),
    decision_interval_minutes: int | None = typer.Option(
        None,
        help="每个标的策略决策最小间隔；LLM 默认 15 分钟，技术策略默认每 tick",
    ),
    position_pct: float | None = typer.Option(None, help="目标仓位比例；ma_cross 默认 30%，LLM 默认 20%"),
    max_position_pct: float | None = typer.Option(None, help="LLM 目标仓位硬裁剪上限，默认 30%"),
) -> None:
    """启动模拟盘 LiveRunner（持续运行）。

    例：
        qtrade run --symbol 000001 --symbol 600519
        qtrade run --once                            # 跑一次 tick 看看
    """
    from qtrade.data.schema import normalize_symbol
    from qtrade.execution.account import Account
    from qtrade.execution.broker.paper import PaperBroker
    from qtrade.persistence import init_db
    from qtrade.risk.engine import RiskEngine
    from qtrade.risk.policies import load_risk_policies
    from qtrade.runtime.live_runner import LiveRunner
    from qtrade.runtime.scheduler import SchedulerService
    from qtrade.runtime.state import StateManager
    from qtrade.strategies.technical.ma_cross import MACrossStrategy

    _validate_common_run_options(
        interval_seconds=interval_seconds,
        decision_interval_minutes=decision_interval_minutes,
        position_pct=position_pct,
        max_position_pct=max_position_pct,
    )

    syms = [normalize_symbol(s) for s in symbols]
    if strategy == "ma_cross":
        strat = MACrossStrategy(
            universe=syms,
            params={
                "fast_window": 5,
                "slow_window": 20,
                "position_pct": position_pct if position_pct is not None else 0.3,
            },
        )
        decision_interval_seconds = (decision_interval_minutes or 0) * 60
    elif strategy == "llm":
        _validate_llm_run_options(
            prompt_version=prompt_version,
            debate_rounds=debate_rounds,
            threshold=threshold,
        )
        settings = _require_deepseek_key_for_llm_run()
        from qtrade.strategies.llm.graph import LLMMultiAgentStrategy
        llm_params: dict = {
            "prompt_version": prompt_version,
            "debate_rounds": debate_rounds,
        }
        if threshold is not None:
            llm_params["decision_threshold"] = threshold
        if position_pct is not None:
            llm_params["position_pct"] = position_pct
        if max_position_pct is not None:
            llm_params["max_position_pct"] = max_position_pct
        strat = LLMMultiAgentStrategy(universe=syms, params=llm_params)
        decision_interval_seconds = (decision_interval_minutes or 15) * 60
        _print_llm_run_preflight(
            settings=settings,
            prompt_version=prompt_version,
            debate_rounds=debate_rounds,
            threshold=threshold,
            decision_interval_seconds=decision_interval_seconds,
            position_pct=position_pct,
            max_position_pct=max_position_pct,
        )
    else:
        raise typer.BadParameter(f"未知策略：{strategy}")

    init_db()

    state = StateManager(strategy_id=strat.id, mode="paper")
    risk = RiskEngine(policies=load_risk_policies())
    broker = PaperBroker(account=Account())   # 占位，start() 会接 state 恢复出的账户

    runner = LiveRunner(
        strategy=strat,
        state=state,
        risk=risk,
        broker=broker,
        decision_interval_seconds=decision_interval_seconds,
    )
    session = runner.start(initial_cash=initial_cash)
    console.print(f"[green]✓[/green] LiveRunner 启动 | run_id={session.run_id} uid={session.run_uid}")
    console.print(f"  策略：{strat.id} | 标的：{syms} | 初始资金：{initial_cash:,.0f}")
    console.print(f"  当前账户：现金 {float(session.account.cash):,.2f} / 总资产 {float(session.account.total_assets):,.2f}")
    if strategy == "llm":
        console.print("  [yellow]LLM 只输出交易意图；订单仍会经过行情新鲜度、价格偏离和硬风控检查[/yellow]")

    try:
        if pre_market:
            runner.pre_market()
            console.print("[green]✓[/green] pre-market done")
        elif post_market:
            runner.post_market()
            console.print("[green]✓[/green] post-market done")
        elif once:
            report = runner.tick()
            console.print(
                f"[green]✓[/green] tick done | quotes={report.quotes_count} "
                f"signals={report.signals_generated} skipped={report.strategy_skipped} "
                f"orders={report.orders_placed} "
                f"risk_rejected={report.risk_rejected}"
            )
            if report.errors:
                console.print(f"[yellow]errors:[/yellow] {report.errors}")
        else:
            sched = SchedulerService(
                runner=runner,
                interval_minutes=interval_minutes,
                interval_seconds=interval_seconds,
            )
            tick_desc = f"{interval_seconds} 秒" if interval_seconds is not None else f"{interval_minutes} 分钟"
            decision_desc = (
                f"{decision_interval_seconds // 60} 分钟"
                if decision_interval_seconds
                else "每 tick"
            )
            console.print(f"[cyan]按 Ctrl+C 退出 | tick 间隔 {tick_desc} | 决策间隔 {decision_desc}[/cyan]")
            sched.start()
    finally:
        runner.stop(note="cli stopped")
        console.print("[green]✓[/green] LiveRunner stopped, state persisted")


@app.command(name="run-paper")
def run_paper(
    strategy: str = typer.Option("ma_cross", help="策略：ma_cross | llm"),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-s"),
    initial_cash: float = typer.Option(100_000.0),
    interval_minutes: int = typer.Option(5, help="盘中 tick 间隔（分钟）"),
    interval_seconds: int | None = typer.Option(None, help="盘中 tick 间隔（秒，1-59；设置后优先于分钟）"),
    once: bool = typer.Option(False, "--once", help="只跑一次 tick 后退出（调试用）"),
    threshold: float = typer.Option(None, help="LLM 决策阈值（覆盖默认 0.45），仅 --strategy llm 有效"),
    prompt_version: str = typer.Option("v2", help="LLM prompt 版本（v1/v2），仅 --strategy llm 有效"),
    debate_rounds: int = typer.Option(2, help="LLM 辩论轮数，仅 --strategy llm 有效"),
    decision_interval_minutes: int | None = typer.Option(
        None,
        help="每个标的策略决策最小间隔；LLM 默认 15 分钟，技术策略默认每 tick",
    ),
    position_pct: float | None = typer.Option(None, help="目标仓位比例；ma_cross 默认 30%，LLM 默认 20%"),
    max_position_pct: float | None = typer.Option(None, help="LLM 目标仓位硬裁剪上限，默认 30%"),
) -> None:
    """启动 AkShare + PaperBroker 模拟实盘；等价于 qtrade run。"""
    run(
        strategy=strategy,
        symbols=symbols,
        initial_cash=initial_cash,
        interval_minutes=interval_minutes,
        interval_seconds=interval_seconds,
        once=once,
        pre_market=False,
        post_market=False,
        threshold=threshold,
        prompt_version=prompt_version,
        debate_rounds=debate_rounds,
        decision_interval_minutes=decision_interval_minutes,
        position_pct=position_pct,
        max_position_pct=max_position_pct,
    )


@app.command(name="run-multi")
def run_multi(
    strategies: list[str] = typer.Option(
        ["ma_cross", "macd", "bollinger"], "--strategy", "-s",
        help="多次 -s 指定多个策略，如 -s ma_cross -s macd",
    ),
    symbols: list[str] = typer.Option(["000001"], "--symbol", "-S"),
    initial_cash: float = typer.Option(100_000.0, help="每个策略独立初始资金"),
    interval_minutes: int = typer.Option(5),
    interval_seconds: int | None = typer.Option(None, help="盘中 tick 间隔（秒，1-59；设置后优先于分钟）"),
    once: bool = typer.Option(False, "--once", help="跑一次 tick 后退出"),
) -> None:
    """多策略并行模拟盘。每个策略独立账户、独立 run。

    例：
        qtrade run-multi -s ma_cross -s macd -s bollinger -S 000001
        qtrade run-multi -s ma_cross --once
    """
    from qtrade.data.schema import normalize_symbol
    from qtrade.execution.account import Account
    from qtrade.execution.broker.paper import PaperBroker
    from qtrade.monitoring.notifiers import build_notifier_from_settings
    from qtrade.persistence import init_db
    from qtrade.risk.engine import RiskEngine
    from qtrade.risk.policies import load_risk_policies
    from qtrade.runtime.live_runner import LiveRunner
    from qtrade.runtime.multi_runner import MultiStrategyRunner
    from qtrade.runtime.scheduler import SchedulerService
    from qtrade.runtime.state import StateManager
    from qtrade.strategies.technical import STRATEGY_REGISTRY

    init_db()
    if interval_seconds is not None and not 1 <= interval_seconds <= 59:
        raise typer.BadParameter("interval_seconds 必须在 1-59 秒之间")
    syms = [normalize_symbol(s) for s in symbols]

    runners: list[LiveRunner] = []
    for strat_name in strategies:
        cls = STRATEGY_REGISTRY.get(strat_name)
        if cls is None:
            raise typer.BadParameter(f"未知策略：{strat_name}（可用：{list(STRATEGY_REGISTRY)}）")
        strat = cls(universe=syms)
        runners.append(LiveRunner(
            strategy=strat,
            state=StateManager(strategy_id=strat.id, mode="paper"),
            risk=RiskEngine(policies=load_risk_policies()),
            broker=PaperBroker(account=Account()),
        ))

    notifier = build_notifier_from_settings()
    multi = MultiStrategyRunner(runners=runners, notifier=notifier)
    sessions = multi.start(initial_cash=initial_cash)

    console.print(f"[green]✓[/green] 已启动 {len(runners)} 个策略：")
    for sid, sess in sessions.items():
        console.print(f"  - {sid} | run_id={sess.run_id} | 现金 {float(sess.account.cash):,.0f}")

    try:
        if once:
            report = multi.tick()
            console.print(
                f"[green]✓[/green] tick done | 共下单 {report.total_orders_placed} | "
                f"风控拒 {report.total_risk_rejected} | 错误 {len(report.errors)}"
            )
            for sid, tr in report.per_strategy.items():
                console.print(f"  - {sid}: signals={tr.signals_generated} orders={tr.orders_placed}")
        else:
            # 用一个 fake "single LiveRunner" 喂 SchedulerService 走多策略 dispatch
            class _MultiAdapter:
                def pre_market(_self): multi.pre_market()
                def post_market(_self): multi.post_market()
                def tick(_self): return multi.tick()

            sched = SchedulerService(
                runner=_MultiAdapter(),
                interval_minutes=interval_minutes,
                interval_seconds=interval_seconds,
            )
            tick_desc = f"{interval_seconds} 秒" if interval_seconds is not None else f"{interval_minutes} 分钟"
            console.print(f"[cyan]按 Ctrl+C 退出 | tick 间隔 {tick_desc}[/cyan]")
            sched.start()
    finally:
        multi.stop(note="cli stopped")
        console.print("[green]✓[/green] 全部停止，状态已持久化")


@app.command(name="compare-prompts")
def compare_prompts(
    symbol: str = typer.Argument(..., help="标的代码，如 600519"),
    versions: list[str] = typer.Option(["v1", "v2"], "--version", "-v"),
    lookback_days: int = typer.Option(120),
    debate_rounds_v2: int = typer.Option(2, help="V2 辩论轮数"),
) -> None:
    """对同一只票、同一时刻，跑多个 prompt 版本做 A/B 对比。

    需要 DEEPSEEK_API_KEY。

    例：
        qtrade compare-prompts 600519
        qtrade compare-prompts 000001 --version v1 --version v2
    """
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    from qtrade.core.config import get_settings
    from qtrade.data.loader import get_loader
    from qtrade.data.schema import normalize_symbol
    from qtrade.persistence import init_db
    from qtrade.strategies.llm.ab_runner import run_ab

    if not get_settings().deepseek_api_key:
        console.print("[red]✗[/red] 未检测到 DEEPSEEK_API_KEY")
        raise typer.Exit(1)

    init_db()
    sym = normalize_symbol(symbol)
    end = _dt.now().date()
    start = end - _td(days=lookback_days * 2)
    df = get_loader().get_daily_bars(sym, start=start, end=end)
    if df.empty:
        console.print(f"[red]✗[/red] {sym} 无可用数据")
        raise typer.Exit(1)

    console.print(f"[cyan]►[/cyan] A/B 跑 {sym} | 版本 {versions}")
    cmp = run_ab(sym, df, versions, debate_rounds_map={"v2": debate_rounds_v2})

    table = Table(title=f"Prompt A/B 对比 — {sym} @ {cmp.as_of:%Y-%m-%d %H:%M}")
    table.add_column("版本", style="cyan")
    table.add_column("辩论轮数")
    table.add_column("决策", style="bold")
    table.add_column("信心")
    table.add_column("仓位 %")
    table.add_column("用时 s")
    table.add_column("Tokens")
    table.add_column("成本 USD")
    table.add_column("理由（截断）")

    for r in cmp.runs:
        if r.error:
            table.add_row(r.version, str(r.debate_rounds), "[red]ERROR[/red]",
                          "-", "-", f"{r.elapsed_s:.1f}", "-", "-", r.error[:60])
        else:
            d = r.decision
            table.add_row(
                r.version, str(r.debate_rounds),
                d.get("action", "?"),
                f"{float(d.get('confidence', 0)):.2f}",
                f"{float(d.get('target_position_pct', 0)):.0%}",
                f"{r.elapsed_s:.1f}",
                str(r.tokens_used),
                f"{r.cost_usd:.4f}",
                (d.get("reasoning") or "")[:80],
            )
    console.print(table)

    cons = cmp.consistency()
    if cons:
        console.print()
        console.print(f"[bold]一致性[/bold]：动作一致 = {cons['all_agree_on_action']} | "
                      f"actions = {cons['actions']} | "
                      f"平均信心 {cons['avg_confidence']:.2f} | "
                      f"信心差 {cons['confidence_spread']:.2f}")


@app.command(name="alert-test")
def alert_test() -> None:
    """测试已配置的告警通道（控制台 / ServerChan / Telegram）。"""
    from qtrade.monitoring.notifiers import NotifyEvent, build_notifier_from_settings
    n = build_notifier_from_settings()
    ok = n.send(NotifyEvent.warn(
        "qtrade 告警测试",
        body="如果你看到这条消息，说明你的告警通道配置正确。",
        platform="qtrade", env="dev",
    ))
    if ok:
        console.print("[green]✓[/green] 至少一个通道发送成功（看上面 log 哪些通道）")
    else:
        console.print("[red]✗[/red] 所有通道都失败，检查 .env 配置")


@app.command()
def decide(
    symbol: str = typer.Argument(..., help="单标的代码，如 600519"),
    lookback_days: int = typer.Option(120, help="给 LLM 的历史 K 线天数"),
    persist: bool = typer.Option(True, help="是否把决策落 agent_decisions 表"),
) -> None:
    """对单标的跑一次 LLM 多 Agent 决策（不下单）。

    用途：
    - 调试 prompt（改完 prompts.py 后 decide 一次看输出）
    - 验证 DeepSeek API key 配置
    - 观察 token 消耗和决策质量

    要求：``.env`` 里已配置 ``DEEPSEEK_API_KEY``。
    """
    from datetime import datetime
    from datetime import timedelta as _td

    from qtrade.core.config import get_settings
    from qtrade.data.loader import get_loader
    from qtrade.data.schema import normalize_symbol
    from qtrade.execution.account import Account
    from qtrade.llm.cost import get_cost_tracker
    from qtrade.persistence import init_db
    from qtrade.strategies.base import StrategyContext
    from qtrade.strategies.llm.graph import LLMMultiAgentStrategy

    settings = get_settings()
    if not settings.deepseek_api_key:
        console.print("[red]✗[/red] 未检测到 DEEPSEEK_API_KEY")
        console.print("  ⤷ 在 .env 里设置后重试。注册 + 充值：https://platform.deepseek.com")
        raise typer.Exit(1)

    init_db()
    sym = normalize_symbol(symbol)
    loader = get_loader()
    end = datetime.now().date()
    start = end - _td(days=lookback_days * 2)
    df = loader.get_daily_bars(sym, start=start, end=end)
    if df.empty:
        console.print(f"[red]✗[/red] {sym} 无可用数据，先 download_history.py")
        raise typer.Exit(1)

    strat = LLMMultiAgentStrategy(universe=[sym])
    if not persist:
        strat._persist_decision = lambda *a, **kw: None     # type: ignore
    strat.on_start()
    ctx = StrategyContext(
        ts=datetime.now(), symbol=sym, history=df,
        account=Account(initial_cash=Decimal("100000")),
    )

    tracker = get_cost_tracker()
    t0 = datetime.now()
    console.print(f"[cyan]►[/cyan] 跑 LLM 决策 | {sym} | 数据 {len(df)} 根 K 线…")
    signals = strat.on_bar(ctx)
    elapsed_s = (datetime.now() - t0).total_seconds()

    # 输出最后一次决策的 trace（agents.py 落进 SQLite，但这里也直接打印一份）
    console.print(f"\n[bold]决策耗时[/bold] {elapsed_s:.1f}s | "
                  f"累计 token {tracker.total_prompt_tokens + tracker.total_completion_tokens} | "
                  f"累计成本 ${tracker.total_cost_usd:.4f}")

    if not signals:
        console.print("[yellow]→ HOLD 或低信心，未产生信号[/yellow]")
    else:
        for sig in signals:
            console.print(f"[green]→[/green] {sig.action} {sig.symbol} x{sig.quantity}  "
                          f"信心={sig.confidence:.2f}  理由={sig.reason[:120]}")

    console.print("\n[dim]完整 trace 已落 agent_decisions 表，UI 上 'Agent 决策' tab 可看[/dim]")


if __name__ == "__main__":
    app()
