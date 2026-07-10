"""UI 展示层回归测试。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtrade.core.time import MARKET_TZ, utcnow
from qtrade.data.adapters.base import Quote
from qtrade.persistence import init_db, session_scope
from qtrade.persistence.models import (
    AccountSnapshot,
    AgentDecision,
    DecisionAction,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyRun,
    TickLog,
)
from qtrade.runtime.paper_check import PaperCheckItem, PaperCheckReport
from qtrade.ui.data_access import (
    market_data_quality,
    paper_run_health,
    performance_runs,
    recent_decisions,
    recent_orders,
    recent_tick_logs,
    run_performance_report,
)
from qtrade.ui.theme import QSS
from qtrade.ui.windows.agent_log import AgentLogWidget
from qtrade.ui.windows.dashboard import DashboardWidget
from qtrade.ui.windows.kline import (
    INITIAL_VISIBLE_BARS,
    _data_source_note,
    _intraday_minute_note,
    _padded_range,
)
from qtrade.ui.windows.market_watch import (
    MarketWatchWidget,
    _chart_start_date,
    _parse_symbols,
    _quote_status,
)
from qtrade.ui.windows.orders import OrdersWidget, _bps_to_pct_text
from qtrade.ui.windows.paper_check import PaperCheckWidget, _status_color
from qtrade.ui.windows.performance import PerformanceWidget
from qtrade.ui.windows.run_control import (
    RunControlWidget,
    build_run_paper_command,
    parse_symbol_list,
)


def _make_dashboard() -> DashboardWidget:
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    return DashboardWidget()


def _ensure_qapp() -> None:
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])


def _metric_text(card) -> str:
    return card._val.text()


def test_dashboard_clears_stale_values_when_no_run():
    init_db()
    widget = _make_dashboard()

    widget.card_total.set_value("999,999.00")
    widget.card_cash.set_value("888,888.00")
    widget.card_market.set_value("777,777.00")
    widget.tbl.setRowCount(1)

    widget.refresh()

    assert "无活跃 run" in widget.lbl_run.text()
    assert "qtrade run" in widget.lbl_run_detail.text()
    assert _metric_text(widget.card_total) == "—"
    assert _metric_text(widget.card_cash) == "—"
    assert _metric_text(widget.card_market) == "—"
    assert _metric_text(widget.card_pos) == "—"
    assert widget.tbl.rowCount() == 0


def test_dashboard_clears_metric_values_when_run_has_no_snapshot():
    init_db()
    with session_scope() as s:
        s.add(
            StrategyRun(
                run_uid="ui-test-run",
                strategy_id="ui_test",
                strategy_version="test",
                mode="paper",
                started_at=utcnow(),
                config_json={},
            )
        )

    widget = _make_dashboard()
    widget.card_total.set_value("999,999.00")
    widget.card_cash.set_value("888,888.00")

    widget.refresh()

    assert "#1 ui_test" in widget.lbl_run.text()
    assert "等待账户快照" in widget.lbl_run_detail.text()
    assert _metric_text(widget.card_total) == "—"
    assert _metric_text(widget.card_cash) == "—"
    assert _metric_text(widget.card_pos) == "0"
    assert widget.tbl.rowCount() == 0


def test_dashboard_shows_latest_snapshot_age():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-snapshot-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add(
            AccountSnapshot(
                run_id=run.id,
                cash=100000,
                frozen_cash=0,
                market_value=0,
                total_assets=100000,
                realized_pnl=0,
                unrealized_pnl=0,
                ts=now,
            )
        )

    widget = _make_dashboard()
    widget.refresh()

    assert "最近快照" in widget.lbl_run_detail.text()
    assert _metric_text(widget.card_total) == "100,000.00"


def test_dashboard_shows_market_data_quality():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-quality-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add(
            AccountSnapshot(
                run_id=run.id,
                cash=100000,
                frozen_cash=0,
                market_value=0,
                total_assets=100000,
                realized_pnl=0,
                unrealized_pnl=0,
                ts=now,
            )
        )
        s.add(
            TickLog(
                run_id=run.id,
                strategy_id="ui_test",
                ts=now,
                quotes_count=1,
                tradeable_quotes=0,
                fallback_quotes=1,
                strategy_skipped=1,
                orders_placed=0,
                risk_rejected=0,
                symbols_json=["000001.SZ"],
                per_symbol_json=[{"symbol": "000001.SZ", "skip_reason": "non_tradeable_quote"}],
                errors_json=[],
            )
        )

    widget = _make_dashboard()
    widget.refresh()

    assert "仅部分可模拟" in widget.lbl_market_quality.text()
    assert "non_tradeable_quote" in widget.lbl_market_quality.text()


def test_paper_run_health_summarizes_active_run():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-health-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add(
            AccountSnapshot(
                run_id=run.id,
                cash=90000,
                frozen_cash=0,
                market_value=10000,
                total_assets=100000,
                realized_pnl=0,
                unrealized_pnl=0,
                ts=now,
            )
        )
        s.add(
            TickLog(
                run_id=run.id,
                strategy_id="ui_test",
                ts=now,
                quotes_count=2,
                tradeable_quotes=2,
                fallback_quotes=0,
                signals_generated=1,
                strategy_skipped=0,
                orders_placed=1,
                risk_rejected=0,
                symbols_json=["000001.SZ", "600519.SH"],
                per_symbol_json=[
                    {"symbol": "000001.SZ", "source": "eastmoney_quote"},
                    {"symbol": "600519.SH", "source": "eastmoney_quote"},
                ],
                errors_json=[],
            )
        )
        s.add_all(
            [
                OrderRecord(
                    run_id=run.id,
                    strategy_id="ui_test",
                    client_order_id="ui-health-open",
                    broker_order_id="PB-ui-health-open",
                    symbol="000001.SZ",
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=100,
                    limit_price=Decimal("10"),
                    status=OrderStatus.SUBMITTED,
                    filled_quantity=0,
                    commission=Decimal("0"),
                    stamp_tax=Decimal("0"),
                    created_at=now,
                ),
                OrderRecord(
                    run_id=run.id,
                    strategy_id="ui_test",
                    client_order_id="ui-health-rejected",
                    broker_order_id="PB-ui-health-rejected",
                    symbol="600519.SH",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=100,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    commission=Decimal("0"),
                    stamp_tax=Decimal("0"),
                    reject_reason="仓位超过上限",
                    created_at=now,
                ),
            ]
        )

    health = paper_run_health(run.id)

    assert health["status"] == "RISK_REJECTED"
    assert health["latest_quote_sources"] == ["eastmoney_quote"]
    assert health["latest_tradeable_quotes"] == 2
    assert health["open_order_count"] == 1
    assert health["rejected_order_count"] == 1
    assert health["account"]["total_assets"] == 100000


def test_dashboard_shows_paper_run_health():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-health-dashboard-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add(
            AccountSnapshot(
                run_id=run.id,
                cash=100000,
                frozen_cash=0,
                market_value=0,
                total_assets=100000,
                realized_pnl=0,
                unrealized_pnl=0,
                ts=now,
            )
        )
        s.add(
            TickLog(
                run_id=run.id,
                strategy_id="ui_test",
                ts=now,
                quotes_count=1,
                tradeable_quotes=1,
                fallback_quotes=0,
                signals_generated=0,
                strategy_skipped=1,
                orders_placed=0,
                risk_rejected=0,
                symbols_json=["000001.SZ"],
                per_symbol_json=[
                    {
                        "symbol": "000001.SZ",
                        "source": "eastmoney_quote",
                        "skip_reason": "decision_interval",
                    }
                ],
                errors_json=[],
            )
        )

    widget = _make_dashboard()
    widget.refresh()

    assert "运行正常" in widget.lbl_run_health.text()
    assert "eastmoney_quote" in widget.lbl_run_health.text()
    assert "decision_interval" in widget.lbl_run_health.text()


def test_run_performance_report_computes_paper_metrics():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-performance-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        for idx, total_assets in enumerate([100000, 110000, 99000, 120000]):
            s.add(
                AccountSnapshot(
                    run_id=run.id,
                    cash=Decimal("80000"),
                    frozen_cash=0,
                    market_value=Decimal(str(total_assets - 80000)),
                    total_assets=Decimal(str(total_assets)),
                    realized_pnl=0,
                    unrealized_pnl=0,
                    ts=now + timedelta(seconds=idx),
                )
            )
        s.add_all(
            [
                OrderRecord(
                    run_id=run.id,
                    strategy_id="ui_test",
                    client_order_id="ui-performance-buy",
                    broker_order_id="PB-ui-performance-buy",
                    symbol="000001.SZ",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=100,
                    status=OrderStatus.FILLED,
                    filled_quantity=100,
                    avg_fill_price=Decimal("10"),
                    commission=Decimal("1"),
                    stamp_tax=Decimal("0"),
                    created_at=now,
                ),
                OrderRecord(
                    run_id=run.id,
                    strategy_id="ui_test",
                    client_order_id="ui-performance-sell",
                    broker_order_id="PB-ui-performance-sell",
                    symbol="000001.SZ",
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=100,
                    status=OrderStatus.FILLED,
                    filled_quantity=100,
                    avg_fill_price=Decimal("12"),
                    commission=Decimal("1"),
                    stamp_tax=Decimal("1"),
                    created_at=now + timedelta(seconds=1),
                ),
                OrderRecord(
                    run_id=run.id,
                    strategy_id="ui_test",
                    client_order_id="ui-performance-reject",
                    broker_order_id="PB-ui-performance-reject",
                    symbol="600519.SH",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=100,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    commission=Decimal("0"),
                    stamp_tax=Decimal("0"),
                    reject_reason="risk rejected",
                    created_at=now + timedelta(seconds=2),
                ),
            ]
        )

    runs = performance_runs(modes=["paper"])
    report = run_performance_report(run.id)
    metrics = report["metrics"]

    assert runs[0]["run_id"] == run.id
    assert metrics["initial_assets"] == 100000
    assert metrics["final_assets"] == 120000
    assert round(metrics["total_return"], 4) == 0.2
    assert round(metrics["max_drawdown"], 4) == -0.1
    assert metrics["filled_order_count"] == 2
    assert metrics["rejected_order_count"] == 1
    assert metrics["completed_trade_count"] == 1
    assert metrics["win_rate"] == 1.0
    assert round(metrics["total_trade_pnl"], 2) == 197.0
    assert round(metrics["turnover"], 4) == round(2200 / ((100000 + 110000 + 99000 + 120000) / 4), 4)


def test_performance_widget_initializes_without_autostart():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-performance-widget-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add(
            AccountSnapshot(
                run_id=run.id,
                cash=100000,
                frozen_cash=0,
                market_value=0,
                total_assets=100000,
                realized_pnl=0,
                unrealized_pnl=0,
                ts=now,
            )
        )

    _ensure_qapp()
    widget = PerformanceWidget()

    assert widget.cbo_run.count() == 1
    assert "run #1" in widget.lbl_status.text()
    assert widget.tbl_orders.rowCount() == 0


def test_theme_uses_qt_available_font_stack():
    assert "-apple-system" not in QSS


def test_kline_initial_view_uses_recent_window():
    assert INITIAL_VISIBLE_BARS["30min"] < 160
    assert INITIAL_VISIBLE_BARS["daily"] >= 100
    assert _padded_range([10.0, 12.0], padding=0.1, floor_zero=False) == (9.8, 12.2)
    assert _padded_range([5.0, 5.0], padding=0.1, floor_zero=True)[0] == 0.0


def test_kline_data_source_note_labels_cache_and_refresh_states():
    assert _data_source_note("refreshed", "eastmoney_kline") == "已刷新(eastmoney_kline)"
    assert _data_source_note("cache_hit", "local_cache") == "本地缓存"
    assert _data_source_note("cache_fallback", "local_cache") == "刷新失败，使用本地缓存"
    assert _data_source_note("derived_15min_from_5min", "local_resample") == "15分由5分重采样"
    assert _data_source_note("unknown", "unknown") == ""


def test_kline_intraday_minute_note_warns_when_today_bars_missing():
    note = _intraday_minute_note(
        "5min",
        end=datetime(2026, 6, 24).date(),
        last=datetime(2026, 6, 23, 15, 0),
        now=datetime(2026, 6, 24, 13, 0),
    )

    assert note == "交易日分钟K线未包含今日数据"
    assert _intraday_minute_note(
        "daily",
        end=datetime(2026, 6, 24).date(),
        last=datetime(2026, 6, 23, 15, 0),
        now=datetime(2026, 6, 24, 13, 0),
    ) == ""


def test_market_watch_parses_symbols_and_marks_cache_quote():
    assert _parse_symbols("000001, 600519 300750") == ["000001.SZ", "600519.SH", "300750.SZ"]
    quote = Quote(
        symbol="000001.SZ",
        price=10.0,
        open=10.0,
        high=10.0,
        low=10.0,
        prev_close=10.0,
        volume=1_000_000,
        amount=10_000_000,
        ts=utcnow(),
        source="cache_daily_close",
    )

    assert _quote_status(quote)[0] == "仅展示"

    stale_ts = utcnow() - timedelta(seconds=120)
    stale = Quote(
        symbol="000001.SZ",
        price=10.0,
        open=10.0,
        high=10.0,
        low=10.0,
        prev_close=10.0,
        volume=1_000_000,
        amount=10_000_000,
        ts=stale_ts,
        fetched_at=stale_ts,
        source="test",
    )
    assert _quote_status(stale)[0] == "过期"


def test_market_watch_embeds_kline_for_selected_symbol():
    _ensure_qapp()
    now = utcnow()

    class _Loader:
        def get_quotes(self, symbols):
            return {
                sym: Quote(
                    symbol=sym,
                    price=10.0,
                    open=9.9,
                    high=10.2,
                    low=9.8,
                    prev_close=9.9,
                    volume=1_000_000,
                    amount=10_000_000,
                    ts=now,
                    fetched_at=now,
                    source="test",
                )
                for sym in symbols
            }

    widget = MarketWatchWidget()
    widget._loader = _Loader()
    calls = []

    def fake_load_symbol(symbol, *, period=None, start=None, end=None):
        calls.append({"symbol": symbol, "period": period, "start": start, "end": end})

    widget.kline_chart.load_symbol = fake_load_symbol
    widget.le_symbols.setText("000001,600519")

    widget.refresh()

    assert widget.tbl.rowCount() == 2
    assert calls[0]["symbol"] == "000001.SZ"
    assert calls[0]["period"] == "5min"

    widget.tbl.selectRow(1)

    assert calls[-1]["symbol"] == "600519.SH"
    assert calls[-1]["period"] == "5min"

    widget._chart_period_buttons["60min"].click()

    assert calls[-1]["symbol"] == "600519.SH"
    assert calls[-1]["period"] == "60min"
    assert widget.kline_chart._current_period == "60min"


def test_market_watch_chart_start_date_by_period():
    today = datetime(2026, 5, 26).date()

    assert _chart_start_date("5min", today=today) == today - timedelta(days=7)
    assert _chart_start_date("15min", today=today) == today - timedelta(days=14)
    assert _chart_start_date("60min", today=today) == today - timedelta(days=35)
    assert _chart_start_date("daily", today=today) == today - timedelta(days=365)
    assert _chart_start_date("weekly", today=today) == today - timedelta(days=365 * 3)
    assert _chart_start_date("monthly", today=today) == today - timedelta(days=365 * 8)


def test_run_control_builds_paper_command():
    cmd = build_run_paper_command(
        strategy="llm",
        symbols=["000001", "600519"],
        initial_cash=100_000,
        interval_seconds=10,
        decision_interval_minutes=15,
        once=True,
        prompt_version="v2",
        debate_rounds=2,
        threshold=0.45,
        position_pct=0.2,
        max_position_pct=0.3,
    )

    assert cmd.args[:3] == ["-m", "qtrade.cli", "run-paper"]
    assert cmd.args.count("--symbol") == 2
    assert "000001.SZ" in cmd.args
    assert "600519.SH" in cmd.args
    assert "--once" in cmd.args
    assert cmd.args[cmd.args.index("--prompt-version") + 1] == "v2"
    assert cmd.args[cmd.args.index("--decision-interval-minutes") + 1] == "15"


def test_run_control_widget_does_not_autostart_process():
    _ensure_qapp()

    widget = RunControlWidget()

    assert widget._process is None
    assert widget.btn_start.isEnabled()
    assert not widget.btn_stop.isEnabled()
    assert not widget.btn_llm_check.isEnabled()
    assert "run-paper" in widget.lbl_command.text()
    assert parse_symbol_list("000001, 600519 300750") == ["000001", "600519", "300750"]


def test_run_control_llm_preflight_reports_missing_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("QTRADE_DEEPSEEK_API_KEY", "")
    from qtrade.core import config as cfg

    cfg.get_settings.cache_clear()
    _ensure_qapp()
    widget = RunControlWidget()
    widget.cbo_strategy.setCurrentText("llm")

    widget.run_llm_preflight()

    assert widget.btn_llm_check.isEnabled()
    assert "LLM 预检失败" in widget.lbl_status.text()
    assert "DEEPSEEK_API_KEY" in widget.txt_log.toPlainText()


def test_paper_check_widget_does_not_autostart_worker():
    _ensure_qapp()

    widget = PaperCheckWidget()

    assert widget._thread is None
    assert widget.btn_run.isEnabled()
    assert not widget.btn_export_json.isEnabled()
    assert not widget.btn_export_markdown.isEnabled()
    assert widget.tbl.rowCount() == 0
    assert "未检查" in widget.lbl_summary.text()


def test_paper_check_widget_renders_report():
    _ensure_qapp()
    widget = PaperCheckWidget()
    report = PaperCheckReport(
        symbols=["000001.SZ"],
        items=[
            PaperCheckItem("实时行情", "PASS", "ok"),
            PaperCheckItem("模拟盘健康", "WARN", "等待 tick"),
        ],
    )

    widget._render_report(report)

    assert widget.tbl.rowCount() == 2
    assert widget.tbl.item(0, 0).text() == "PASS"
    assert widget.tbl.item(1, 1).text() == "模拟盘健康"
    assert "WARN=1" in widget.lbl_summary.text()
    assert widget.btn_export_json.isEnabled()
    assert widget.btn_export_markdown.isEnabled()
    assert _status_color("FAIL") == "#d04949"


def test_paper_check_widget_exports_latest_report(tmp_path):
    _ensure_qapp()
    widget = PaperCheckWidget()
    report = PaperCheckReport(
        symbols=["000001.SZ"],
        items=[PaperCheckItem("实时行情", "PASS", "ok")],
    )
    widget._render_report(report)

    json_path = widget.export_json(tmp_path / "paper-check.json")
    md_path = widget.export_markdown(tmp_path / "paper-check.md")

    assert json_path is not None
    assert md_path is not None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["symbols"] == ["000001.SZ"]
    assert payload["overall_status"] == "PASS"
    markdown = md_path.read_text(encoding="utf-8")
    assert "# 模拟实盘验收报告" in markdown
    assert "Markdown 已导出" in widget.lbl_export_status.text()


def test_recent_orders_returns_local_display_time():
    init_db()
    utc_ts = datetime(2026, 5, 9, 16, 30)
    expected_local = utc_ts.replace(tzinfo=timezone.utc).astimezone(MARKET_TZ).replace(tzinfo=None)
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-time-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=utc_ts,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add(
            OrderRecord(
                run_id=run.id,
                strategy_id="ui_test",
                client_order_id="ui-time-order",
                broker_order_id="PB-ui-time",
                symbol="000001.SZ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                avg_fill_price=Decimal("10"),
                commission=Decimal("0"),
                stamp_tax=Decimal("0"),
                created_at=utc_ts,
            )
        )

    order = recent_orders(limit=1)[0]

    assert order["ts"] == expected_local


def test_recent_decisions_includes_execution_summary():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-decision-exec-run",
            strategy_id="llm_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        filled_decision = AgentDecision(
            run_id=run.id,
            strategy_id="llm_test",
            symbol="000001.SZ",
            action=DecisionAction.BUY,
            confidence=Decimal("0.8000"),
            target_position_pct=Decimal("0.2000"),
            reasoning="AI buy",
            trace_json={},
            ts=now,
        )
        blocked_decision = AgentDecision(
            run_id=run.id,
            strategy_id="llm_test",
            symbol="600519.SH",
            action=DecisionAction.BUY,
            confidence=Decimal("0.7000"),
            target_position_pct=Decimal("0.5000"),
            reasoning="AI oversized buy",
            trace_json={},
            risk_passed=False,
            risk_reject_reason="仓位超过上限",
            ts=now + timedelta(seconds=1),
        )
        s.add_all([filled_decision, blocked_decision])
        s.flush()
        s.add(
            OrderRecord(
                run_id=run.id,
                strategy_id="llm_test",
                client_order_id="ui-decision-exec-order",
                broker_order_id="PB-ui-decision-exec",
                symbol="000001.SZ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1000,
                status=OrderStatus.FILLED,
                filled_quantity=1000,
                avg_fill_price=Decimal("10"),
                commission=Decimal("0"),
                stamp_tax=Decimal("0"),
                created_at=now,
                decision_id=filled_decision.id,
            )
        )

    rows = recent_decisions(run.id, limit=10)
    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["000001.SZ"]["execution_status"] == "FILLED"
    assert by_symbol["000001.SZ"]["execution_summary"] == "已成交 1000 股"
    assert by_symbol["600519.SH"]["execution_status"] == "BLOCKED"
    assert "仓位超过上限" in by_symbol["600519.SH"]["execution_summary"]


def test_agent_log_shows_today_llm_usage():
    init_db()
    now = utcnow()
    with session_scope() as s:
        s.add(
            AgentDecision(
                strategy_id="llm_test",
                symbol="000001.SZ",
                action=DecisionAction.HOLD,
                confidence=Decimal("0.1000"),
                target_position_pct=Decimal("0.0000"),
                reasoning="budget display",
                trace_json={},
                total_tokens=1234,
                total_cost_usd=Decimal("0.0123"),
                ts=now,
            )
        )

    _ensure_qapp()
    widget = AgentLogWidget()
    widget.refresh()

    assert "今日 AI" in widget.lbl_status.text()
    assert "1/" in widget.lbl_status.text()
    assert "1,234/" in widget.lbl_status.text()
    assert "$0.0123" in widget.lbl_status.text()


def test_recent_tick_logs_returns_display_rows():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-tick-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add(
            TickLog(
                run_id=run.id,
                strategy_id="ui_test",
                ts=now,
                quotes_count=1,
                tradeable_quotes=1,
                fallback_quotes=0,
                signals_generated=0,
                strategy_skipped=1,
                orders_placed=0,
                risk_rejected=0,
                symbols_json=["000001.SZ"],
                per_symbol_json=[{"symbol": "000001.SZ", "skip_reason": "decision_interval"}],
                errors_json=[],
            )
        )

    rows = recent_tick_logs(limit=1)

    assert rows[0]["strategy_id"] == "ui_test"
    assert rows[0]["symbols"] == ["000001.SZ"]
    assert rows[0]["per_symbol"][0]["skip_reason"] == "decision_interval"
    assert rows[0]["latest_reason"] == "decision_interval"


def test_recent_tick_logs_extracts_order_issue_reason():
    init_db()
    now = utcnow()
    long_reason = (
        "price_deviation 100.0bp exceeds 50bp "
        "(reference=10.000, quote=10.100)"
    )
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-tick-reason-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add(
            TickLog(
                run_id=run.id,
                strategy_id="ui_test",
                ts=now,
                quotes_count=1,
                tradeable_quotes=1,
                fallback_quotes=0,
                signals_generated=1,
                strategy_skipped=0,
                orders_placed=0,
                risk_rejected=1,
                symbols_json=["000001.SZ"],
                per_symbol_json=[
                    {
                        "symbol": "000001.SZ",
                        "order_rejections": [
                            {"kind": "risk", "action": "BUY", "quantity": 1000, "reason": long_reason}
                        ],
                    }
                ],
                errors_json=[f"risk[000001.SZ BUY x1000]: {long_reason}"],
            )
        )

    row = recent_tick_logs(limit=1)[0]

    assert row["latest_reason"].startswith("price_deviation 100.0bp")
    assert len(row["latest_reason"]) <= 72


def test_market_data_quality_summarizes_recent_tick_logs():
    init_db()
    now = utcnow()
    with session_scope() as s:
        run = StrategyRun(
            run_uid="ui-quality-summary-run",
            strategy_id="ui_test",
            strategy_version="test",
            mode="paper",
            started_at=now,
            config_json={},
        )
        s.add(run)
        s.flush()
        s.add_all(
            [
                TickLog(
                    run_id=run.id,
                    strategy_id="ui_test",
                    ts=now,
                    quotes_count=2,
                    tradeable_quotes=2,
                    fallback_quotes=0,
                    strategy_skipped=0,
                    symbols_json=["000001.SZ", "600519.SH"],
                    per_symbol_json=[],
                    errors_json=[],
                ),
                TickLog(
                    run_id=run.id,
                    strategy_id="ui_test",
                    ts=now,
                    quotes_count=2,
                    tradeable_quotes=1,
                    fallback_quotes=1,
                    strategy_skipped=1,
                    symbols_json=["000001.SZ", "600519.SH"],
                    per_symbol_json=[
                        {"symbol": "600519.SH", "skip_reason": "non_tradeable_quote"}
                    ],
                    errors_json=[],
                ),
            ]
        )

    quality = market_data_quality(run.id, limit=20)

    assert quality["window_ticks"] == 2
    assert quality["quotes_total"] == 4
    assert quality["tradeable_total"] == 3
    assert quality["fallback_total"] == 1
    assert quality["tradeable_ratio"] == 0.75
    assert quality["top_skip_reasons"] == [{"reason": "non_tradeable_quote", "count": 1}]


def test_orders_widget_does_not_fetch_quote_while_hidden(monkeypatch):
    init_db()
    _ensure_qapp()
    calls = 0

    def fake_get_quote(self, symbol):
        nonlocal calls
        calls += 1
        raise AssertionError("hidden widget should not fetch quotes")

    monkeypatch.setattr("qtrade.runtime.manual_trading.ManualTradingService.get_quote", fake_get_quote)

    OrdersWidget()

    assert calls == 0


def test_orders_widget_uses_last_displayed_quote_as_expected_price(monkeypatch):
    init_db()
    _ensure_qapp()

    quote = Quote(
        symbol="000001.SZ",
        price=10.0,
        open=10.0,
        high=10.2,
        low=9.9,
        prev_close=9.9,
        volume=1_000_000,
        amount=10_000_000,
        ts=utcnow(),
        source="test",
    )

    monkeypatch.setattr("qtrade.runtime.manual_trading.ManualTradingService.get_quote", lambda self, symbol: quote)

    widget = OrdersWidget()
    widget.le_manual_symbol.setText("000001")
    widget._refresh_manual_quote()

    assert widget._expected_manual_price("000001") == 10.0
    assert widget._expected_manual_price("600519") is None
    assert _bps_to_pct_text(50) == "0.50%"
    assert "价格偏离保护 ≤ 0.50%" in widget.lbl_price_guard.text()
