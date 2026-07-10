"""CLI 参数和启动前检查测试。"""
from __future__ import annotations

import json
import subprocess
from datetime import date

import pytest
import typer
from typer.testing import CliRunner

from qtrade.cli import (
    _build_factor_strategy_param_grid,
    _checkpoint_csv_writer,
    _parse_param_grid,
    _validate_common_run_options,
    _validate_llm_run_options,
    app,
)
from qtrade.core import config as cfg


def test_llm_run_rejects_invalid_prompt_version():
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["run", "--strategy", "llm", "--prompt-version", "v9", "--once"],
    )

    assert result.exit_code != 0
    assert "prompt_version" in result.output


def test_llm_run_missing_key_fails_before_start(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("QTRADE_DEEPSEEK_API_KEY", "")
    cfg.get_settings.cache_clear()
    runner = CliRunner()

    result = runner.invoke(app, ["run", "--strategy", "llm", "--symbol", "000001", "--once"])

    assert result.exit_code == 1
    assert "DEEPSEEK_API_KEY" in result.output
    assert "LiveRunner 启动" not in result.output


def test_run_paper_help_exposes_llm_safety_parameters():
    runner = CliRunner()

    result = runner.invoke(app, ["run-paper", "--help"])

    assert result.exit_code == 0
    assert "--decision-interval-minutes" in result.output
    assert "--position-pct" in result.output
    assert "--max-position-pct" in result.output


def test_llm_check_help_exposes_preflight_parameters():
    runner = CliRunner()

    result = runner.invoke(app, ["llm-check", "--help"])

    assert result.exit_code == 0
    assert "--prompt-version" in result.output
    assert "--decision-interval-minutes" in result.output
    assert "--position-pct" in result.output


def test_llm_check_missing_key_fails_before_llm_call(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("QTRADE_DEEPSEEK_API_KEY", "")
    cfg.get_settings.cache_clear()
    runner = CliRunner()

    result = runner.invoke(app, ["llm-check", "--symbol", "000001"])

    assert result.exit_code == 1
    assert "DEEPSEEK_API_KEY" in result.output
    assert "LLM 模拟盘启动前检查" in result.output


def test_paper_smoke_help_exposes_isolation_flag():
    runner = CliRunner()

    result = runner.invoke(app, ["paper-smoke", "--help"])

    assert result.exit_code == 0
    assert "--write-db" in result.output
    assert "--symbol" in result.output


def test_paper_smoke_runs_in_isolated_temp_db_by_default():
    runner = CliRunner()

    result = runner.invoke(app, ["paper-smoke", "--symbol", "000001"])

    assert result.exit_code == 0
    assert "模拟实盘链路自检" in result.output
    assert "临时隔离库" in result.output
    assert "PASS=" in result.output


def test_paper_check_help_exposes_safe_run_once_controls():
    runner = CliRunner()

    result = runner.invoke(app, ["paper-check", "--help"])

    assert result.exit_code == 0
    assert "--skip-bars" in result.output
    assert "--run-once" in result.output
    assert "--period" in result.output
    assert "--json-output" in result.output
    assert "--markdown-output" in result.output


def test_data_check_help_exposes_health_options():
    runner = CliRunner()

    result = runner.invoke(app, ["data-check", "--help"])

    assert result.exit_code == 0
    assert "--period" in result.output
    assert "--skip-quotes" in result.output
    assert "--force-refresh" in result.output
    assert "--json-output" in result.output


def test_data_check_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.data import health as health_mod
    from qtrade.data.health import DataHealthItem, DataHealthReport

    def _fake_run_data_health_check(**kwargs):
        assert kwargs["symbols"] == ["000001"]
        assert kwargs["periods"] == ["daily", "15min"]
        assert kwargs["start"] == "2026-01-01"
        assert kwargs["end"] == "2026-01-31"
        assert kwargs["include_quotes"] is False
        assert kwargs["force_refresh"] is True
        return DataHealthReport(
            symbols=["000001.SZ"],
            periods=["daily", "15min"],
            start=date(2026, 1, 1),
            end=date(2026, 1, 31),
            adjust="qfq",
            include_quotes=False,
            force_refresh=True,
            items=[DataHealthItem("K 线 000001.SZ 日", "PASS", "10 根")],
        )

    monkeypatch.setattr(health_mod, "run_data_health_check", _fake_run_data_health_check)
    runner = CliRunner()
    json_path = tmp_path / "data-health.json"
    md_path = tmp_path / "data-health.md"

    result = runner.invoke(
        app,
        [
            "data-check",
            "--symbol",
            "000001",
            "--period",
            "daily",
            "--period",
            "15min",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
            "--skip-quotes",
            "--force-refresh",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
    )

    assert result.exit_code == 0
    assert "行情源健康检查" in result.output
    assert "data-health.json" in result.output
    assert "data-health.md" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "PASS"
    assert payload["periods"] == ["daily", "15min"]
    assert md_path.read_text(encoding="utf-8").startswith("# 行情源健康检查报告")


def test_net_check_help_exposes_network_options():
    runner = CliRunner()

    result = runner.invoke(app, ["net-check", "--help"])

    assert result.exit_code == 0
    assert "--period" in result.output
    assert "--timeout" in result.output
    assert "--json-output" in result.output


def test_net_check_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.data import network_check as net_mod
    from qtrade.data.network_check import NetworkCheckItem, NetworkCheckReport

    def _fake_run_network_check(**kwargs):
        assert kwargs["symbol"] == "000001"
        assert kwargs["periods"] == ["daily"]
        assert kwargs["timeout_s"] == 2.0
        return NetworkCheckReport(
            symbol="000001.SZ",
            timeout_s=2.0,
            periods=["daily"],
            items=[NetworkCheckItem("EastMoney K线 daily", "PASS", "OK", "接口可达", 12.0)],
        )

    monkeypatch.setattr(net_mod, "run_network_check", _fake_run_network_check)
    runner = CliRunner()
    json_path = tmp_path / "net.json"

    result = runner.invoke(
        app,
        [
            "net-check",
            "--symbol",
            "000001",
            "--period",
            "daily",
            "--timeout",
            "2",
            "--json-output",
            str(json_path),
        ],
    )

    assert result.exit_code == 0
    assert "行情网络诊断" in result.output
    assert "net.json" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "PASS"
    assert payload["items"][0]["category"] == "OK"


def test_doctor_help_exposes_basic_readiness_options():
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    assert "--period" in result.output
    assert "--timeout" in result.output
    assert "--skip-smoke" in result.output
    assert "--json-output" in result.output
    assert "--verbose-logs" in result.output


def test_doctor_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.runtime import doctor as doctor_mod
    from qtrade.runtime.doctor import DoctorItem, DoctorReport

    def _fake_run_doctor(**kwargs):
        assert kwargs["symbol"] == "000001"
        assert kwargs["periods"] == ["daily"]
        assert kwargs["timeout_s"] == 2.0
        assert kwargs["run_smoke"] is False
        return DoctorReport(
            symbol="000001.SZ",
            periods=["daily"],
            items=[DoctorItem("LLM/DeepSeek", "PASS", "doctor 不调用 LLM API")],
        )

    monkeypatch.setattr(doctor_mod, "run_doctor", _fake_run_doctor)
    runner = CliRunner()
    json_path = tmp_path / "doctor.json"

    result = runner.invoke(
        app,
        [
            "doctor",
            "--symbol",
            "000001",
            "--period",
            "daily",
            "--timeout",
            "2",
            "--skip-smoke",
            "--json-output",
            str(json_path),
        ],
    )

    assert result.exit_code == 0
    assert "基础验收 Doctor" in result.output
    assert "doctor.json" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "PASS"
    assert payload["items"][0]["name"] == "LLM/DeepSeek"


def test_paper_check_cli_prints_report(monkeypatch, tmp_path):
    from qtrade import persistence as persistence_mod
    from qtrade.runtime import paper_check as paper_check_mod
    from qtrade.runtime.paper_check import PaperCheckItem, PaperCheckReport

    captured = {"init_db": False}

    def _fake_init_db():
        captured["init_db"] = True

    def _fake_run_paper_check(**kwargs):
        assert kwargs["symbols"] == ["000001"]
        assert kwargs["include_bars"] is False
        assert kwargs["run_once"] is False
        return PaperCheckReport(
            symbols=["000001.SZ"],
            items=[PaperCheckItem("实时行情", "PASS", "ok")],
        )

    monkeypatch.setattr(persistence_mod, "init_db", _fake_init_db)
    monkeypatch.setattr(paper_check_mod, "run_paper_check", _fake_run_paper_check)
    runner = CliRunner()
    json_path = tmp_path / "report.json"

    result = runner.invoke(app, ["paper-check", "--skip-bars", "--json-output", str(json_path)])

    assert result.exit_code == 0
    assert "模拟实盘验收" in result.output
    assert "PASS=1" in result.output
    assert "report.json" in result.output
    assert captured["init_db"] is True
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "PASS"
    assert payload["items"][0]["name"] == "实时行情"


def test_backtest_data_error_is_concise(monkeypatch):
    from qtrade.backtest import engine as engine_mod
    from qtrade.core.exceptions import DataError

    class _FailingBacktestEngine:
        def __init__(self, **kwargs):
            pass

        def run(self):
            raise DataError("daily bars unavailable")

    monkeypatch.setattr(engine_mod, "BacktestEngine", _FailingBacktestEngine)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "ma_cross",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )

    assert result.exit_code != 0
    assert "daily bars unavailable" in result.output
    assert "Traceback" not in result.output


def test_backtest_compare_help_exposes_export_options():
    runner = CliRunner()

    result = runner.invoke(app, ["backtest-compare", "--help"])

    assert result.exit_code == 0
    assert "--strategy" in result.output
    assert "--keep-trades" in result.output
    assert "--continue-on-error" in result.output
    assert "--checkpoint-csv" in result.output
    assert "--resume-from-check" in result.output
    assert "--csv-output" in result.output
    assert "--markdown-output" in result.output


def test_backtest_compare_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.backtest import comparison as comparison_mod
    from qtrade.backtest.comparison import BacktestComparison, BacktestSummary
    from qtrade.backtest.metrics import Metrics

    def _fake_run_backtest_comparison(**kwargs):
        assert kwargs["strategies"] == ["ma_cross", "macd"]
        assert kwargs["symbols"] == ["000001.SZ"]
        assert kwargs["keep_trades"] is False
        assert kwargs["continue_on_error"] is False
        assert kwargs["skip_task_keys"] == set()
        assert callable(kwargs["progress_callback"])
        kwargs["progress_callback"](1, 2, {"strategy": "ma_cross"})
        return BacktestComparison(
            rows=[
                BacktestSummary(
                    strategy_name="ma_cross",
                    strategy_id="ma_cross_v1",
                    symbols=["000001.SZ"],
                    start=date(2024, 1, 1),
                    end=date(2024, 1, 31),
                    metrics=Metrics(
                        total_return=0.1,
                        annual_return=0.2,
                        annual_volatility=0.3,
                        sharpe=0.4,
                        sortino=0.5,
                        calmar=0.6,
                        max_drawdown=-0.07,
                        max_drawdown_duration_days=3,
                        win_rate=0.8,
                        profit_loss_ratio=1.2,
                        n_trades=4,
                        final_equity=110_000.0,
                        initial_equity=100_000.0,
                    ),
                    trades=[],
                )
            ],
        )

    monkeypatch.setattr(comparison_mod, "run_backtest_comparison", _fake_run_backtest_comparison)
    runner = CliRunner()
    csv_path = tmp_path / "comparison.csv"
    md_path = tmp_path / "comparison.md"

    result = runner.invoke(
        app,
        [
            "backtest-compare",
            "--strategy",
            "ma_cross",
            "--strategy",
            "macd",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--csv-output",
            str(csv_path),
            "--markdown-output",
            str(md_path),
        ],
    )

    assert result.exit_code == 0
    assert "多策略回测对比" in result.output
    assert "批量质量" in result.output
    assert "comparison.csv" in result.output
    assert "comparison.md" in result.output
    assert csv_path.read_text(encoding="utf-8").startswith("strategy,strategy_id")
    assert md_path.read_text(encoding="utf-8").startswith("# 回测策略对比报告")


def test_backtest_grid_help_exposes_grid_options():
    runner = CliRunner()

    result = runner.invoke(app, ["backtest-grid", "--help"])

    assert result.exit_code == 0
    assert "--param" in result.output
    assert "--sort-by" in result.output
    assert "--top" in result.output
    assert "--max-param" in result.output
    assert "--keep-trades" in result.output
    assert "--continue-on-error" in result.output
    assert "--checkpoint-csv" in result.output
    assert "--resume-from-check" in result.output
    assert "--csv-output" in result.output


def test_backtest_grid_resource_guard_blocks_large_grid_before_run():
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "backtest-grid",
            "--param",
            "fast_window=5,10",
            "--param",
            "slow_window=20",
            "--max-param-combinations",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "参数组合 2 超过资源保护上限" in result.output
    assert "--max-param-combinations" in result.output


def test_backtest_grid_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.backtest import comparison as comparison_mod
    from qtrade.backtest.comparison import BacktestComparison, BacktestSummary
    from qtrade.backtest.metrics import Metrics

    def _metrics(total_return: float, sharpe: float) -> Metrics:
        return Metrics(
            total_return=total_return,
            annual_return=total_return,
            annual_volatility=0.2,
            sharpe=sharpe,
            sortino=sharpe,
            calmar=sharpe,
            max_drawdown=-0.05,
            max_drawdown_duration_days=2,
            win_rate=0.5,
            profit_loss_ratio=1.0,
            n_trades=2,
            final_equity=100_000.0 * (1 + total_return),
            initial_equity=100_000.0,
        )

    def _fake_run_backtest_param_grid(**kwargs):
        assert kwargs["strategy_name"] == "ma_cross"
        assert kwargs["symbols"] == ["000001.SZ"]
        assert kwargs["param_grid"] == {
            "fast_window": [5, 10],
            "slow_window": [20],
        }
        assert kwargs["keep_trades"] is False
        assert kwargs["continue_on_error"] is False
        assert kwargs["skip_task_keys"] == set()
        assert callable(kwargs["progress_callback"])
        rows = [
            BacktestSummary(
                strategy_name="ma_cross",
                strategy_id="ma_cross_v1",
                symbols=["000001.SZ"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                metrics=_metrics(0.02, 0.2),
                trades=[],
                params={"fast_window": 5, "slow_window": 20},
            ),
            BacktestSummary(
                strategy_name="ma_cross",
                strategy_id="ma_cross_v1",
                symbols=["000001.SZ"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                metrics=_metrics(0.08, 0.9),
                trades=[],
                params={"fast_window": 10, "slow_window": 20},
            ),
        ]
        kwargs["progress_callback"](
            1,
            2,
            {"strategy": "ma_cross", "params": {"fast_window": 5}, "summary": rows[0]},
        )
        return BacktestComparison(rows=rows)

    monkeypatch.setattr(comparison_mod, "run_backtest_param_grid", _fake_run_backtest_param_grid)
    runner = CliRunner()
    csv_path = tmp_path / "grid.csv"
    md_path = tmp_path / "grid.md"
    checkpoint_path = tmp_path / "grid-checkpoint.csv"
    experiment_log = tmp_path / "experiments.jsonl"

    result = runner.invoke(
        app,
        [
            "backtest-grid",
            "--strategy",
            "ma_cross",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--param",
            "fast_window=5,10",
            "--param",
            "slow_window=20",
            "--sort-by",
            "sharpe",
            "--top",
            "1",
            "--csv-output",
            str(csv_path),
            "--markdown-output",
            str(md_path),
            "--checkpoint-csv",
            str(checkpoint_path),
            "--experiment-log",
            str(experiment_log),
        ],
    )

    assert result.exit_code == 0
    assert "多策略回测对比" in result.output
    assert "批量质量" in result.output
    assert "参数敏感性" in result.output
    assert "grid.csv" in result.output
    assert "grid.md" in result.output
    assert "grid-checkpoint.csv" in result.output
    assert "experiments.jsonl" in result.output
    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
    assert "params" in csv_text.splitlines()[0]
    assert 'fast_window"":10' in csv_text
    assert 'fast_window"":5' not in csv_text
    assert "| 策略 | 参数 | 总收益 |" in md_text
    assert "## 参数敏感性" in md_text
    assert checkpoint_text.startswith("done,total,fold,stage,status,error_type,error,strategy,strategy_id")
    assert 'fast_window"":5' in checkpoint_text
    experiment = json.loads(experiment_log.read_text(encoding="utf-8").splitlines()[0])
    assert experiment["record_type"] == "backtest_grid"
    assert experiment["strategy"] == "ma_cross"
    assert experiment["best_by_sharpe"]["params"]["fast_window"] == 10
    assert experiment["parameter_sensitivity"]


def test_checkpoint_csv_writer_records_failures(tmp_path):
    from qtrade.backtest.comparison import BacktestFailure

    checkpoint_path = tmp_path / "failures.csv"
    failure = BacktestFailure(
        strategy_name="ma_cross",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        params={"fast_window": 20, "slow_window": 20},
        error_type="ValueError",
        error="fast_window must be < slow_window",
    )

    with _checkpoint_csv_writer(checkpoint_path) as callback:
        assert callback is not None
        callback(1, 1, {"strategy": "ma_cross", "failure": failure})

    text = checkpoint_path.read_text(encoding="utf-8")
    assert "failed" in text
    assert "ValueError" in text
    assert "fast_window must be < slow_window" in text


def test_backtest_grid_cli_resumes_from_checkpoint(monkeypatch, tmp_path):
    from qtrade.backtest import comparison as comparison_mod
    from qtrade.backtest.comparison import BacktestComparison, BacktestSummary
    from qtrade.backtest.metrics import Metrics

    def _metrics(total_return: float, sharpe: float) -> Metrics:
        return Metrics(
            total_return=total_return,
            annual_return=total_return,
            annual_volatility=0.2,
            sharpe=sharpe,
            sortino=sharpe,
            calmar=sharpe,
            max_drawdown=-0.05,
            max_drawdown_duration_days=2,
            win_rate=0.5,
            profit_loss_ratio=1.0,
            n_trades=2,
            final_equity=100_000.0 * (1 + total_return),
            initial_equity=100_000.0,
        )

    old_summary = BacktestSummary(
        strategy_name="ma_cross",
        strategy_id="ma_cross_v1",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        metrics=_metrics(0.02, 0.2),
        trades=[],
        params={"fast_window": 5, "slow_window": 20},
    )
    new_summary = BacktestSummary(
        strategy_name="ma_cross",
        strategy_id="ma_cross_v1",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        metrics=_metrics(0.08, 0.9),
        trades=[],
        params={"fast_window": 10, "slow_window": 20},
    )
    checkpoint_path = tmp_path / "grid-checkpoint.csv"
    with _checkpoint_csv_writer(checkpoint_path) as callback:
        assert callback is not None
        callback(1, 2, {"strategy": "ma_cross", "params": old_summary.params, "summary": old_summary})

    captured = {}

    def _fake_run_backtest_param_grid(**kwargs):
        captured.update(kwargs)
        assert len(kwargs["skip_task_keys"]) == 1
        assert callable(kwargs["progress_callback"])
        kwargs["progress_callback"](
            1,
            2,
            {"strategy": "ma_cross", "params": old_summary.params, "skipped": True},
        )
        kwargs["progress_callback"](
            2,
            2,
            {"strategy": "ma_cross", "params": new_summary.params, "summary": new_summary},
        )
        return BacktestComparison(rows=[new_summary])

    monkeypatch.setattr(comparison_mod, "run_backtest_param_grid", _fake_run_backtest_param_grid)
    runner = CliRunner()
    csv_path = tmp_path / "grid.csv"

    result = runner.invoke(
        app,
        [
            "backtest-grid",
            "--strategy",
            "ma_cross",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--param",
            "fast_window=5,10",
            "--param",
            "slow_window=20",
            "--top",
            "0",
            "--checkpoint-csv",
            str(checkpoint_path),
            "--resume-from-checkpoint",
            "--csv-output",
            str(csv_path),
        ],
    )

    assert result.exit_code == 0
    assert "从 checkpoint 恢复已完成结果：1 条" in result.output
    assert "grid.csv" in result.output
    assert captured["skip_task_keys"]
    csv_text = csv_path.read_text(encoding="utf-8")
    assert 'fast_window"":5' in csv_text
    assert 'fast_window"":10' in csv_text
    checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
    assert checkpoint_text.count("done,total,fold,stage") == 1
    assert 'fast_window"":5' in checkpoint_text
    assert 'fast_window"":10' in checkpoint_text


def test_parse_param_grid_casts_values_and_rejects_bad_format():
    assert _parse_param_grid(["fast_window=5,10", "position_pct=0.2", "enabled=true"]) == {
        "fast_window": [5, 10],
        "position_pct": [0.2],
        "enabled": [True],
    }
    with pytest.raises(ValueError, match="参数格式错误"):
        _parse_param_grid(["fast_window"])


def test_factor_mine_help_exposes_research_options():
    runner = CliRunner()

    result = runner.invoke(app, ["factor-mine", "--help"])

    assert result.exit_code == 0
    assert "--window" in result.output
    assert "--include-interactions" in result.output
    assert "--factor-sort-by" in result.output
    assert "--csv-output" in result.output


def test_factor_mine_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.backtest import factors as factor_mod
    from qtrade.backtest.factors import FactorMiningReport, FactorScore, FactorWindowScore

    def _fake_run_factor_mining(**kwargs):
        assert kwargs["symbols"] == ["000001.SZ"]
        assert kwargs["windows"] == [5, 20]
        assert kwargs["horizon_days"] == 5
        assert kwargs["include_interactions"] is True
        return FactorMiningReport(
            symbols=["000001.SZ"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            horizon_days=5,
            train_ratio=0.7,
            rows=[
                FactorScore(
                    name="ret_5",
                    expression="ret_5",
                    horizon_days=5,
                    train=FactorWindowScore(20, 0.1, 0.2, 0.03, -0.01, 0.04),
                    test=FactorWindowScore(10, 0.2, 0.3, 0.04, -0.02, 0.06),
                    train_start=date(2024, 1, 1),
                    train_end=date(2024, 1, 20),
                    test_start=date(2024, 1, 21),
                    test_end=date(2024, 1, 31),
                )
            ],
        )

    monkeypatch.setattr(factor_mod, "run_factor_mining", _fake_run_factor_mining)
    runner = CliRunner()
    csv_path = tmp_path / "factors.csv"
    md_path = tmp_path / "factors.md"

    result = runner.invoke(
        app,
        [
            "factor-mine",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--window",
            "5",
            "--window",
            "20",
            "--horizon-days",
            "5",
            "--include-interactions",
            "--top",
            "1",
            "--csv-output",
            str(csv_path),
            "--markdown-output",
            str(md_path),
        ],
    )

    assert result.exit_code == 0
    assert "因子挖掘" in result.output
    assert "ret_5" in result.output
    assert "factors.csv" in result.output
    assert "factors.md" in result.output
    assert csv_path.read_text(encoding="utf-8").startswith("factor,expression")
    assert md_path.read_text(encoding="utf-8").startswith("# 因子挖掘报告")


def test_factor_layer_help_exposes_layer_options():
    runner = CliRunner()

    result = runner.invoke(app, ["factor-layer", "--help"])

    assert result.exit_code == 0
    assert "--factor" in result.output
    assert "--groups" in result.output
    assert "--window" in result.output
    assert "--csv-output" in result.output


def test_factor_layer_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.backtest import factors as factor_mod
    from qtrade.backtest.factors import (
        FactorLayerBucket,
        FactorLayerReport,
        FactorLayerWindow,
    )

    def _fake_run_factor_layering(**kwargs):
        assert kwargs["symbols"] == ["000001.SZ"]
        assert kwargs["factor_name"] == "ret_5"
        assert kwargs["groups"] == 3
        return FactorLayerReport(
            symbols=["000001.SZ"],
            factor="ret_5",
            expression="ret_5",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            horizon_days=5,
            train_ratio=0.7,
            groups=3,
            train=FactorLayerWindow(
                n_obs=30,
                rank_ic=0.2,
                top_return=0.03,
                bottom_return=-0.01,
                spread_return=0.04,
                monotonicity=0.8,
                turnover=0.5,
                buckets=[
                    FactorLayerBucket(1, 10, -0.01),
                    FactorLayerBucket(2, 10, 0.01),
                    FactorLayerBucket(3, 10, 0.03),
                ],
            ),
            test=FactorLayerWindow(
                n_obs=15,
                rank_ic=0.3,
                top_return=0.04,
                bottom_return=-0.02,
                spread_return=0.06,
                monotonicity=1.0,
                turnover=0.4,
                buckets=[
                    FactorLayerBucket(1, 5, -0.02),
                    FactorLayerBucket(2, 5, 0.00),
                    FactorLayerBucket(3, 5, 0.04),
                ],
            ),
        )

    monkeypatch.setattr(factor_mod, "run_factor_layering", _fake_run_factor_layering)
    runner = CliRunner()
    csv_path = tmp_path / "layer.csv"
    md_path = tmp_path / "layer.md"

    result = runner.invoke(
        app,
        [
            "factor-layer",
            "--factor",
            "ret_5",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--groups",
            "3",
            "--csv-output",
            str(csv_path),
            "--markdown-output",
            str(md_path),
        ],
    )

    assert result.exit_code == 0
    assert "因子分层验证" in result.output
    assert "ret_5" in result.output
    assert "layer.csv" in result.output
    assert "layer.md" in result.output
    assert csv_path.read_text(encoding="utf-8").startswith("factor,expression,window,bucket")
    assert md_path.read_text(encoding="utf-8").startswith("# 因子分层验证报告")


def test_factor_backtest_help_exposes_signal_options():
    runner = CliRunner()

    result = runner.invoke(app, ["factor-backtest", "--help"])

    assert result.exit_code == 0
    assert "--factor" in result.output
    assert "--entry-quantile" in result.output
    assert "--exit-quantile" in result.output
    assert "--direction" in result.output


def test_factor_backtest_cli_runs_without_live_network(monkeypatch):
    from qtrade.backtest import engine as engine_mod
    from qtrade.backtest.engine import BacktestResult
    from qtrade.backtest.metrics import Metrics

    captured = {}

    class _FakeBacktestEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return BacktestResult(
                metrics=Metrics(
                    total_return=0.05,
                    annual_return=0.1,
                    annual_volatility=0.2,
                    sharpe=0.5,
                    sortino=0.6,
                    calmar=0.7,
                    max_drawdown=-0.03,
                    max_drawdown_duration_days=2,
                    win_rate=0.5,
                    profit_loss_ratio=1.2,
                    n_trades=2,
                    final_equity=105_000.0,
                    initial_equity=100_000.0,
                ),
                equity_curve=[],
                trades=[],
                final_account={},
            )

    monkeypatch.setattr(engine_mod, "BacktestEngine", _FakeBacktestEngine)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "factor-backtest",
            "--factor",
            "ret_5",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--window",
            "5",
            "--lookback",
            "20",
            "--entry-quantile",
            "0.8",
            "--exit-quantile",
            "0.5",
            "--direction",
            "high",
        ],
    )

    assert result.exit_code == 0
    assert "回测结果" in result.output
    assert "factor=ret_5" in result.output
    assert captured["strategy"].factor_name == "ret_5"
    assert captured["strategy"].windows == [5]
    assert captured["initial_cash"] == 100_000.0


def test_factor_grid_help_exposes_grid_options():
    runner = CliRunner()

    result = runner.invoke(app, ["factor-grid", "--help"])

    assert result.exit_code == 0
    assert "--factor" in result.output
    assert "--lookback" in result.output
    assert "--entry-quantile" in result.output
    assert "--position-pct" in result.output
    assert "--max-symbols" in result.output
    assert "--max-param" in result.output
    assert "--keep-trades" in result.output
    assert "--continue-on-error" in result.output
    assert "--checkpoint-csv" in result.output
    assert "--resume-from-check" in result.output
    assert "--csv-output" in result.output
    assert "--experiment-log" in result.output


def test_factor_grid_resource_guard_blocks_large_grid_before_run():
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "factor-grid",
            "--factor",
            "ret_5",
            "--entry-quantile",
            "0.7",
            "--entry-quantile",
            "0.8",
            "--exit-quantile",
            "0.5",
            "--position-pct",
            "0.2",
            "--max-param-combinations",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "参数组合 4 超过资源保护上限" in result.output
    assert "--max-param-combinations" in result.output


def test_factor_grid_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.backtest import comparison as comparison_mod
    from qtrade.backtest.comparison import BacktestComparison, BacktestSummary
    from qtrade.backtest.metrics import Metrics

    def _metrics(total_return: float, sharpe: float) -> Metrics:
        return Metrics(
            total_return=total_return,
            annual_return=total_return,
            annual_volatility=0.2,
            sharpe=sharpe,
            sortino=sharpe,
            calmar=sharpe,
            max_drawdown=-0.05,
            max_drawdown_duration_days=2,
            win_rate=0.5,
            profit_loss_ratio=1.0,
            n_trades=2,
            final_equity=100_000.0 * (1 + total_return),
            initial_equity=100_000.0,
        )

    def _fake_run_backtest_param_grid(**kwargs):
        assert kwargs["strategy_name"] == "factor"
        assert kwargs["symbols"] == ["000001.SZ"]
        assert kwargs["param_grid"] == {
            "factor_name": ["ret_5"],
            "windows": [[5]],
            "lookback": [20],
            "entry_quantile": [0.7, 0.8],
            "exit_quantile": [0.5],
            "direction": ["high"],
            "position_pct": [0.2],
            "include_interactions": [False],
            "max_interactions": [50],
        }
        assert kwargs["keep_trades"] is False
        assert kwargs["continue_on_error"] is False
        assert kwargs["skip_task_keys"] == set()
        assert callable(kwargs["progress_callback"])
        kwargs["progress_callback"](1, 2, {"strategy": "factor", "params": {"entry_quantile": 0.7}})
        return BacktestComparison(
            rows=[
                BacktestSummary(
                    strategy_name="factor",
                    strategy_id="factor_threshold_v1",
                    symbols=["000001.SZ"],
                    start=date(2024, 1, 1),
                    end=date(2024, 1, 31),
                    metrics=_metrics(0.01, 0.2),
                    trades=[],
                    params={"factor_name": "ret_5", "entry_quantile": 0.7},
                ),
                BacktestSummary(
                    strategy_name="factor",
                    strategy_id="factor_threshold_v1",
                    symbols=["000001.SZ"],
                    start=date(2024, 1, 1),
                    end=date(2024, 1, 31),
                    metrics=_metrics(0.06, 0.8),
                    trades=[],
                    params={"factor_name": "ret_5", "entry_quantile": 0.8},
                ),
            ],
        )

    monkeypatch.setattr(comparison_mod, "run_backtest_param_grid", _fake_run_backtest_param_grid)
    runner = CliRunner()
    csv_path = tmp_path / "factor-grid.csv"
    md_path = tmp_path / "factor-grid.md"
    experiment_log = tmp_path / "factor-experiments.jsonl"

    result = runner.invoke(
        app,
        [
            "factor-grid",
            "--factor",
            "ret_5",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--window",
            "5",
            "--lookback",
            "20",
            "--entry-quantile",
            "0.7",
            "--entry-quantile",
            "0.8",
            "--exit-quantile",
            "0.5",
            "--direction",
            "high",
            "--position-pct",
            "0.2",
            "--top",
            "1",
            "--csv-output",
            str(csv_path),
            "--markdown-output",
            str(md_path),
            "--experiment-log",
            str(experiment_log),
        ],
    )

    assert result.exit_code == 0
    assert "多策略回测对比" in result.output
    assert "factor-grid.csv" in result.output
    assert "factor-grid.md" in result.output
    assert "factor-experiments.jsonl" in result.output
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "params" in csv_text.splitlines()[0]
    assert 'entry_quantile"":0.8' in csv_text
    assert 'entry_quantile"":0.7' not in csv_text
    assert "| 策略 | 参数 | 总收益 |" in md_path.read_text(encoding="utf-8")
    experiment = json.loads(experiment_log.read_text(encoding="utf-8").splitlines()[0])
    assert experiment["record_type"] == "factor_grid"
    assert experiment["param_grid"]["factor_name"] == ["ret_5"]


def test_factor_walk_forward_help_exposes_window_options():
    runner = CliRunner()

    result = runner.invoke(app, ["factor-walk-forward", "--help"])

    assert result.exit_code == 0
    assert "--factor" in result.output
    assert "--train-days" in result.output
    assert "--test-days" in result.output
    assert "--step-days" in result.output
    assert "--max-backtest-runs" in result.output
    assert "--keep-trades" in result.output
    assert "--continue-on-error" in result.output
    assert "--checkpoint-csv" in result.output
    assert "--csv-output" in result.output
    assert "--experiment-log" in result.output


def test_factor_walk_forward_resource_guard_blocks_too_many_internal_runs():
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "factor-walk-forward",
            "--factor",
            "ret_5",
            "--start",
            "2024-01-01",
            "--end",
            "2024-07-31",
            "--lookback",
            "20",
            "--entry-quantile",
            "0.8",
            "--exit-quantile",
            "0.5",
            "--position-pct",
            "0.2",
            "--train-days",
            "90",
            "--test-days",
            "30",
            "--max-backtest-runs",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "内部回测次数" in result.output
    assert "超过资源保护上限 1" in result.output


def test_factor_walk_forward_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.backtest import comparison as comparison_mod
    from qtrade.backtest.comparison import BacktestSummary, WalkForwardFold, WalkForwardReport
    from qtrade.backtest.metrics import Metrics

    captured = {}

    def _metrics(total_return: float, sharpe: float) -> Metrics:
        return Metrics(
            total_return=total_return,
            annual_return=total_return,
            annual_volatility=0.2,
            sharpe=sharpe,
            sortino=sharpe,
            calmar=sharpe,
            max_drawdown=-0.05,
            max_drawdown_duration_days=2,
            win_rate=0.5,
            profit_loss_ratio=1.0,
            n_trades=2,
            final_equity=100_000.0 * (1 + total_return),
            initial_equity=100_000.0,
        )

    def _summary(total_return: float, sharpe: float, params: dict) -> BacktestSummary:
        return BacktestSummary(
            strategy_name="factor",
            strategy_id="factor_threshold_v1",
            symbols=["000001.SZ"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            metrics=_metrics(total_return, sharpe),
            trades=[],
            params=params,
        )

    def _fake_run_walk_forward_param_grid(**kwargs):
        captured.update(kwargs)
        assert kwargs["strategy_name"] == "factor"
        assert kwargs["symbols"] == ["000001.SZ"]
        assert kwargs["train_days"] == 90
        assert kwargs["test_days"] == 30
        assert kwargs["step_days"] == 30
        assert kwargs["param_grid"] == {
            "factor_name": ["ret_5"],
            "windows": [[5]],
            "lookback": [20],
            "entry_quantile": [0.8],
            "exit_quantile": [0.5],
            "direction": ["high"],
            "position_pct": [0.2],
            "include_interactions": [False],
            "max_interactions": [50],
        }
        assert kwargs["keep_trades"] is False
        assert kwargs["continue_on_error"] is False
        assert callable(kwargs["progress_callback"])
        kwargs["progress_callback"](1, 2, {"fold": 1, "stage": "train", "strategy": "factor"})
        params = {"factor_name": "ret_5", "entry_quantile": 0.8, "position_pct": 0.2}
        return WalkForwardReport(
            strategy_name="factor",
            symbols=["000001.SZ"],
            start=date(2024, 1, 1),
            end=date(2024, 7, 31),
            train_days=90,
            test_days=30,
            step_days=30,
            sort_by="sharpe",
            folds=[
                WalkForwardFold(
                    fold=1,
                    train_start=date(2024, 1, 1),
                    train_end=date(2024, 3, 30),
                    test_start=date(2024, 3, 31),
                    test_end=date(2024, 4, 29),
                    selected_params=params,
                    train=_summary(0.05, 0.8, params),
                    test=_summary(0.02, 0.3, params),
                )
            ],
        )

    monkeypatch.setattr(comparison_mod, "run_walk_forward_param_grid", _fake_run_walk_forward_param_grid)
    runner = CliRunner()
    csv_path = tmp_path / "walk-forward.csv"
    md_path = tmp_path / "walk-forward.md"
    experiment_log = tmp_path / "walk-forward-experiments.jsonl"

    result = runner.invoke(
        app,
        [
            "factor-walk-forward",
            "--factor",
            "ret_5",
            "--symbol",
            "000001",
            "--start",
            "2024-01-01",
            "--end",
            "2024-07-31",
            "--window",
            "5",
            "--lookback",
            "20",
            "--entry-quantile",
            "0.8",
            "--exit-quantile",
            "0.5",
            "--direction",
            "high",
            "--position-pct",
            "0.2",
            "--train-days",
            "90",
            "--test-days",
            "30",
            "--step-days",
            "30",
            "--csv-output",
            str(csv_path),
            "--markdown-output",
            str(md_path),
            "--experiment-log",
            str(experiment_log),
        ],
    )

    assert result.exit_code == 0
    assert "Walk-forward 验证" in result.output
    assert "walk-forward.csv" in result.output
    assert "walk-forward.md" in result.output
    assert "实验记录已追加" in result.output
    assert captured["sort_by"] == "sharpe"
    assert csv_path.read_text(encoding="utf-8").startswith("fold,train_start")
    md_text = md_path.read_text(encoding="utf-8")
    assert md_text.startswith("# Walk-forward 验证报告")
    assert "验证/训练 Sharpe 保留率" in md_text
    experiment = json.loads(experiment_log.read_text(encoding="utf-8").splitlines()[0])
    assert experiment["record_type"] == "factor_walk_forward"
    assert experiment["fold_count"] == 1
    assert experiment["folds"][0]["selected_params"]["entry_quantile"] == 0.8


def test_experiment_leaderboard_help_exposes_options():
    runner = CliRunner()

    result = runner.invoke(app, ["experiment-leaderboard", "--help"])

    assert result.exit_code == 0
    assert "--experiment-log" in result.output
    assert "--record-type" in result.output
    assert "--csv-output" in result.output


def test_experiment_leaderboard_reads_candidates_and_exports_csv(tmp_path):
    def _summary(params: dict, *, total_return: float, sharpe: float) -> dict:
        return {
            "strategy": "ma_cross",
            "strategy_id": "ma_cross_v1",
            "symbols": ["000001.SZ"],
            "start": "2024-01-01",
            "end": "2024-01-31",
            "params": params,
            "metrics": {
                "total_return": total_return,
                "annual_return": total_return,
                "annual_volatility": 0.2,
                "sharpe": sharpe,
                "sortino": sharpe,
                "calmar": 1.0,
                "max_drawdown": -0.05,
                "n_trades": 4,
                "win_rate": 0.75,
                "final_equity": 100_000 * (1 + total_return),
            },
        }

    log_path = tmp_path / "experiments.jsonl"
    csv_path = tmp_path / "leaderboard.csv"
    records = [
        {
            "schema_version": 1,
            "record_type": "backtest_grid",
            "generated_at": "2026-06-06T01:00:00",
            "strategy": "ma_cross",
            "symbols": ["000001.SZ"],
            "start": "2024-01-01",
            "end": "2024-01-31",
            "failure_rate": 0.0,
            "best_by_sharpe": _summary({"fast_window": 5}, total_return=0.08, sharpe=1.2),
            "best_by_return": _summary({"fast_window": 10}, total_return=0.11, sharpe=0.9),
            "top_results": [_summary({"fast_window": 5}, total_return=0.08, sharpe=1.2)],
        },
        {
            "schema_version": 1,
            "record_type": "factor_walk_forward",
            "generated_at": "2026-06-06T02:00:00",
            "strategy": "factor",
            "symbols": ["000001.SZ"],
            "start": "2024-01-01",
            "end": "2024-07-31",
            "compounded_test_return": 0.12,
            "average_test_sharpe": 1.5,
            "average_train_sharpe": 1.8,
            "average_sharpe_decay": 0.3,
            "test_sharpe_retention": 0.83,
            "positive_test_return_rate": 1.0,
            "folds": [
                {
                    "selected_params": {"factor_name": "ret_5", "entry_quantile": 0.8},
                    "test": {
                        "metrics": {
                            "max_drawdown": -0.04,
                            "n_trades": 3,
                        }
                    },
                }
            ],
        },
    ]
    log_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiment-leaderboard",
            "--experiment-log",
            str(log_path),
            "--sort-by",
            "sharpe",
            "--top",
            "2",
            "--csv-output",
            str(csv_path),
        ],
    )

    assert result.exit_code == 0
    assert "实验候选榜单" in result.output
    assert "候选数量" in result.output
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert csv_text.startswith("record_type,generated_at")
    assert "factor_walk_forward" in csv_text
    assert "backtest_grid" in csv_text
    assert csv_text.index("factor_walk_forward") < csv_text.index("backtest_grid")


def test_experiment_plan_help_exposes_options():
    runner = CliRunner()

    result = runner.invoke(app, ["experiment-plan", "--help"])

    assert result.exit_code == 0
    assert "--experiment-log" in result.output
    assert "--validation-start" in result.output
    assert "--next-experiment-log" in result.output
    assert "--markdown-output" in result.output


def test_experiment_plan_generates_next_validation_commands(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    plan_path = tmp_path / "plan.md"
    next_log = tmp_path / "next-experiments.jsonl"
    record = {
        "schema_version": 1,
        "record_type": "factor_grid",
        "generated_at": "2026-06-06T01:00:00",
        "strategy": "factor",
        "symbols": ["000001.SZ"],
        "start": "2024-01-01",
        "end": "2024-12-31",
        "failure_rate": 0.0,
        "best_by_sharpe": {
            "strategy": "factor",
            "strategy_id": "factor_threshold",
            "symbols": ["000001.SZ"],
            "start": "2024-01-01",
            "end": "2024-12-31",
            "params": {
                "factor_name": "ret_5",
                "windows": [5, 20],
                "lookback": 60,
                "entry_quantile": 0.8,
                "exit_quantile": 0.5,
                "direction": "high",
                "position_pct": 0.2,
            },
            "metrics": {
                "total_return": 0.1,
                "sharpe": 1.3,
                "max_drawdown": -0.05,
                "n_trades": 4,
                "win_rate": 0.75,
            },
        },
    }
    log_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiment-plan",
            "--experiment-log",
            str(log_path),
            "--validation-start",
            "2025-01-01",
            "--validation-end",
            "2025-12-31",
            "--train-days",
            "180",
            "--test-days",
            "60",
            "--next-experiment-log",
            str(next_log),
            "--markdown-output",
            str(plan_path),
        ],
    )

    assert result.exit_code == 0
    assert "下一轮验证计划" in result.output
    assert "验证命令" in result.output
    text = plan_path.read_text(encoding="utf-8")
    assert "qtrade factor-walk-forward" in text
    assert "--factor ret_5" in text
    assert "--start 2025-01-01" in text
    assert "--end 2025-12-31" in text
    assert "--train-days 180" in text
    assert "--test-days 60" in text
    assert f"--experiment-log {next_log}" in text


def test_experiment_run_plan_defaults_to_dry_run(tmp_path):
    plan_path = tmp_path / "plan.md"
    output_dir = tmp_path / "run-output"
    plan_path.write_text(
        "\n".join(
            [
                "# plan",
                "",
                "```bash",
                "qtrade factor-backtest --factor ret_5 --symbol 000001.SZ --start 2025-01-01 --end 2025-12-31",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["experiment-run-plan", str(plan_path), "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "验证计划执行" in result.output
    assert "dry-run" in result.output
    summary = json.loads((output_dir / "validation-run-summary.json").read_text(encoding="utf-8"))
    assert summary["record_type"] == "validation_plan_run"
    assert summary["execute"] is False
    assert summary["total_commands"] == 1
    assert summary["processed_commands"] == 1
    assert summary["status_counts"] == {"dry_run": 1}
    assert summary["results"][0]["command"].startswith("qtrade factor-backtest")


def test_experiment_run_plan_rejects_non_whitelisted_commands(tmp_path):
    plan_path = tmp_path / "bad-plan.md"
    plan_path.write_text(
        "\n".join(["```bash", "qtrade run --strategy ma_cross --once", "```"]),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["experiment-run-plan", str(plan_path)])

    assert result.exit_code != 0
    assert "验证白名单" in result.output


def test_experiment_run_plan_execute_uses_subprocess(monkeypatch, tmp_path):
    from qtrade.backtest import experiment_log as experiment_log_mod

    plan_path = tmp_path / "plan.md"
    plan_path.write_text(
        "\n".join(
            [
                "```bash",
                "qtrade factor-backtest --factor ret_5 --symbol 000001.SZ --start 2025-01-01 --end 2025-12-31",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(experiment_log_mod.subprocess, "run", _fake_run)
    runner = CliRunner()

    result = runner.invoke(app, ["experiment-run-plan", str(plan_path), "--execute"])

    assert result.exit_code == 0
    assert captured["argv"][:2] == ["qtrade", "factor-backtest"]
    assert captured["kwargs"]["shell"] is not True if "shell" in captured["kwargs"] else True
    assert "success" in result.output


def test_experiment_run_plan_help_exposes_output_options():
    runner = CliRunner()

    result = runner.invoke(app, ["experiment-run-plan", "--help"])

    assert result.exit_code == 0
    assert "--output-dir" in result.output
    assert "--json-output" in result.output
    assert "--execute" in result.output


def test_factor_portfolio_help_exposes_portfolio_options():
    runner = CliRunner()

    result = runner.invoke(app, ["factor-portfolio", "--help"])

    assert result.exit_code == 0
    assert "--factor" in result.output
    assert "--select-count" in result.output
    assert "--select-quantile" in result.output
    assert "--rebalance-days" in result.output
    assert "--min-volume" in result.output
    assert "--price-limit-pct" in result.output
    assert "--max-symbols" in result.output
    assert "--csv-output" in result.output


def test_factor_portfolio_resource_guard_blocks_too_many_symbols_before_run():
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "factor-portfolio",
            "--factor",
            "ret_5",
            "--symbol",
            "000001",
            "--symbol",
            "000002",
            "--max-symbols",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "标的数量 2 超过资源保护上限 1" in result.output


def test_factor_portfolio_cli_exports_without_live_network(monkeypatch, tmp_path):
    from qtrade.backtest import factor_portfolio as portfolio_mod
    from qtrade.backtest.factor_portfolio import FactorPortfolioDay, FactorPortfolioReport
    from qtrade.backtest.metrics import Metrics

    captured = {}

    def _fake_run_factor_portfolio_backtest(**kwargs):
        captured.update(kwargs)
        assert kwargs["symbols"] == ["000001.SZ", "000002.SZ"]
        assert kwargs["factor_name"] == "ret_5"
        assert kwargs["windows"] == [5]
        assert kwargs["select_count"] == 1
        assert kwargs["respect_tradeability"] is True
        assert kwargs["price_limit_pct"] is None
        return FactorPortfolioReport(
            symbols=["000001.SZ", "000002.SZ"],
            factor="ret_5",
            start=date(2024, 1, 1),
            end=date(2024, 8, 31),
            windows=[5],
            direction="high",
            rebalance_days=5,
            select_count=1,
            select_quantile=0.2,
            fee_bps=3.0,
            slippage_bps=1.0,
            respect_tradeability=True,
            min_volume=1.0,
            price_limit_pct=None,
            limit_buffer_bps=5.0,
            initial_equity=100_000.0,
            metrics=Metrics(
                total_return=0.08,
                annual_return=0.12,
                annual_volatility=0.2,
                sharpe=0.5,
                sortino=0.6,
                calmar=0.7,
                max_drawdown=-0.04,
                max_drawdown_duration_days=3,
                win_rate=0.0,
                profit_loss_ratio=0.0,
                n_trades=0,
                final_equity=108_000.0,
                initial_equity=100_000.0,
            ),
            days=[
                FactorPortfolioDay(
                    date=date(2024, 1, 8),
                    selected_symbols=["000001.SZ"],
                    blocked_buy_symbols=["000002.SZ"],
                    blocked_sell_symbols=[],
                    gross_return=0.01,
                    cost_return=0.0004,
                    net_return=0.0096,
                    turnover=1.0,
                    holdings=1,
                    eligible_count=1,
                    equity=100_960.0,
                )
            ],
        )

    monkeypatch.setattr(portfolio_mod, "run_factor_portfolio_backtest", _fake_run_factor_portfolio_backtest)
    runner = CliRunner()
    csv_path = tmp_path / "factor-portfolio.csv"
    md_path = tmp_path / "factor-portfolio.md"

    result = runner.invoke(
        app,
        [
            "factor-portfolio",
            "--factor",
            "ret_5",
            "--symbol",
            "000001",
            "--symbol",
            "000002",
            "--start",
            "2024-01-01",
            "--end",
            "2024-08-31",
            "--window",
            "5",
            "--select-count",
            "1",
            "--rebalance-days",
            "5",
            "--csv-output",
            str(csv_path),
            "--markdown-output",
            str(md_path),
        ],
    )

    assert result.exit_code == 0
    assert "因子组合回测" in result.output
    assert "factor-portfolio.csv" in result.output
    assert "factor-portfolio.md" in result.output
    assert captured["direction"] == "high"
    assert csv_path.read_text(encoding="utf-8").startswith("date,selected_symbols")
    assert md_path.read_text(encoding="utf-8").startswith("# 因子组合回测报告")


def test_build_factor_strategy_param_grid_validates_thresholds():
    grid = _build_factor_strategy_param_grid(
        factor_name="ret_5",
        windows=[5],
        lookbacks=[20],
        entry_quantiles=[0.8],
        exit_quantiles=[0.5],
        directions=["high", "low"],
        position_pcts=[0.2],
        include_interactions=False,
        max_interactions=50,
    )
    assert grid["factor_name"] == ["ret_5"]
    assert grid["direction"] == ["high", "low"]
    assert grid["windows"] == [[5]]

    with pytest.raises(ValueError, match="entry_quantile 必须大于 exit_quantile"):
        _build_factor_strategy_param_grid(
            factor_name="ret_5",
            windows=[5],
            lookbacks=[20],
            entry_quantiles=[0.5],
            exit_quantiles=[0.8],
            directions=["high"],
            position_pcts=[0.2],
            include_interactions=False,
            max_interactions=50,
        )


def test_run_option_validators_reject_unsafe_values():
    with pytest.raises(typer.BadParameter):
        _validate_llm_run_options(prompt_version="v1", debate_rounds=5, threshold=0.5)

    with pytest.raises(typer.BadParameter):
        _validate_llm_run_options(prompt_version="v1", debate_rounds=2, threshold=1.5)

    with pytest.raises(typer.BadParameter):
        _validate_common_run_options(
            interval_seconds=60,
            decision_interval_minutes=0,
        )

    with pytest.raises(typer.BadParameter):
        _validate_common_run_options(
            interval_seconds=10,
            decision_interval_minutes=0,
            position_pct=1.5,
        )
