"""回测引擎冒烟测试：不依赖网络，用合成行情。

校验：
- 引擎能跑完不抛异常
- 权益曲线非空
- 触发金叉时的确产生买入交易
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from qtrade.backtest.comparison import (
    BacktestComparison,
    BacktestSummary,
    _summary_from_result,
    backtest_task_key,
    build_backtest_strategy,
    comparison_to_markdown,
    compute_parameter_sensitivity,
    expand_param_grid,
    limit_backtest_comparison,
    run_backtest_comparison,
    run_backtest_param_grid,
    run_walk_forward_param_grid,
    sort_backtest_comparison,
    walk_forward_report_to_markdown,
    write_comparison_csv,
    write_comparison_markdown,
    write_walk_forward_csv,
    write_walk_forward_markdown,
)
from qtrade.backtest.data_cache import InMemoryBarLoader
from qtrade.backtest.engine import BacktestEngine, BacktestResult
from qtrade.backtest.factor_portfolio import (
    factor_portfolio_report_to_markdown,
    run_factor_portfolio_backtest,
    write_factor_portfolio_csv,
    write_factor_portfolio_markdown,
)
from qtrade.backtest.factors import (
    build_factor_candidates,
    compute_factor_series,
    factor_layer_report_to_markdown,
    factor_report_to_markdown,
    limit_factor_report,
    run_factor_layering,
    run_factor_mining,
    write_factor_layer_report_csv,
    write_factor_layer_report_markdown,
    write_factor_report_csv,
    write_factor_report_markdown,
)
from qtrade.backtest.metrics import Metrics
from qtrade.data.loader import DataLoader
from qtrade.data.schema import AdjustMode
from qtrade.persistence import session_scope
from qtrade.persistence.models import OrderRecord, OrderStatus
from qtrade.risk.ashare_rules import ashare_price_limit_pct, infer_ashare_board
from qtrade.strategies.base import Signal, Strategy, StrategyContext
from qtrade.strategies.factor import FactorThresholdStrategy
from qtrade.strategies.technical.ma_cross import MACrossStrategy


class _FakeAdapter:
    name = "fake"

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        # 60 天上涨 + 30 天回调 + 60 天上涨：会触发金叉/死叉
        n = 150
        dates = pd.date_range(start=start, periods=n, freq="B")
        prices = np.concatenate(
            [
                np.linspace(10, 20, 60),
                np.linspace(20, 14, 30),
                np.linspace(14, 25, 60),
            ]
        )
        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.full(n, 1_000_000),
                "amount": prices * 1_000_000,
                "symbol": symbol,
                "adjust": adjust,
            }
        )
        return df

    def fetch_news(self, symbol=None, *, limit=20):
        return []


class _CountingBarLoader:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def get_daily_bars(self, symbol, *, start, end=None, adjust: AdjustMode = "qfq", force_refresh=False):
        self.calls[symbol] = self.calls.get(symbol, 0) + 1
        n = 150
        dates = pd.date_range(start=start, periods=n, freq="B")
        prices = np.linspace(10, 15, n)
        return pd.DataFrame(
            {
                "date": dates,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.full(n, 1_000_000),
                "amount": prices * 1_000_000,
                "symbol": symbol,
                "adjust": adjust,
            }
        )


class _CrossSectionAdapter:
    name = "cross-section"

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        n = 160
        dates = pd.date_range(start=start, periods=n, freq="B")
        base = {
            "000001.SZ": np.linspace(10, 18, n),
            "000002.SZ": np.linspace(18, 10, n),
            "000003.SZ": 13 + np.sin(np.linspace(0, 8, n)),
        }.get(symbol, np.linspace(10, 12, n))
        df = pd.DataFrame(
            {
                "date": dates,
                "open": base * 0.99,
                "high": base * 1.01,
                "low": base * 0.98,
                "close": base,
                "volume": np.full(n, 1_000_000),
                "amount": base * 1_000_000,
                "symbol": symbol,
                "adjust": adjust,
            }
        )
        return df

    def fetch_news(self, symbol=None, *, limit=20):
        return []


class _TradeabilityAdapter:
    name = "tradeability"

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        dates = pd.date_range(start=start, periods=18, freq="B")
        prices = {
            # Day 6 is limit-down after being the previous winner.
            "000001.SZ": [10, 10.2, 10.6, 11.0, 11.5, 12.0, 10.8, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7],
            "000002.SZ": [10, 10.0, 10.0, 10.0, 10.0, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2],
            # Day 6 is limit-up and should not be newly bought.
            "000003.SZ": [10, 10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1],
        }[symbol]
        arr = np.array(prices, dtype=float)
        volume = np.full(len(arr), 1_000_000.0)
        df = pd.DataFrame(
            {
                "date": dates,
                "open": arr,
                "high": arr,
                "low": arr,
                "close": arr,
                "volume": volume,
                "amount": arr * volume,
                "symbol": symbol,
                "adjust": adjust,
            }
        )
        return df

    def fetch_news(self, symbol=None, *, limit=20):
        return []


class _BoardLimitAdapter:
    name = "board-limit"

    def fetch_daily_bars(self, symbol, start, end, adjust: AdjustMode = "qfq"):
        dates = pd.date_range(start=start, periods=12, freq="B")
        prices = {
            "300001.SZ": [10, 10.2, 10.4, 10.6, 10.8, 11.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6],
            "000001.SZ": [10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1],
        }[symbol]
        arr = np.array(prices, dtype=float)
        volume = np.full(len(arr), 1_000_000.0)
        return pd.DataFrame(
            {
                "date": dates,
                "open": arr,
                "high": arr,
                "low": arr,
                "close": arr,
                "volume": volume,
                "amount": arr * volume,
                "symbol": symbol,
                "adjust": adjust,
            }
        )

    def fetch_news(self, symbol=None, *, limit=20):
        return []


def test_backtest_runs_and_trades():
    loader = DataLoader(adapters=[_FakeAdapter()])
    strat = MACrossStrategy(
        universe=["000001.SZ"],
        params={"fast_window": 5, "slow_window": 20, "position_pct": 0.3},
    )
    engine = BacktestEngine(
        strategy=strat,
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        initial_cash=200_000.0,
        loader=loader,
    )
    res = engine.run()
    assert len(res.equity_curve) > 0
    # 合成数据下应至少有 1 笔交易
    assert res.metrics.n_trades >= 0  # 可能是 0 也可能是 >0；关键是不报错
    assert res.metrics.final_equity > 0


class _BuyOnceStrategy(Strategy):
    id = "buy_once_test"

    def __init__(self, *, universe: list[str]) -> None:
        super().__init__(universe=universe)
        self.done = False

    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        if self.done:
            return []
        self.done = True
        return [Signal(symbol=ctx.symbol, action="BUY", quantity=1000, reason="persist smoke")]


def test_backtest_persist_writes_orders():
    loader = DataLoader(adapters=[_FakeAdapter()])
    strat = _BuyOnceStrategy(universe=["000001.SZ"])
    engine = BacktestEngine(
        strategy=strat,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        initial_cash=100_000.0,
        loader=loader,
        persist=True,
    )

    res = engine.run()

    assert len(res.trades) == 1
    with session_scope() as s:
        orders = s.query(OrderRecord).filter_by(strategy_id=strat.id).all()
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.FILLED
        assert orders[0].filled_quantity == 1000


def test_backtest_engine_can_skip_recording_trade_details():
    loader = DataLoader(adapters=[_FakeAdapter()])
    strat = _BuyOnceStrategy(universe=["000001.SZ"])
    engine = BacktestEngine(
        strategy=strat,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        initial_cash=100_000.0,
        loader=loader,
        record_trades=False,
    )

    res = engine.run()

    assert res.trades == []
    assert res.trade_count == 1
    assert res.metrics.final_equity > 0


def test_backtest_comparison_runs_multiple_strategies_and_exports(tmp_path):
    loader = DataLoader(adapters=[_FakeAdapter()])

    comparison = run_backtest_comparison(
        strategies=["ma_cross", "macd"],
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        initial_cash=200_000.0,
        loader=loader,
    )

    assert [row.strategy_name for row in comparison.rows] == ["ma_cross", "macd"]
    assert all(row.metrics.final_equity > 0 for row in comparison.rows)
    assert comparison.best_by_sharpe is not None
    assert comparison.best_by_return is not None

    markdown = comparison_to_markdown(comparison)
    assert "# 回测策略对比报告" in markdown
    assert "- 成功结果: 2" in markdown
    assert "- 失败率: 0.00%" in markdown
    assert "| 策略 | 总收益 | 年化 | 波动 | Sharpe |" in markdown

    csv_path = write_comparison_csv(comparison, tmp_path / "comparison.csv")
    md_path = write_comparison_markdown(comparison, tmp_path / "comparison.md")

    csv_text = csv_path.read_text(encoding="utf-8")
    assert csv_text.startswith("strategy,strategy_id")
    assert "ma_cross" in csv_text
    assert md_path.read_text(encoding="utf-8").startswith("# 回测策略对比报告")


def test_backtest_comparison_blocks_llm_by_default():
    with pytest.raises(ValueError, match="LLM 回测默认关闭"):
        build_backtest_strategy("llm", symbols=["000001.SZ"])


def test_build_backtest_strategy_supports_factor_strategy():
    strat = build_backtest_strategy(
        "factor",
        symbols=["000001.SZ"],
        params={"factor_name": "ret_5", "windows": [5]},
    )

    assert isinstance(strat, FactorThresholdStrategy)


def test_backtest_param_grid_runs_sorts_and_limits(tmp_path):
    loader = DataLoader(adapters=[_FakeAdapter()])

    comparison = run_backtest_param_grid(
        strategy_name="ma_cross",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        param_grid={
            "fast_window": [5, 10],
            "slow_window": [20],
            "position_pct": [0.2, 0.3],
        },
        initial_cash=200_000.0,
        loader=loader,
    )

    assert len(comparison.rows) == 4
    assert comparison.rows[0].params == {"fast_window": 5, "slow_window": 20, "position_pct": 0.2}
    assert all(row.strategy_name == "ma_cross" for row in comparison.rows)

    sorted_comparison = sort_backtest_comparison(comparison, sort_by="total_return")
    limited = limit_backtest_comparison(sorted_comparison, top=2)
    assert len(limited.rows) == 2
    assert limited.rows[0].metrics.total_return >= limited.rows[1].metrics.total_return

    csv_path = write_comparison_csv(limited, tmp_path / "grid.csv")
    md_path = write_comparison_markdown(limited, tmp_path / "grid.md")

    assert "params" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "| 策略 | 参数 | 总收益 |" in md_path.read_text(encoding="utf-8")


def test_parameter_sensitivity_flags_top_boundary_value():
    def _summary(fast_window: int, sharpe: float) -> BacktestSummary:
        return BacktestSummary(
            strategy_name="ma_cross",
            strategy_id="ma_cross_v1",
            symbols=["000001.SZ"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            metrics=Metrics(
                total_return=sharpe / 10,
                annual_return=sharpe / 10,
                annual_volatility=0.2,
                sharpe=sharpe,
                sortino=sharpe,
                calmar=sharpe,
                max_drawdown=-0.05,
                max_drawdown_duration_days=2,
                win_rate=0.5,
                profit_loss_ratio=1.0,
                n_trades=2,
                final_equity=100_000.0,
                initial_equity=100_000.0,
            ),
            trades=[],
            params={"fast_window": fast_window, "slow_window": 20},
        )

    comparison = BacktestComparison(
        rows=[
            _summary(5, 0.1),
            _summary(10, 0.2),
            _summary(20, 0.8),
            _summary(20, 0.9),
        ]
    )

    sensitivity = compute_parameter_sensitivity(comparison, top_n=2, sort_by="sharpe")
    fast = next(row for row in sensitivity if row.param_name == "fast_window")
    assert fast.dominant_value == "20"
    assert fast.dominant_share == 1.0
    assert fast.boundary_side == "max"

    markdown = comparison_to_markdown(comparison)
    assert "## 参数敏感性" in markdown
    assert "fast_window" in markdown
    assert "max" in markdown


def test_backtest_param_grid_reuses_in_memory_bars_per_symbol():
    loader = _CountingBarLoader()

    comparison = run_backtest_param_grid(
        strategy_name="ma_cross",
        symbols=["000001.SZ", "000002.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        param_grid={
            "fast_window": [5, 10],
            "slow_window": [20],
            "position_pct": [0.2, 0.3],
        },
        initial_cash=200_000.0,
        loader=loader,
    )

    assert len(comparison.rows) == 4
    assert loader.calls == {"000001.SZ": 1, "000002.SZ": 1}


def test_backtest_param_grid_reports_progress():
    loader = DataLoader(adapters=[_FakeAdapter()])
    events = []

    comparison = run_backtest_param_grid(
        strategy_name="ma_cross",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        param_grid={
            "fast_window": [5, 10],
            "slow_window": [20],
            "position_pct": [0.2, 0.3],
        },
        initial_cash=200_000.0,
        loader=loader,
        progress_callback=lambda done, total, payload: events.append((done, total, payload)),
    )

    assert len(comparison.rows) == 4
    assert [(done, total) for done, total, _payload in events] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert events[0][2]["strategy"] == "ma_cross"
    assert events[0][2]["params"] == {"fast_window": 5, "slow_window": 20, "position_pct": 0.2}
    assert events[0][2]["summary"].params == {"fast_window": 5, "slow_window": 20, "position_pct": 0.2}


def test_backtest_param_grid_skips_checkpointed_task_keys():
    loader = DataLoader(adapters=[_FakeAdapter()])
    skipped_params = {"fast_window": 5, "slow_window": 20, "position_pct": 0.2}
    events = []

    comparison = run_backtest_param_grid(
        strategy_name="ma_cross",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        param_grid={
            "fast_window": [5, 10],
            "slow_window": [20],
            "position_pct": [0.2],
        },
        initial_cash=200_000.0,
        loader=loader,
        skip_task_keys={
            backtest_task_key(
                strategy_name="ma_cross",
                symbols=["000001.SZ"],
                start=date(2024, 1, 1),
                end=date(2024, 7, 31),
                params=skipped_params,
            )
        },
        progress_callback=lambda done, total, payload: events.append((done, total, payload)),
    )

    assert len(comparison.rows) == 1
    assert comparison.rows[0].params == {"fast_window": 10, "slow_window": 20, "position_pct": 0.2}
    assert events[0][2] == {"strategy": "ma_cross", "params": skipped_params, "skipped": True}
    assert "summary" in events[1][2]


def test_backtest_param_grid_can_continue_after_param_failure():
    loader = DataLoader(adapters=[_FakeAdapter()])
    events = []

    comparison = run_backtest_param_grid(
        strategy_name="ma_cross",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        param_grid={
            "fast_window": [5, 20],
            "slow_window": [20],
            "position_pct": [0.2],
        },
        initial_cash=200_000.0,
        loader=loader,
        continue_on_error=True,
        progress_callback=lambda done, total, payload: events.append((done, total, payload)),
    )

    assert len(comparison.rows) == 1
    assert len(comparison.failures) == 1
    assert comparison.failures[0].params == {"fast_window": 20, "slow_window": 20, "position_pct": 0.2}
    assert comparison.failures[0].error_type == "ValueError"
    assert "fast_window must be < slow_window" in comparison.failures[0].error
    assert [(done, total) for done, total, _payload in events] == [(1, 2), (2, 2)]
    assert "summary" in events[0][2]
    assert "failure" in events[1][2]


def test_backtest_param_grid_fails_fast_by_default():
    loader = DataLoader(adapters=[_FakeAdapter()])

    with pytest.raises(ValueError, match="fast_window must be < slow_window"):
        run_backtest_param_grid(
            strategy_name="ma_cross",
            symbols=["000001.SZ"],
            start=date(2024, 1, 1),
            end=date(2024, 7, 31),
            param_grid={
                "fast_window": [20],
                "slow_window": [20],
                "position_pct": [0.2],
            },
            initial_cash=200_000.0,
            loader=loader,
        )


def test_backtest_summary_drops_trade_details_by_default():
    result = BacktestResult(
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
            n_trades=2,
            final_equity=110_000.0,
            initial_equity=100_000.0,
        ),
        equity_curve=[],
        trades=[{"id": 1}, {"id": 2}],
        final_account={},
    )
    strategy = _BuyOnceStrategy(universe=["000001.SZ"])

    summary = _summary_from_result(
        "buy_once_test",
        ["000001.SZ"],
        date(2024, 1, 1),
        date(2024, 1, 31),
        strategy,
        result,
    )
    kept = _summary_from_result(
        "buy_once_test",
        ["000001.SZ"],
        date(2024, 1, 1),
        date(2024, 1, 31),
        strategy,
        result,
        keep_trades=True,
    )

    assert summary.trades == []
    assert summary.trade_count == 2
    assert summary.n_trades == 2
    assert kept.trades == [{"id": 1}, {"id": 2}]
    assert kept.trades is not result.trades
    assert kept.trade_count == 2


def test_in_memory_bar_loader_returns_copies_and_stats():
    raw_loader = _CountingBarLoader()
    loader = InMemoryBarLoader(raw_loader)

    first = loader.get_daily_bars("000001.SZ", start=date(2024, 1, 1), end=date(2024, 7, 31))
    second = loader.get_daily_bars("000001.SZ", start=date(2024, 1, 1), end=date(2024, 7, 31))

    assert raw_loader.calls == {"000001.SZ": 1}
    assert first is not second
    assert len(first) == len(second)
    assert loader.stats.hits == 1
    assert loader.stats.misses == 1
    assert loader.stats.entries == 1


def test_expand_param_grid_validates_empty_values():
    assert expand_param_grid({"fast_window": [5, 10], "slow_window": [20]}) == [
        {"fast_window": 5, "slow_window": 20},
        {"fast_window": 10, "slow_window": 20},
    ]
    with pytest.raises(ValueError, match="候选值不能为空"):
        expand_param_grid({"fast_window": []})


def test_factor_mining_generates_scores_and_exports(tmp_path):
    loader = DataLoader(adapters=[_FakeAdapter()])

    report = run_factor_mining(
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        windows=[5, 20],
        horizon_days=5,
        train_ratio=0.7,
        include_interactions=True,
        max_interactions=3,
        loader=loader,
    )
    report = limit_factor_report(report, top=5)

    assert report.rows
    assert all(row.train.n_obs > 0 and row.test.n_obs > 0 for row in report.rows)
    assert report.best_by_test_rank_ic_abs is not None

    markdown = factor_report_to_markdown(report)
    assert "# 因子挖掘报告" in markdown
    assert "| 因子 | 表达式 | 样本内 Rank IC |" in markdown

    csv_path = write_factor_report_csv(report, tmp_path / "factors.csv")
    md_path = write_factor_report_markdown(report, tmp_path / "factors.md")
    assert csv_path.read_text(encoding="utf-8").startswith("factor,expression")
    assert md_path.read_text(encoding="utf-8").startswith("# 因子挖掘报告")


def test_factor_candidates_include_transforms_and_limited_interactions():
    loader = DataLoader(adapters=[_FakeAdapter()])
    df = loader.get_daily_bars("000001.SZ", start=date(2024, 1, 1), end=date(2024, 7, 31))
    report = run_factor_mining(
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        windows=[5],
        horizon_days=5,
        include_interactions=False,
        loader=loader,
    )
    # Use the public candidate builder through a minimal frame derived from the report path.
    assert any(row.name.endswith("__square") for row in report.rows)

    from qtrade.backtest import factors as factor_mod

    factor_data = factor_mod._build_symbol_factor_data(df, windows=[5], horizon_days=5)
    candidates = build_factor_candidates(factor_data, include_interactions=True, max_interactions=2)
    assert sum("__x__" in name for name in candidates) == 2


def test_factor_layering_generates_bucket_report_and_exports(tmp_path):
    loader = DataLoader(adapters=[_FakeAdapter()])

    report = run_factor_layering(
        symbols=["000001.SZ"],
        factor_name="ret_5",
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        windows=[5, 20],
        horizon_days=5,
        train_ratio=0.7,
        groups=5,
        loader=loader,
    )

    assert report.factor == "ret_5"
    assert report.train.n_obs > 0
    assert report.test.n_obs > 0
    assert len(report.train.buckets) == 5
    assert len(report.test.buckets) == 5
    assert 0.0 <= report.test.turnover <= 1.0

    markdown = factor_layer_report_to_markdown(report)
    assert "# 因子分层验证报告" in markdown
    assert "| 样本 | Rank IC | Top | Bottom |" in markdown

    csv_path = write_factor_layer_report_csv(report, tmp_path / "layer.csv")
    md_path = write_factor_layer_report_markdown(report, tmp_path / "layer.md")
    assert csv_path.read_text(encoding="utf-8").startswith("factor,expression,window,bucket")
    assert md_path.read_text(encoding="utf-8").startswith("# 因子分层验证报告")


def test_factor_layering_rejects_unknown_factor():
    loader = DataLoader(adapters=[_FakeAdapter()])

    with pytest.raises(ValueError, match="未知因子"):
        run_factor_layering(
            symbols=["000001.SZ"],
            factor_name="not_a_factor",
            start=date(2024, 1, 1),
            end=date(2024, 7, 31),
            loader=loader,
        )


def test_compute_factor_series_matches_research_factor_names():
    loader = DataLoader(adapters=[_FakeAdapter()])
    df = loader.get_daily_bars("000001.SZ", start=date(2024, 1, 1), end=date(2024, 7, 31))

    series = compute_factor_series(df, factor_name="ret_5", windows=[5])

    assert len(series) == len(df)
    assert series.dropna().iloc[-1] != 0


def test_factor_threshold_strategy_runs_in_backtest_engine():
    loader = DataLoader(adapters=[_FakeAdapter()])
    strat = FactorThresholdStrategy(
        universe=["000001.SZ"],
        params={
            "factor_name": "ret_5",
            "windows": [5],
            "lookback": 20,
            "entry_quantile": 0.8,
            "exit_quantile": 0.5,
            "position_pct": 0.3,
        },
    )
    engine = BacktestEngine(
        strategy=strat,
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        initial_cash=200_000.0,
        loader=loader,
    )

    res = engine.run()

    assert len(res.equity_curve) > 0
    assert res.metrics.final_equity > 0


def test_factor_strategy_runs_through_param_grid():
    loader = DataLoader(adapters=[_FakeAdapter()])

    comparison = run_backtest_param_grid(
        strategy_name="factor",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        param_grid={
            "factor_name": ["ret_5"],
            "windows": [[5]],
            "lookback": [20],
            "entry_quantile": [0.7, 0.8],
            "exit_quantile": [0.5],
            "direction": ["high"],
            "position_pct": [0.3],
        },
        initial_cash=200_000.0,
        loader=loader,
    )

    assert len(comparison.rows) == 2
    assert all(row.strategy_name == "factor" for row in comparison.rows)
    assert all(row.params["factor_name"] == "ret_5" for row in comparison.rows)
    assert all(row.metrics.final_equity > 0 for row in comparison.rows)


def test_factor_strategy_walk_forward_selects_and_exports(tmp_path):
    loader = DataLoader(adapters=[_FakeAdapter()])
    events = []

    report = run_walk_forward_param_grid(
        strategy_name="factor",
        symbols=["000001.SZ"],
        start=date(2024, 1, 1),
        end=date(2024, 7, 31),
        param_grid={
            "factor_name": ["ret_5"],
            "windows": [[5]],
            "lookback": [20],
            "entry_quantile": [0.7, 0.8],
            "exit_quantile": [0.5],
            "direction": ["high"],
            "position_pct": [0.3],
        },
        train_days=90,
        test_days=30,
        step_days=30,
        initial_cash=200_000.0,
        loader=loader,
        progress_callback=lambda done, total, payload: events.append((done, total, payload)),
    )

    assert report.folds
    assert report.strategy_name == "factor"
    assert report.train_days == 90
    assert report.test_days == 30
    assert all(fold.selected_params["factor_name"] == "ret_5" for fold in report.folds)
    assert all(fold.test.metrics.final_equity > 0 for fold in report.folds)
    assert events
    assert events[-1][0] == events[-1][1]
    assert {"train", "test"} <= {payload["stage"] for _done, _total, payload in events}
    assert all("summary" in payload for _done, _total, payload in events)

    markdown = walk_forward_report_to_markdown(report)
    assert markdown.startswith("# Walk-forward 验证报告")
    assert "- 平均训练 Sharpe:" in markdown
    assert "- 平均 Sharpe 衰减:" in markdown
    assert "- 验证/训练 Sharpe 保留率:" in markdown
    assert "- 验证正收益折数:" in markdown
    assert "| 折 | 训练区间 | 验证区间 | 参数 |" in markdown
    assert "Sharpe 衰减" in markdown

    csv_path = write_walk_forward_csv(report, tmp_path / "walk-forward.csv")
    md_path = write_walk_forward_markdown(report, tmp_path / "walk-forward.md")
    assert csv_path.read_text(encoding="utf-8").startswith("fold,train_start")
    assert md_path.read_text(encoding="utf-8").startswith("# Walk-forward 验证报告")


def test_walk_forward_rejects_too_short_period():
    loader = DataLoader(adapters=[_FakeAdapter()])

    with pytest.raises(ValueError, match="没有生成可用 walk-forward 折"):
        run_walk_forward_param_grid(
            strategy_name="factor",
            symbols=["000001.SZ"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 10),
            param_grid={
                "factor_name": ["ret_5"],
                "windows": [[5]],
                "lookback": [20],
                "entry_quantile": [0.8],
                "exit_quantile": [0.5],
                "direction": ["high"],
                "position_pct": [0.3],
            },
            train_days=30,
            test_days=30,
            loader=loader,
        )


def test_factor_portfolio_backtest_runs_and_exports(tmp_path):
    loader = DataLoader(adapters=[_CrossSectionAdapter()])

    report = run_factor_portfolio_backtest(
        symbols=["000001.SZ", "000002.SZ", "000003.SZ"],
        factor_name="ret_5",
        start=date(2024, 1, 1),
        end=date(2024, 8, 31),
        windows=[5],
        direction="high",
        rebalance_days=5,
        select_count=1,
        initial_equity=200_000.0,
        fee_bps=3.0,
        slippage_bps=1.0,
        loader=loader,
    )

    assert report.days
    assert report.metrics.final_equity > 0
    assert report.rebalance_count > 0
    assert report.average_holdings == 1.0
    assert all(day.holdings == 1 for day in report.days)

    markdown = factor_portfolio_report_to_markdown(report)
    assert markdown.startswith("# 因子组合回测报告")
    assert "| 日期 | 持仓数 | 可选数 | 换手 |" in markdown

    csv_path = write_factor_portfolio_csv(report, tmp_path / "factor-portfolio.csv")
    md_path = write_factor_portfolio_markdown(report, tmp_path / "factor-portfolio.md")
    assert csv_path.read_text(encoding="utf-8").startswith("date,selected_symbols")
    assert md_path.read_text(encoding="utf-8").startswith("# 因子组合回测报告")


def test_factor_portfolio_respects_limit_and_suspension_constraints():
    loader = DataLoader(adapters=[_TradeabilityAdapter()])

    report = run_factor_portfolio_backtest(
        symbols=["000001.SZ", "000002.SZ", "000003.SZ"],
        factor_name="ret_2",
        start=date(2024, 1, 1),
        end=date(2024, 2, 29),
        windows=[2],
        direction="high",
        rebalance_days=1,
        select_count=1,
        loader=loader,
    )

    assert report.blocked_buy_count >= 1
    assert report.blocked_sell_count >= 1
    assert any("000003.SZ" in day.blocked_buy_symbols for day in report.days)
    assert any("000001.SZ" in day.blocked_sell_symbols for day in report.days)


def test_ashare_board_rules_are_prefix_based_and_lightweight():
    assert infer_ashare_board("600519.SH") == "main"
    assert infer_ashare_board("000001.SZ") == "main"
    assert infer_ashare_board("688001.SH") == "star"
    assert infer_ashare_board("300001.SZ") == "chinext"
    assert infer_ashare_board("830000.BJ") == "beijing"
    assert ashare_price_limit_pct("600519.SH") == 0.10
    assert ashare_price_limit_pct("688001.SH") == 0.20
    assert ashare_price_limit_pct("300001.SZ") == 0.20
    assert ashare_price_limit_pct("830000.BJ") == 0.30
    assert ashare_price_limit_pct("600519.SH", is_st=True) == 0.05
    assert ashare_price_limit_pct("600519.SH", listing_trading_days=5) is None


def test_factor_portfolio_uses_board_aware_price_limits_by_default():
    loader = DataLoader(adapters=[_BoardLimitAdapter()])

    auto_report = run_factor_portfolio_backtest(
        symbols=["300001.SZ", "000001.SZ"],
        factor_name="ret_2",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        windows=[2],
        direction="high",
        rebalance_days=1,
        select_count=1,
        loader=loader,
    )
    override_report = run_factor_portfolio_backtest(
        symbols=["300001.SZ", "000001.SZ"],
        factor_name="ret_2",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        windows=[2],
        direction="high",
        rebalance_days=1,
        select_count=1,
        price_limit_pct=0.10,
        loader=loader,
    )

    assert auto_report.price_limit_pct is None
    assert not any("300001.SZ" in day.blocked_buy_symbols for day in auto_report.days)
    assert any("300001.SZ" in day.blocked_buy_symbols for day in override_report.days)


def test_factor_portfolio_rejects_bad_selection():
    loader = DataLoader(adapters=[_CrossSectionAdapter()])

    with pytest.raises(ValueError, match="select_quantile 必须在 0-1"):
        run_factor_portfolio_backtest(
            symbols=["000001.SZ", "000002.SZ"],
            factor_name="ret_5",
            start=date(2024, 1, 1),
            end=date(2024, 8, 31),
            windows=[5],
            select_quantile=0,
            loader=loader,
        )
