"""Factor mining helpers for historical research.

This module is deliberately model-free: it generates simple price/volume factors
from historical bars and evaluates them against future returns with an
in-sample / out-of-sample split. It does not trade, simulate orders, or touch
runtime state.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtrade.core.time import utcnow
from qtrade.data.loader import DataLoader, get_loader


@dataclass(frozen=True)
class FactorWindowScore:
    n_obs: int
    ic: float
    rank_ic: float
    top_return: float
    bottom_return: float
    spread_return: float


@dataclass(frozen=True)
class FactorScore:
    name: str
    expression: str
    horizon_days: int
    train: FactorWindowScore
    test: FactorWindowScore
    train_start: date | None
    train_end: date | None
    test_start: date | None
    test_end: date | None

    @property
    def stable_rank_ic_sign(self) -> bool:
        return self.train.rank_ic == 0 or self.test.rank_ic == 0 or self.train.rank_ic * self.test.rank_ic > 0

    def to_row(self) -> dict[str, Any]:
        return {
            "factor": self.name,
            "expression": self.expression,
            "horizon_days": self.horizon_days,
            "train_start": self.train_start.isoformat() if self.train_start else "",
            "train_end": self.train_end.isoformat() if self.train_end else "",
            "test_start": self.test_start.isoformat() if self.test_start else "",
            "test_end": self.test_end.isoformat() if self.test_end else "",
            "train_n": self.train.n_obs,
            "test_n": self.test.n_obs,
            "train_ic": self.train.ic,
            "test_ic": self.test.ic,
            "train_rank_ic": self.train.rank_ic,
            "test_rank_ic": self.test.rank_ic,
            "train_spread_return": self.train.spread_return,
            "test_spread_return": self.test.spread_return,
            "stable_rank_ic_sign": self.stable_rank_ic_sign,
        }


@dataclass(frozen=True)
class FactorMiningReport:
    symbols: list[str]
    start: date
    end: date
    horizon_days: int
    train_ratio: float
    rows: list[FactorScore]
    generated_at: datetime = field(default_factory=utcnow)

    @property
    def best_by_test_rank_ic_abs(self) -> FactorScore | None:
        if not self.rows:
            return None
        return max(self.rows, key=lambda row: abs(row.test.rank_ic))


@dataclass(frozen=True)
class FactorLayerBucket:
    bucket: int
    n_obs: int
    avg_forward_return: float


@dataclass(frozen=True)
class FactorLayerWindow:
    n_obs: int
    rank_ic: float
    top_return: float
    bottom_return: float
    spread_return: float
    monotonicity: float
    turnover: float
    buckets: list[FactorLayerBucket]


@dataclass(frozen=True)
class FactorLayerReport:
    symbols: list[str]
    factor: str
    expression: str
    start: date
    end: date
    horizon_days: int
    train_ratio: float
    groups: int
    train: FactorLayerWindow
    test: FactorLayerWindow
    generated_at: datetime = field(default_factory=utcnow)

    @property
    def stable_spread_sign(self) -> bool:
        return (
            self.train.spread_return == 0
            or self.test.spread_return == 0
            or self.train.spread_return * self.test.spread_return > 0
        )

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for window_name, window in (("train", self.train), ("test", self.test)):
            for bucket in window.buckets:
                rows.append(
                    {
                        "factor": self.factor,
                        "expression": self.expression,
                        "window": window_name,
                        "bucket": bucket.bucket,
                        "n_obs": bucket.n_obs,
                        "avg_forward_return": bucket.avg_forward_return,
                        "rank_ic": window.rank_ic,
                        "top_return": window.top_return,
                        "bottom_return": window.bottom_return,
                        "spread_return": window.spread_return,
                        "monotonicity": window.monotonicity,
                        "turnover": window.turnover,
                        "stable_spread_sign": self.stable_spread_sign,
                    }
                )
        return rows


def run_factor_mining(
    *,
    symbols: list[str],
    start: date,
    end: date,
    windows: list[int] | None = None,
    horizon_days: int = 5,
    train_ratio: float = 0.7,
    include_interactions: bool = False,
    max_interactions: int = 50,
    loader: DataLoader | None = None,
) -> FactorMiningReport:
    if not symbols:
        raise ValueError("至少需要一个标的")
    if horizon_days <= 0:
        raise ValueError("horizon_days 必须为正数")
    if not 0.1 <= train_ratio <= 0.9:
        raise ValueError("train_ratio 必须在 0.1-0.9 之间")
    windows = sorted(set(windows or [5, 20, 60]))
    if any(window <= 1 for window in windows):
        raise ValueError("window 必须大于 1")

    loader = loader or get_loader()
    factor_data = _load_factor_data(
        symbols=symbols,
        start=start,
        end=end,
        windows=windows,
        horizon_days=horizon_days,
        loader=loader,
    )
    candidates = build_factor_candidates(
        factor_data,
        include_interactions=include_interactions,
        max_interactions=max_interactions,
    )
    rows = [
        evaluate_factor(
            factor_data,
            name=name,
            values=values,
            expression=expression,
            horizon_days=horizon_days,
            train_ratio=train_ratio,
        )
        for name, (expression, values) in candidates.items()
    ]
    rows = [row for row in rows if row.train.n_obs > 0 and row.test.n_obs > 0]
    return FactorMiningReport(
        symbols=list(symbols),
        start=start,
        end=end,
        horizon_days=horizon_days,
        train_ratio=train_ratio,
        rows=sort_factor_scores(rows),
    )


def run_factor_layering(
    *,
    symbols: list[str],
    factor_name: str,
    start: date,
    end: date,
    windows: list[int] | None = None,
    horizon_days: int = 5,
    train_ratio: float = 0.7,
    groups: int = 5,
    include_interactions: bool = False,
    max_interactions: int = 50,
    loader: DataLoader | None = None,
) -> FactorLayerReport:
    if not symbols:
        raise ValueError("至少需要一个标的")
    if not factor_name:
        raise ValueError("factor_name 不能为空")
    if horizon_days <= 0:
        raise ValueError("horizon_days 必须为正数")
    if groups < 2:
        raise ValueError("groups 必须至少为 2")
    if not 0.1 <= train_ratio <= 0.9:
        raise ValueError("train_ratio 必须在 0.1-0.9 之间")
    windows = sorted(set(windows or [5, 20, 60]))
    if any(window <= 1 for window in windows):
        raise ValueError("window 必须大于 1")

    loader = loader or get_loader()
    factor_data = _load_factor_data(
        symbols=symbols,
        start=start,
        end=end,
        windows=windows,
        horizon_days=horizon_days,
        loader=loader,
    )
    candidates = build_factor_candidates(
        factor_data,
        include_interactions=include_interactions,
        max_interactions=max_interactions,
    )
    if factor_name not in candidates:
        raise ValueError(f"未知因子：{factor_name}（可用示例：{list(candidates)[:12]}）")
    expression, values = candidates[factor_name]
    sample = _factor_sample(factor_data, values)
    train, test = _split_by_date(sample, train_ratio=train_ratio)
    return FactorLayerReport(
        symbols=list(symbols),
        factor=factor_name,
        expression=expression,
        start=start,
        end=end,
        horizon_days=horizon_days,
        train_ratio=train_ratio,
        groups=groups,
        train=_layer_window(train, groups=groups),
        test=_layer_window(test, groups=groups),
    )


def build_factor_candidates(
    factor_data: pd.DataFrame,
    *,
    include_interactions: bool = False,
    max_interactions: int = 50,
) -> dict[str, tuple[str, pd.Series]]:
    base_cols = [
        col
        for col in factor_data.columns
        if col not in {"date", "symbol", "target_return"}
    ]
    candidates: dict[str, tuple[str, pd.Series]] = {}
    for col in base_cols:
        s = factor_data[col].astype(float)
        candidates[col] = (col, s)
        candidates[f"{col}__square"] = (f"{col}^2", s.pow(2))
        candidates[f"{col}__signed_square"] = (f"{col}*abs({col})", s * s.abs())

    if include_interactions:
        for idx, (left, right) in enumerate(combinations(base_cols, 2)):
            if idx >= max_interactions:
                break
            name = f"{left}__x__{right}"
            candidates[name] = (f"{left}*{right}", factor_data[left].astype(float) * factor_data[right].astype(float))
    return candidates


def compute_factor_series(
    df: pd.DataFrame,
    *,
    factor_name: str,
    windows: list[int] | None = None,
    include_interactions: bool = False,
    max_interactions: int = 50,
) -> pd.Series:
    """Compute one factor series from historical bars.

    The returned series is aligned to ``df`` rows and uses only information from
    each row and earlier rows. It is safe for strategy code to call this on
    ``ctx.history``.
    """
    if df.empty:
        return pd.Series(dtype=float)
    windows = sorted(set(windows or [5, 20, 60]))
    if any(window <= 1 for window in windows):
        raise ValueError("window 必须大于 1")
    factor_data = _build_symbol_factor_data(df, windows=windows, horizon_days=1)
    candidates = build_factor_candidates(
        factor_data,
        include_interactions=include_interactions,
        max_interactions=max_interactions,
    )
    if factor_name not in candidates:
        raise ValueError(f"未知因子：{factor_name}（可用示例：{list(candidates)[:12]}）")
    return candidates[factor_name][1]


def evaluate_factor(
    factor_data: pd.DataFrame,
    *,
    name: str,
    values: pd.Series,
    expression: str,
    horizon_days: int,
    train_ratio: float,
) -> FactorScore:
    sample = _factor_sample(factor_data, values)
    train, test = _split_by_date(sample, train_ratio=train_ratio)
    return FactorScore(
        name=name,
        expression=expression,
        horizon_days=horizon_days,
        train=_score_window(train),
        test=_score_window(test),
        train_start=_min_date(train),
        train_end=_max_date(train),
        test_start=_min_date(test),
        test_end=_max_date(test),
    )


def sort_factor_scores(rows: list[FactorScore], *, sort_by: str = "test_rank_ic_abs") -> list[FactorScore]:
    key_func = _FACTOR_SORT_KEYS.get(sort_by)
    if key_func is None:
        raise ValueError(f"未知因子排序指标：{sort_by}（可用：{list(_FACTOR_SORT_KEYS)}）")
    return sorted(rows, key=key_func, reverse=True)


def limit_factor_report(report: FactorMiningReport, *, top: int | None) -> FactorMiningReport:
    if top is None or top <= 0:
        return report
    return FactorMiningReport(
        symbols=report.symbols,
        start=report.start,
        end=report.end,
        horizon_days=report.horizon_days,
        train_ratio=report.train_ratio,
        rows=report.rows[:top],
        generated_at=report.generated_at,
    )


def sort_factor_report(report: FactorMiningReport, *, sort_by: str = "test_rank_ic_abs") -> FactorMiningReport:
    return FactorMiningReport(
        symbols=report.symbols,
        start=report.start,
        end=report.end,
        horizon_days=report.horizon_days,
        train_ratio=report.train_ratio,
        rows=sort_factor_scores(report.rows, sort_by=sort_by),
        generated_at=report.generated_at,
    )


def factor_report_to_markdown(report: FactorMiningReport) -> str:
    lines = [
        "# 因子挖掘报告",
        "",
        f"- 生成时间: {report.generated_at.isoformat()}",
        f"- 标的: {','.join(report.symbols)}",
        f"- 区间: {report.start.isoformat()} → {report.end.isoformat()}",
        f"- 预测目标: 未来 {report.horizon_days} 日收益",
        f"- 样本切分: {report.train_ratio:.0%} 样本内 / {1 - report.train_ratio:.0%} 样本外",
    ]
    best = report.best_by_test_rank_ic_abs
    if best:
        lines.append(f"- 样本外 Rank IC 绝对值最高: {best.name} ({best.test.rank_ic:.4f})")
    lines.extend(
        [
            "",
            "| 因子 | 表达式 | 样本内 Rank IC | 样本外 Rank IC | 样本外 IC | 样本外 Top-Bottom | 稳定同向 | 样本外 N |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in report.rows:
        lines.append(
            "| "
            f"{row.name} | "
            f"{row.expression} | "
            f"{row.train.rank_ic:.4f} | "
            f"{row.test.rank_ic:.4f} | "
            f"{row.test.ic:.4f} | "
            f"{row.test.spread_return:.4%} | "
            f"{'yes' if row.stable_rank_ic_sign else 'no'} | "
            f"{row.test.n_obs} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_factor_report_markdown(report: FactorMiningReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(factor_report_to_markdown(report), encoding="utf-8")
    return out


def write_factor_report_csv(report: FactorMiningReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.to_row() for row in report.rows]
    fieldnames = list(rows[0]) if rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def factor_layer_report_to_markdown(report: FactorLayerReport) -> str:
    lines = [
        "# 因子分层验证报告",
        "",
        f"- 生成时间: {report.generated_at.isoformat()}",
        f"- 标的: {','.join(report.symbols)}",
        f"- 因子: {report.factor}",
        f"- 表达式: {report.expression}",
        f"- 区间: {report.start.isoformat()} → {report.end.isoformat()}",
        f"- 预测目标: 未来 {report.horizon_days} 日收益",
        f"- 分组数: {report.groups}",
        f"- 样本切分: {report.train_ratio:.0%} 样本内 / {1 - report.train_ratio:.0%} 样本外",
        f"- 样本外 Top-Bottom: {report.test.spread_return:.4%}",
        f"- 样本外换手率估算: {report.test.turnover:.2%}",
        f"- 样本内/样本外价差同向: {'yes' if report.stable_spread_sign else 'no'}",
        "",
        "| 样本 | Rank IC | Top | Bottom | Top-Bottom | 单调性 | 换手率 | N |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _layer_summary_row("train", report.train),
        _layer_summary_row("test", report.test),
        "",
        "| 样本 | 分组 | N | 平均未来收益 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for window_name, window in (("train", report.train), ("test", report.test)):
        for bucket in window.buckets:
            lines.append(
                "| "
                f"{window_name} | "
                f"{bucket.bucket} | "
                f"{bucket.n_obs} | "
                f"{bucket.avg_forward_return:.4%} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_factor_layer_report_markdown(report: FactorLayerReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(factor_layer_report_to_markdown(report), encoding="utf-8")
    return out


def write_factor_layer_report_csv(report: FactorLayerReport, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = report.to_rows()
    fieldnames = list(rows[0]) if rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _load_factor_data(
    *,
    symbols: list[str],
    start: date,
    end: date,
    windows: list[int],
    horizon_days: int,
    loader: DataLoader,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = loader.get_daily_bars(symbol, start=start, end=end)
        if df.empty:
            continue
        frames.append(_build_symbol_factor_data(df, windows=windows, horizon_days=horizon_days))
    if not frames:
        raise ValueError("没有可用历史 K 线，无法挖掘因子")
    return pd.concat(frames, ignore_index=True)


def _factor_sample(factor_data: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(factor_data["date"]).dt.date,
            "symbol": factor_data["symbol"],
            "factor": values,
            "target_return": factor_data["target_return"],
        }
    )
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
    return sample.sort_values(["date", "symbol"])


def _build_symbol_factor_data(df: pd.DataFrame, *, windows: list[int], horizon_days: int) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").reset_index(drop=True)
    close = work["close"].astype(float)
    volume = work["volume"].astype(float)
    one_day_return = close.pct_change()
    out = work[["date", "symbol"]].copy()
    for window in windows:
        out[f"ret_{window}"] = close.pct_change(window)
        out[f"volatility_{window}"] = one_day_return.rolling(window).std()
        out[f"ma_gap_{window}"] = close / close.rolling(window).mean() - 1
        out[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean() - 1
        out[f"range_mean_{window}"] = ((work["high"].astype(float) - work["low"].astype(float)) / close).rolling(window).mean()
    out["target_return"] = close.shift(-horizon_days) / close - 1
    return out


def _split_by_date(sample: pd.DataFrame, *, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(sample["date"].dropna().unique())
    if len(unique_dates) < 2:
        return sample.iloc[0:0], sample.iloc[0:0]
    split_idx = int(len(unique_dates) * train_ratio)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    train_dates = set(unique_dates[:split_idx])
    return sample[sample["date"].isin(train_dates)], sample[~sample["date"].isin(train_dates)]


def _layer_window(sample: pd.DataFrame, *, groups: int) -> FactorLayerWindow:
    n = len(sample)
    if n < groups:
        return FactorLayerWindow(
            n_obs=n,
            rank_ic=0.0,
            top_return=0.0,
            bottom_return=0.0,
            spread_return=0.0,
            monotonicity=0.0,
            turnover=0.0,
            buckets=[],
        )
    grouped = sample.copy()
    grouped["bucket"] = _assign_factor_buckets(grouped["factor"], groups=groups)
    bucket_rows: list[FactorLayerBucket] = []
    for bucket in range(1, groups + 1):
        part = grouped[grouped["bucket"] == bucket]
        bucket_rows.append(
            FactorLayerBucket(
                bucket=bucket,
                n_obs=len(part),
                avg_forward_return=_safe_mean(part["target_return"]),
            )
        )
    top = bucket_rows[-1].avg_forward_return
    bottom = bucket_rows[0].avg_forward_return
    bucket_numbers = pd.Series([row.bucket for row in bucket_rows], dtype=float)
    bucket_returns = pd.Series([row.avg_forward_return for row in bucket_rows], dtype=float)
    return FactorLayerWindow(
        n_obs=n,
        rank_ic=_safe_corr(
            grouped["factor"].astype(float).rank(method="average"),
            grouped["target_return"].astype(float).rank(method="average"),
        ),
        top_return=top,
        bottom_return=bottom,
        spread_return=top - bottom,
        monotonicity=_safe_corr(bucket_numbers, bucket_returns),
        turnover=_estimate_top_bucket_turnover(grouped, top_bucket=groups),
        buckets=bucket_rows,
    )


def _assign_factor_buckets(series: pd.Series, *, groups: int) -> pd.Series:
    pct_rank = series.astype(float).rank(pct=True, method="first")
    buckets = np.ceil(pct_rank * groups).astype(int)
    return buckets.clip(lower=1, upper=groups)


def _estimate_top_bucket_turnover(sample: pd.DataFrame, *, top_bucket: int) -> float:
    dates = sorted(sample["date"].dropna().unique())
    if len(dates) < 2:
        return 0.0
    top_by_date = {
        day: set(sample[(sample["date"] == day) & (sample["bucket"] == top_bucket)]["symbol"].astype(str))
        for day in dates
    }
    turnovers: list[float] = []
    prev = top_by_date[dates[0]]
    for day in dates[1:]:
        curr = top_by_date[day]
        union = prev | curr
        if union:
            turnovers.append(1 - len(prev & curr) / len(union))
        prev = curr
    return _safe_mean(pd.Series(turnovers, dtype=float))


def _score_window(sample: pd.DataFrame, *, quantile: float = 0.2) -> FactorWindowScore:
    n = len(sample)
    if n < 3:
        return FactorWindowScore(n_obs=n, ic=0.0, rank_ic=0.0, top_return=0.0, bottom_return=0.0, spread_return=0.0)
    factor = sample["factor"].astype(float)
    target = sample["target_return"].astype(float)
    ic = _safe_corr(factor, target)
    rank_ic = _safe_corr(factor.rank(method="average"), target.rank(method="average"))
    ranks = factor.rank(pct=True, method="average")
    top = _safe_mean(target[ranks >= 1 - quantile])
    bottom = _safe_mean(target[ranks <= quantile])
    return FactorWindowScore(
        n_obs=n,
        ic=ic,
        rank_ic=rank_ic,
        top_return=top,
        bottom_return=bottom,
        spread_return=top - bottom,
    )


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or len(right) < 3:
        return 0.0
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return 0.0
    value = left.corr(right)
    if value is None or not math.isfinite(value):
        return 0.0
    return float(value)


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    value = float(series.mean())
    return value if math.isfinite(value) else 0.0


def _min_date(sample: pd.DataFrame) -> date | None:
    if sample.empty:
        return None
    return min(sample["date"])


def _max_date(sample: pd.DataFrame) -> date | None:
    if sample.empty:
        return None
    return max(sample["date"])


def _layer_summary_row(name: str, window: FactorLayerWindow) -> str:
    return (
        "| "
        f"{name} | "
        f"{window.rank_ic:.4f} | "
        f"{window.top_return:.4%} | "
        f"{window.bottom_return:.4%} | "
        f"{window.spread_return:.4%} | "
        f"{window.monotonicity:.4f} | "
        f"{window.turnover:.2%} | "
        f"{window.n_obs} |"
    )


_FACTOR_SORT_KEYS = {
    "test_rank_ic_abs": lambda row: abs(row.test.rank_ic),
    "test_rank_ic": lambda row: row.test.rank_ic,
    "test_ic_abs": lambda row: abs(row.test.ic),
    "test_ic": lambda row: row.test.ic,
    "test_spread_return_abs": lambda row: abs(row.test.spread_return),
    "test_spread_return": lambda row: row.test.spread_return,
    "train_rank_ic_abs": lambda row: abs(row.train.rank_ic),
}
