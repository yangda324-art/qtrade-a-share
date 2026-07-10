"""Per-run in-memory bar cache for backtests.

This cache is intentionally short-lived: callers create it for one research
command and discard it afterwards. It reduces repeated disk/network reads when
parameter grids or walk-forward loops request the same bars many times.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from qtrade.data.schema import AdjustMode, normalize_symbol


@dataclass(frozen=True)
class BarCacheStats:
    hits: int
    misses: int
    entries: int
    rows: int


class InMemoryBarLoader:
    """Duck-typed DataLoader wrapper with a per-command bar cache."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._cache: dict[tuple[str, date, date | None, AdjustMode, str], pd.DataFrame] = {}
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> BarCacheStats:
        return BarCacheStats(
            hits=self._hits,
            misses=self._misses,
            entries=len(self._cache),
            rows=sum(len(df) for df in self._cache.values()),
        )

    def get_daily_bars(
        self,
        symbol: str,
        *,
        start: date | datetime | str,
        end: date | datetime | str | None = None,
        adjust: AdjustMode = "qfq",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return self.get_bars(
            symbol,
            start=start,
            end=end,
            adjust=adjust,
            period="daily",
            force_refresh=force_refresh,
        )

    def get_bars(
        self,
        symbol: str,
        *,
        start: date | datetime | str,
        end: date | datetime | str | None = None,
        adjust: AdjustMode = "qfq",
        period: str = "daily",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        key = (
            normalize_symbol(symbol),
            pd.to_datetime(start).date(),
            pd.to_datetime(end).date() if end is not None else None,
            adjust,
            period,
        )
        if not force_refresh and key in self._cache:
            self._hits += 1
            return self._cache[key].copy(deep=False)

        self._misses += 1
        if hasattr(self._loader, "get_bars"):
            df = self._loader.get_bars(
                symbol,
                start=start,
                end=end,
                adjust=adjust,
                period=period,
                force_refresh=force_refresh,
            )
        elif period == "daily":
            df = self._loader.get_daily_bars(
                symbol,
                start=start,
                end=end,
                adjust=adjust,
                force_refresh=force_refresh,
            )
        else:
            raise AttributeError(f"wrapped loader does not support period {period}")
        self._cache[key] = df.copy(deep=False)
        return df.copy(deep=False)

    def get_quotes(self, symbols: list[str]) -> dict:
        return self._loader.get_quotes(symbols)

    def get_news(self, symbol: str | None = None, *, limit: int = 20):
        return self._loader.get_news(symbol, limit=limit)


def ensure_in_memory_bar_loader(loader: Any) -> InMemoryBarLoader:
    if isinstance(loader, InMemoryBarLoader):
        return loader
    return InMemoryBarLoader(loader)
