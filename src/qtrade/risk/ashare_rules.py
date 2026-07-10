"""Lightweight China A-share trading rule helpers.

These helpers intentionally avoid external calendars or security master
dependencies. They infer the common board-level defaults from symbol prefixes
and allow callers to override when they have more precise metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qtrade.data.schema import normalize_symbol, split_symbol

AshareBoard = Literal["main", "star", "chinext", "beijing", "unknown"]


@dataclass(frozen=True)
class AShareInstrumentRule:
    symbol: str
    board: AshareBoard
    price_limit_pct: float | None
    lot_size: int
    tick_size: float
    note: str = ""


def infer_ashare_board(symbol: str) -> AshareBoard:
    code, exchange = split_symbol(symbol)
    if exchange == "BJ" or code.startswith(("4", "8", "92")):
        return "beijing"
    if exchange == "SH" and code.startswith("68"):
        return "star"
    if exchange == "SZ" and code.startswith("30"):
        return "chinext"
    if code.startswith(("60", "00")):
        return "main"
    return "unknown"


def ashare_price_limit_pct(
    symbol: str,
    *,
    is_st: bool = False,
    listing_trading_days: int | None = None,
) -> float | None:
    """Return the board-level daily price limit percentage.

    ``None`` means no daily price limit should be assumed. The first five
    trading days after listing are modeled as no-limit because recent A-share
    registration-system rules use that broad treatment across major boards.
    Callers with exact listing metadata should pass ``listing_trading_days``.
    """
    if listing_trading_days is not None and listing_trading_days <= 5:
        return None
    if is_st:
        return 0.05
    board = infer_ashare_board(symbol)
    if board in {"star", "chinext"}:
        return 0.20
    if board == "beijing":
        return 0.30
    return 0.10


def ashare_instrument_rule(
    symbol: str,
    *,
    is_st: bool = False,
    listing_trading_days: int | None = None,
) -> AShareInstrumentRule:
    normalized = normalize_symbol(symbol)
    board = infer_ashare_board(normalized)
    price_limit_pct = ashare_price_limit_pct(
        normalized,
        is_st=is_st,
        listing_trading_days=listing_trading_days,
    )
    return AShareInstrumentRule(
        symbol=normalized,
        board=board,
        price_limit_pct=price_limit_pct,
        lot_size=100,
        tick_size=0.01,
        note="prefix-inferred board rule",
    )
