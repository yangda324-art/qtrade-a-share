#!/usr/bin/env python3
"""批量下载 A 股日线历史到本地 Parquet 缓存。

用法::

    python scripts/download_history.py --symbols 600519 000001 --start 2023-01-01

不传 ``--symbols`` 时会下载一个默认演示池。
"""
from __future__ import annotations

import argparse
from datetime import date

from qtrade.core.logging import get_logger
from qtrade.data.loader import get_loader
from qtrade.data.schema import normalize_symbol

DEFAULT_POOL = ["600519", "000001", "601318", "300750", "000333"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_POOL)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true", help="忽略缓存重新拉取")
    args = parser.parse_args()

    log = get_logger(__name__)
    loader = get_loader()
    end = date.fromisoformat(args.end) if args.end else date.today()

    for sym in args.symbols:
        s = normalize_symbol(sym)
        df = loader.get_daily_bars(s, start=args.start, end=end, force_refresh=args.force)
        log.info("{} : {} rows ({} → {})",
                 s, len(df), df["date"].min().date() if not df.empty else "?",
                 df["date"].max().date() if not df.empty else "?")
        print(f"✓ {s}: {len(df)} rows")


if __name__ == "__main__":
    main()
