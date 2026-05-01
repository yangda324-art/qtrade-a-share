#!/usr/bin/env python3
"""初始化 SQLite 数据库（建表）。

幂等：已存在的表不会重建。

用法::

    python scripts/init_db.py
"""
from __future__ import annotations

from qtrade.core.config import get_settings
from qtrade.core.logging import get_logger
from qtrade.persistence import init_db


def main() -> None:
    log = get_logger(__name__)
    settings = get_settings()
    init_db()
    log.info("✓ database initialized at {}", settings.paths.db_path)
    print(f"✓ database initialized at {settings.paths.db_path}")


if __name__ == "__main__":
    main()
