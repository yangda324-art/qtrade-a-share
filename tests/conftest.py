"""pytest 公共 fixtures。

所有 IO（数据库、缓存、日志）都重定向到临时目录，避免污染开发数据。
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path: Path, monkeypatch):
    """每个测试用独立的 data/log 目录。"""
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    cache_dir = data_dir / "cache"
    db_path = data_dir / "qtrade.sqlite"
    for p in (data_dir, log_dir, cache_dir):
        p.mkdir(parents=True, exist_ok=True)

    # 把 config_dir 指到一个空目录 → 不加载 default.yaml → BaseSettings 用 env 默认值
    empty_cfg = tmp_path / "cfg"
    empty_cfg.mkdir()
    monkeypatch.setenv("QTRADE_CONFIG_DIR", str(empty_cfg))
    monkeypatch.setenv("QTRADE_PATHS__DATA_DIR", str(data_dir))
    monkeypatch.setenv("QTRADE_PATHS__LOG_DIR", str(log_dir))
    monkeypatch.setenv("QTRADE_PATHS__CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("QTRADE_PATHS__DB_PATH", str(db_path))
    monkeypatch.setenv("QTRADE_LOG_LEVEL", "WARNING")

    # 重新加载 settings + db engine（engine 在 import 时已绑死路径，需要 rebuild）
    from qtrade.core import config as cfg
    from qtrade.persistence import db as dbmod
    cfg.get_settings.cache_clear()
    dbmod.engine.dispose()
    new_engine = dbmod._build_engine()
    dbmod.engine = new_engine
    dbmod.SessionLocal.configure(bind=new_engine)
    yield
    cfg.get_settings.cache_clear()
