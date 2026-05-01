"""数据库连接与会话管理。

设计：
- SQLite 即可满足单机量化场景。多账户/多策略并发也基本无压力。
- 用 ``session_scope()`` 上下文管理器统一处理 commit/rollback，避免业务代码到处写 try。
- ``init_db()`` 调一次即可建表（首次启动或 ``scripts/init_db.py``）。
- 后续如果加新字段，用 Alembic 做迁移；早期开发阶段直接 ``init_db`` 重建也行。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from qtrade.core.config import get_settings
from qtrade.persistence.models import Base


def _build_engine() -> Engine:
    settings = get_settings()
    url = f"sqlite:///{settings.paths.db_path}"
    engine = create_engine(
        url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},  # 多线程读
    )

    # SQLite 默认外键关闭，必须显式开启
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")  # 并发读写更友好
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """建表（幂等）。首次启动或重置时调用。"""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务作用域。

    用法::

        with session_scope() as s:
            s.add(order)
            # 自动 commit；异常时自动 rollback
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
