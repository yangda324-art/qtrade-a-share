"""结构化日志。

设计：
- 用 ``loguru`` 替代标准库 logging，零配置开箱即用。
- 控制台彩色输出 + 滚动文件（按天 + 大小双触发）。
- ``bind(strategy=..., symbol=...)`` 可附加结构化上下文，便于事后按维度检索。
- 关键审计事件（订单、风控、决策）走单独文件，与运行日志分离。
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from qtrade.core.config import get_settings

_INITIALIZED = False


def setup_logging(level: str | None = None) -> None:
    """初始化全局日志。幂等。"""
    global _INITIALIZED
    if _INITIALIZED:
        return

    settings = get_settings()
    log_level = (level or settings.log_level or "INFO").upper()
    log_dir: Path = settings.paths.log_dir

    _logger.remove()  # 清掉默认 handler

    # 控制台：彩色简洁
    _logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}:{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=False,  # 生产关掉避免泄露变量值
    )

    # 运行日志：JSON-line 格式便于机器解析
    _logger.add(
        log_dir / "qtrade.log",
        level=log_level,
        rotation="50 MB",
        retention="30 days",
        compression="gz",
        serialize=True,        # 输出 JSON line
        enqueue=True,          # 多进程安全
        backtrace=True,
        diagnose=False,
    )

    # 审计日志：独立文件，不滚动压缩，方便复盘和监管
    _logger.add(
        log_dir / "audit.log",
        level="INFO",
        rotation="100 MB",
        retention="365 days",
        compression="gz",
        serialize=True,
        enqueue=True,
        filter=lambda r: r["extra"].get("audit") is True,
    )

    _INITIALIZED = True
    _logger.debug("logging initialized | level={} | dir={}", log_level, log_dir)


def get_logger(name: str | None = None, **context):
    """获取一个绑定上下文的 logger。

    用法::

        log = get_logger(__name__, strategy="ma_cross_v1", symbol="600519.SH")
        log.info("entry signal | price={}", 1750.0)

    审计场景::

        get_logger(__name__).bind(audit=True).info("ORDER_FILLED ...")
    """
    setup_logging()
    bound = _logger.bind(**context) if context else _logger
    return bound.opt(depth=1) if name is None else bound


# 暴露原始 logger 给希望直接 ``from qtrade.core.logging import logger`` 的代码
logger = _logger
