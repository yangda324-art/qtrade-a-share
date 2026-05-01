"""PySide6 主窗口骨架。

5 个 tab 占位：Dashboard / K 线 / 回测 / 订单 / Agent 决策。
每个 tab 先放一个简单的占位组件，后续会话里再迭代填充内容。

启动::

    python -m qtrade.ui.main
"""
from __future__ import annotations

import sys

from qtrade.core.logging import get_logger

log = get_logger(__name__)


def _import_qt():
    try:
        from PySide6.QtWidgets import (
            QApplication,
            QLabel,
            QMainWindow,
            QStatusBar,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as e:  # pragma: no cover
        raise SystemExit("PySide6 未安装：pip install PySide6") from e
    return QApplication, QLabel, QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget


def _placeholder(label: str, hint: str = ""):
    QApplication, QLabel, QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget = _import_qt()
    w = QWidget()
    layout = QVBoxLayout(w)
    title = QLabel(f"<h2>{label}</h2>")
    layout.addWidget(title)
    if hint:
        layout.addWidget(QLabel(hint))
    layout.addStretch()
    return w


def _build_dashboard():
    return _placeholder(
        "Dashboard",
        "运行状态 / 当日 P&L / 持仓总览 / 风险预警一览。\n（Phase 4 完成）",
    )


def _build_kline():
    QApplication, QLabel, QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget = _import_qt()
    w = QWidget()
    layout = QVBoxLayout(w)
    layout.addWidget(QLabel("<h2>K 线</h2>"))
    try:
        import pyqtgraph as pg
        plot = pg.PlotWidget(title="K 线（占位）")
        plot.showGrid(x=True, y=True)
        layout.addWidget(plot)
    except ImportError:
        layout.addWidget(QLabel("pyqtgraph 未安装；安装后这里会显示 K 线"))
    layout.addStretch()
    return w


def _build_backtest():
    return _placeholder(
        "回测",
        "选择策略 + 标的 + 区间 → 跑回测 → 看权益曲线和指标。\n"
        "Phase 1 已可用 CLI 跑回测：qtrade backtest --strategy ma_cross --symbol 600519",
    )


def _build_orders():
    return _placeholder(
        "订单 / 持仓",
        "实时订单状态 + 当前持仓 + 资金账本。\n（接入模拟盘 / 实盘后可用）",
    )


def _build_agent_log():
    return _placeholder(
        "Agent 决策",
        "每次多 Agent 决策的完整链路：每个 Agent 的输入 / 输出 / 用时 / token。\n"
        "数据来自 SQLite agent_decisions 表。",
    )


def run() -> None:
    QApplication, QLabel, QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget = _import_qt()

    app = QApplication.instance() or QApplication(sys.argv)

    win = QMainWindow()
    win.setWindowTitle("qtrade — A 股 LLM 多 Agent 量化平台")
    win.resize(1200, 800)

    tabs = QTabWidget()
    tabs.addTab(_build_dashboard(), "Dashboard")
    tabs.addTab(_build_kline(), "K 线")
    tabs.addTab(_build_backtest(), "回测")
    tabs.addTab(_build_orders(), "订单 / 持仓")
    tabs.addTab(_build_agent_log(), "Agent 决策")
    win.setCentralWidget(tabs)

    bar = QStatusBar()
    bar.showMessage("Phase 0：UI 骨架。后续会话会逐 tab 填充。")
    win.setStatusBar(bar)

    win.show()
    log.info("UI started")
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
