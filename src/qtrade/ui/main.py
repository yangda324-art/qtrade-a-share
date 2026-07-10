"""PySide6 主窗口（极简深色 dashboard 风格）。

布局：
    +-------------------------------------------------+
    | [Sidebar 220px] | [Stacked Content]             |
    |                 |                               |
    |                 |   ← 各 view widget 切换显示    |
    |                 |                               |
    +-------------------------------------------------+
    | StatusBar                                       |
    +-------------------------------------------------+

启动::

    qtrade ui
    python -m qtrade.ui.main      # 等价
"""
from __future__ import annotations

import sys

from qtrade.core.logging import get_logger

log = get_logger(__name__)


def _import_qt():
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QStackedWidget,
            QStatusBar,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as e:  # pragma: no cover
        raise SystemExit("PySide6 未安装：pip install PySide6") from e
    return (
        QApplication, Qt, QHBoxLayout, QLabel, QMainWindow,
        QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
    )


def run() -> None:
    (
        QApplication, Qt, QHBoxLayout, QLabel, QMainWindow,
        QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
    ) = _import_qt()

    # 业务 widgets 延迟 import
    from qtrade.ui.sidebar import NavGroup, NavItem, Sidebar
    from qtrade.ui.theme import apply_theme
    from qtrade.ui.windows.agent_log import AgentLogWidget
    from qtrade.ui.windows.backtest import BacktestWidget
    from qtrade.ui.windows.compare import CompareWidget
    from qtrade.ui.windows.dashboard import DashboardWidget
    from qtrade.ui.windows.kline import KlineWidget
    from qtrade.ui.windows.market_watch import MarketWatchWidget
    from qtrade.ui.windows.orders import OrdersWidget
    from qtrade.ui.windows.paper_check import PaperCheckWidget
    from qtrade.ui.windows.performance import PerformanceWidget
    from qtrade.ui.windows.run_control import RunControlWidget
    from qtrade.ui.windows.tick_log import TickLogWidget

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)

    win = QMainWindow()
    win.setWindowTitle("qtrade")
    win.resize(1440, 900)

    # ---- 中央容器：左 sidebar + 右 stack ----
    central = QWidget()
    root = QHBoxLayout(central)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    nav_groups = [
        NavGroup(
            title="交易",
            items=(
                NavItem("dashboard", "▦", "总览"),
                NavItem("run_control", "▶", "模拟盘"),
                NavItem("paper_check", "✓", "模拟验收"),
                NavItem("market_watch", "◌", "实时行情"),
                NavItem("tick_log", "▣", "运行日志"),
                NavItem("performance", "↗", "绩效报告"),
                NavItem("compare",   "≡", "策略对比"),
                NavItem("orders",    "▤", "订单 / 持仓"),
            ),
        ),
        NavGroup(
            title="分析",
            items=(
                NavItem("kline",     "∿", "K 线"),
                NavItem("backtest",  "↻", "回测"),
            ),
        ),
        NavGroup(
            title="智能",
            items=(
                NavItem("agent_log", "✦", "Agent 决策"),
            ),
        ),
    ]
    sidebar = Sidebar(nav_groups)

    stack = QStackedWidget()
    # 内容区有自己的内边距，让卡片不贴边
    stack_wrap = QWidget()
    wrap_layout = QVBoxLayout(stack_wrap)
    wrap_layout.setContentsMargins(20, 18, 20, 18)
    wrap_layout.addWidget(stack)

    # 注册视图
    views: dict[str, QWidget] = {
        "dashboard": DashboardWidget(),
        "run_control": RunControlWidget(),
        "paper_check": PaperCheckWidget(),
        "market_watch": MarketWatchWidget(),
        "tick_log": TickLogWidget(),
        "performance": PerformanceWidget(),
        "compare":   CompareWidget(),
        "orders":    OrdersWidget(),
        "kline":     KlineWidget(),
        "backtest":  BacktestWidget(),
        "agent_log": AgentLogWidget(),
    }
    for w in views.values():
        stack.addWidget(w)

    def _switch(key: str) -> None:
        if key in views:
            stack.setCurrentWidget(views[key])

    sidebar.activated.connect(_switch)

    root.addWidget(sidebar)
    root.addWidget(stack_wrap, stretch=1)
    win.setCentralWidget(central)

    # 默认进入 Dashboard
    sidebar.set_active("dashboard")
    stack.setCurrentWidget(views["dashboard"])

    # 状态栏
    bar = QStatusBar()
    bar.showMessage("qtrade · LLM 多 Agent 量化平台 · Phase 4")
    win.setStatusBar(bar)

    win.show()
    log.info("UI started")
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
