"""多策略对比 tab：并排显示 N 个活跃 run 的核心指标 + 权益曲线叠加。

数据源：
- 表：``all_active_runs()`` + 每个 run 的 ``run_summary()``
- 图：每个 run 的 ``equity_curve()``，叠加显示

关键判断指标：累计收益、Sharpe、最终权益、当前持仓数 —— 一眼看出哪个策略真有 edge。
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qtrade.ui.data_access import (
    all_active_runs,
    equity_curve,
    recent_runs,
    run_summary,
)
from qtrade.ui.widgets.chart_utils import lock_view_to_data

REFRESH_MS = 5000
PALETTE = ["#5b8bd6", "#caa14a", "#3aa055", "#d04949", "#b07cc6", "#7cc6b0", "#d09a4a"]


def _pnl_color(v: float) -> str:
    if v > 0:
        return "#d04949"
    if v < 0:
        return "#3aa055"
    return "#aaaaaa"


class CompareWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)

        topbar = QHBoxLayout()
        topbar.addWidget(QLabel("<h3>多策略对比</h3>"))
        topbar.addStretch()
        self.cb_active_only = QCheckBox("只看活跃 run（未 ended）")
        self.cb_active_only.setChecked(False)     # 默认显示最近 24h 全部
        self.cb_active_only.stateChanged.connect(self.refresh)
        topbar.addWidget(self.cb_active_only)
        root.addLayout(topbar)

        splitter = QSplitter(Qt.Vertical)

        # 上：表格
        cols = ["Run", "策略", "模式", "状态", "总资产", "累计收益", "Sharpe", "持仓", "现金", "已实现", "未实现"]
        self.tbl = QTableWidget(0, len(cols))
        self.tbl.setHorizontalHeaderLabels(cols)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        splitter.addWidget(self.tbl)

        # 下：权益曲线叠加
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "总资产")
        self.plot.addLegend(offset=(10, 10))
        splitter.addWidget(self.plot)

        splitter.setSizes([300, 400])
        root.addWidget(splitter, stretch=1)

        self.lbl_status = QLabel("等待数据…")
        self.lbl_status.setStyleSheet("color:#888;")
        root.addWidget(self.lbl_status)

    def refresh(self):
        if self.cb_active_only.isChecked():
            runs = all_active_runs()
            scope_msg = "活跃 run"
        else:
            runs = recent_runs(hours=24)
            scope_msg = "最近 24h 内启动"

        if not runs:
            self.lbl_status.setText(f"无{scope_msg}。先 qtrade run-multi 启动几个策略")
            self.tbl.setRowCount(0)
            self.plot.clear()
            return

        # 表格
        self.tbl.setRowCount(len(runs))
        for r, run in enumerate(runs):
            summary = run_summary(run["run_id"])
            ended = run.get("ended_at")
            status = "● running" if ended is None else "○ ended"
            cells = [
                f"#{run['run_id']}",
                run["strategy_id"],
                run["mode"],
                status,
                f"{summary.get('total_assets', 0):,.2f}",
                f"{summary.get('cum_return', 0):+.2%}",
                f"{summary.get('sharpe', 0):.2f}",
                str(summary.get("n_positions", 0)),
                f"{summary.get('cash', 0):,.0f}",
                f"{summary.get('realized_pnl', 0):+,.2f}",
                f"{summary.get('unrealized_pnl', 0):+,.2f}",
            ]
            for c, v in enumerate(cells):
                item = QTableWidgetItem(str(v))
                if c == 3:    # 状态列
                    item.setForeground(QColor("#3aa055" if ended is None else "#888"))
                elif c == 5 or c == 6:    # 累计收益 / Sharpe
                    val = summary.get("cum_return", 0) if c == 5 else summary.get("sharpe", 0)
                    item.setForeground(QColor(_pnl_color(val)))
                if c >= 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl.setItem(r, c, item)

        # 权益曲线叠加
        self.plot.clear()
        all_xs: list[float] = []
        all_ys: list[float] = []
        for i, run in enumerate(runs):
            curve = equity_curve(run["run_id"], limit=500)
            if not curve:
                continue
            ys = [c["total_assets"] for c in curve]
            xs = list(range(len(ys)))
            self.plot.plot(
                xs, ys,
                pen=pg.mkPen(PALETTE[i % len(PALETTE)], width=2),
                name=f"#{run['run_id']} {run['strategy_id']}",
            )
            all_xs.extend(xs)
            all_ys.extend(ys)
        # 锁视图：用所有 run 的并集
        if all_xs and all_ys:
            lock_view_to_data(
                self.plot, xs=all_xs, ys=all_ys,
                x_padding=0.02, y_padding=0.05, min_visible_x=5,
            )

        active = sum(1 for r in runs if r.get("ended_at") is None)
        self.lbl_status.setText(f"{len(runs)} 个 run（{active} 活跃 / {len(runs) - active} 已结束）")
