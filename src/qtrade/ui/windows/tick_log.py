"""运行日志页：展示 LiveRunner 每次 tick 的摘要和每标的状态。"""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from qtrade.ui.data_access import recent_tick_logs

REFRESH_MS = 3000

_ERROR_COLOR = "#d04949"
_OK_COLOR = "#3aa055"
_WARN_COLOR = "#caa14a"


class TickLogWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._logs: list[dict] = []
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(QLabel("<h3>运行日志</h3>"))

        splitter = QSplitter(Qt.Vertical)

        cols = ["时间", "Run", "策略", "标的", "行情", "Fallback", "跳过", "信号", "订单", "风控", "错误", "原因"]
        self.tbl = QTableWidget(0, len(cols))
        self.tbl.setHorizontalHeaderLabels(cols)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.itemSelectionChanged.connect(self._show_detail)
        splitter.addWidget(self.tbl)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setPlaceholderText("选中上方 tick 查看每个标的的详细状态")
        splitter.addWidget(self.txt)

        splitter.setSizes([420, 260])
        root.addWidget(splitter, stretch=1)

        self.lbl_status = QLabel("等待 tick 日志…")
        self.lbl_status.setProperty("secondary", "true")
        root.addWidget(self.lbl_status)

    def refresh(self) -> None:
        self._logs = recent_tick_logs(limit=100)
        self.tbl.setRowCount(len(self._logs))

        for row, log in enumerate(self._logs):
            symbols = ",".join(log["symbols"][:3])
            if len(log["symbols"]) > 3:
                symbols += f" +{len(log['symbols']) - 3}"
            errors = len(log["errors"])
            values = [
                log["ts"].strftime("%m-%d %H:%M:%S") if log["ts"] else "-",
                str(log["run_id"] or "-"),
                log["strategy_id"],
                symbols,
                f"{log['tradeable_quotes']}/{log['quotes_count']}",
                str(log["fallback_quotes"]),
                str(log["strategy_skipped"]),
                str(log["signals_generated"]),
                str(log["orders_placed"]),
                str(log["risk_rejected"]),
                str(errors),
                log.get("latest_reason") or "-",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in {1, 4, 5, 6, 7, 8, 9, 10}:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 4:
                    item.setForeground(QColor(_OK_COLOR if log["tradeable_quotes"] else _WARN_COLOR))
                elif col in {5, 6} and int(value) > 0:
                    item.setForeground(QColor(_WARN_COLOR))
                elif col in {9, 10} and int(value) > 0:
                    item.setForeground(QColor(_ERROR_COLOR))
                elif col == 11 and value != "-":
                    item.setForeground(QColor(_ERROR_COLOR if log["risk_rejected"] or errors else _WARN_COLOR))
                self.tbl.setItem(row, col, item)

        self.lbl_status.setText(f"最近 {len(self._logs)} 条 tick")

    def _show_detail(self) -> None:
        rows = self.tbl.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if idx >= len(self._logs):
            return
        log = self._logs[idx]
        chunks = [
            f"# Tick #{log['id']}  Run {log['run_id']}  {log['strategy_id']}\n",
            f"时间：{log['ts']}\n",
            f"行情：{log['tradeable_quotes']}/{log['quotes_count']} 可交易，fallback {log['fallback_quotes']}\n",
            f"策略：signals={log['signals_generated']} skipped={log['strategy_skipped']} "
            f"orders={log['orders_placed']} risk_rejected={log['risk_rejected']}\n",
        ]
        if log["errors"]:
            chunks.append("\n## Errors\n")
            chunks.append(json.dumps(log["errors"], ensure_ascii=False, indent=2))
            chunks.append("\n")
        chunks.append("\n## Per Symbol\n")
        chunks.append(json.dumps(log["per_symbol"], ensure_ascii=False, indent=2))
        self.txt.setPlainText("".join(chunks))
