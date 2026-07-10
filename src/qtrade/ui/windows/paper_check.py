"""Paper trading readiness check page."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qtrade.runtime.paper_check import (
    PaperCheckReport,
    run_paper_check,
    write_report_json,
    write_report_markdown,
)
from qtrade.ui.theme import TEXT_MUTED, TEXT_SECONDARY
from qtrade.ui.windows.run_control import parse_symbol_list


def _status_color(status: str) -> str:
    return {
        "PASS": "#52c285",
        "WARN": "#caa14a",
        "FAIL": "#d04949",
    }.get(status, TEXT_SECONDARY)


class _PaperCheckWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        symbols: list[str],
        include_bars: bool,
        run_once: bool,
    ) -> None:
        super().__init__()
        self.symbols = symbols
        self.include_bars = include_bars
        self.run_once = run_once

    @Slot()
    def run(self) -> None:
        try:
            report = run_paper_check(
                symbols=self.symbols,
                include_bars=self.include_bars,
                run_once=self.run_once,
            )
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self.failed.emit(str(exc))
            return
        self.finished.emit(report)


class PaperCheckWidget(QWidget):
    """UI wrapper around ``qtrade paper-check`` logic."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _PaperCheckWorker | None = None
        self._last_report: PaperCheckReport | None = None
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(14)

        root.addWidget(QLabel("<h3>模拟实盘验收</h3>"))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("标的"))
        self.le_symbols = QLineEdit("000001")
        self.le_symbols.setPlaceholderText("000001, 600519")
        controls.addWidget(self.le_symbols, stretch=1)

        self.cb_include_bars = QCheckBox("检查 K 线周期")
        self.cb_include_bars.setChecked(True)
        controls.addWidget(self.cb_include_bars)

        self.cb_run_once = QCheckBox("跑一次 tick")
        controls.addWidget(self.cb_run_once)

        self.btn_run = QPushButton("开始验收")
        self.btn_run.clicked.connect(self.start_check)
        controls.addWidget(self.btn_run)
        root.addLayout(controls)

        export_buttons = QHBoxLayout()
        self.btn_export_json = QPushButton("导出 JSON")
        self.btn_export_json.setProperty("secondary", "true")
        self.btn_export_json.setEnabled(False)
        self.btn_export_json.clicked.connect(lambda: self.export_json())
        export_buttons.addWidget(self.btn_export_json)

        self.btn_export_markdown = QPushButton("导出 Markdown")
        self.btn_export_markdown.setProperty("secondary", "true")
        self.btn_export_markdown.setEnabled(False)
        self.btn_export_markdown.clicked.connect(lambda: self.export_markdown())
        export_buttons.addWidget(self.btn_export_markdown)
        export_buttons.addStretch()
        root.addLayout(export_buttons)

        self.lbl_hint = QLabel(
            "默认不跑策略 tick；勾选“跑一次 tick”会写入隔离的 paper_check_ma_cross_v1 验收记录。"
        )
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        root.addWidget(self.lbl_hint)

        self.lbl_summary = QLabel("未检查")
        self.lbl_summary.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        root.addWidget(self.lbl_summary)

        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        self.lbl_export_status.setWordWrap(True)
        root.addWidget(self.lbl_export_status)

        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["状态", "检查项", "结果"])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setShowGrid(False)
        root.addWidget(self.tbl, stretch=1)

    def start_check(self) -> None:
        if self._thread is not None:
            return
        symbols = parse_symbol_list(self.le_symbols.text())
        if not symbols:
            self.lbl_summary.setText("请输入至少一个标的")
            return

        self.btn_run.setEnabled(False)
        self._set_export_enabled(False)
        self._last_report = None
        self.lbl_summary.setText("验收中：正在拉取行情和检查状态…")
        self.lbl_export_status.setText("")
        self.tbl.setRowCount(0)

        thread = QThread(self)
        worker = _PaperCheckWorker(
            symbols=symbols,
            include_bars=self.cb_include_bars.isChecked(),
            run_once=self.cb_run_once.isChecked(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_report)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def _on_report(self, report: PaperCheckReport) -> None:
        self._render_report(report)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.lbl_summary.setText(f"验收失败：{message}")
        self._last_report = None
        self._set_export_enabled(False)
        self.tbl.setRowCount(0)

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.btn_run.setEnabled(True)

    def _render_report(self, report: PaperCheckReport) -> None:
        self._last_report = report
        self._set_export_enabled(True)
        self.tbl.setRowCount(len(report.items))
        for row, item in enumerate(report.items):
            color = _status_color(item.status)
            cells = [item.status, item.name, item.detail]
            for col, text in enumerate(cells):
                qitem = QTableWidgetItem(text)
                qitem.setForeground(QColor(color if col == 0 else TEXT_SECONDARY))
                self.tbl.setItem(row, col, qitem)

        color = _status_color(report.overall_status)
        self.lbl_summary.setText(
            f"<span style='color:{color}; font-weight:600;'>{report.overall_status}</span>"
            f" · 标的 {','.join(report.symbols)}"
            f" · PASS={len(report.passed)}"
            f" · WARN={len(report.warnings)}"
            f" · FAIL={len(report.failures)}"
        )

    def export_json(self, path: str | Path | None = None) -> Path | None:
        if self._last_report is None:
            self.lbl_export_status.setText("暂无可导出的验收结果")
            return None
        out = self._resolve_export_path(path, suffix="json", title="导出 JSON 验收报告")
        if out is None:
            return None
        try:
            saved = write_report_json(self._last_report, out)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self.lbl_export_status.setText(f"JSON 导出失败：{exc}")
            return None
        self.lbl_export_status.setText(f"JSON 已导出：{saved}")
        return saved

    def export_markdown(self, path: str | Path | None = None) -> Path | None:
        if self._last_report is None:
            self.lbl_export_status.setText("暂无可导出的验收结果")
            return None
        out = self._resolve_export_path(path, suffix="md", title="导出 Markdown 验收报告")
        if out is None:
            return None
        try:
            saved = write_report_markdown(self._last_report, out)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self.lbl_export_status.setText(f"Markdown 导出失败：{exc}")
            return None
        self.lbl_export_status.setText(f"Markdown 已导出：{saved}")
        return saved

    def _resolve_export_path(
        self,
        path: str | Path | None,
        *,
        suffix: str,
        title: str,
    ) -> Path | None:
        if path is not None:
            return Path(path).expanduser().resolve()
        default = self._default_export_path(suffix)
        file_filter = "JSON (*.json)" if suffix == "json" else "Markdown (*.md)"
        selected, _ = QFileDialog.getSaveFileName(self, title, str(default), file_filter)
        return Path(selected).expanduser().resolve() if selected else None

    def _default_export_path(self, suffix: str) -> Path:
        report = self._last_report
        ts = report.generated_at.strftime("%Y%m%d_%H%M%S") if report else "latest"
        return Path.cwd() / "reports" / f"paper-check_{ts}.{suffix}"

    def _set_export_enabled(self, enabled: bool) -> None:
        self.btn_export_json.setEnabled(enabled)
        self.btn_export_markdown.setEnabled(enabled)
