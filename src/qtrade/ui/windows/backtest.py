"""回测 tab：表单 + 后台跑回测 + 结果可视化。

设计：
- 回测在 ``QThread`` 里跑，避免大区间回测时 UI 卡死
- 用 ``Signal/Slot`` 把进度和结果回传 UI 线程
- 结果三块：
    1) 顶部指标卡（总收益/年化/Sharpe/MDD/...）
    2) 中部权益曲线
    3) 底部交易明细表
- 点行交易明细可在右侧弹气泡显示策略 reason
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qtrade.backtest.engine import BacktestEngine
from qtrade.backtest.metrics import Metrics
from qtrade.core.exceptions import DataError
from qtrade.core.logging import get_logger
from qtrade.data.schema import normalize_symbol
from qtrade.strategies.technical.ma_cross import MACrossStrategy
from qtrade.ui.widgets.chart_utils import lock_view_to_data

log = get_logger(__name__)


# -------------------- 后台 Worker --------------------


@dataclass
class _BacktestRequest:
    strategy: str
    symbols: list[str]
    start: date
    end: date
    initial_cash: float
    fast: int
    slow: int


class _BacktestWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)        # BacktestResult
    failed = Signal(str)

    def __init__(self, req: _BacktestRequest) -> None:
        super().__init__()
        self.req = req

    def run(self):
        try:
            self.progress.emit("准备数据…")
            r = self.req
            if r.strategy == "ma_cross":
                strat = MACrossStrategy(
                    universe=r.symbols,
                    params={"fast_window": r.fast, "slow_window": r.slow, "position_pct": 0.3},
                )
            elif r.strategy == "llm":
                from qtrade.strategies.llm.graph import LLMMultiAgentStrategy
                strat = LLMMultiAgentStrategy(universe=r.symbols)
            else:
                raise ValueError(f"未知策略：{r.strategy}")

            self.progress.emit("跑回测中…")
            engine = BacktestEngine(
                strategy=strat,
                start=r.start, end=r.end,
                initial_cash=r.initial_cash,
            )
            result = engine.run()
            self.finished.emit(result)
        except DataError as e:
            self.failed.emit(f"数据错误：{e}")
        except Exception as e:
            log.exception("backtest failed")
            self.failed.emit(f"{type(e).__name__}: {e}")


# -------------------- Widget --------------------


def _kv_card(label: str) -> tuple[QWidget, QLabel]:
    w = QWidget()
    layout = QVBoxLayout(w)
    layout.setContentsMargins(10, 6, 10, 6)
    layout.addWidget(QLabel(f"<span style='color:#888;font-size:11px'>{label}</span>"))
    val = QLabel("—")
    val.setStyleSheet("font-size:16px;font-weight:600;")
    layout.addWidget(val)
    return w, val


def _pnl_color(v: float) -> str:
    if v > 0:
        return "#d04949"
    if v < 0:
        return "#3aa055"
    return "#aaaaaa"


class BacktestWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _BacktestWorker | None = None
        self._thread: QThread | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)

        # ----- 表单 -----
        form_box = QWidget()
        form = QFormLayout(form_box)
        form.setContentsMargins(0, 0, 0, 0)

        self.cmb_strategy = QComboBox()
        self.cmb_strategy.addItems(["ma_cross", "llm"])
        form.addRow("策略", self.cmb_strategy)

        from PySide6.QtWidgets import QLineEdit
        self.le_symbols = QLineEdit("000001")
        self.le_symbols.setToolTip("多个标的逗号分隔，如：000001, 600519")
        form.addRow("标的", self.le_symbols)

        self.de_start = QDateEdit()
        self.de_start.setCalendarPopup(True)
        self.de_start.setDate(date.today() - timedelta(days=365 * 2))
        form.addRow("起始日", self.de_start)

        self.de_end = QDateEdit()
        self.de_end.setCalendarPopup(True)
        self.de_end.setDate(date.today())
        form.addRow("终止日", self.de_end)

        self.sp_cash = QDoubleSpinBox()
        self.sp_cash.setRange(1000, 100_000_000)
        self.sp_cash.setSingleStep(10_000)
        self.sp_cash.setValue(100_000)
        self.sp_cash.setSuffix(" 元")
        form.addRow("初始资金", self.sp_cash)

        self.sp_fast = QDoubleSpinBox()
        self.sp_fast.setRange(2, 200)
        self.sp_fast.setDecimals(0)
        self.sp_fast.setValue(5)
        self.sp_slow = QDoubleSpinBox()
        self.sp_slow.setRange(3, 500)
        self.sp_slow.setDecimals(0)
        self.sp_slow.setValue(20)
        ma_row = QWidget()
        ma_layout = QHBoxLayout(ma_row)
        ma_layout.setContentsMargins(0, 0, 0, 0)
        ma_layout.addWidget(QLabel("快"))
        ma_layout.addWidget(self.sp_fast)
        ma_layout.addWidget(QLabel("慢"))
        ma_layout.addWidget(self.sp_slow)
        ma_layout.addStretch()
        form.addRow("MA 参数", ma_row)

        self.btn_run = QPushButton("跑回测")
        self.btn_run.clicked.connect(self.run_backtest)
        form.addRow("", self.btn_run)

        root.addWidget(form_box)

        # ----- 状态 -----
        self.lbl_status = QLabel("配置好参数后点 [跑回测]")
        self.lbl_status.setStyleSheet("color:#888;")
        root.addWidget(self.lbl_status)

        # ----- 指标卡 -----
        self.cards: dict[str, QLabel] = {}
        cards_box = QWidget()
        grid = QGridLayout(cards_box)
        for i, name in enumerate(["总收益", "年化收益", "Sharpe", "Sortino", "Calmar",
                                  "最大回撤", "MDD 天数", "交易次数", "胜率", "盈亏比", "最终权益"]):
            w, lbl = _kv_card(name)
            grid.addWidget(w, i // 6, i % 6)
            self.cards[name] = lbl
        root.addWidget(cards_box)

        # ----- 权益曲线 + 交易表 -----
        splitter = QSplitter(Qt.Vertical)
        self.plot = pg.PlotWidget(title="权益曲线")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "总资产")
        splitter.addWidget(self.plot)

        self.tbl = QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(["时间", "标的", "方向", "数量", "成交价", "费用", "现金前", "理由"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        splitter.addWidget(self.tbl)
        splitter.setSizes([300, 300])
        root.addWidget(splitter, stretch=1)

    # ---------------- 跑回测 ----------------

    def run_backtest(self):
        if self._thread and self._thread.isRunning():
            self.lbl_status.setText("已有回测在跑，等它结束")
            return
        try:
            symbols = [normalize_symbol(s.strip()) for s in self.le_symbols.text().split(",") if s.strip()]
            if not symbols:
                raise ValueError("请输入至少一个标的")
        except Exception as e:
            self.lbl_status.setText(f"标的格式错：{e}")
            return

        req = _BacktestRequest(
            strategy=self.cmb_strategy.currentText(),
            symbols=symbols,
            start=self.de_start.date().toPython(),
            end=self.de_end.date().toPython(),
            initial_cash=float(self.sp_cash.value()),
            fast=int(self.sp_fast.value()),
            slow=int(self.sp_slow.value()),
        )

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("启动回测线程…")

        self._thread = QThread(self)
        self._worker = _BacktestWorker(req)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.lbl_status.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.btn_run.setEnabled(True))
        self._thread.start()

    def _on_finished(self, result):
        m: Metrics = result.metrics
        self.lbl_status.setText(
            f"完成 | 初始 {m.initial_equity:,.0f} → 终值 {m.final_equity:,.0f} | "
            f"{m.n_trades} 笔交易"
        )
        # 指标卡
        rows = {
            "总收益": (f"{m.total_return:+.2%}", _pnl_color(m.total_return)),
            "年化收益": (f"{m.annual_return:+.2%}", _pnl_color(m.annual_return)),
            "Sharpe": (f"{m.sharpe:.2f}", _pnl_color(m.sharpe)),
            "Sortino": (f"{m.sortino:.2f}", _pnl_color(m.sortino)),
            "Calmar": (f"{m.calmar:.2f}", _pnl_color(m.calmar)),
            "最大回撤": (f"{m.max_drawdown:.2%}", _pnl_color(m.max_drawdown)),
            "MDD 天数": (str(m.max_drawdown_duration_days), "#aaa"),
            "交易次数": (str(m.n_trades), "#aaa"),
            "胜率": (f"{m.win_rate:.1%}", "#aaa"),
            "盈亏比": (f"{m.profit_loss_ratio:.2f}", "#aaa"),
            "最终权益": (f"{m.final_equity:,.0f}", _pnl_color(m.final_equity - m.initial_equity)),
        }
        for k, (text, color) in rows.items():
            self.cards[k].setText(text)
            self.cards[k].setStyleSheet(f"font-size:16px;font-weight:600;color:{color};")

        # 权益曲线
        self.plot.clear()
        if result.equity_curve:
            ys = [float(v) for _, v in result.equity_curve]
            xs = list(range(len(ys)))
            self.plot.plot(xs, ys, pen=pg.mkPen("#5b8bd6", width=2))
            # 初始资金水平参考线
            self.plot.addLine(y=m.initial_equity, pen=pg.mkPen("#888", style=Qt.DashLine))
            # 锁视图：把初始资金参考线也算进 Y 范围
            lock_view_to_data(
                self.plot, xs=xs, ys=ys + [m.initial_equity],
                x_padding=0.02, y_padding=0.05, min_visible_x=5,
            )

        # 交易表
        self.tbl.setRowCount(len(result.trades))
        for r, t in enumerate(result.trades):
            cells = [
                str(t.get("ts", ""))[:19],
                t["symbol"],
                t["side"],
                str(t["quantity"]),
                f"{t['price']:.2f}",
                f"{t['commission'] + t['stamp_tax']:.2f}",
                f"{t.get('cash_before', 0):,.0f}",
                t.get("reason", "")[:60],
            ]
            for c, v in enumerate(cells):
                item = QTableWidgetItem(str(v))
                if c == 2:
                    item.setForeground(QColor("#d04949" if t["side"] == "BUY" else "#3aa055"))
                elif c >= 3 and c <= 6:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl.setItem(r, c, item)

    def _on_failed(self, msg: str):
        self.lbl_status.setText(f"失败：{msg}")
