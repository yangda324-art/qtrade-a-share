"""模拟盘绩效报告页。"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
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

from qtrade.ui.data_access import performance_runs, run_performance_report
from qtrade.ui.theme import TEXT_MUTED, pnl_color
from qtrade.ui.widgets.card import MetricCard
from qtrade.ui.widgets.chart_utils import lock_view_to_data

REFRESH_MS = 5000


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _pct(value: float) -> str:
    return f"{value:+.2%}"


def _plain_pct(value: float) -> str:
    return f"{value:.1%}"


def _ratio(value: float) -> str:
    return f"{value:.2f}"


class PerformanceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._runs: list[dict] = []
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)

        topbar = QHBoxLayout()
        topbar.addWidget(QLabel("<h3>绩效报告</h3>"))
        topbar.addStretch()
        topbar.addWidget(QLabel("Run"))
        self.cbo_run = QComboBox()
        self.cbo_run.setMinimumWidth(320)
        self.cbo_run.currentIndexChanged.connect(self.refresh_report)
        topbar.addWidget(self.cbo_run)
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setProperty("secondary", "true")
        self.btn_refresh.clicked.connect(self.refresh)
        topbar.addWidget(self.btn_refresh)
        root.addLayout(topbar)

        cards = QWidget()
        grid = QGridLayout(cards)
        grid.setSpacing(10)
        self.card_total_return = MetricCard("总收益", "—")
        self.card_final_assets = MetricCard("当前权益", "—")
        self.card_drawdown = MetricCard("最大回撤", "—")
        self.card_current_dd = MetricCard("当前回撤", "—")
        self.card_turnover = MetricCard("换手", "—")
        self.card_exposure = MetricCard("平均仓位", "—")
        self.card_trades = MetricCard("成交订单", "—")
        self.card_win_rate = MetricCard("胜率", "—")
        self.card_avg_win = MetricCard("平均盈利", "—")
        self.card_avg_loss = MetricCard("平均亏损", "—")
        self.card_pl_ratio = MetricCard("盈亏比", "—")
        self.card_rejected = MetricCard("拒单/挂单", "—")
        for idx, card in enumerate(
            [
                self.card_total_return,
                self.card_final_assets,
                self.card_drawdown,
                self.card_current_dd,
                self.card_turnover,
                self.card_exposure,
                self.card_trades,
                self.card_win_rate,
                self.card_avg_win,
                self.card_avg_loss,
                self.card_pl_ratio,
                self.card_rejected,
            ]
        ):
            grid.addWidget(card, idx // 6, idx % 6)
        root.addWidget(cards)

        splitter = QSplitter(Qt.Vertical)
        self.plot_equity = pg.PlotWidget(title="权益曲线")
        self.plot_equity.showGrid(x=True, y=True, alpha=0.25)
        self.plot_equity.setLabel("left", "总资产")
        splitter.addWidget(self.plot_equity)

        self.plot_drawdown = pg.PlotWidget(title="回撤")
        self.plot_drawdown.showGrid(x=True, y=True, alpha=0.25)
        self.plot_drawdown.setLabel("left", "Drawdown")
        splitter.addWidget(self.plot_drawdown)

        cols = ["时间", "标的", "方向", "类型", "数量", "已成", "成交价", "状态", "费用", "拒绝原因"]
        self.tbl_orders = QTableWidget(0, len(cols))
        self.tbl_orders.setHorizontalHeaderLabels(cols)
        self.tbl_orders.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_orders.horizontalHeader().setStretchLastSection(True)
        self.tbl_orders.setEditTriggers(QTableWidget.NoEditTriggers)
        splitter.addWidget(self.tbl_orders)

        splitter.setSizes([280, 180, 260])
        root.addWidget(splitter, stretch=1)

        self.lbl_status = QLabel("等待数据…")
        self.lbl_status.setStyleSheet(f"color:{TEXT_MUTED};")
        root.addWidget(self.lbl_status)

    def refresh(self, *_args) -> None:
        current_run_id = self._selected_run_id()
        self._runs = performance_runs(limit=50, modes=["paper"])
        self.cbo_run.blockSignals(True)
        self.cbo_run.clear()
        for run in self._runs:
            status = "running" if run.get("ended_at") is None else "ended"
            self.cbo_run.addItem(
                f"#{run['run_id']} {run['strategy_id']} · {status}",
                run["run_id"],
            )
        if current_run_id is not None:
            for i in range(self.cbo_run.count()):
                if self.cbo_run.itemData(i) == current_run_id:
                    self.cbo_run.setCurrentIndex(i)
                    break
        self.cbo_run.blockSignals(False)
        self.refresh_report()

    def refresh_report(self, *_args) -> None:
        run_id = self._selected_run_id()
        if run_id is None:
            self._clear()
            self.lbl_status.setText("暂无 paper run。先启动一次模拟盘后再看绩效报告。")
            return

        report = run_performance_report(run_id)
        if not report:
            self._clear()
            self.lbl_status.setText(f"run #{run_id} 暂无绩效数据")
            return

        metrics = report["metrics"]
        self._set_metric_cards(metrics)
        self._plot_equity(report["equity_curve"])
        self._plot_drawdown(report["equity_curve"])
        self._set_orders(report["orders"])

        run = report["run"]
        first_ts = metrics.get("first_ts")
        last_ts = metrics.get("last_ts")
        scope = f"{first_ts:%m-%d %H:%M}" if first_ts else "-"
        scope += f" → {last_ts:%m-%d %H:%M}" if last_ts else " → -"
        self.lbl_status.setText(
            f"run #{run['run_id']} · {run['strategy_id']} · {scope} · "
            f"{metrics['snapshot_count']} 个账户快照"
        )

    def _selected_run_id(self) -> int | None:
        if self.cbo_run.count() <= 0:
            return None
        value = self.cbo_run.currentData()
        return int(value) if value is not None else None

    def _set_metric_cards(self, metrics: dict) -> None:
        self.card_total_return.set_value(
            _pct(metrics["total_return"]),
            color=pnl_color(metrics["total_return"]),
        )
        self.card_final_assets.set_value(_money(metrics["final_assets"]))
        self.card_drawdown.set_value(_pct(metrics["max_drawdown"]), color=pnl_color(metrics["max_drawdown"]))
        self.card_current_dd.set_value(
            _pct(metrics["current_drawdown"]),
            color=pnl_color(metrics["current_drawdown"]),
        )
        self.card_turnover.set_value(_plain_pct(metrics["turnover"]))
        self.card_exposure.set_value(_plain_pct(metrics["avg_exposure"]))
        self.card_trades.set_value(str(metrics["filled_order_count"]))
        self.card_trades.set_sub(
            f"买 {metrics['buy_order_count']} / 卖 {metrics['sell_order_count']} / 完整 {metrics['completed_trade_count']}"
        )
        self.card_win_rate.set_value(_plain_pct(metrics["win_rate"]))
        self.card_win_rate.set_sub(f"赢 {metrics['win_count']} / 亏 {metrics['loss_count']}")
        self.card_avg_win.set_value(_money(metrics["avg_win"]), color=pnl_color(metrics["avg_win"]))
        self.card_avg_loss.set_value(_money(metrics["avg_loss"]), color=pnl_color(metrics["avg_loss"]))
        self.card_pl_ratio.set_value(_ratio(metrics["profit_loss_ratio"]))
        self.card_pl_ratio.set_sub(
            f"最好 {_money(metrics['best_trade_pnl'])} / 最差 {_money(metrics['worst_trade_pnl'])}"
        )
        self.card_rejected.set_value(f"{metrics['rejected_order_count']} / {metrics['open_order_count']}")
        self.card_rejected.set_sub("拒单 / 挂单")

    def _plot_equity(self, curve: list[dict]) -> None:
        self.plot_equity.clear()
        if not curve:
            return
        ys = [float(p["total_assets"]) for p in curve]
        xs = list(range(len(ys)))
        self.plot_equity.plot(xs, ys, pen=pg.mkPen("#5b8bd6", width=2))
        lock_view_to_data(
            self.plot_equity,
            xs=xs,
            ys=ys,
            x_padding=0.02,
            y_padding=0.05,
            min_visible_x=5,
        )

    def _plot_drawdown(self, curve: list[dict]) -> None:
        self.plot_drawdown.clear()
        if not curve:
            return
        ys: list[float] = []
        peak = 0.0
        for point in curve:
            value = float(point["total_assets"])
            peak = max(peak, value)
            ys.append(value / peak - 1 if peak > 0 else 0.0)
        xs = list(range(len(ys)))
        self.plot_drawdown.plot(xs, ys, pen=pg.mkPen("#d04949", width=2))
        lock_view_to_data(
            self.plot_drawdown,
            xs=xs,
            ys=ys + [0.0],
            x_padding=0.02,
            y_padding=0.10,
            min_visible_x=5,
        )

    def _set_orders(self, orders: list[dict]) -> None:
        self.tbl_orders.setRowCount(len(orders))
        for row, order in enumerate(orders):
            fee = float(order["commission"] + order["stamp_tax"])
            values = [
                order["ts"].strftime("%m-%d %H:%M:%S") if order["ts"] else "-",
                order["symbol"],
                order["side"],
                order["order_type"],
                str(order["quantity"]),
                str(order["filled_quantity"]),
                f"{order['avg_fill_price']:.3f}" if order["avg_fill_price"] else "-",
                order["status"],
                f"{fee:.2f}",
                (order["reject_reason"] or "")[:80],
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 2:
                    item.setForeground(QColor("#d04949" if order["side"] == "BUY" else "#3aa055"))
                elif col == 7 and order["status"] == "REJECTED":
                    item.setForeground(QColor("#d04949"))
                elif col == 7 and order["status"] == "FILLED":
                    item.setForeground(QColor("#3aa055"))
                if col in {4, 5, 6, 8}:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_orders.setItem(row, col, item)

    def _clear(self) -> None:
        for card in (
            self.card_total_return,
            self.card_final_assets,
            self.card_drawdown,
            self.card_current_dd,
            self.card_turnover,
            self.card_exposure,
            self.card_trades,
            self.card_win_rate,
            self.card_avg_win,
            self.card_avg_loss,
            self.card_pl_ratio,
            self.card_rejected,
        ):
            card.set_value("—")
            card.set_sub("")
        self.plot_equity.clear()
        self.plot_drawdown.clear()
        self.tbl_orders.setRowCount(0)
