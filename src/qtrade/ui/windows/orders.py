"""Orders tab：实时订单 + 成交 + 风控拒单。

数据源：``recent_orders()``——按 created_at 倒序的全部订单。
颜色规则：FILLED 绿、REJECTED 红、其他默认。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qtrade.core.exceptions import QtradeError
from qtrade.data.adapters.base import CACHE_QUOTE_SOURCE
from qtrade.data.schema import normalize_symbol
from qtrade.runtime.manual_trading import ManualTradingService
from qtrade.ui.data_access import latest_active_run, recent_orders

REFRESH_MS = 3000
QUOTE_REFRESH_INTERVAL = timedelta(seconds=15)
LIMIT_MATCH_INTERVAL = timedelta(seconds=15)


_STATUS_COLOR = {
    "FILLED":    "#3aa055",
    "PARTIAL":   "#caa14a",
    "PENDING":   "#888888",
    "SUBMITTED": "#5b8bd6",
    "REJECTED":  "#d04949",
    "CANCELLED": "#888888",
    "EXPIRED":   "#888888",
}


class OrdersWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._manual = ManualTradingService()
        self._last_manual_quote_at: datetime | None = None
        self._last_manual_quote_symbol: str | None = None
        self._last_manual_quote_price: float | None = None
        self._last_limit_match_at: datetime | None = None
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self._last_manual_quote_at = None
        self._refresh_manual_quote_if_due()

    def _build(self):
        root = QVBoxLayout(self)

        topbar = QHBoxLayout()
        topbar.addWidget(QLabel("<h3>订单 / 成交</h3>"))
        topbar.addStretch()
        self.cb_only_today = QCheckBox("只看今日")
        self.cb_only_today.setChecked(True)
        self.cb_only_today.stateChanged.connect(self.refresh)
        topbar.addWidget(self.cb_only_today)
        self.cb_current_run = QCheckBox("只看当前 run")
        self.cb_current_run.setChecked(True)
        self.cb_current_run.stateChanged.connect(self.refresh)
        topbar.addWidget(self.cb_current_run)
        root.addLayout(topbar)

        manual_bar = QHBoxLayout()
        manual_bar.addWidget(QLabel("手动模拟交易"))

        self.le_manual_symbol = QLineEdit("000001")
        self.le_manual_symbol.setMaximumWidth(120)
        self.le_manual_symbol.setToolTip("标的代码，如 000001 或 600519")
        manual_bar.addWidget(self.le_manual_symbol)

        self.lbl_manual_quote = QLabel("最新价 —")
        self.lbl_manual_quote.setStyleSheet("color:#888;")
        manual_bar.addWidget(self.lbl_manual_quote)

        self.lbl_price_guard = QLabel(
            f"价格偏离保护 ≤ {_bps_to_pct_text(self._manual.max_price_deviation_bps)}"
        )
        self.lbl_price_guard.setStyleSheet("color:#888;")
        manual_bar.addWidget(self.lbl_price_guard)

        self.cbo_order_type = QComboBox()
        self.cbo_order_type.addItems(["市价", "限价"])
        self.cbo_order_type.setMaximumWidth(90)
        self.cbo_order_type.currentIndexChanged.connect(self._sync_order_type_controls)
        manual_bar.addWidget(self.cbo_order_type)

        self.sp_manual_qty = QSpinBox()
        self.sp_manual_qty.setRange(100, 10_000_000)
        self.sp_manual_qty.setSingleStep(100)
        self.sp_manual_qty.setValue(1000)
        self.sp_manual_qty.setSuffix(" 股")
        manual_bar.addWidget(self.sp_manual_qty)

        self.sp_limit_price = QDoubleSpinBox()
        self.sp_limit_price.setRange(0.001, 1_000_000)
        self.sp_limit_price.setDecimals(3)
        self.sp_limit_price.setSingleStep(0.01)
        self.sp_limit_price.setValue(10.0)
        self.sp_limit_price.setPrefix("¥ ")
        self.sp_limit_price.setEnabled(False)
        manual_bar.addWidget(self.sp_limit_price)

        self.btn_quote = QPushButton("刷新行情")
        self.btn_quote.setProperty("secondary", "true")
        self.btn_quote.clicked.connect(self._refresh_manual_quote)
        manual_bar.addWidget(self.btn_quote)

        self.btn_buy = QPushButton("模拟买入")
        self.btn_buy.clicked.connect(lambda: self._place_manual_order("BUY"))
        manual_bar.addWidget(self.btn_buy)

        self.btn_sell = QPushButton("模拟卖出")
        self.btn_sell.setProperty("secondary", "true")
        self.btn_sell.clicked.connect(lambda: self._place_manual_order("SELL"))
        manual_bar.addWidget(self.btn_sell)

        self.btn_cancel = QPushButton("撤销挂单")
        self.btn_cancel.setProperty("secondary", "true")
        self.btn_cancel.clicked.connect(self._cancel_selected_order)
        manual_bar.addWidget(self.btn_cancel)

        manual_bar.addStretch()
        root.addLayout(manual_bar)

        self.lbl_manual_status = QLabel("手动交易账户：manual_paper · 市价/限价 · 走风控和 OMS")
        self.lbl_manual_status.setStyleSheet("color:#888;")
        root.addWidget(self.lbl_manual_status)

        cols = ["时间", "标的", "方向", "类型", "限价", "数量", "已成", "成交价", "状态", "佣金", "印花税", "拒绝原因 / coid"]
        self.tbl = QTableWidget(0, len(cols))
        self.tbl.setHorizontalHeaderLabels(cols)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl.setSortingEnabled(False)
        root.addWidget(self.tbl, stretch=1)

        self.lbl_status = QLabel("等待数据…")
        self.lbl_status.setStyleSheet("color:#888;")
        root.addWidget(self.lbl_status)
        self._sync_order_type_controls()

    def refresh(self, *_args):
        self._match_open_limits_if_due()

        run = latest_active_run() if self.cb_current_run.isChecked() else None
        run_id = run.get("run_id") if run else None
        orders = recent_orders(run_id=run_id, limit=200)

        if self.cb_only_today.isChecked():
            today = datetime.now().date()
            orders = [o for o in orders if o["ts"] and o["ts"].date() == today]

        self.tbl.setRowCount(len(orders))
        for r, o in enumerate(orders):
            cells = [
                o["ts"].strftime("%H:%M:%S") if o["ts"] else "-",
                o["symbol"],
                o["side"],
                o["order_type"],
                f"{o['limit_price']:.3f}" if o["limit_price"] else "-",
                str(o["quantity"]),
                str(o["filled_quantity"]),
                f"{o['avg_fill_price']:.3f}" if o["avg_fill_price"] else "-",
                o["status"],
                f"{o['commission']:.2f}",
                f"{o['stamp_tax']:.2f}",
                (o["reject_reason"] or o.get("client_order_id", ""))[:80],
            ]
            for c, v in enumerate(cells):
                item = QTableWidgetItem(str(v))
                if c == 0:
                    item.setData(Qt.UserRole, o["id"])
                if c == 8:        # 状态列上色
                    color = _STATUS_COLOR.get(o["status"], "#aaa")
                    item.setForeground(QColor(color))
                elif c == 2:       # 方向列：BUY 红、SELL 绿
                    item.setForeground(QColor("#d04949" if o["side"] == "BUY" else "#3aa055"))
                elif c in (4, 5, 6, 7, 9, 10):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl.setItem(r, c, item)

        suffix = f" | run #{run['run_id']}" if run else ""
        self.lbl_status.setText(f"共 {len(orders)} 条订单{suffix} | {datetime.now().strftime('%H:%M:%S')}")

        if self.isVisible():
            self._refresh_manual_quote_if_due()

    def _refresh_manual_quote_if_due(self):
        now = datetime.now()
        if self._last_manual_quote_at and now - self._last_manual_quote_at < QUOTE_REFRESH_INTERVAL:
            return
        self._refresh_manual_quote(silent=True)

    def _refresh_manual_quote(self, silent: bool = False):
        self._last_manual_quote_at = datetime.now()
        sym = self.le_manual_symbol.text().strip()
        if not sym:
            self.lbl_manual_quote.setText("最新价 —")
            self._last_manual_quote_symbol = None
            self._last_manual_quote_price = None
            return
        try:
            quote = self._manual.get_quote(sym)
            self._last_manual_quote_symbol = quote.symbol
            self._last_manual_quote_price = quote.price
            reject_reason = quote.realtime_reject_reason(
                max_age_seconds=self._manual.max_quote_age_seconds
            )
            if quote.source == CACHE_QUOTE_SOURCE:
                source_note = " | 仅展示：缓存价"
            elif reject_reason == "stale_quote":
                source_note = " | 过期：不可实时成交"
            elif reject_reason is not None:
                source_note = f" | 不可实时成交：{reject_reason}"
            else:
                source_note = " | 可模拟"
            self.lbl_manual_quote.setText(
                f"{quote.symbol} 最新价 {quote.price:.3f} | 今开 {quote.open:.3f} | 高 {quote.high:.3f} | 低 {quote.low:.3f}{source_note}"
            )
            if not silent:
                self.lbl_manual_status.setText(f"行情已刷新 | {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            self.lbl_manual_quote.setText("最新价 —")
            self._last_manual_quote_symbol = None
            self._last_manual_quote_price = None
            if not silent:
                self.lbl_manual_status.setText(f"行情刷新失败：{e}")

    def _place_manual_order(self, side: str):
        sym = self.le_manual_symbol.text().strip()
        qty = int(self.sp_manual_qty.value())
        order_type = "LIMIT" if self.cbo_order_type.currentText() == "限价" else "MARKET"
        limit_price = float(self.sp_limit_price.value()) if order_type == "LIMIT" else None
        self.btn_buy.setEnabled(False)
        self.btn_sell.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        try:
            result = self._manual.place_order(
                symbol=sym,
                side=side,
                quantity=qty,
                order_type=order_type,
                limit_price=limit_price,
                expected_price=self._expected_manual_price(sym),
            )
            if result.ok:
                self.lbl_manual_status.setText(
                    f"{result.order_type} {result.side} {result.symbol} x{result.filled_quantity} 成交 "
                    f"@ {result.avg_fill_price:.3f} | run #{result.run_id}"
                )
            elif result.pending:
                self.lbl_manual_status.setText(
                    f"{result.order_type} {result.side} {result.symbol} x{result.quantity} 已挂单 "
                    f"@ {result.limit_price:.3f} | run #{result.run_id}"
                )
            else:
                self.lbl_manual_status.setText(
                    f"{result.side} {result.symbol} x{result.quantity} 未成交：{result.reject_reason or result.status}"
                )
            self.cb_current_run.setChecked(False)
            self.refresh()
        except QtradeError as e:
            self.lbl_manual_status.setText(f"下单失败：{e}")
        except Exception as e:
            self.lbl_manual_status.setText(f"下单异常：{type(e).__name__}: {e}")
        finally:
            self.btn_buy.setEnabled(True)
            self.btn_sell.setEnabled(True)
            self.btn_cancel.setEnabled(True)

    def _cancel_selected_order(self):
        row = self.tbl.currentRow()
        if row < 0:
            self.lbl_manual_status.setText("请先在订单表里选中一笔未成交限价单")
            return
        item = self.tbl.item(row, 0)
        order_id = item.data(Qt.UserRole) if item else None
        if not order_id:
            self.lbl_manual_status.setText("无法识别选中订单")
            return
        self.btn_cancel.setEnabled(False)
        try:
            result = self._manual.cancel_order(int(order_id))
            self.lbl_manual_status.setText(
                f"已撤销 {result.side} {result.symbol} x{result.quantity} 限价单 | run #{result.run_id}"
            )
            self.refresh()
        except QtradeError as e:
            self.lbl_manual_status.setText(f"撤单失败：{e}")
        except Exception as e:
            self.lbl_manual_status.setText(f"撤单异常：{type(e).__name__}: {e}")
        finally:
            self.btn_cancel.setEnabled(True)

    def _match_open_limits_if_due(self):
        now = datetime.now()
        if self._last_limit_match_at and now - self._last_limit_match_at < LIMIT_MATCH_INTERVAL:
            return
        self._last_limit_match_at = now
        try:
            matched = self._manual.match_open_limit_orders()
        except Exception:
            return
        if matched:
            self.lbl_manual_status.setText(
                f"限价单撮合 {len(matched)} 笔 | {datetime.now().strftime('%H:%M:%S')}"
            )

    def _sync_order_type_controls(self):
        is_limit = self.cbo_order_type.currentText() == "限价"
        self.sp_limit_price.setEnabled(is_limit)
        self.btn_buy.setText("限价买入" if is_limit else "模拟买入")
        self.btn_sell.setText("限价卖出" if is_limit else "模拟卖出")

    def _expected_manual_price(self, symbol: str) -> float | None:
        try:
            normalized = normalize_symbol(symbol)
        except Exception:
            return None
        if normalized != self._last_manual_quote_symbol:
            return None
        return self._last_manual_quote_price


def _bps_to_pct_text(bps: int | float) -> str:
    pct = float(bps) / 100
    return f"{pct:.2f}%"
