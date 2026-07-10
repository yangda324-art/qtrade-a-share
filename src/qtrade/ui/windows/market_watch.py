"""实时行情看板：AkShare 准实时轮询快照。"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qtrade.core.time import utc_naive_to_market_time
from qtrade.data.adapters.base import DEFAULT_REALTIME_MAX_QUOTE_AGE_SECONDS, Quote
from qtrade.data.loader import get_loader
from qtrade.data.schema import PERIOD_LABELS, normalize_symbol
from qtrade.ui.theme import DOWN, NEUTRAL, TEXT_MUTED, UP
from qtrade.ui.windows.kline import KlineWidget

REFRESH_MS = 5000
STALE_AGE_SECONDS = DEFAULT_REALTIME_MAX_QUOTE_AGE_SECONDS
KLINE_REFRESH_SECONDS = 60
MARKET_WATCH_PERIODS = ("5min", "15min", "30min", "60min", "daily", "weekly", "monthly")
CHART_LOOKBACK_DAYS = {
    "5min": 7,
    "15min": 14,
    "30min": 35,
    "60min": 35,
    "daily": 365,
    "weekly": 365 * 3,
    "monthly": 365 * 8,
}


def _parse_symbols(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[,，\s]+", text) if p.strip()]
    return [normalize_symbol(p) for p in parts]


def _quote_status(quote: Quote) -> tuple[str, str]:
    reason = quote.realtime_reject_reason(max_age_seconds=STALE_AGE_SECONDS)
    if reason == "fallback_quote":
        return ("仅展示", "#caa14a")
    if reason == "stale_quote":
        return ("过期", "#d04949")
    if reason is not None:
        return ("异常", "#d04949")
    return ("可模拟", "#3aa055")


class MarketWatchWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._loader = get_loader()
        self._refreshing = False
        self._chart_symbol: str | None = None
        self._chart_period = "5min"
        self._last_chart_refresh_at: datetime | None = None
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.setInterval(REFRESH_MS)

    def showEvent(self, event):
        super().showEvent(event)
        self._restart_timer()
        self.refresh()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._timer.stop()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        topbar = QHBoxLayout()
        topbar.addWidget(QLabel("<h3>实时行情</h3>"))

        self.le_symbols = QLineEdit("000001,600519,300750")
        self.le_symbols.setMinimumWidth(260)
        topbar.addWidget(self.le_symbols)

        self.sp_interval = QSpinBox()
        self.sp_interval.setRange(1, 60)
        self.sp_interval.setValue(5)
        self.sp_interval.setSuffix(" 秒")
        self.sp_interval.valueChanged.connect(self._restart_timer)
        topbar.addWidget(self.sp_interval)

        topbar.addWidget(QLabel("K线周期"))
        self._chart_period_group = QButtonGroup(self)
        self._chart_period_group.setExclusive(True)
        self._chart_period_buttons: dict[str, QPushButton] = {}
        for period in MARKET_WATCH_PERIODS:
            btn = QPushButton(PERIOD_LABELS[period])
            btn.setCheckable(True)
            btn.setProperty("secondary", "true")
            btn.setMaximumWidth(62)
            if period == self._chart_period:
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked=False, p=period: self._set_chart_period(p))
            self._chart_period_group.addButton(btn)
            self._chart_period_buttons[period] = btn
            topbar.addWidget(btn)

        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self.refresh)
        topbar.addWidget(self.btn_refresh)

        topbar.addStretch()
        root.addLayout(topbar)

        splitter = QSplitter(Qt.Vertical)

        cols = ["标的", "最新价", "涨跌幅", "今开", "最高", "最低", "成交量", "数据源", "获取时间", "年龄", "状态"]
        self.tbl = QTableWidget(0, len(cols))
        self.tbl.setHorizontalHeaderLabels(cols)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.itemSelectionChanged.connect(self._on_table_selection_changed)
        splitter.addWidget(self.tbl)

        self.kline_chart = KlineWidget()
        self.kline_chart.set_controls_visible(False)
        self.kline_chart.set_period(self._chart_period, load=False)
        self.kline_chart.de_start.setDate(_chart_start_date(self._chart_period))
        self.kline_chart.de_end.setDate(date.today())
        self.kline_chart.lbl_status.setText("选择上方标的后加载 K 线")
        splitter.addWidget(self.kline_chart)
        splitter.setSizes([260, 620])
        root.addWidget(splitter, stretch=1)

        self.lbl_status = QLabel("等待行情…")
        self.lbl_status.setProperty("secondary", "true")
        root.addWidget(self.lbl_status)

    def _restart_timer(self) -> None:
        self._timer.setInterval(int(self.sp_interval.value()) * 1000)
        if self.isVisible():
            self._timer.start()

    def _set_chart_period(self, period: str) -> None:
        if period not in MARKET_WATCH_PERIODS:
            return
        self._chart_period = period
        btn = self._chart_period_buttons.get(period)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self.kline_chart.set_period(period, load=False)
        self._last_chart_refresh_at = None
        symbol = self._selected_symbol() or self._chart_symbol
        if symbol:
            self._refresh_chart_for_symbol(symbol, force=True)

    def refresh(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        self.btn_refresh.setEnabled(False)
        try:
            try:
                symbols = _parse_symbols(self.le_symbols.text())
            except Exception as exc:
                self.lbl_status.setText(f"标的格式错误：{exc}")
                self.tbl.setRowCount(0)
                return
            if not symbols:
                self.lbl_status.setText("请输入至少一个标的")
                self.tbl.setRowCount(0)
                return

            selected = self._selected_symbol()
            quotes = self._loader.get_quotes(symbols)
            self.tbl.setRowCount(len(symbols))
            ok_count = 0
            fallback_count = 0
            for row, sym in enumerate(symbols):
                q = quotes.get(sym)
                if q is None:
                    self._set_missing_row(row, sym)
                    continue
                if q.is_realtime_tradeable(max_age_seconds=STALE_AGE_SECONDS):
                    ok_count += 1
                else:
                    fallback_count += 1
                self._set_quote_row(row, q)
            self.lbl_status.setText(
                f"已刷新 {len(symbols)} 个标的 | 可模拟 {ok_count} | 不可实时成交 {fallback_count}"
            )
            chart_symbol = selected if selected in symbols else (self._chart_symbol if self._chart_symbol in symbols else symbols[0])
            self._select_symbol_row(chart_symbol)
            self._refresh_chart_for_symbol(chart_symbol)
        except Exception as exc:
            self.lbl_status.setText(f"行情刷新失败：{type(exc).__name__}: {exc}")
        finally:
            self.btn_refresh.setEnabled(True)
            self._refreshing = False

    def _selected_symbol(self) -> str | None:
        rows = self.tbl.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.tbl.item(rows[0].row(), 0)
        return item.text() if item else None

    def _select_symbol_row(self, symbol: str) -> None:
        for row in range(self.tbl.rowCount()):
            item = self.tbl.item(row, 0)
            if item is not None and item.text() == symbol:
                if self.tbl.currentRow() != row:
                    self.tbl.selectRow(row)
                return

    def _on_table_selection_changed(self) -> None:
        symbol = self._selected_symbol()
        if symbol:
            self._refresh_chart_for_symbol(symbol, force=True)

    def _refresh_chart_for_symbol(self, symbol: str, *, force: bool = False) -> None:
        now = datetime.now()
        if (
            not force
            and symbol == self._chart_symbol
            and self._last_chart_refresh_at is not None
            and (now - self._last_chart_refresh_at).total_seconds() < KLINE_REFRESH_SECONDS
        ):
            return
        self._chart_symbol = symbol
        self._last_chart_refresh_at = now
        try:
            self.kline_chart.load_symbol(
                symbol,
                period=self._chart_period,
                start=_chart_start_date(self._chart_period),
                end=date.today(),
            )
        except Exception as exc:
            self.kline_chart.lbl_status.setText(f"{symbol} K 线加载失败：{type(exc).__name__}: {exc}")

    def _set_missing_row(self, row: int, symbol: str) -> None:
        values = [symbol, "-", "-", "-", "-", "-", "-", "-", "-", "-", "无行情"]
        for col, value in enumerate(values):
            color = "#d04949" if col == len(values) - 1 else TEXT_MUTED
            self.tbl.setItem(row, col, _item(value, color=color))

    def _set_quote_row(self, row: int, quote: Quote) -> None:
        pct = ((quote.price - quote.prev_close) / quote.prev_close) if quote.prev_close else 0.0
        fetched = utc_naive_to_market_time(quote.fetched_at) if quote.fetched_at else None
        status, status_color = _quote_status(quote)
        pct_color = UP if pct > 0 else DOWN if pct < 0 else NEUTRAL
        values = [
            quote.symbol,
            f"{quote.price:.3f}",
            f"{pct:.2%}",
            f"{quote.open:.3f}",
            f"{quote.high:.3f}",
            f"{quote.low:.3f}",
            f"{quote.volume:,.0f}",
            quote.source,
            fetched.strftime("%H:%M:%S") if fetched else "-",
            f"{quote.age_seconds():.1f}s",
            status,
        ]
        for col, value in enumerate(values):
            color = None
            align = Qt.AlignRight | Qt.AlignVCenter if col in {1, 2, 3, 4, 5, 6, 9} else None
            if col == 2:
                color = pct_color
            elif col == 7 and not quote.is_realtime_tradeable(max_age_seconds=STALE_AGE_SECONDS):
                color = "#caa14a"
            elif col == 10:
                color = status_color
            self.tbl.setItem(row, col, _item(value, color=color, align=align))


def _item(text: str, *, color: str | None = None, align=None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if color:
        item.setForeground(QColor(color))
    if align is not None:
        item.setTextAlignment(align)
    return item


def _chart_start_date(period: str, *, today: date | None = None) -> date:
    ref = today or date.today()
    return ref - timedelta(days=CHART_LOOKBACK_DAYS.get(period, 365))
