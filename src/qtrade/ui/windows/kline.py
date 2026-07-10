"""K 线 tab：蜡烛图 + 成交量 + 均线 + 历史买卖标记。

布局：
- 顶部表单：标的输入、起止日期、均线参数、加载按钮
- 主图：K 线 + MA5/MA20/MA60 + 成交标记（▲ 买、▼ 卖）
- 副图：成交量
- 下方状态栏：数据范围、加载耗时

数据来源：通过 ``DataLoader`` 读取本地缓存并按需刷新；订单标记读 SQLite。
外部行情源不可用但本地有缓存时，图表会降级展示缓存并在状态栏说明。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import ceil, floor

import pandas as pd
import pyqtgraph as pg
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from qtrade.core.calendar import is_trading_day
from qtrade.core.exceptions import DataError
from qtrade.core.logging import get_logger
from qtrade.core.time import MARKET_TZ
from qtrade.data.loader import BAR_SOURCE_ATTR, BAR_STATUS_ATTR, get_loader
from qtrade.data.schema import PERIOD_LABELS, normalize_symbol
from qtrade.ui.data_access import recent_orders
from qtrade.ui.widgets.candlestick import (
    DOWN_COLOR,
    UP_COLOR,
    CandlestickItem,
    DateAxisItem,
)
from qtrade.ui.widgets.chart_utils import lock_view_to_data

log = get_logger(__name__)

# 默认演示池
PRESET_SYMBOLS = ["000001", "600519", "601318", "300750", "000333"]
MINUTE_PERIODS = {"5min", "15min", "30min", "60min"}
# akshare 分钟线接口只稳定返回近期数据。UI 侧提前收窄请求范围，
# 避免用户选了一年却只看到几十天，误以为图表漏画。
MAX_MINUTE_LOOKBACK_DAYS = 35
INITIAL_VISIBLE_BARS = {
    "5min": 120,
    "15min": 96,
    "30min": 80,
    "60min": 60,
    "daily": 120,
    "weekly": 80,
    "monthly": 80,
}


@dataclass
class _ChartData:
    df: pd.DataFrame                 # 已排序好的日线
    candles: list[tuple]             # CandlestickItem 输入
    ma_lines: dict[int, list[float]]  # window → MA 序列
    dates: list[datetime]
    volumes: list[float]
    trade_markers: list[dict]        # {x, price, side}
    data_status: str = "unknown"
    data_source: str = "unknown"


def _load_chart_data(
    symbol: str, start: date, end: date, ma_windows: tuple[int, ...], period: str = "daily",
) -> _ChartData:
    df = get_loader().get_bars(symbol, start=start, end=end, period=period)
    if df.empty:
        raise DataError(f"{symbol} 在 {start}~{end} 没有 {period} 数据")

    df = df.sort_values("date").reset_index(drop=True)
    candles = [
        (i, float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
        for i, r in df.iterrows()
    ]
    ma = {w: df["close"].rolling(w).mean().tolist() for w in ma_windows}
    return _ChartData(
        df=df,
        candles=candles,
        ma_lines=ma,
        dates=[d.to_pydatetime() for d in df["date"]],
        volumes=df["volume"].tolist(),
        trade_markers=[],
        data_status=str(df.attrs.get(BAR_STATUS_ATTR) or "unknown"),
        data_source=str(df.attrs.get(BAR_SOURCE_ATTR) or "unknown"),
    )


def _effective_request_range(start: date, end: date, period: str) -> tuple[date, bool]:
    """返回实际向数据源请求的 start，以及是否因为分钟周期被自动收窄。"""
    if period not in MINUTE_PERIODS:
        return start, False
    earliest = end - timedelta(days=MAX_MINUTE_LOOKBACK_DAYS)
    if start < earliest:
        return earliest, True
    return start, False


def _data_source_note(status: str, source: str) -> str:
    if status == "refreshed":
        return f"已刷新({source})"
    if status == "cache_hit":
        return "本地缓存"
    if status == "cache_fallback":
        return "刷新失败，使用本地缓存"
    if status == "derived_15min_from_5min":
        return "15分由5分重采样"
    return ""


def _intraday_minute_note(
    period: str,
    *,
    end: date,
    last: datetime,
    now: datetime | None = None,
) -> str:
    if not period.endswith("min"):
        return ""
    local_now = now or datetime.now(MARKET_TZ).replace(tzinfo=None)
    today = local_now.date()
    if end < today or last.date() >= today:
        return ""
    if not is_trading_day(today):
        return ""
    if local_now.time() < time(9, 35):
        return ""
    return "交易日分钟K线未包含今日数据"


def _attach_trade_markers(data: _ChartData, symbol: str) -> None:
    """从 SQLite 读这个 symbol 的历史订单，找出在数据时间范围内的成交，标到图上。"""
    if data.df.empty:
        return
    orders = recent_orders(limit=1000)
    date_to_x = {pd.Timestamp(d.date()).normalize(): i for i, d in enumerate(data.dates)}
    for o in orders:
        if o["symbol"] != symbol:
            continue
        if o["status"] != "FILLED" or not o["ts"]:
            continue
        # 把订单时间对齐到当日交易日 X
        d = pd.Timestamp(o["ts"].date()).normalize()
        x = date_to_x.get(d)
        if x is None:
            continue
        price = o["avg_fill_price"] or 0
        data.trade_markers.append({"x": x, "price": float(price), "side": o["side"]})


class KlineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        from PySide6.QtWidgets import QButtonGroup

        root = QVBoxLayout(self)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        self._controls_widget = controls

        # ----- 顶部表单 -----
        form = QHBoxLayout()
        form.addWidget(QLabel("标的"))
        self.cmb_symbol = QComboBox()
        self.cmb_symbol.setEditable(True)
        self.cmb_symbol.addItems(PRESET_SYMBOLS)
        self.cmb_symbol.setCurrentText("000001")
        form.addWidget(self.cmb_symbol)

        form.addWidget(QLabel("起"))
        self.de_start = QDateEdit()
        self.de_start.setCalendarPopup(True)
        self.de_start.setDate(date.today() - timedelta(days=365))
        form.addWidget(self.de_start)

        form.addWidget(QLabel("止"))
        self.de_end = QDateEdit()
        self.de_end.setCalendarPopup(True)
        self.de_end.setDate(date.today())
        form.addWidget(self.de_end)

        form.addWidget(QLabel("均线"))
        self.le_ma = QLineEdit("5,20,60")
        self.le_ma.setMaximumWidth(120)
        form.addWidget(self.le_ma)

        self.btn_load = QPushButton("加载")
        self.btn_load.clicked.connect(self.load)
        form.addWidget(self.btn_load)
        form.addStretch()
        controls_layout.addLayout(form)

        # ----- 周期切换按钮组 -----
        period_bar = QHBoxLayout()
        period_bar.addWidget(QLabel("周期"))
        self._period_group = QButtonGroup(self)
        self._period_group.setExclusive(True)
        self._current_period = "daily"
        self._period_buttons: dict[str, QPushButton] = {}
        for code in ("5min", "15min", "30min", "60min", "daily", "weekly", "monthly"):
            btn = QPushButton(PERIOD_LABELS[code])
            btn.setCheckable(True)
            btn.setProperty("secondary", "true")
            btn.setMaximumWidth(70)
            if code == "daily":
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked=False, c=code: self._on_period_change(c))
            self._period_group.addButton(btn)
            self._period_buttons[code] = btn
            period_bar.addWidget(btn)
        period_bar.addStretch()
        controls_layout.addLayout(period_bar)
        root.addWidget(controls)

        # ----- 双图 -----
        # 共享 X 轴的两个 PlotWidget（pyqtgraph 自带 setXLink）
        self._date_axis_main = DateAxisItem(orientation="bottom")
        self.plot_main = pg.PlotWidget(axisItems={"bottom": self._date_axis_main})
        self.plot_main.showGrid(x=True, y=True, alpha=0.3)
        self.plot_main.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_main.setMouseEnabled(x=True, y=True)
        self.plot_main.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot_main.getViewBox().sigXRangeChanged.connect(self._on_main_x_range_changed)
        self.plot_main.addLegend(offset=(10, 10))
        root.addWidget(self.plot_main, stretch=3)

        self._date_axis_vol = DateAxisItem(orientation="bottom")
        self.plot_vol = pg.PlotWidget(axisItems={"bottom": self._date_axis_vol})
        self.plot_vol.showGrid(x=True, y=True, alpha=0.3)
        self.plot_vol.setLabel("left", "成交量")
        self.plot_vol.setMaximumHeight(160)
        self.plot_vol.setXLink(self.plot_main)
        self.plot_vol.setMouseEnabled(x=True, y=False)
        self.plot_vol.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        root.addWidget(self.plot_vol)

        # ----- 状态栏 -----
        self.lbl_status = QLabel("点[加载]开始")
        self.lbl_status.setStyleSheet("color:#888;")
        root.addWidget(self.lbl_status)

        # 占位 items 引用，避免反复重建
        self._candle_item: CandlestickItem | None = None
        self._ma_curves: list = []
        self._marker_items: list = []
        self._vol_item = None
        self._chart_data: _ChartData | None = None

    # ---------------- 加载 ----------------

    def set_controls_visible(self, visible: bool) -> None:
        """显示/隐藏表单和周期按钮；嵌入实时行情页时由外层提供控制。"""
        self._controls_widget.setVisible(visible)

    def _on_period_change(self, period: str) -> None:
        """切换周期。**不修改用户的日期选择**——让他们自己控制范围。

        如果用户选的范围对当前周期不合适（如分钟级 + 1 年范围），
        akshare 会自然只返回它能给的部分，状态栏会显示真实数据范围。
        """
        self.set_period(period, load=True)

    def set_period(self, period: str, *, load: bool = False) -> None:
        """设置周期；供实时行情页复用 K 线控件。"""
        if period not in PERIOD_LABELS:
            raise ValueError(f"unknown period: {period}")
        self._current_period = period
        btn = self._period_buttons.get(period)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        if load:
            self.load()

    def load_symbol(
        self,
        symbol: str,
        *,
        period: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> None:
        """设置标的和日期后加载图表；供实时行情页联动选中标的。"""
        self.cmb_symbol.setCurrentText(symbol)
        if start is not None:
            self.de_start.setDate(start)
        if end is not None:
            self.de_end.setDate(end)
        if period is not None:
            self.set_period(period, load=False)
        self.load()

    def load(self):
        sym_raw = self.cmb_symbol.currentText().strip()
        try:
            symbol = normalize_symbol(sym_raw)
        except Exception as e:
            self.lbl_status.setText(f"标的格式错：{e}")
            return

        raw_start = self.de_start.date().toPython()
        end = self.de_end.date().toPython()
        if raw_start > end:
            self.lbl_status.setText("日期范围错：起始日不能晚于终止日")
            return
        start, clamped = _effective_request_range(raw_start, end, self._current_period)
        if clamped:
            self.de_start.setDate(start)
        try:
            ma_windows = tuple(int(x.strip()) for x in self.le_ma.text().split(",") if x.strip())
        except ValueError:
            self.lbl_status.setText("均线参数格式错，应为如 5,20,60")
            return

        t0 = datetime.now()
        try:
            data = _load_chart_data(symbol, start, end, ma_windows, period=self._current_period)
        except DataError as e:
            self.lbl_status.setText(f"加载失败：{e}")
            return
        _attach_trade_markers(data, symbol)
        self._render(symbol, data)
        elapsed_ms = (datetime.now() - t0).total_seconds() * 1000

        # 比较"实际请求范围"和"实际拿到范围"，如果差距大就给提示
        first = data.dates[0]
        last = data.dates[-1]
        first_str = first.strftime("%Y-%m-%d %H:%M") if self._current_period.endswith("min") else first.date()
        last_str = last.strftime("%Y-%m-%d %H:%M") if self._current_period.endswith("min") else last.date()

        requested_days = (end - start).days
        actual_days = (last.date() - first.date()).days if hasattr(last, "date") else 0
        notes: list[str] = []
        if clamped:
            notes.append(
                f"分钟周期最多自动请求近 {MAX_MINUTE_LOOKBACK_DAYS} 天"
                f"（已从 {raw_start} 收窄）"
            )
        if self._current_period.endswith("min") and requested_days > actual_days + 5 and requested_days > 7:
            notes.append(f"数据源实际返回约 {actual_days} 天")
        intraday_note = _intraday_minute_note(self._current_period, end=end, last=last)
        if intraday_note:
            notes.append(intraday_note)
        source_note = _data_source_note(data.data_status, data.data_source)
        if source_note:
            notes.append(source_note)
        note_text = f" | {'；'.join(notes)}" if notes else ""

        self.lbl_status.setText(
            f"{symbol} · {PERIOD_LABELS[self._current_period]} | "
            f"请求 {start} → {end} | 实际 {first_str} → {last_str} | "
            f"{len(data.df)} 根 | 标记 {len(data.trade_markers)} | 加载 {elapsed_ms:.0f}ms{note_text}"
        )

    # ---------------- 渲染 ----------------

    def _render(self, symbol: str, data: _ChartData) -> None:
        self._chart_data = data

        # 清理旧 item
        self.plot_main.clear()
        self.plot_vol.clear()
        self._ma_curves.clear()
        self._marker_items.clear()

        # 蜡烛
        self._candle_item = CandlestickItem(data.candles, width=0.6)
        self.plot_main.addItem(self._candle_item)

        # 均线
        ma_colors = ["#5b8bd6", "#caa14a", "#b07cc6", "#7cc6b0"]
        for i, (w, series) in enumerate(data.ma_lines.items()):
            xs = list(range(len(series)))
            curve = pg.PlotCurveItem(
                xs, series, pen=pg.mkPen(ma_colors[i % len(ma_colors)], width=1.5),
                name=f"MA{w}",
            )
            self.plot_main.addItem(curve)
            self._ma_curves.append(curve)

        # 买卖标记
        if data.trade_markers:
            buys = [(m["x"], m["price"]) for m in data.trade_markers if m["side"] == "BUY"]
            sells = [(m["x"], m["price"]) for m in data.trade_markers if m["side"] == "SELL"]
            if buys:
                xs, ys = zip(*buys, strict=False)
                self.plot_main.plot(
                    xs, ys, pen=None, symbol="t1",   # ▲
                    symbolBrush=pg.mkBrush(UP_COLOR), symbolSize=12, name="买入",
                )
            if sells:
                xs, ys = zip(*sells, strict=False)
                self.plot_main.plot(
                    xs, ys, pen=None, symbol="t",    # ▼
                    symbolBrush=pg.mkBrush(DOWN_COLOR), symbolSize=12, name="卖出",
                )

        # X 轴日期
        self._date_axis_main.set_dates(data.dates)
        self._date_axis_vol.set_dates(data.dates)

        # 成交量：上涨日红、下跌日绿
        bg_brushes = []
        for _i, r in data.df.iterrows():
            color = UP_COLOR if r["close"] >= r["open"] else DOWN_COLOR
            bg_brushes.append(pg.mkBrush(QColor(color)))
        x_idx = list(range(len(data.df)))
        self._vol_item = pg.BarGraphItem(
            x=x_idx, height=data.volumes, width=0.6, brushes=bg_brushes, pen=None,
        )
        self.plot_vol.addItem(self._vol_item)

        # 标题
        self.plot_main.setTitle(f"{symbol} K 线", color="w")

        # 关键：切标的时 Y 轴量级可能差 100 倍（10 元股 vs 1700 元茅台），
        # pyqtgraph 在 clear() 后不会自动重置已有 view range，必须显式 autoRange。
        # 同时锁定 view 范围在数据内 —— 防止滚轮 / 拖动把视图飞出数据范围找不回来。
        n = len(data.df)
        all_highs = data.df["high"].astype(float).tolist()
        all_lows = data.df["low"].astype(float).tolist()
        # MA 也算进 Y 范围（避免均线被裁掉）
        for series in data.ma_lines.values():
            for v in series:
                if v == v:    # 跳过 NaN
                    all_highs.append(float(v))
                    all_lows.append(float(v))
        lock_view_to_data(
            self.plot_main,
            xs=range(n),
            ys=all_lows + all_highs,
            x_padding=0.02, y_padding=0.12,
            min_visible_x=10,
        )
        # 成交量副图：X 共享，Y 从 0 起
        lock_view_to_data(
            self.plot_vol,
            xs=range(n),
            ys=[float(v) for v in data.volumes],
            x_padding=0.02, y_padding=0.05,
            min_visible_x=10,
            fix_y_zero=True,
        )
        self._focus_latest_window(data)

    def _focus_latest_window(self, data: _ChartData) -> None:
        """加载后默认聚焦最近一段 K 线，保留左右拖动和滚轮缩放空间。"""
        n = len(data.df)
        if n <= 0:
            return
        window = min(n, INITIAL_VISIBLE_BARS.get(self._current_period, 100))
        start = max(0, n - window)
        end = n - 1
        x_left = max(-0.5, start - 0.5)
        x_right = end + 0.8

        self.plot_main.setXRange(x_left, x_right, padding=0)
        self._fit_main_y_range(data, x_left, x_right)

        vol_values = [float(v) for v in data.volumes[start:n]]
        vol_lo, vol_hi = _padded_range(vol_values, padding=0.08, floor_zero=True)
        self.plot_vol.setYRange(vol_lo, vol_hi, padding=0)

    def _on_main_x_range_changed(self, _view_box, x_range) -> None:
        data = self._chart_data
        if data is None or not x_range:
            return
        self._fit_main_y_range(data, float(x_range[0]), float(x_range[1]))

    def _fit_main_y_range(self, data: _ChartData, x_left: float, x_right: float) -> None:
        start, end = _visible_slice_bounds(len(data.df), x_left, x_right)
        if start >= end:
            return
        visible = data.df.iloc[start:end]
        y_values = visible["low"].astype(float).tolist() + visible["high"].astype(float).tolist()
        for series in data.ma_lines.values():
            for v in series[start:end]:
                if v == v:
                    y_values.append(float(v))
        y_lo, y_hi = _padded_range(y_values, padding=0.12, floor_zero=False)
        self.plot_main.setYRange(y_lo, y_hi, padding=0)


def _padded_range(values: list[float], *, padding: float, floor_zero: bool) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    y_lo = 0.0 if floor_zero else lo - span * padding
    y_hi = hi + span * padding
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    return y_lo, y_hi


def _visible_slice_bounds(n: int, x_left: float, x_right: float) -> tuple[int, int]:
    if n <= 0:
        return (0, 0)
    start = max(0, floor(x_left))
    end = min(n, ceil(x_right) + 1)
    if end <= start:
        end = min(n, start + 1)
    return (start, end)
