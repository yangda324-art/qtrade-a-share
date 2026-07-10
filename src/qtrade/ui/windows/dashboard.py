"""Dashboard：极简深色风格。

布局：
    Header（当前 run + 状态）
    指标卡片网格 (2 行 × 3 列)
    持仓表
    权益曲线 (面积图)

数据每 ``REFRESH_MS`` 毫秒刷新。
"""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qtrade.core.time import MARKET_TZ
from qtrade.ui.data_access import (
    equity_curve,
    latest_account_snapshot,
    latest_active_run,
    latest_positions,
    market_data_quality,
    paper_run_health,
)
from qtrade.ui.theme import (
    ACCENT,
    BG_CARD,
    BORDER,
    TEXT_MUTED,
    TEXT_SECONDARY,
    pnl_color,
)
from qtrade.ui.widgets.card import Card, MetricCard, SectionTitle

REFRESH_MS = 3000
SNAPSHOT_STALE_SECONDS = 60


def _money(v: float) -> str:
    return f"{v:,.2f}"


def _fmt_dt(ts) -> str:
    if not ts:
        return "-"
    return ts.strftime("%m-%d %H:%M:%S")


def _seconds_since(ts) -> float | None:
    if not ts:
        return None
    now = datetime.now(MARKET_TZ).replace(tzinfo=None)
    return max((now - ts).total_seconds(), 0.0)


def _age_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} 分钟"
    return f"{minutes / 60:.1f} 小时"


def _source_text(sources: list[str]) -> str:
    if not sources:
        return "-"
    shown = sources[:3]
    suffix = f" +{len(sources) - len(shown)}" if len(sources) > len(shown) else ""
    return ", ".join(shown) + suffix


class DashboardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(0, 0, 0, 0)

        # ---- Header ----
        header = QHBoxLayout()
        title = QLabel("总览")
        title.setObjectName("SectionTitle")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()

        self.lbl_run = QLabel("—")
        self.lbl_run.setProperty("muted", "true")
        self.lbl_run.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        header.addWidget(self.lbl_run)
        root.addLayout(header)

        self.lbl_run_detail = QLabel("—")
        self.lbl_run_detail.setProperty("secondary", "true")
        self.lbl_run_detail.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        root.addWidget(self.lbl_run_detail)

        # ---- 指标卡片 6 个 (2×3 grid) ----
        grid = QGridLayout()
        grid.setSpacing(14)
        self.card_total    = MetricCard("总资产",   "—", large=True)
        self.card_cash     = MetricCard("可用现金", "—")
        self.card_market   = MetricCard("持仓市值", "—")
        self.card_realized = MetricCard("已实现盈亏", "—")
        self.card_unreal   = MetricCard("浮动盈亏", "—")
        self.card_pos      = MetricCard("持仓只数", "—")

        grid.addWidget(self.card_total,    0, 0)
        grid.addWidget(self.card_cash,     0, 1)
        grid.addWidget(self.card_market,   0, 2)
        grid.addWidget(self.card_realized, 1, 0)
        grid.addWidget(self.card_unreal,   1, 1)
        grid.addWidget(self.card_pos,      1, 2)
        root.addLayout(grid)

        # ---- 模拟盘健康 ----
        root.addWidget(SectionTitle("模拟盘健康"))
        quality_card = Card()
        self.lbl_run_health = QLabel("—")
        self.lbl_run_health.setWordWrap(True)
        self.lbl_run_health.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        quality_card.add(self.lbl_run_health)
        self.lbl_market_quality = QLabel("—")
        self.lbl_market_quality.setWordWrap(True)
        self.lbl_market_quality.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        quality_card.add(self.lbl_market_quality)
        root.addWidget(quality_card, stretch=0)

        # ---- 持仓 ----
        root.addWidget(SectionTitle("持仓"))
        self.tbl = QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(["标的", "持仓", "可卖", "成本", "现价", "市值", "浮动盈亏"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setShowGrid(False)
        self.tbl.setAlternatingRowColors(False)
        self.tbl.setMinimumHeight(180)
        root.addWidget(self.tbl, stretch=0)

        # ---- 权益曲线 ----
        root.addWidget(SectionTitle("权益曲线"))
        try:
            import pyqtgraph as pg
            pg.setConfigOption("background", QColor(BG_CARD))
            pg.setConfigOption("foreground", QColor(TEXT_SECONDARY))

            self.plot = pg.PlotWidget()
            self.plot.setBackground(QColor(BG_CARD))
            self.plot.showGrid(x=False, y=True, alpha=0.15)
            self.plot.setMenuEnabled(False)
            self.plot.hideButtons()
            self.plot.setMouseEnabled(x=True, y=False)

            # 隐藏左边轴的数字外的刻度线
            for axis in ("left", "bottom"):
                ax = self.plot.getAxis(axis)
                ax.setPen(BORDER)
                ax.setTextPen(TEXT_MUTED)
                ax.setStyle(tickLength=0)

            self._curve = pg.PlotCurveItem(pen=pg.mkPen(ACCENT, width=2))
            self._curve_baseline = pg.PlotCurveItem()       # 用于填充的下限
            self._fill_brush = pg.mkBrush(QColor(122, 92, 255, 40))   # ACCENT 半透明
            self._fill = pg.FillBetweenItem(self._curve, self._curve_baseline, brush=self._fill_brush)
            self.plot.addItem(self._fill)
            self.plot.addItem(self._curve)
            self._has_plot = True
        except ImportError:
            self.plot = QLabel("pyqtgraph 未安装")
            self._has_plot = False
        # 卡片包裹图表
        plot_card = Card()
        if self._has_plot:
            plot_card.add(self.plot)
            self.plot.setMinimumHeight(220)
        else:
            plot_card.add(self.plot)
        root.addWidget(plot_card, stretch=1)

    # ---------------- 刷新 ----------------

    def refresh(self):
        run = latest_active_run()
        if not run:
            self.lbl_run.setText("无活跃 run · 跑一次 qtrade run --once 后再来")
            self.lbl_run_detail.setText(
                "启动模拟实盘：qtrade run --strategy ma_cross --symbol 000001 --interval-seconds 10"
            )
            self._clear_account_display()
            self._clear_market_quality()
            return

        run_id = run["run_id"]
        active = "● running" if run["ended_at"] is None else "○ ended"
        active_color = "#52c285" if run["ended_at"] is None else TEXT_MUTED
        self.lbl_run.setText(
            f"<span style='color:{active_color};'>{active}</span> · "
            f"<span style='color:{TEXT_SECONDARY};'>#{run_id} {run['strategy_id']} ({run['mode']})</span>"
        )

        snap = latest_account_snapshot(run_id)
        if snap:
            self._set_run_detail(run, snap)
            self.card_total.set_value(_money(snap["total_assets"]))
            self.card_cash.set_value(_money(snap["cash"]))
            self.card_market.set_value(_money(snap["market_value"]))
            self.card_realized.set_pnl(snap["realized_pnl"])
            self.card_unreal.set_pnl(snap["unrealized_pnl"])
        else:
            self.lbl_run_detail.setText("等待账户快照写入…")
            self._clear_metric_cards()

        positions = latest_positions(run_id)
        self.card_pos.set_value(str(len(positions)))
        self._set_run_health(run_id)
        self._set_market_quality(run_id)

        # ---- 持仓表 ----
        self.tbl.setRowCount(len(positions))
        for r, p in enumerate(positions):
            pnl = p["unrealized_pnl"]
            cells = [
                p["symbol"],
                str(p["quantity"]),
                str(p["available"]),
                f"{p['avg_cost']:.3f}",
                f"{p['market_price']:.3f}",
                _money(p["market_value"]),
                f"{pnl:+,.2f}",
            ]
            for c, v in enumerate(cells):
                item = QTableWidgetItem(v)
                if c == 6:
                    item.setForeground(QColor(pnl_color(pnl)))
                if c >= 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.tbl.setItem(r, c, item)

        # ---- 权益曲线 ----
        if self._has_plot:
            curve = equity_curve(run_id, limit=500)
            if curve:
                xs = list(range(len(curve)))
                ys = [c["total_assets"] for c in curve]
                self._curve.setData(xs, ys)
                # 下限用 min - 5% 形成填充
                ymin = min(ys) - (max(ys) - min(ys)) * 0.1 - 1
                self._curve_baseline.setData(xs, [ymin] * len(xs))
                self._fill.setCurves(self._curve, self._curve_baseline)
                self.plot.enableAutoRange()
            else:
                self._clear_equity_plot()

    def _clear_account_display(self) -> None:
        self._clear_metric_cards()
        self.tbl.setRowCount(0)
        self._clear_equity_plot()
        self._clear_run_health()
        self._clear_market_quality()

    def _set_run_detail(self, run: dict, snap: dict) -> None:
        started = _fmt_dt(run.get("started_at"))
        snap_ts = snap.get("ts")
        snap_age = _seconds_since(snap_ts)
        if snap_age is None:
            freshness = "快照时间未知"
            color = TEXT_MUTED
        elif snap_age > SNAPSHOT_STALE_SECONDS and run.get("ended_at") is None:
            freshness = f"最近快照 {_age_text(snap_age)} 前，后台可能未持续 tick"
            color = "#caa14a"
        else:
            freshness = f"最近快照 {_age_text(snap_age)} 前"
            color = "#52c285" if run.get("ended_at") is None else TEXT_MUTED

        end_note = f" · 已结束 {_fmt_dt(run.get('ended_at'))}" if run.get("ended_at") else ""
        self.lbl_run_detail.setText(
            f"开始 {started}{end_note} · <span style='color:{color};'>{freshness}</span>"
        )

    def _clear_metric_cards(self) -> None:
        for card in (
            self.card_total,
            self.card_cash,
            self.card_market,
            self.card_realized,
            self.card_unreal,
            self.card_pos,
        ):
            card.set_value("—")

    def _set_market_quality(self, run_id: int) -> None:
        quality = market_data_quality(run_id, limit=20)
        if not quality:
            self._clear_market_quality("暂无 tick 日志；启动模拟盘后这里会显示行情源质量。")
            return

        age = _seconds_since(quality.get("latest_ts"))
        age_text = _age_text(age) if age is not None else "未知"
        latest_tradeable = quality["latest_tradeable_quotes"]
        latest_total = quality["latest_quotes_count"]
        fallback = quality["latest_fallback_quotes"]
        ratio = quality["tradeable_ratio"]

        if latest_total == 0:
            status = "无行情"
            color = "#caa14a"
        elif latest_tradeable == latest_total and fallback == 0:
            status = "实时可模拟"
            color = "#52c285"
        else:
            status = "仅部分可模拟"
            color = "#caa14a"

        reasons = ", ".join(
            f"{item['reason']}×{item['count']}" for item in quality["top_skip_reasons"]
        ) or "无"
        self.lbl_market_quality.setText(
            f"<span style='color:{color}; font-weight:600;'>{status}</span>"
            f" · 最近 tick {age_text} 前"
            f" · 最新可交易 {latest_tradeable}/{latest_total}"
            f" · fallback {fallback}"
            f" · 近 {quality['window_ticks']} tick 可交易率 {ratio:.0%}"
            f" · 跳过原因：{reasons}"
        )

    def _set_run_health(self, run_id: int) -> None:
        health = paper_run_health(run_id, tick_limit=20, order_limit=50)
        if not health:
            self._clear_run_health("健康状态读取失败")
            return

        color = {
            "ok": "#52c285",
            "warning": "#caa14a",
            "error": "#d04949",
            "muted": TEXT_MUTED,
        }.get(health.get("severity"), TEXT_MUTED)

        tick_age = health.get("latest_tick_age_seconds")
        tick_text = f"最近 tick {_age_text(tick_age)} 前" if tick_age is not None else "暂无 tick"
        sources = _source_text(health.get("latest_quote_sources") or [])
        reason = health.get("latest_reason") or "-"
        reason_part = "原因：无" if reason == "-" else f"原因：{reason}"

        self.lbl_run_health.setText(
            f"<span style='color:{color}; font-weight:600;'>{health['status_text']}</span>"
            f" · {tick_text}"
            f" · 行情 {health['latest_tradeable_quotes']}/{health['latest_quotes_count']} 可模拟"
            f" · 源 {sources}"
            f" · 挂单 {health['open_order_count']}"
            f" · 近 {health['recent_order_count']} 单"
            f"（成 {health['filled_order_count']} / 拒 {health['rejected_order_count']}）"
            f" · 风控 {health['risk_rejected_total']}"
            f" · {reason_part}"
        )

    def _clear_run_health(self, text: str = "—") -> None:
        self.lbl_run_health.setText(text)

    def _clear_market_quality(self, text: str = "—") -> None:
        self.lbl_market_quality.setText(text)

    def _clear_equity_plot(self) -> None:
        if not self._has_plot:
            return
        self._curve.setData([], [])
        self._curve_baseline.setData([], [])
        self._fill.setCurves(self._curve, self._curve_baseline)
