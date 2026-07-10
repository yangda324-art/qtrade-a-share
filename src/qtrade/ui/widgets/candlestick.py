"""pyqtgraph 蜡烛图控件。

pyqtgraph 没有内置 candlestick，自己用 ``QGraphicsObject`` 实现一个。
所有蜡烛在一个 ``QPicture`` 里画完后整体绘制——百根级别也很流畅。

输入数据：``list[tuple]`` ``(t_index, open, high, low, close)``，``t_index`` 是 X 轴整数。
- 不直接用 datetime 做 X：浮点时间戳和 GUI 渲染配合不太好；用整数索引 + AxisItem 自定义 tick 显示日期更稳。

颜色：A 股习惯——红涨绿跌。
"""
from __future__ import annotations

from datetime import datetime

import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainter, QPicture
from PySide6.QtWidgets import QGraphicsObject

UP_COLOR = "#d04949"        # 红：涨
DOWN_COLOR = "#3aa055"      # 绿：跌
DOJI_COLOR = "#888888"      # 同价


class CandlestickItem(QGraphicsObject):
    """一组蜡烛。

    Args:
        data: list of ``(x, open, high, low, close)``，``x`` 是 X 轴整数索引
        width: 单根蜡烛宽度（占 X 单位的比例，0–1），默认 0.6
    """

    def __init__(self, data: list[tuple[int, float, float, float, float]], width: float = 0.6):
        super().__init__()
        self.data = data
        self.width = width
        self._picture: QPicture | None = None
        self._generate_picture()

    def set_data(self, data: list[tuple[int, float, float, float, float]]) -> None:
        self.data = data
        self._generate_picture()
        self.update()

    def _generate_picture(self) -> None:
        self._picture = QPicture()
        if not self.data:
            return
        p = QPainter(self._picture)
        try:
            half = self.width / 2.0
            for x, o, h, low, c in self.data:
                if c > o:
                    color = UP_COLOR
                elif c < o:
                    color = DOWN_COLOR
                else:
                    color = DOJI_COLOR
                pen = pg.mkPen(color)
                brush = pg.mkBrush(color)
                p.setPen(pen)
                # 上下影线
                p.drawLine(QPointF(x, low), QPointF(x, h))
                # 实体（A 股风格：涨为红实心、跌为绿实心；空心也 OK，这里用实心更直观）
                p.setBrush(brush)
                top = max(o, c)
                bot = min(o, c)
                if top == bot:
                    # 一字线，画一条横线
                    p.drawLine(QPointF(x - half, o), QPointF(x + half, o))
                else:
                    p.drawRect(QRectF(x - half, bot, self.width, top - bot))
        finally:
            p.end()

    def paint(self, painter, option, widget=None):
        if self._picture is not None:
            painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        if self._picture is None:
            return QRectF()
        return QRectF(self._picture.boundingRect())


class DateAxisItem(pg.AxisItem):
    """X 轴：把整数索引映射回日期字符串显示。

    自动根据数据跨度切换格式：
    - 跨多年   → ``YY-MM``
    - 跨多天   → ``MM-DD``
    - 同一天   → ``HH:MM``
    """

    def __init__(self, dates: list[datetime] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.dates: list[datetime] = dates or []

    def set_dates(self, dates: list[datetime]) -> None:
        self.dates = dates
        self.picture = None
        self.update()

    def _fmt(self) -> str:
        if not self.dates:
            return "%m-%d"
        span = self.dates[-1] - self.dates[0]
        if span.days > 365:
            return "%y-%m"
        if span.days >= 1:
            return "%m-%d"
        return "%H:%M"

    def tickStrings(self, values, scale, spacing) -> list[str]:
        fmt = self._fmt()
        out: list[str] = []
        for v in values:
            i = int(v)
            if 0 <= i < len(self.dates):
                out.append(self.dates[i].strftime(fmt))
            else:
                out.append("")
        return out
