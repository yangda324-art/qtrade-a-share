"""图表通用辅助：把 pyqtgraph view 锁在数据范围内。

解决的问题：默认情况下 pyqtgraph 允许你把视图拖到无穷远 / 缩到一个像素，
经常一不小心就找不回数据。``lock_view_to_data`` 给一个 PlotWidget 设
xMin/xMax/yMin/yMax + 最小最大缩放区间，让滚轮和拖动都不能"飞出去"。
"""
from __future__ import annotations

from collections.abc import Iterable


def lock_view_to_data(
    plot,
    *,
    xs: Iterable[float],
    ys: Iterable[float],
    x_padding: float = 0.02,
    y_padding: float = 0.08,
    min_visible_x: int = 8,
    fix_y_zero: bool = False,
) -> None:
    """把 plot 的 view 锁定在数据范围内。

    Args:
        plot: ``pyqtgraph.PlotWidget`` 或 ``ViewBox``
        xs:   X 轴所有数据点
        ys:   Y 轴所有数据点
        x_padding: X 两侧留白比例
        y_padding: Y 两侧留白比例
        min_visible_x: 最小可见 X 数（防止缩放到一个像素）
        fix_y_zero: 是否把 Y 下限固定为 0（成交量图用）
    """
    xs_list = list(xs)
    ys_list = list(ys)
    if not xs_list or not ys_list:
        return

    xmin, xmax = float(min(xs_list)), float(max(xs_list))
    ymin, ymax = float(min(ys_list)), float(max(ys_list))
    xrange = xmax - xmin
    yrange = ymax - ymin
    if xrange <= 0:
        xrange = 1.0
    if yrange <= 0:
        yrange = max(abs(ymax), 1.0) * 0.1

    xpad = xrange * x_padding
    ypad = yrange * y_padding
    y_lo = 0.0 if fix_y_zero else ymin - ypad
    y_hi = ymax + ypad

    plot.setLimits(
        xMin=xmin - xpad,
        xMax=xmax + xpad,
        yMin=y_lo,
        yMax=y_hi,
        # 缩放区间下限：至少要看到 min_visible_x 个点；防止缩到一格
        minXRange=float(min(min_visible_x, max(int(xrange), 1))),
        # 缩放区间上限：最多就是数据全宽 + 两侧 padding
        maxXRange=xrange + xpad * 2,
        minYRange=max(yrange * 0.05, 1e-6),
        maxYRange=yrange + ypad * 2,
    )
    plot.setRange(
        xRange=(xmin - xpad, xmax + xpad),
        yRange=(y_lo, y_hi),
        padding=0,
    )
