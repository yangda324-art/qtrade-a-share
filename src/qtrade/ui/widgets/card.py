"""可复用的卡片组件：MetricCard / Card / SectionTitle。

样式由 theme.QSS 控制，所有视觉调整都集中在主题表里。
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from qtrade.ui.theme import pnl_color


class Card(QFrame):
    """通用卡片容器：圆角 + 边框 + 内边距。子内容自由布局。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setFrameShape(QFrame.NoFrame)
        # 主内容布局，子类/外部用 self.content_layout 加内容
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(16, 14, 16, 14)
        self._outer.setSpacing(8)

    def add(self, w: QWidget) -> None:
        self._outer.addWidget(w)

    def addLayout(self, layout) -> None:
        self._outer.addLayout(layout)


class MetricCard(Card):
    """单指标卡片：上方小标签 + 下方大数字 + 可选副文本。"""

    def __init__(
        self,
        label: str,
        value: str = "—",
        sub: str = "",
        large: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._lbl = QLabel(label)
        self._lbl.setObjectName("CardLabel")

        self._val = QLabel(value)
        self._val.setObjectName("CardValueLg" if large else "CardValue")

        self._sub = QLabel(sub)
        self._sub.setObjectName("CardSub")
        self._sub.setVisible(bool(sub))

        self._outer.addWidget(self._lbl)
        self._outer.addWidget(self._val)
        self._outer.addWidget(self._sub)
        self._outer.addStretch()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(96)

    def set_value(self, value: str, color: str | None = None) -> None:
        self._val.setText(value)
        if color:
            self._val.setStyleSheet(f"color: {color};")
        else:
            self._val.setStyleSheet("")

    def set_sub(self, text: str, color: str | None = None) -> None:
        self._sub.setText(text)
        self._sub.setVisible(bool(text))
        if color:
            self._sub.setStyleSheet(f"color: {color};")
        else:
            self._sub.setStyleSheet("")

    def set_pnl(self, value: float, prefix: str = "", suffix: str = "", as_pct: bool = False) -> None:
        """便捷：用涨跌色染数字。as_pct=True 时按百分比格式化。"""
        text = f"{prefix}{value:+.2%}{suffix}" if as_pct else f"{prefix}{value:+,.2f}{suffix}"
        self.set_value(text, color=pnl_color(value))


class SectionTitle(QLabel):
    """章节小标题。"""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SectionTitle")
