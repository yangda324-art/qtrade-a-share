"""左侧分组导航栏。

设计：
- 顶部 logo
- 多个分组（每组 ``GROUP NAME`` 小标题 + 若干 item）
- 选中 item 左侧紫色 border + 浅色背景
- 点击 item 触发 ``activated(view_key: str)`` 信号

不依赖任何业务逻辑——把 view_key 给主窗口由它去切 ``QStackedWidget``。
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class NavItem:
    key: str          # 视图标识，如 "dashboard"
    icon: str         # Unicode 图标，如 "▦"
    label: str        # 显示名


@dataclass(frozen=True)
class NavGroup:
    title: str
    items: tuple[NavItem, ...]


class _NavButton(QPushButton):
    def __init__(self, item: NavItem, parent: QWidget | None = None) -> None:
        super().__init__(f"  {item.icon}    {item.label}", parent)
        self.setObjectName("SidebarItem")
        self.item = item
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_active(self, active: bool) -> None:
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class Sidebar(QFrame):
    activated = Signal(str)  # view_key

    def __init__(self, groups: list[NavGroup], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(220)
        self._buttons: dict[str, _NavButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo
        logo = QLabel("◆ qtrade")
        logo.setObjectName("SidebarLogo")
        layout.addWidget(logo)

        # 分组
        for grp in groups:
            title = QLabel(grp.title)
            title.setObjectName("SidebarGroupTitle")
            layout.addWidget(title)
            for it in grp.items:
                btn = _NavButton(it)
                btn.clicked.connect(lambda _checked=False, k=it.key: self._on_click(k))
                layout.addWidget(btn)
                self._buttons[it.key] = btn

        layout.addStretch()

        # 底部副信息
        footer = QLabel("v0.1.0 · Phase 4")
        footer.setObjectName("SidebarGroupTitle")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

    def _on_click(self, key: str) -> None:
        self.set_active(key)
        self.activated.emit(key)

    def set_active(self, key: str) -> None:
        for k, btn in self._buttons.items():
            btn.set_active(k == key)
