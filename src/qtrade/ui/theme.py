"""全局深色主题：色板 + Qt 样式表 (QSS)。

设计参考：现代 SaaS dashboard（Linear / Vercel / Stripe Dashboard 风格）
- 极简、深色、留白多
- 卡片用细微边框而非阴影，避免 Mac 上模糊
- 强调色用偏冷的紫蓝 (#7a5cff)
- 涨跌色保留 A 股红涨绿跌

约定：所有 widget 通过 ``apply_theme(app)`` 一次性应用，子组件不要再自定义 QSS，
否则主题切换/微调时无法集中管理。
"""
from __future__ import annotations

# ------------------ 色板 ------------------

# 背景层（从外到内）
BG_WINDOW = "#0a0c11"        # 窗口底色，最深
BG_SIDEBAR = "#11141a"       # 侧边栏
BG_CARD = "#161a22"          # 卡片
BG_HOVER = "#1c2029"         # 悬停 / 表头
BG_ACTIVE = "#23283299"      # 选中（带透明）

# 边框
BORDER = "#22262f"
BORDER_STRONG = "#2d323d"

# 文字
TEXT_PRIMARY = "#eaecef"     # 主要数字 / 标题
TEXT_SECONDARY = "#9aa0aa"   # 普通正文
TEXT_MUTED = "#5a606b"       # 标签 / 提示

# 强调
ACCENT = "#7a5cff"           # 主强调（紫蓝）
ACCENT_DIM = "#7a5cff44"

# 涨跌（A 股习惯）
UP = "#e85d5d"               # 红涨
DOWN = "#52c285"             # 绿跌
NEUTRAL = "#7a8090"


# ------------------ QSS ------------------

QSS = f"""
/* ============ 全局 ============ */
QWidget {{
    background-color: {BG_WINDOW};
    color: {TEXT_PRIMARY};
    font-family: "PingFang SC", "Hiragino Sans GB", "Helvetica Neue", "Arial", sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {BG_WINDOW};
}}

/* ============ 侧边栏 ============ */
#Sidebar {{
    background-color: {BG_SIDEBAR};
    border: none;
    border-right: 1px solid {BORDER};
}}

#SidebarLogo {{
    color: {TEXT_PRIMARY};
    font-size: 16px;
    font-weight: 600;
    padding: 18px 20px 10px 20px;
}}

#SidebarGroupTitle {{
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 14px 20px 6px 20px;
}}

#SidebarItem {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    padding: 8px 20px;
    border: none;
    border-left: 2px solid transparent;
    text-align: left;
    font-size: 13px;
}}
#SidebarItem:hover {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}
#SidebarItem[active="true"] {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
    border-left: 2px solid {ACCENT};
    font-weight: 500;
}}

/* ============ 卡片 ============ */
#Card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

#CardLabel {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.3px;
}}

#CardValue {{
    color: {TEXT_PRIMARY};
    font-size: 22px;
    font-weight: 600;
}}

#CardValueLg {{
    color: {TEXT_PRIMARY};
    font-size: 28px;
    font-weight: 600;
}}

#CardSub {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}

#SectionTitle {{
    color: {TEXT_PRIMARY};
    font-size: 15px;
    font-weight: 600;
    padding: 4px 0px;
}}

/* ============ 表格 ============ */
QTableWidget, QTableView {{
    background-color: {BG_CARD};
    alternate-background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT_DIM};
    selection-color: {TEXT_PRIMARY};
    outline: 0;
}}
QTableWidget::item, QTableView::item {{
    padding: 8px 6px;
    border: none;
}}
QHeaderView::section {{
    background-color: {BG_CARD};
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px 6px;
    font-size: 11px;
    font-weight: 500;
    text-align: left;
}}
QHeaderView {{
    background-color: {BG_CARD};
}}
QTableCornerButton::section {{
    background-color: {BG_CARD};
    border: none;
}}

/* ============ 输入控件 ============ */
QLineEdit, QComboBox, QDateEdit, QDoubleSpinBox, QSpinBox {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QComboBox:focus, QDateEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_SECONDARY};
    margin-right: 8px;
}}

/* ============ 按钮 ============ */
QPushButton {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: #8a6cff;
}}
QPushButton:pressed {{
    background-color: #6a4cef;
}}
QPushButton:disabled {{
    background-color: {BG_HOVER};
    color: {TEXT_MUTED};
}}
QPushButton[secondary="true"] {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
}}
QPushButton[secondary="true"]:hover {{
    border: 1px solid {ACCENT};
    color: {TEXT_PRIMARY};
}}
QPushButton[secondary="true"]:checked {{
    background-color: {ACCENT};
    color: white;
    border: 1px solid {ACCENT};
}}

QCheckBox {{
    color: {TEXT_SECONDARY};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 3px;
    background-color: {BG_CARD};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
}}

/* ============ 滚动条 ============ */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ============ 文本编辑 ============ */
QTextEdit, QPlainTextEdit {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px;
    font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: 12px;
}}

/* ============ Splitter ============ */
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QSplitter::handle:vertical {{
    height: 1px;
}}

/* ============ 状态栏 ============ */
QStatusBar {{
    background-color: {BG_SIDEBAR};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
    font-size: 11px;
}}

/* ============ 标签 ============ */
QLabel {{
    background: transparent;
    color: {TEXT_PRIMARY};
}}
QLabel[muted="true"] {{
    color: {TEXT_MUTED};
}}
QLabel[secondary="true"] {{
    color: {TEXT_SECONDARY};
}}
"""


def apply_theme(app) -> None:
    """对 QApplication 应用全局主题。"""
    app.setStyleSheet(QSS)


def pnl_color(value: float) -> str:
    """统一的涨跌着色（中国习惯：红涨绿跌）。"""
    if value > 0:
        return UP
    if value < 0:
        return DOWN
    return NEUTRAL
