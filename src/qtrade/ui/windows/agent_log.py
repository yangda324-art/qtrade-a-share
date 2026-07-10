"""Agent 决策 tab：从 SQLite 读 LLM 多 Agent 的完整决策链路。

上半：决策列表（时间 / 标的 / 动作 / 置信度 / token）
下半：选中行展开 trace（每个 Agent 的输入输出原文）
"""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from qtrade.ui.data_access import llm_usage_today, recent_decisions

REFRESH_MS = 5000

_ACTION_COLOR = {"BUY": "#d04949", "SELL": "#3aa055", "HOLD": "#888888"}
_EXECUTION_COLOR = {
    "FILLED": "#3aa055",
    "SUBMITTED": "#caa14a",
    "BLOCKED": "#d04949",
    "REJECTED": "#d04949",
    "NO_ORDER": "#888888",
}


class AgentLogWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._decisions: list[dict] = []
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)
        root.addWidget(QLabel("<h3>多 Agent 决策日志</h3>"))

        splitter = QSplitter(Qt.Vertical)

        cols = ["时间", "标的", "动作", "置信度", "目标仓位", "执行", "Tokens", "成本(USD)", "理由"]
        self.tbl = QTableWidget(0, len(cols))
        self.tbl.setHorizontalHeaderLabels(cols)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.itemSelectionChanged.connect(self._show_trace)
        splitter.addWidget(self.tbl)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setPlaceholderText("选中上方任一行查看完整 Agent trace")
        splitter.addWidget(self.txt)

        splitter.setSizes([400, 400])
        root.addWidget(splitter, stretch=1)

        self.lbl_status = QLabel("等待数据…")
        self.lbl_status.setStyleSheet("color:#888;")
        root.addWidget(self.lbl_status)

    def refresh(self):
        # 不按 run_id 过滤，展示全局最近 N 条决策。
        # 历史 standalone qtrade decide 写的决策 run_id 是 None，按 run 严格过滤会漏掉。
        self._decisions = recent_decisions(run_id=None, limit=100)

        self.tbl.setRowCount(len(self._decisions))
        for r, d in enumerate(self._decisions):
            cells = [
                d["ts"].strftime("%m-%d %H:%M:%S") if d["ts"] else "-",
                d["symbol"],
                d["action"],
                f"{d['confidence']:.2f}",
                f"{d['target_pct']:.0%}",
                d.get("execution_summary") or "-",
                str(d["total_tokens"]),
                f"{d['total_cost_usd']:.4f}",
                (d["reasoning"] or "")[:120],
            ]
            for c, v in enumerate(cells):
                item = QTableWidgetItem(str(v))
                if c == 2:
                    item.setForeground(QColor(_ACTION_COLOR.get(d["action"], "#aaa")))
                elif c == 5:
                    item.setForeground(QColor(_EXECUTION_COLOR.get(d.get("execution_status"), "#aaa")))
                elif c in (3, 4, 6, 7):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl.setItem(r, c, item)

        usage = llm_usage_today()
        if usage:
            self.lbl_status.setText(
                f"共 {len(self._decisions)} 条决策 | 今日 AI "
                f"{usage['decision_count']}/{_limit_text(usage['daily_max_decisions'])} 次 · "
                f"{usage['total_tokens']:,}/{_limit_text(usage['daily_max_tokens'])} token · "
                f"${usage['total_cost_usd']:.4f}/${_limit_text(usage['daily_max_cost_usd'], decimals=4)}"
            )
        else:
            self.lbl_status.setText(f"共 {len(self._decisions)} 条决策")

    def _show_trace(self):
        rows = self.tbl.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if idx >= len(self._decisions):
            return
        d = self._decisions[idx]

        # 兼容 trace_json 两种格式：
        #   旧（Phase 2）：直接是 list[{agent, output}]
        #   新（Phase 4）：{prompt_version, debate_rounds, trace: list[...]}
        raw = d.get("trace") or []
        meta_lines: list[str] = []
        if isinstance(raw, dict):
            if "prompt_version" in raw:
                meta_lines.append(f"Prompt 版本：{raw['prompt_version']}")
            if "debate_rounds" in raw:
                meta_lines.append(f"辩论轮数：{raw['debate_rounds']}")
            entries = raw.get("trace", [])
        else:
            entries = raw

        chunks = [
            f"# 决策 #{d['id']}  {d['symbol']}  {d['action']}  置信度 {d['confidence']:.2f}\n",
            f"理由：{d['reasoning']}\n",
            f"执行：{d.get('execution_summary') or '-'}\n",
            f"总 tokens：{d['total_tokens']}  /  成本：${d['total_cost_usd']:.4f}\n",
        ]
        if d.get("risk_reject_reason"):
            chunks.append(f"拦截原因：{d['risk_reject_reason']}\n")
        if meta_lines:
            chunks.append("\n".join(meta_lines) + "\n")
        chunks.append("\n--- Agent trace ---\n")

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = entry.get("agent", "?")
            if "round" in entry:
                label += f" (round {entry['round']})"
            chunks.append(f"\n## {label}\n")
            chunks.append(json.dumps(entry.get("output"), indent=2, ensure_ascii=False))
        self.txt.setPlainText("".join(chunks))


def _limit_text(value: int | float | None, *, decimals: int = 0) -> str:
    if value is None or value <= 0:
        return "∞"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return f"{value:,}"
