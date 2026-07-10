"""模拟盘启动页：从 UI 管理 run-paper 子进程。"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qtrade.data.schema import normalize_symbol
from qtrade.runtime.llm_preflight import LLMPreflightReport, run_llm_paper_preflight


@dataclass(frozen=True)
class PaperRunCommand:
    program: str
    args: list[str]

    def display(self) -> str:
        return " ".join([self.program, *self.args])


def build_run_paper_command(
    *,
    strategy: str,
    symbols: list[str],
    initial_cash: float,
    interval_seconds: int,
    decision_interval_minutes: int,
    once: bool = False,
    prompt_version: str = "v2",
    debate_rounds: int = 2,
    threshold: float | None = None,
    position_pct: float | None = None,
    max_position_pct: float | None = None,
) -> PaperRunCommand:
    normalized = [normalize_symbol(s) for s in symbols if s.strip()]
    args = [
        "-m",
        "qtrade.cli",
        "run-paper",
        "--strategy",
        strategy,
        "--initial-cash",
        f"{initial_cash:.2f}",
        "--interval-seconds",
        str(interval_seconds),
    ]
    for symbol in normalized:
        args.extend(["--symbol", symbol])
    if once:
        args.append("--once")
    if strategy == "llm":
        args.extend(
            [
                "--prompt-version",
                prompt_version,
                "--debate-rounds",
                str(debate_rounds),
                "--decision-interval-minutes",
                str(decision_interval_minutes),
            ]
        )
        if threshold is not None:
            args.extend(["--threshold", f"{threshold:.4f}"])
        if position_pct is not None:
            args.extend(["--position-pct", f"{position_pct:.4f}"])
        if max_position_pct is not None:
            args.extend(["--max-position-pct", f"{max_position_pct:.4f}"])
    return PaperRunCommand(program=sys.executable, args=args)


def parse_symbol_list(text: str) -> list[str]:
    return [part.strip() for part in text.replace("，", ",").replace(" ", ",").split(",") if part.strip()]


class RunControlWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._build()
        self._sync_strategy_controls()
        self._refresh_command_preview()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        root.addWidget(QLabel("<h3>模拟盘启动</h3>"))

        form = QFormLayout()
        self.cbo_strategy = QComboBox()
        self.cbo_strategy.addItems(["ma_cross", "llm"])
        self.cbo_strategy.currentTextChanged.connect(self._sync_strategy_controls)
        self.cbo_strategy.currentTextChanged.connect(self._refresh_command_preview)
        form.addRow("策略", self.cbo_strategy)

        self.le_symbols = QLineEdit("000001")
        self.le_symbols.textChanged.connect(self._refresh_command_preview)
        form.addRow("标的", self.le_symbols)

        self.sp_initial_cash = QDoubleSpinBox()
        self.sp_initial_cash.setRange(1_000, 100_000_000)
        self.sp_initial_cash.setDecimals(2)
        self.sp_initial_cash.setSingleStep(10_000)
        self.sp_initial_cash.setValue(100_000)
        self.sp_initial_cash.valueChanged.connect(self._refresh_command_preview)
        form.addRow("初始资金", self.sp_initial_cash)

        self.sp_interval = QSpinBox()
        self.sp_interval.setRange(1, 59)
        self.sp_interval.setValue(10)
        self.sp_interval.setSuffix(" 秒")
        self.sp_interval.valueChanged.connect(self._refresh_command_preview)
        form.addRow("行情 tick", self.sp_interval)

        self.cb_once = QCheckBox("只跑一次 tick")
        self.cb_once.stateChanged.connect(self._refresh_command_preview)
        form.addRow("调试", self.cb_once)

        self.cbo_prompt = QComboBox()
        self.cbo_prompt.addItems(["v2", "v1"])
        self.cbo_prompt.currentTextChanged.connect(self._refresh_command_preview)
        form.addRow("Prompt", self.cbo_prompt)

        self.sp_debate_rounds = QSpinBox()
        self.sp_debate_rounds.setRange(1, 4)
        self.sp_debate_rounds.setValue(2)
        self.sp_debate_rounds.valueChanged.connect(self._refresh_command_preview)
        form.addRow("辩论轮数", self.sp_debate_rounds)

        self.sp_decision_interval = QSpinBox()
        self.sp_decision_interval.setRange(0, 240)
        self.sp_decision_interval.setValue(15)
        self.sp_decision_interval.setSuffix(" 分钟")
        self.sp_decision_interval.valueChanged.connect(self._refresh_command_preview)
        form.addRow("决策间隔", self.sp_decision_interval)

        self.sp_threshold = QDoubleSpinBox()
        self.sp_threshold.setRange(0, 1)
        self.sp_threshold.setDecimals(2)
        self.sp_threshold.setSingleStep(0.05)
        self.sp_threshold.setValue(0.45)
        self.sp_threshold.valueChanged.connect(self._refresh_command_preview)
        form.addRow("信心阈值", self.sp_threshold)

        self.sp_position_pct = QDoubleSpinBox()
        self.sp_position_pct.setRange(0.01, 1.0)
        self.sp_position_pct.setDecimals(2)
        self.sp_position_pct.setSingleStep(0.05)
        self.sp_position_pct.setValue(0.20)
        self.sp_position_pct.valueChanged.connect(self._refresh_command_preview)
        form.addRow("目标仓位", self.sp_position_pct)

        self.sp_max_position_pct = QDoubleSpinBox()
        self.sp_max_position_pct.setRange(0.01, 1.0)
        self.sp_max_position_pct.setDecimals(2)
        self.sp_max_position_pct.setSingleStep(0.05)
        self.sp_max_position_pct.setValue(0.30)
        self.sp_max_position_pct.valueChanged.connect(self._refresh_command_preview)
        form.addRow("仓位上限", self.sp_max_position_pct)

        root.addLayout(form)

        self.lbl_command = QLabel("命令预览")
        self.lbl_command.setWordWrap(True)
        self.lbl_command.setProperty("secondary", "true")
        root.addWidget(self.lbl_command)

        buttons = QHBoxLayout()
        self.btn_llm_check = QPushButton("LLM 预检")
        self.btn_llm_check.setProperty("secondary", "true")
        self.btn_llm_check.clicked.connect(self.run_llm_preflight)
        buttons.addWidget(self.btn_llm_check)

        self.btn_start = QPushButton("启动模拟盘")
        self.btn_start.clicked.connect(self.start_run)
        buttons.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setProperty("secondary", "true")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_run)
        buttons.addWidget(self.btn_stop)
        buttons.addStretch()
        root.addLayout(buttons)

        self.lbl_status = QLabel("未启动")
        self.lbl_status.setProperty("secondary", "true")
        root.addWidget(self.lbl_status)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("模拟盘输出会显示在这里")
        root.addWidget(self.txt_log, stretch=1)

    def _sync_strategy_controls(self) -> None:
        is_llm = self.cbo_strategy.currentText() == "llm"
        for widget in (
            self.cbo_prompt,
            self.sp_debate_rounds,
            self.sp_decision_interval,
            self.sp_threshold,
            self.sp_position_pct,
            self.sp_max_position_pct,
        ):
            widget.setEnabled(is_llm)
        self.btn_llm_check.setEnabled(is_llm)
        self._refresh_command_preview()

    def _build_command(self) -> PaperRunCommand:
        symbols = parse_symbol_list(self.le_symbols.text())
        return build_run_paper_command(
            strategy=self.cbo_strategy.currentText(),
            symbols=symbols,
            initial_cash=float(self.sp_initial_cash.value()),
            interval_seconds=int(self.sp_interval.value()),
            decision_interval_minutes=int(self.sp_decision_interval.value()),
            once=self.cb_once.isChecked(),
            prompt_version=self.cbo_prompt.currentText(),
            debate_rounds=int(self.sp_debate_rounds.value()),
            threshold=float(self.sp_threshold.value()),
            position_pct=float(self.sp_position_pct.value()),
            max_position_pct=float(self.sp_max_position_pct.value()),
        )

    def _refresh_command_preview(self) -> None:
        try:
            cmd = self._build_command()
            self.lbl_command.setText(f"命令：{cmd.display()}")
        except Exception as exc:
            self.lbl_command.setText(f"命令参数错误：{exc}")

    def start_run(self) -> None:
        if self._process is not None:
            return
        try:
            cmd = self._build_command()
        except Exception as exc:
            self.lbl_status.setText(f"参数错误：{exc}")
            return

        if self.cbo_strategy.currentText() == "llm":
            preflight = self._build_llm_preflight()
            self._append_llm_preflight(preflight)
            if preflight.has_failures:
                self.lbl_status.setText("LLM 预检失败；已拦截启动")
                return

        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(Path.cwd()))
        proc.setProgram(cmd.program)
        proc.setArguments(cmd.args)
        proc.readyReadStandardOutput.connect(self._append_stdout)
        proc.readyReadStandardError.connect(self._append_stderr)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        self.txt_log.clear()
        self.txt_log.appendPlainText(f"$ {cmd.display()}\n")
        proc.start()
        if not proc.waitForStarted(3000):
            self.lbl_status.setText("启动失败：子进程未启动")
            return

        self._process = proc
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("模拟盘运行中")

    def stop_run(self) -> None:
        if self._process is None:
            return
        self.lbl_status.setText("正在停止模拟盘…")
        self._process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._process.kill()

    def _append_stdout(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.txt_log.appendPlainText(text.rstrip())

    def _append_stderr(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self.txt_log.appendPlainText(text.rstrip())

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        self.txt_log.appendPlainText(f"\n[process exited: {exit_code}]")
        self._process = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText(f"已停止，退出码 {exit_code}")

    def _on_error(self, error) -> None:
        self.lbl_status.setText(f"子进程错误：{error}")

    def run_llm_preflight(self) -> None:
        if self.cbo_strategy.currentText() != "llm":
            self.lbl_status.setText("LLM 预检只适用于 LLM 策略")
            return
        report = self._build_llm_preflight()
        self._append_llm_preflight(report)
        if report.has_failures:
            self.lbl_status.setText("LLM 预检失败")
        elif report.warnings:
            self.lbl_status.setText("LLM 预检通过，但存在警告")
        else:
            self.lbl_status.setText("LLM 预检通过")

    def _build_llm_preflight(self) -> LLMPreflightReport:
        return run_llm_paper_preflight(
            symbols=parse_symbol_list(self.le_symbols.text()),
            prompt_version=self.cbo_prompt.currentText(),
            debate_rounds=int(self.sp_debate_rounds.value()),
            threshold=float(self.sp_threshold.value()),
            decision_interval_minutes=int(self.sp_decision_interval.value()),
            position_pct=float(self.sp_position_pct.value()),
            max_position_pct=float(self.sp_max_position_pct.value()),
        )

    def _append_llm_preflight(self, report: LLMPreflightReport) -> None:
        lines = [
            "",
            f"[LLM preflight] overall={report.overall_status} "
            f"PASS={len(report.passed)} WARN={len(report.warnings)} FAIL={len(report.failures)}",
        ]
        for item in report.items:
            lines.append(f"- {item.status} | {item.name} | {item.detail}")
        self.txt_log.appendPlainText("\n".join(lines))
