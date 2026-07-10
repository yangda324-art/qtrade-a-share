"""MultiStrategyRunner：多策略并行 + 独立账户。

用途：
- 同时跑 N 个策略，每个策略**独立 Account / Run / PaperBroker**
- 共享行情拉取（quote 只拉一次喂给所有 broker）
- 任一策略报错不影响其他

为什么独立账户：
- 业绩归因要干净——LLM 策略下的单不能"沾光"双均线持仓
- 风控也独立，每个策略各自维护回撤 / 当日亏损上限
- 结果数据（AccountSnapshot / OrderRecord）按 run_id 隔离，UI 可独立展示

整合：
- 调度器只管"该 tick 了"，调用 MultiStrategyRunner.tick() 一次即可
- 告警（Notifier）注入：订单成交、风控否决、错误统一推送
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from qtrade.core.logging import get_logger
from qtrade.core.time import utcnow
from qtrade.data.loader import DataLoader, get_loader
from qtrade.monitoring.notifiers import (
    ConsoleNotifier,
    Notifier,
    NotifyEvent,
)
from qtrade.runtime.live_runner import LiveRunner, TickReport
from qtrade.runtime.state import RunSession

log = get_logger(__name__)


@dataclass
class MultiTickReport:
    ts: datetime
    per_strategy: dict[str, TickReport]
    total_orders_placed: int = 0
    total_risk_rejected: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class MultiStrategyRunner:
    runners: list[LiveRunner]
    loader: DataLoader = field(default_factory=get_loader)
    notifier: Notifier = field(default_factory=ConsoleNotifier)

    _sessions: dict[str, RunSession] = field(default_factory=dict)

    # ---------------- 生命周期 ----------------

    def start(self, *, initial_cash: float = 100_000.0) -> dict[str, RunSession]:
        for r in self.runners:
            session = r.start(initial_cash=initial_cash)
            self._sessions[r.strategy.id] = session
        log.info("MultiStrategyRunner started | strategies={}",
                 [r.strategy.id for r in self.runners])
        self.notifier.send(NotifyEvent.info(
            "qtrade 启动",
            body=f"策略数 = {len(self.runners)}",
            strategies=",".join(r.strategy.id for r in self.runners),
        ))
        return dict(self._sessions)

    def stop(self, note: str | None = None) -> None:
        for r in self.runners:
            try:
                r.stop(note=note)
            except Exception as e:
                log.warning("stop {} failed: {}", r.strategy.id, e)
        self.notifier.send(NotifyEvent.info("qtrade 停止", body=note or ""))

    # ---------------- 三段式 ----------------

    def pre_market(self) -> None:
        for r in self.runners:
            try:
                r.pre_market()
            except Exception as e:
                log.exception("pre_market {} failed", r.strategy.id)
                self.notifier.send(NotifyEvent.error(
                    f"{r.strategy.id} 盘前任务失败", body=str(e),
                ))

    def post_market(self) -> None:
        summary = []
        for r in self.runners:
            try:
                r.post_market()
                if r._session:
                    a = r._session.account
                    summary.append(
                        f"- **{r.strategy.id}**: 总资产 {float(a.total_assets):,.0f} | "
                        f"现金 {float(a.cash):,.0f} | 已实现 {float(a.realized_pnl):+,.2f}"
                    )
            except Exception as e:
                log.exception("post_market {} failed", r.strategy.id)
                summary.append(f"- **{r.strategy.id}**: 失败 {e}")

        self.notifier.send(NotifyEvent.info(
            "盘后日终汇总", body="\n".join(summary) if summary else "无活跃策略",
        ))

    # ---------------- 盘中 tick ----------------

    def tick(self) -> MultiTickReport:
        ts = utcnow()
        report = MultiTickReport(ts=ts, per_strategy={})

        # 1) 收集所有 universe 标的
        all_symbols: set[str] = set()
        for r in self.runners:
            all_symbols.update(r.strategy.universe)

        # 2) 共享拉一次 quotes
        try:
            quotes = self.loader.get_quotes(list(all_symbols))
        except Exception as e:
            log.exception("quote fetch failed: {}", e)
            self.notifier.send(NotifyEvent.warn("行情拉取失败", body=str(e)))
            report.errors.append(f"quotes: {e}")
            quotes = {}

        # 3) 每策略跑一遍 tick，把预拉的 quotes 注入 → 避免重复 fetch
        for r in self.runners:
            try:
                tr = r.tick(quotes=quotes)
                report.per_strategy[r.strategy.id] = tr
                report.total_orders_placed += tr.orders_placed
                report.total_risk_rejected += tr.risk_rejected

                # 订单成交告警（INFO 级，可被静音）
                if tr.orders_placed > 0:
                    self.notifier.send(NotifyEvent.info(
                        f"{r.strategy.id} 下单 {tr.orders_placed} 笔",
                        body=f"信号 {tr.signals_generated} / 订单 {tr.orders_placed} / "
                             f"风控拒 {tr.risk_rejected}",
                    ))
            except Exception as e:
                log.exception("tick {} failed", r.strategy.id)
                report.errors.append(f"{r.strategy.id}: {e}")
                self.notifier.send(NotifyEvent.error(
                    f"{r.strategy.id} tick 异常", body=str(e),
                ))
        return report
