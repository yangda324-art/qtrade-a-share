"""SchedulerService：APScheduler 把 LiveRunner 绑到 A 股交易日历上。

3 类任务：
- ``pre_market``：每个交易日 09:00 跑一次。``LiveRunner.pre_market()``
- ``intraday_tick``：每个交易日 09:30–11:30 / 13:00–15:00，每 ``interval_minutes`` 一次
- ``post_market``：每个交易日 15:30 跑一次。``LiveRunner.post_market()``

每次触发前都用 ``is_trading_day`` 拦一下——APScheduler 的 cron 不认中国节假日，
我们自己挡。

设计：scheduler 不持有任何业务状态；所有状态在 LiveRunner / StateManager 里。
所以"重启进程"是安全的——重启后只要再 start，就接着上次的 run 继续跑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Event

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from qtrade.core.calendar import is_in_trading_session, is_trading_day
from qtrade.core.logging import get_logger
from qtrade.core.time import utcnow
from qtrade.runtime.live_runner import LiveRunner

log = get_logger(__name__)


@dataclass
class SchedulerService:
    runner: LiveRunner
    interval_minutes: int = 5             # 盘中 tick 间隔
    interval_seconds: int | None = None   # 盘中秒级 tick；设置后优先于 interval_minutes
    pre_market_hour: int = 9
    pre_market_minute: int = 0
    post_market_hour: int = 15
    post_market_minute: int = 30

    _scheduler: BlockingScheduler | None = None
    _stop_event: Event = field(default_factory=Event)

    def start(self) -> None:
        """阻塞调用：进入调度循环，直到收到 SIGINT。"""
        sched = BlockingScheduler(timezone="Asia/Shanghai")
        self._scheduler = sched

        sched.add_job(
            self._wrap(self.runner.pre_market, "pre_market"),
            CronTrigger(
                day_of_week="mon-fri",
                hour=self.pre_market_hour,
                minute=self.pre_market_minute,
                timezone="Asia/Shanghai",
            ),
            id="pre_market",
        )
        sched.add_job(
            self._wrap_tick,
            self._tick_trigger(),
            id="intraday_tick",
        )
        sched.add_job(
            self._wrap(self.runner.post_market, "post_market"),
            CronTrigger(
                day_of_week="mon-fri",
                hour=self.post_market_hour,
                minute=self.post_market_minute,
                timezone="Asia/Shanghai",
            ),
            id="post_market",
        )

        log.info(
            "scheduler started | pre={:02d}:{:02d} | tick={} | post={:02d}:{:02d}",
            self.pre_market_hour, self.pre_market_minute,
            self._tick_label(),
            self.post_market_hour, self.post_market_minute,
        )
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("scheduler interrupted, shutting down...")
            self.stop()

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._stop_event.set()

    def _tick_trigger(self) -> CronTrigger:
        if self.interval_seconds is not None:
            if not 1 <= self.interval_seconds <= 59:
                raise ValueError("interval_seconds must be between 1 and 59")
            return CronTrigger(
                day_of_week="mon-fri",
                hour="9-14",
                minute="*",
                second=f"*/{self.interval_seconds}",
                timezone="Asia/Shanghai",
            )
        return CronTrigger(
            day_of_week="mon-fri",
            hour="9-14",
            minute=f"*/{self.interval_minutes}",
            second=0,
            timezone="Asia/Shanghai",
        )

    def _tick_label(self) -> str:
        if self.interval_seconds is not None:
            return f"*/{self.interval_seconds}s"
        return f"*/{self.interval_minutes}min"

    # ---------------- 任务包装：节假日过滤 + 异常隔离 ----------------

    def _wrap(self, fn, name: str):
        def runner():
            now = utcnow()
            if not is_trading_day(now):
                log.debug("[{}] skip non-trading day", name)
                return
            try:
                fn()
                log.info("[{}] done @ {}", name, now.isoformat())
            except Exception as e:
                log.exception("[{}] failed: {}", name, e)
        return runner

    def _wrap_tick(self):
        now = datetime.now()       # APScheduler 给的就是本地时间
        if not is_trading_day(now):
            return
        if not is_in_trading_session(now):
            return
        try:
            report = self.runner.tick()
            if hasattr(report, "per_strategy"):
                log.info(
                    "[tick] @ {} | strategies={} orders={} risk_rejected={} errors={}",
                    now.strftime("%H:%M"),
                    len(report.per_strategy),
                    report.total_orders_placed,
                    report.total_risk_rejected,
                    len(report.errors),
                )
            else:
                log.info(
                    "[tick] @ {} | quotes={} signals={} skipped={} orders={} risk_rejected={} errors={}",
                    now.strftime("%H:%M"), report.quotes_count, report.signals_generated,
                    report.strategy_skipped, report.orders_placed, report.risk_rejected, len(report.errors),
                )
        except Exception as e:
            log.exception("[tick] failed: {}", e)
