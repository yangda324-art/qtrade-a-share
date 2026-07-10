"""运行时层：实时盯盘 + 调度 + 状态管理。

设计思路：把"在交易日内持续运行的策略系统"拆成几个可独立测试的组件：

- ``StateManager``   账户/持仓的持久化与恢复（重启不丢状态）
- ``LiveRunner``      tick：拉行情 → 跑策略 → 走风控/OMS → 落库
- ``SchedulerService`` 用 APScheduler 把 tick 绑到 A 股交易日历

UI 通过读 SQLite 拿到最新状态——不与 runner 直接耦合。
"""
from qtrade.runtime.live_runner import LiveRunner
from qtrade.runtime.scheduler import SchedulerService
from qtrade.runtime.state import RunSession, StateManager

__all__ = ["StateManager", "RunSession", "LiveRunner", "SchedulerService"]
