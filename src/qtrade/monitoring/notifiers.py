"""告警通知。

设计：
- ``Notifier`` 抽象基类：定义 ``send(event)``
- 三种实现：``ConsoleNotifier``（始终启用）/ ``ServerChanNotifier``（微信）/ ``TelegramNotifier``
- ``MultiNotifier`` 同时往多个通道推
- 所有 notifier 都有"失败不影响主流程"的契约：发送异常被捕获，只 log 不抛

事件分级（``NotifyLevel``）：
- INFO     普通事件（订单已成交）
- WARN     预警（风控触发、数据源失败）
- ERROR    错误（broker 报错、流程异常）
- CRITICAL 紧急（账户级熔断、连续多次失败）

集成示例：
    notifier = build_notifier_from_settings()
    notifier.send(NotifyEvent.info("BUY 000001 ×1000 @ 11.05"))
"""
from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from qtrade.core.logging import get_logger
from qtrade.core.time import utcnow

log = get_logger(__name__)


class NotifyLevel(str, enum.Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class NotifyEvent:
    title: str
    body: str = ""
    level: NotifyLevel = NotifyLevel.INFO
    ts: datetime = field(default_factory=utcnow)
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def info(cls, title: str, body: str = "", **tags) -> NotifyEvent:
        return cls(title=title, body=body, level=NotifyLevel.INFO, tags=tags)

    @classmethod
    def warn(cls, title: str, body: str = "", **tags) -> NotifyEvent:
        return cls(title=title, body=body, level=NotifyLevel.WARN, tags=tags)

    @classmethod
    def error(cls, title: str, body: str = "", **tags) -> NotifyEvent:
        return cls(title=title, body=body, level=NotifyLevel.ERROR, tags=tags)

    @classmethod
    def critical(cls, title: str, body: str = "", **tags) -> NotifyEvent:
        return cls(title=title, body=body, level=NotifyLevel.CRITICAL, tags=tags)

    def render_md(self) -> str:
        """渲染成 Markdown 字符串（ServerChan / Telegram 都支持）。"""
        lines = [f"**[{self.level.value}]** {self.title}"]
        if self.body:
            lines.append("")
            lines.append(self.body)
        if self.tags:
            lines.append("")
            for k, v in self.tags.items():
                lines.append(f"- {k}: `{v}`")
        lines.append(f"\n_时间：{self.ts.strftime('%Y-%m-%d %H:%M:%S')} UTC_")
        return "\n".join(lines)


# -------------------- Notifier 实现 --------------------


class Notifier:
    name: str = "base"

    def send(self, event: NotifyEvent) -> bool:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """打印到日志：始终启用，作为兜底通道。"""

    name = "console"

    def send(self, event: NotifyEvent) -> bool:
        msg = f"[NOTIFY {event.level.value}] {event.title}"
        if event.body:
            msg += f" | {event.body}"
        if event.tags:
            msg += " | " + " ".join(f"{k}={v}" for k, v in event.tags.items())
        if event.level in (NotifyLevel.ERROR, NotifyLevel.CRITICAL):
            log.error(msg)
        elif event.level == NotifyLevel.WARN:
            log.warning(msg)
        else:
            log.info(msg)
        return True


class ServerChanNotifier(Notifier):
    """微信 ServerChan（https://sct.ftqq.com/）。

    用法：申请 SendKey（免费），填到 ``.env`` 的 ``SERVERCHAN_KEY``。
    """

    name = "serverchan"
    BASE_URL = "https://sctapi.ftqq.com/{key}.send"

    def __init__(self, send_key: str) -> None:
        self.send_key = send_key

    def send(self, event: NotifyEvent) -> bool:
        try:
            import httpx
            url = self.BASE_URL.format(key=self.send_key)
            resp = httpx.post(
                url,
                data={"title": f"[{event.level.value}] {event.title[:50]}", "desp": event.render_md()},
                timeout=10.0,
            )
            ok = resp.status_code == 200 and resp.json().get("code", -1) == 0
            if not ok:
                log.warning("serverchan failed: status={} body={}", resp.status_code, resp.text[:200])
            return ok
        except Exception as e:
            log.warning("serverchan exception: {}", e)
            return False


class TelegramNotifier(Notifier):
    """Telegram Bot 推送。

    要先建 bot（@BotFather）拿 token；让 bot 跟你建个对话拿 chat_id。
    """

    name = "telegram"
    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, event: NotifyEvent) -> bool:
        try:
            import httpx
            url = self.API.format(token=self.bot_token)
            resp = httpx.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": event.render_md(),
                    "parse_mode": "Markdown",
                    "disable_notification": event.level == NotifyLevel.INFO,
                },
                timeout=10.0,
            )
            ok = resp.status_code == 200 and resp.json().get("ok") is True
            if not ok:
                log.warning("telegram failed: status={} body={}", resp.status_code, resp.text[:200])
            return ok
        except Exception as e:
            log.warning("telegram exception: {}", e)
            return False


class MultiNotifier(Notifier):
    """组合多个 Notifier，逐个发，互不影响。"""

    name = "multi"

    def __init__(self, notifiers: Iterable[Notifier]) -> None:
        self.notifiers = list(notifiers)

    def send(self, event: NotifyEvent) -> bool:
        results: list[bool] = []
        for notifier in self.notifiers:
            try:
                results.append(notifier.send(event))
            except Exception as e:
                log.warning("{} notifier exception: {}", getattr(notifier, "name", notifier), e)
                results.append(False)
        return any(results)


def build_notifier_from_settings() -> Notifier:
    """根据 settings/.env 自动构建 Notifier。

    始终包含 ConsoleNotifier；其他通道按 key 是否配置启用。
    """
    from qtrade.core.config import get_settings

    s = get_settings()
    notifiers: list[Notifier] = [ConsoleNotifier()]

    if s.monitoring.alerts.enable_serverchan and s.serverchan_key:
        notifiers.append(ServerChanNotifier(s.serverchan_key))

    if s.monitoring.alerts.enable_telegram and s.telegram_bot_token and s.telegram_chat_id:
        notifiers.append(TelegramNotifier(s.telegram_bot_token, s.telegram_chat_id))

    return MultiNotifier(notifiers)
