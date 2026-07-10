"""Notifier 测试：用 mock httpx 验证 ServerChan / Telegram；console 总能发。"""
from __future__ import annotations

from qtrade.monitoring.notifiers import (
    ConsoleNotifier,
    MultiNotifier,
    NotifyEvent,
    ServerChanNotifier,
    TelegramNotifier,
)


def test_console_always_succeeds():
    n = ConsoleNotifier()
    assert n.send(NotifyEvent.info("hello")) is True
    assert n.send(NotifyEvent.critical("crash", body="full account drawdown")) is True


def test_event_render_md_includes_level_and_tags():
    e = NotifyEvent.warn("risk hit", body="max_order_value", rule="max_order_value", value="50000")
    md = e.render_md()
    assert "[WARN]" in md
    assert "risk hit" in md
    assert "rule" in md and "max_order_value" in md


def test_serverchan_calls_correct_endpoint(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200
        text = '{"code": 0}'
        def json(self):
            return {"code": 0}

    def fake_post(url, data=None, json=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return _FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)

    n = ServerChanNotifier("SCT_TEST_KEY")
    ok = n.send(NotifyEvent.warn("hello", body="body"))
    assert ok is True
    assert "SCT_TEST_KEY" in captured["url"]
    assert "hello" in captured["data"]["title"]


def test_telegram_calls_api(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200
        text = '{"ok": true}'
        def json(self):
            return {"ok": True}

    def fake_post(url, data=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)

    n = TelegramNotifier("BOT_TOKEN_X", "12345")
    assert n.send(NotifyEvent.error("oops", body="something"))
    assert "BOT_TOKEN_X" in captured["url"]
    assert captured["json"]["chat_id"] == "12345"
    assert "oops" in captured["json"]["text"]


def test_multi_notifier_calls_all_even_if_one_fails(monkeypatch):
    calls = []

    class GoodN:
        def send(self, e):
            calls.append("good")
            return True

    class BadN:
        def send(self, e):
            calls.append("bad")
            return False

    multi = MultiNotifier([GoodN(), BadN(), GoodN()])
    assert multi.send(NotifyEvent.info("x")) is True
    assert calls == ["good", "bad", "good"]


def test_multi_notifier_isolates_exceptions():
    calls = []

    class GoodN:
        name = "good"

        def send(self, e):
            calls.append("good")
            return True

    class ExplodingN:
        name = "boom"

        def send(self, e):
            calls.append("boom")
            raise RuntimeError("network broke")

    multi = MultiNotifier([ExplodingN(), GoodN()])
    assert multi.send(NotifyEvent.info("x")) is True
    assert calls == ["boom", "good"]
