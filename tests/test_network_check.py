"""Network diagnostic tests."""
from __future__ import annotations

import requests

from qtrade.data.network_check import run_network_check


class _Resp:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_network_check_passes_quote_and_kline(monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    session = _Session(
        [
            _Resp({"data": {"diff": [{"f12": "000001"}]}}),
            _Resp({"data": {"klines": ["2026-06-24,10,11,12,9,1,100"]}}),
        ]
    )

    report = run_network_check(symbol="000001", periods=["daily"], timeout_s=3, session=session)

    assert report.overall_status == "PASS"
    assert report.items[0].name == "代理环境"
    assert report.items[0].status == "PASS"
    assert report.items[2].name == "EastMoney 快照"
    assert report.items[2].status == "PASS"
    assert report.items[3].name == "EastMoney K线 daily"
    assert report.items[3].status == "PASS"
    assert session.calls[0]["params"]["secids"] == "0.000001"
    assert session.calls[1]["params"]["klt"] == "101"
    assert session.calls[1]["timeout"] == 3


def test_network_check_classifies_proxy_error(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    session = _Session(
        [
            requests.exceptions.ProxyError("proxy down"),
            _Resp({"data": {"klines": ["2026-06-24,10,11,12,9,1,100"]}}),
        ]
    )

    report = run_network_check(symbol="000001", periods=["daily"], session=session)

    assert report.overall_status == "FAIL"
    assert report.items[0].status == "WARN"
    assert report.items[2].category == "PROXY_ERROR"
    assert "代理连接失败" in report.items[2].detail


def test_network_check_classifies_empty_and_json_error(monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    session = _Session(
        [
            _Resp({"data": {"diff": []}}),
            _Resp(ValueError("not json")),
        ]
    )

    report = run_network_check(symbol="000001", periods=["daily"], session=session)

    assert report.overall_status == "FAIL"
    assert report.items[2].category == "EMPTY_RESPONSE"
    assert report.items[3].category == "JSON_ERROR"
