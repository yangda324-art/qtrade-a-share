from __future__ import annotations

from sqlalchemy import select

from qtrade.persistence import session_scope
from qtrade.persistence.models import OrderRecord, TickLog
from qtrade.runtime.paper_smoke import run_paper_smoke_check


def test_paper_smoke_exercises_market_limit_and_stale_paths():
    report = run_paper_smoke_check(symbol="000001", isolated=False)

    assert report.overall_status == "PASS"
    assert not report.has_failures
    assert [item.name for item in report.items] == ["市价单链路", "限价单挂单/成交", "过期行情保护"]
    assert all(item.status == "PASS" for item in report.items)

    with session_scope() as s:
        orders = s.scalars(select(OrderRecord)).all()
        ticks = s.scalars(select(TickLog)).all()

    assert len(orders) == 2
    assert len(ticks) == 4
    assert any(tick.per_symbol_json[0].get("skip_reason") == "stale_quote" for tick in ticks)
