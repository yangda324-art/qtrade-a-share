"""持仓和 T+1 测试。"""
from decimal import Decimal

import pytest

from qtrade.execution.position import Position


def test_buy_then_t1_unlock():
    p = Position(symbol="600519.SH")
    p.add_buy(200, Decimal("1700"))
    assert p.quantity == 200
    assert p.available == 0       # 当日不可卖
    assert p.today_buy == 200

    p.settle_t1()
    assert p.available == 200
    assert p.today_buy == 0


def test_avg_cost_weighted():
    p = Position(symbol="600519.SH")
    p.add_buy(100, Decimal("1000"))
    p.add_buy(100, Decimal("1200"))
    assert p.avg_cost == Decimal("1100")


def test_sell_more_than_available_raises():
    p = Position(symbol="600519.SH")
    p.add_buy(100, Decimal("100"))
    with pytest.raises(ValueError):
        p.add_sell(200, Decimal("110"))


def test_realized_pnl_on_sell():
    p = Position(symbol="600519.SH")
    p.add_buy(100, Decimal("100"))
    p.settle_t1()
    p.add_sell(100, Decimal("110"))
    assert p.realized_pnl == Decimal("1000")
    assert p.quantity == 0
    assert p.avg_cost == Decimal("0")
