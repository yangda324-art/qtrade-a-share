"""模拟券商测试。"""
from decimal import Decimal

from qtrade.execution.account import Account
from qtrade.execution.broker.paper import PaperBroker
from qtrade.persistence.models import OrderSide, OrderStatus, OrderType


def test_market_buy_fills():
    acct = Account(initial_cash=Decimal("100000"))
    b = PaperBroker(account=acct)
    b.update_market({"000001.SZ": 10.0})

    res = b.submit_order(
        client_order_id="t1",
        symbol="000001.SZ",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
    )
    assert res.status == OrderStatus.FILLED
    assert res.filled_quantity == 100
    pos = acct.positions.get("000001.SZ")
    assert pos.quantity == 100
    assert pos.available == 0          # T+1 锁仓
    assert acct.cash < Decimal("100000")


def test_market_sell_blocked_by_t1():
    """当日买入当日卖出 → broker 自身（available=0）会拒绝。"""
    acct = Account(initial_cash=Decimal("100000"))
    b = PaperBroker(account=acct)
    b.update_market({"000001.SZ": 10.0})
    b.submit_order(
        client_order_id="t1",
        symbol="000001.SZ",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
    )
    res = b.submit_order(
        client_order_id="t2",
        symbol="000001.SZ",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=100,
    )
    assert res.status == OrderStatus.REJECTED


def test_limit_order_rests_then_fills():
    acct = Account(initial_cash=Decimal("100000"))
    b = PaperBroker(account=acct)
    b.update_market({"000001.SZ": 10.0})

    res = b.submit_order(
        client_order_id="lim1",
        symbol="000001.SZ",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        limit_price=Decimal("9.5"),
    )
    assert res.status == OrderStatus.SUBMITTED          # 高于 limit 暂不成交

    # 价格跌到 9.4 → 触发
    b.update_market({"000001.SZ": 9.4})
    pos = acct.positions.get("000001.SZ")
    assert pos.quantity == 100
