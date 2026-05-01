"""风控引擎测试。"""
from decimal import Decimal

from qtrade.execution.account import Account
from qtrade.risk.engine import RiskEngine
from qtrade.risk.policies import RiskPolicies


def _engine(**override):
    return RiskEngine(policies=RiskPolicies(**override))


def test_max_order_value_blocks_large_orders():
    e = _engine()
    acct = Account(initial_cash=Decimal("1000000"))
    d = e.check_pre_trade(symbol="600519.SH", side="BUY", quantity=100, price=2000.0, account=acct)
    # 单笔 200000 > 默认 50000 上限
    assert not d.passed
    assert d.rule == "max_order_value"


def test_lot_size_must_be_100x():
    e = _engine()
    acct = Account(initial_cash=Decimal("1000000"))
    d = e.check_pre_trade(symbol="600519.SH", side="BUY", quantity=150, price=10.0, account=acct)
    assert not d.passed
    assert d.rule == "lot_size"


def test_min_cash_reserve_blocks_full_position():
    e = _engine()
    acct = Account(initial_cash=Decimal("10000"))
    # 全仓买 → 无现金保留 → 失败
    d = e.check_pre_trade(symbol="000001.SZ", side="BUY", quantity=900, price=10.0, account=acct)
    assert not d.passed
    assert d.rule in {"min_cash_reserve", "max_pct_per_symbol"}


def test_passes_normal_order():
    e = _engine()
    acct = Account(initial_cash=Decimal("100000"))
    d = e.check_pre_trade(symbol="000001.SZ", side="BUY", quantity=200, price=10.0, account=acct)
    assert d.passed, d.detail
