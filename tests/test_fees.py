"""费用模型测试。"""
from decimal import Decimal

from qtrade.execution.fees import AStockFeeModel


def test_buy_sh_stock_fees():
    """买入 1000 股茅台 @ 1700：佣金、过户费有，印花税没。"""
    m = AStockFeeModel()
    fb = m.calc(symbol="600519.SH", side="BUY", quantity=1000, price=1700.0)
    notional = Decimal("1700000")  # 170 万
    assert fb.commission == max(
        notional * Decimal("0.00025"),
        Decimal("5"),
    ).quantize(Decimal("0.0001"))
    assert fb.stamp_tax == Decimal("0")
    assert fb.transfer_fee > 0


def test_sell_sz_stock_has_stamp_tax():
    m = AStockFeeModel()
    fb = m.calc(symbol="000001.SZ", side="SELL", quantity=1000, price=10.0)
    notional = Decimal("10000")
    assert fb.stamp_tax == (notional * Decimal("0.0005")).quantize(Decimal("0.0001"))
    assert fb.transfer_fee == Decimal("0")        # 深市无过户费


def test_min_commission():
    """小额买入，最少 5 元佣金。"""
    m = AStockFeeModel()
    fb = m.calc(symbol="000001.SZ", side="BUY", quantity=100, price=10.0)
    assert fb.commission == Decimal("5")
