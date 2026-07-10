"""数据源适配器：把各家 API 的输出统一转为 ``data.schema`` 标准格式。"""
from qtrade.data.adapters.akshare_adapter import AkshareAdapter
from qtrade.data.adapters.base import DataAdapter, Quote
from qtrade.data.adapters.eastmoney_kline_adapter import EastMoneyKlineAdapter
from qtrade.data.adapters.eastmoney_quote_adapter import EastMoneyQuoteAdapter

# Tushare 适配器是可选的（需要 token + tushare 包），延迟 import
try:
    from qtrade.data.adapters.tushare_adapter import TushareAdapter
    _has_tushare = True
except Exception:
    TushareAdapter = None    # type: ignore
    _has_tushare = False

__all__ = [
    "DataAdapter",
    "AkshareAdapter",
    "EastMoneyKlineAdapter",
    "EastMoneyQuoteAdapter",
    "Quote",
    "TushareAdapter",
]
