"""数据层：行情/财报/新闻的统一入口。"""
from qtrade.data.loader import DataLoader, get_loader
from qtrade.data.schema import Bar, NewsItem, normalize_symbol

__all__ = ["Bar", "NewsItem", "DataLoader", "get_loader", "normalize_symbol"]
