"""Tushare Pro 适配器。

为什么加 Tushare：
- akshare 接口偶尔抖动（前面的 RemoteDisconnected）
- Tushare Pro 数据质量高、接口稳定，但需要 token（注册后免费 120 积分够用，
  日线接口需要 ≥ 2000 积分，可以做些任务积分换）
- 多源 fallback 让系统鲁棒：akshare 失败自动切 tushare

接入：
- 在 .env 设置 ``TUSHARE_TOKEN=xxx``
- DataLoader 启动时自动检测：有 token 就把 TushareAdapter 加入 adapters 链
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from qtrade.core.config import get_settings
from qtrade.core.exceptions import DataError
from qtrade.core.logging import get_logger
from qtrade.data.adapters.base import DataAdapter
from qtrade.data.schema import AdjustMode, normalize_symbol

log = get_logger(__name__)


class TushareAdapter(DataAdapter):
    name = "tushare"

    def __init__(self, token: str | None = None) -> None:
        token = token or get_settings().tushare_token
        if not token:
            raise DataError("Tushare token 未配置（设置 TUSHARE_TOKEN）")
        try:
            import tushare as ts
        except ImportError as e:
            raise DataError("tushare 未安装：pip install tushare") from e
        ts.set_token(token)
        self._pro = ts.pro_api()
        self._ts = ts

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjust: AdjustMode = "qfq",
    ) -> pd.DataFrame:
        sym = normalize_symbol(symbol)
        # tushare 用 ``600519.SH`` 这种格式，正好和我们一致
        ts_code = sym

        try:
            # ``pro_bar`` 同时支持复权和原始；``adj`` = ``qfq``/``hfq``/None
            adj = adjust if adjust in ("qfq", "hfq") else None
            df = self._ts.pro_bar(
                ts_code=ts_code, adj=adj,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                freq="D",
            )
        except Exception as e:
            raise DataError(f"tushare fetch failed for {sym}: {e}") from e

        if df is None or df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]
            )

        # tushare 字段：trade_date / open / high / low / close / vol / amount
        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = df["volume"].astype(float) * 100      # tushare vol 单位是手
        df["amount"] = df["amount"].astype(float) * 1000     # tushare amount 单位是千元
        df["symbol"] = sym
        df["adjust"] = adjust
        return df[["date", "open", "high", "low", "close", "volume", "amount", "symbol", "adjust"]]
