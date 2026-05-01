"""风控参数 schema + 加载。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class AccountPolicy(BaseModel):
    max_drawdown_pct: float = 0.15
    daily_loss_limit_pct: float = 0.05
    min_cash_reserve_pct: float = 0.05


class OrderPolicy(BaseModel):
    max_order_value: float = 50_000.0
    max_orders_per_day: int = 50
    min_order_value: float = 1_000.0
    forbid_t0: bool = True


class PositionPolicy(BaseModel):
    max_pct_per_symbol: float = 0.3
    max_symbols: int = 10
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.10
    take_profit_pct: float = 0.25


class MarketPolicy(BaseModel):
    skip_if_volatility_index_above: float | None = None
    skip_first_n_minutes: int = 5


class RiskPolicies(BaseModel):
    account: AccountPolicy = Field(default_factory=AccountPolicy)
    order: OrderPolicy = Field(default_factory=OrderPolicy)
    position: PositionPolicy = Field(default_factory=PositionPolicy)
    blacklist: list[str] = Field(default_factory=list)
    blacklist_match_mode: Literal["name_prefix", "exact", "regex"] = "name_prefix"
    market: MarketPolicy = Field(default_factory=MarketPolicy)

    def is_blacklisted(self, name: str) -> bool:
        if not self.blacklist:
            return False
        for pat in self.blacklist:
            if self.blacklist_match_mode == "exact" and name == pat:
                return True
            if self.blacklist_match_mode == "name_prefix":
                # 支持简单的 ``*ST*`` 通配
                core = pat.replace("*", "")
                if core in name:
                    return True
            if self.blacklist_match_mode == "regex" and re.search(pat, name):
                return True
        return False


def load_risk_policies(path: str | Path = "./config/risk.yaml") -> RiskPolicies:
    p = Path(path)
    if not p.exists():
        return RiskPolicies()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return RiskPolicies(**raw.get("risk", {}))
