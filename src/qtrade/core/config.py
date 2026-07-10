"""配置加载。

分层：
1. ``config/default.yaml`` —— 默认参数，进 git。
2. ``config/{env}.yaml`` —— 环境覆盖（可选），进 git。
3. ``.env`` —— 密钥和敏感配置，**不进 git**。
4. 进程环境变量 —— 优先级最高。

用法::

    from qtrade.core.config import get_settings
    settings = get_settings()
    print(settings.llm.deep_model)

约定：所有"路径"字段都会被 ``Path`` 化并自动 ``mkdir(parents=True, exist_ok=True)``。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _alias(*names: str) -> AliasChoices:
    """同时接受多个环境变量名。

    用途：让密钥既能用业内标准名（``DEEPSEEK_API_KEY``）也能用项目前缀名
    （``QTRADE_DEEPSEEK_API_KEY``）。前者更符合各家 SDK 习惯，后者保留向后兼容。
    """
    return AliasChoices(*names)

# ---------------- 嵌套 schema ----------------


class PathsCfg(BaseModel):
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./logs")
    cache_dir: Path = Path("./data/cache")
    db_path: Path = Path("./data/qtrade.sqlite")

    @field_validator("data_dir", "log_dir", "cache_dir", mode="after")
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v = v.expanduser().resolve()
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("db_path", mode="after")
    @classmethod
    def _ensure_parent_dir(cls, v: Path) -> Path:
        v = v.expanduser().resolve()
        v.parent.mkdir(parents=True, exist_ok=True)
        return v


class FeesCfg(BaseModel):
    commission_rate: float = 0.00025
    commission_min: float = 5.0
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001


class MarketCfg(BaseModel):
    exchange: str = "SSE_SZSE"
    timezone: str = "Asia/Shanghai"
    trading_hours: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "morning": ["09:30", "11:30"],
            "afternoon": ["13:00", "15:00"],
        }
    )
    fees: FeesCfg = Field(default_factory=FeesCfg)


class LLMCfg(BaseModel):
    provider: str = "deepseek"
    deep_model: str = "deepseek-reasoner"
    quick_model: str = "deepseek-chat"
    temperature: float = 0.2
    max_retries: int = 3
    request_timeout_s: int = 120
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    daily_max_decisions: int = 50
    daily_max_tokens: int = 200_000
    daily_max_cost_usd: float = 5.0


class AgentsCfg(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    debate_rounds: int = 1
    max_news_items: int = 10


class SchedulerCfg(BaseModel):
    pre_market_time: str = "09:00"
    intraday_check: str = "*/30"
    post_market_time: str = "15:30"


class AlertsCfg(BaseModel):
    enable_telegram: bool = False
    enable_serverchan: bool = False


class MonitoringCfg(BaseModel):
    alerts: AlertsCfg = Field(default_factory=AlertsCfg)
    metrics_retention_days: int = 90


class RealtimeTradingCfg(BaseModel):
    max_quote_age_seconds: int = 15
    max_price_deviation_bps: int = 50


class TradingCfg(BaseModel):
    realtime: RealtimeTradingCfg = Field(default_factory=RealtimeTradingCfg)


class AppCfg(BaseModel):
    name: str = "qtrade"
    env: str = "dev"


# ---------------- 顶层 Settings ----------------


class Settings(BaseSettings):
    """全局设置单例。环境变量以 ``QTRADE_`` 前缀注入。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="QTRADE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ---- 来自 YAML ----
    app: AppCfg = Field(default_factory=AppCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    market: MarketCfg = Field(default_factory=MarketCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    agents: AgentsCfg = Field(default_factory=AgentsCfg)
    scheduler: SchedulerCfg = Field(default_factory=SchedulerCfg)
    monitoring: MonitoringCfg = Field(default_factory=MonitoringCfg)
    trading: TradingCfg = Field(default_factory=TradingCfg)

    # ---- 来自 .env / 环境变量（敏感）----
    # 关键：密钥既支持业内标准名（DEEPSEEK_API_KEY），也支持加 QTRADE_ 前缀的版本。
    # 普通用户在 .env 里写 ``DEEPSEEK_API_KEY=sk-xxx`` 就能用，不需要懂前缀机制。
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias=_alias("DEEPSEEK_API_KEY", "QTRADE_DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=_alias("DEEPSEEK_BASE_URL", "QTRADE_DEEPSEEK_BASE_URL"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=_alias("ANTHROPIC_API_KEY", "QTRADE_ANTHROPIC_API_KEY"),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=_alias("OPENAI_API_KEY", "QTRADE_OPENAI_API_KEY"),
    )
    tushare_token: str | None = Field(
        default=None,
        validation_alias=_alias("TUSHARE_TOKEN", "QTRADE_TUSHARE_TOKEN"),
    )

    gm_token: str | None = Field(
        default=None,
        validation_alias=_alias("GM_TOKEN", "QTRADE_GM_TOKEN"),
    )
    gm_strategy_id: str | None = Field(
        default=None,
        validation_alias=_alias("GM_STRATEGY_ID", "QTRADE_GM_STRATEGY_ID"),
    )
    gm_account_id: str | None = Field(
        default=None,
        validation_alias=_alias("GM_ACCOUNT_ID", "QTRADE_GM_ACCOUNT_ID"),
    )

    telegram_bot_token: str | None = Field(
        default=None,
        validation_alias=_alias("TELEGRAM_BOT_TOKEN", "QTRADE_TELEGRAM_BOT_TOKEN"),
    )
    telegram_chat_id: str | None = Field(
        default=None,
        validation_alias=_alias("TELEGRAM_CHAT_ID", "QTRADE_TELEGRAM_CHAT_ID"),
    )
    serverchan_key: str | None = Field(
        default=None,
        validation_alias=_alias("SERVERCHAN_KEY", "QTRADE_SERVERCHAN_KEY"),
    )

    log_level: str = Field(
        default="INFO",
        validation_alias=_alias("LOG_LEVEL", "QTRADE_LOG_LEVEL"),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """递归 merge：``over`` 覆盖 ``base``。"""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def get_settings(config_dir: str | Path | None = None) -> Settings:
    """加载并缓存 Settings。

    解析顺序：default.yaml → {env}.yaml → .env / 环境变量。
    """
    cfg_dir = Path(config_dir or os.environ.get("QTRADE_CONFIG_DIR", "./config"))
    default = _load_yaml(cfg_dir / "default.yaml")

    env = os.environ.get("QTRADE_ENV", default.get("app", {}).get("env", "dev"))
    env_specific = _load_yaml(cfg_dir / f"{env}.yaml")

    merged = _deep_merge(default, env_specific)
    # YAML 里没有的字段（如密钥）由 BaseSettings 从 .env / env 读取。
    return Settings(**merged)


def reload_settings() -> Settings:
    """清除缓存并重新加载。测试时有用。"""
    get_settings.cache_clear()
    return get_settings()
