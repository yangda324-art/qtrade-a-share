"""SQLAlchemy 模型。

落库原则：
- 订单 / 成交 / 持仓快照 / 资金快照 / Agent 决策都要写库——监管与复盘的基础。
- 所有金额字段用 ``Numeric(18, 4)`` 而不是 ``Float``，避免浮点误差累积。
- 时间统一存 ``DateTime``（UTC naive）—— 展示层再做时区转换。
- 任何"业务关键 ID"（订单号、客户端订单号、决策 ID）建唯一索引；
  幂等性靠 ``client_order_id`` 唯一约束保证：同一信号重试不会下两次单。
"""
from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# -------------------- 基类 + 枚举 --------------------


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class OrderSide(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, enum.Enum):
    PENDING = "PENDING"          # 已提交本地 OMS，未发到券商
    SUBMITTED = "SUBMITTED"      # 已发到券商，等待回报
    PARTIAL = "PARTIAL"          # 部分成交
    FILLED = "FILLED"            # 全部成交
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class DecisionAction(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# -------------------- 业务表 --------------------


class StrategyRun(Base):
    """一次策略运行的元数据。每次回测/模拟/实盘运行起一条。"""

    __tablename__ = "strategy_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_uid: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    strategy_id: Mapped[str] = mapped_column(String(80), index=True)
    strategy_version: Mapped[str] = mapped_column(String(40))     # git sha + cfg hash
    mode: Mapped[str] = mapped_column(String(20))                  # backtest/paper/live
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    orders: Mapped[list["OrderRecord"]] = relationship(back_populates="run")


class OrderRecord(Base):
    """订单（含本地状态 + 券商返回）。"""

    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("client_order_id", name="uq_orders_client_order_id"),
        Index("ix_orders_symbol_time", "symbol", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_runs.id"), nullable=True)
    strategy_id: Mapped[str] = mapped_column(String(80), index=True)
    client_order_id: Mapped[str] = mapped_column(String(64), index=True)  # 幂等键
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[OrderSide] = mapped_column(Enum(OrderSide))
    order_type: Mapped[OrderType] = mapped_column(Enum(OrderType))
    quantity: Mapped[int] = mapped_column(Integer)             # A 股按手 = 100 股，这里直接存股数
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)

    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus), default=OrderStatus.PENDING)
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    avg_fill_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    commission: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal("0"))
    stamp_tax: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal("0"))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_decisions.id"), nullable=True
    )

    run: Mapped[StrategyRun | None] = relationship(back_populates="orders")
    fills: Mapped[list["TradeFill"]] = relationship(back_populates="order")
    decision: Mapped["AgentDecision | None"] = relationship(back_populates="orders")


class TradeFill(Base):
    """成交回报（一笔订单可能对应多笔分段成交）。"""

    __tablename__ = "trade_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    commission: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal("0"))
    stamp_tax: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal("0"))
    filled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    order: Mapped[OrderRecord] = relationship(back_populates="fills")


class PositionSnapshot(Base):
    """持仓快照（按 ``run_id + symbol + ts`` 留痕，便于事后归因）。"""

    __tablename__ = "position_snapshots"
    __table_args__ = (Index("ix_pos_run_symbol_ts", "run_id", "symbol", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_runs.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[int] = mapped_column(Integer)
    available_quantity: Mapped[int] = mapped_column(Integer)   # T+1 锁仓后真正可卖
    avg_cost: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    market_price: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    market_value: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AccountSnapshot(Base):
    """账户级快照：现金 + 总市值 + 累计盈亏。每个交易日收盘后写一条。"""

    __tablename__ = "account_snapshots"
    __table_args__ = (Index("ix_acct_run_ts", "run_id", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_runs.id"), nullable=True)
    cash: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    frozen_cash: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal("0"))
    market_value: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    total_assets: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal("0"))
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=Decimal("0"))
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentDecision(Base):
    """每次多 Agent 决策的完整快照。

    审计是这张表的核心：当事后问"为什么 4 月 15 日 10:32 我买了茅台"，
    必须能查到这一行，并通过 ``trace_json`` 还原所有 Agent 的输入/输出。
    """

    __tablename__ = "agent_decisions"
    __table_args__ = (
        Index("ix_decision_strategy_symbol_ts", "strategy_id", "symbol", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_runs.id"), nullable=True)
    strategy_id: Mapped[str] = mapped_column(String(80))
    symbol: Mapped[str] = mapped_column(String(20))

    action: Mapped[DecisionAction] = mapped_column(Enum(DecisionAction))
    confidence: Mapped[float] = mapped_column(Numeric(6, 4))   # 0~1
    target_position_pct: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    reasoning: Mapped[str] = mapped_column(Text)               # 人类可读结论

    # 完整链路：每个 Agent 的 prompt/output、token 用量、用时
    trace_json: Mapped[dict] = mapped_column(JSON, default=dict)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=Decimal("0"))

    risk_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    risk_reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    orders: Mapped[list[OrderRecord]] = relationship(back_populates="decision")
