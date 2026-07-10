"""UI 数据查询层。

UI 不直接写 ORM 查询；这里提供"最新账户快照"、"今日订单"等高层函数，
让 UI 控件只关心展示。

设计：
- 所有查询都返回 ``dict`` / ``list[dict]``（不返回 ORM 对象），避免跨 session 访问 detached 实例
- 没有数据返回空 dict / 空 list，而不是 None
- 不抛异常：DB 错误就返回空集 + log warning（UI 不应因为读不到数据就崩）
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median

from sqlalchemy import desc, select

from qtrade.core.logging import get_logger
from qtrade.core.time import utc_naive_to_market_time, utcnow
from qtrade.persistence import session_scope
from qtrade.persistence.models import (
    AccountSnapshot,
    AgentDecision,
    OrderRecord,
    PositionSnapshot,
    StrategyRun,
    TickLog,
)

log = get_logger(__name__)

_OPEN_ORDER_STATUSES = {"PENDING", "SUBMITTED", "PARTIAL"}
_DEFAULT_TICK_STALE_SECONDS = 600.0


@dataclass
class _Lot:
    quantity: int
    cost_per_share: float


def latest_active_run() -> dict:
    """最近的活跃（未结束）运行；没有则返回最近一次结束的 run。

    ``demo_seed`` 只用于 UI 演示数据，不能长期遮挡真实 paper/live run。
    """
    try:
        with session_scope() as s:
            run = s.scalars(
                select(StrategyRun)
                .where(StrategyRun.ended_at.is_(None), StrategyRun.strategy_id != "demo_seed")
                .order_by(desc(StrategyRun.started_at))
            ).first()
            if not run:
                run = s.scalars(
                    select(StrategyRun).order_by(desc(StrategyRun.started_at))
                ).first()
            if not run:
                return {}
            return {
                "run_id": run.id,
                "run_uid": run.run_uid,
                "strategy_id": run.strategy_id,
                "mode": run.mode,
                "started_at": utc_naive_to_market_time(run.started_at),
                "ended_at": utc_naive_to_market_time(run.ended_at),
            }
    except Exception as e:
        log.warning("latest_active_run query failed: {}", e)
        return {}


def all_active_runs(*, modes: list[str] | None = None) -> list[dict]:
    """所有未结束的 run；用于多策略对比视图。"""
    try:
        with session_scope() as s:
            stmt = select(StrategyRun).where(StrategyRun.ended_at.is_(None))
            if modes:
                stmt = stmt.where(StrategyRun.mode.in_(modes))
            stmt = stmt.order_by(StrategyRun.started_at.desc())
            runs = s.scalars(stmt).all()
            return [
                {
                    "run_id": r.id,
                    "run_uid": r.run_uid,
                    "strategy_id": r.strategy_id,
                    "mode": r.mode,
                    "started_at": utc_naive_to_market_time(r.started_at),
                    "ended_at": utc_naive_to_market_time(r.ended_at),
                }
                for r in runs
            ]
    except Exception as e:
        log.warning("all_active_runs failed: {}", e)
        return []


def recent_runs(*, hours: int = 24, modes: list[str] | None = None) -> list[dict]:
    """最近 N 小时内启动的所有 run（含已结束），按启动时间倒序。

    默认 24h 涵盖：活跃的 + 今天测试过但已 stop 的，方便事后看 --once 跑过的对比。
    """
    from datetime import timedelta as _td
    try:
        with session_scope() as s:
            since = utcnow() - _td(hours=hours)
            stmt = select(StrategyRun).where(StrategyRun.started_at >= since)
            if modes:
                stmt = stmt.where(StrategyRun.mode.in_(modes))
            stmt = stmt.order_by(StrategyRun.started_at.desc())
            runs = s.scalars(stmt).all()
            return [
                {
                    "run_id": r.id,
                    "run_uid": r.run_uid,
                    "strategy_id": r.strategy_id,
                    "mode": r.mode,
                    "started_at": utc_naive_to_market_time(r.started_at),
                    "ended_at": utc_naive_to_market_time(r.ended_at),
                }
                for r in runs
            ]
    except Exception as e:
        log.warning("recent_runs failed: {}", e)
        return []


def run_summary(run_id: int) -> dict:
    """汇总一个 run 的关键指标：总资产、累计收益、Sharpe、当前持仓数。

    简单近似：用所有 AccountSnapshot 算 Sharpe（假设相邻 snapshot 间隔大致相等）。
    """
    snap = latest_account_snapshot(run_id)
    if not snap:
        return {}

    initial = _initial_assets(run_id)
    total_assets = snap["total_assets"]
    cum_return = (total_assets / initial - 1) if initial > 0 else 0.0

    # 算 Sharpe：拿权益曲线
    curve = equity_curve(run_id, limit=2000)
    if len(curve) >= 5:
        import numpy as np
        vals = [c["total_assets"] for c in curve]
        rets = np.diff(vals) / np.array(vals[:-1])
        mean = float(np.mean(rets)) if len(rets) > 0 else 0.0
        std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        # 简化年化：假设 snapshot 间隔等同，简单乘 252（日级）
        sharpe = (mean - 0.02 / 252) / std * (252 ** 0.5) if std > 1e-12 else 0.0
    else:
        sharpe = 0.0

    positions = latest_positions(run_id)
    return {
        "total_assets": total_assets,
        "initial_assets": initial,
        "cum_return": cum_return,
        "sharpe": sharpe,
        "n_positions": len(positions),
        "cash": snap["cash"],
        "realized_pnl": snap["realized_pnl"],
        "unrealized_pnl": snap["unrealized_pnl"],
    }


def performance_runs(*, limit: int = 50, modes: list[str] | None = None) -> list[dict]:
    """可生成绩效报告的 run 列表，默认覆盖 paper run。"""
    try:
        with session_scope() as s:
            stmt = select(StrategyRun).order_by(desc(StrategyRun.started_at)).limit(limit)
            stmt = stmt.where(StrategyRun.strategy_id != "demo_seed")
            if modes is None:
                stmt = stmt.where(StrategyRun.mode.in_(["paper", "live", "backtest"]))
            elif modes:
                stmt = stmt.where(StrategyRun.mode.in_(modes))
            rows = s.scalars(stmt).all()
            return [
                {
                    "run_id": r.id,
                    "run_uid": r.run_uid,
                    "strategy_id": r.strategy_id,
                    "mode": r.mode,
                    "started_at": utc_naive_to_market_time(r.started_at),
                    "ended_at": utc_naive_to_market_time(r.ended_at),
                }
                for r in rows
            ]
    except Exception as e:
        log.warning("performance_runs failed: {}", e)
        return []


def run_performance_report(run_id: int | None = None, *, equity_limit: int = 5000) -> dict:
    """按 run 汇总模拟盘绩效。

    口径：
    - 权益、收益、回撤、暴露来自 ``AccountSnapshot``。
    - 成交统计来自已成交 ``OrderRecord``。
    - 单笔盈亏用同一 run 内 BUY/SELL 成交 FIFO 估算，适合模拟盘复盘；真实券商接入后应以 broker
      成交回报和结算数据为准。
    """
    try:
        with session_scope() as s:
            run = _load_run_for_health(s, run_id)
            if not run:
                return {}

            snapshots_desc = s.scalars(
                select(AccountSnapshot)
                .where(AccountSnapshot.run_id == run.id)
                .order_by(desc(AccountSnapshot.ts))
                .limit(equity_limit)
            ).all()
            snapshots = list(reversed(snapshots_desc))
            orders = s.scalars(
                select(OrderRecord)
                .where(OrderRecord.run_id == run.id)
                .order_by(OrderRecord.created_at.asc())
            ).all()
            filled_orders = [o for o in orders if o.status.value == "FILLED" and int(o.filled_quantity or 0) > 0]
            order_counts = Counter(o.status.value for o in orders)
            trade_stats = _fifo_trade_stats(filled_orders)
            equity_stats = _equity_stats(snapshots)

            total_traded_value = sum(_order_notional(o) for o in filled_orders)
            avg_equity = equity_stats["avg_equity"]
            turnover = total_traded_value / avg_equity if avg_equity > 0 else 0.0

            latest = snapshots[-1] if snapshots else None
            return {
                "run": {
                    "run_id": run.id,
                    "run_uid": run.run_uid,
                    "strategy_id": run.strategy_id,
                    "mode": run.mode,
                    "started_at": utc_naive_to_market_time(run.started_at),
                    "ended_at": utc_naive_to_market_time(run.ended_at),
                },
                "equity_curve": [
                    {
                        "ts": utc_naive_to_market_time(snap.ts),
                        "total_assets": float(snap.total_assets),
                        "cash": float(snap.cash),
                        "market_value": float(snap.market_value),
                    }
                    for snap in snapshots
                ],
                "metrics": {
                    **equity_stats,
                    "cash": float(latest.cash) if latest else 0.0,
                    "market_value": float(latest.market_value) if latest else 0.0,
                    "realized_pnl": float(latest.realized_pnl) if latest else 0.0,
                    "unrealized_pnl": float(latest.unrealized_pnl) if latest else 0.0,
                    "filled_order_count": len(filled_orders),
                    "buy_order_count": sum(1 for o in filled_orders if o.side.value == "BUY"),
                    "sell_order_count": sum(1 for o in filled_orders if o.side.value == "SELL"),
                    "open_order_count": sum(order_counts.get(k, 0) for k in _OPEN_ORDER_STATUSES),
                    "rejected_order_count": order_counts.get("REJECTED", 0),
                    "total_traded_value": total_traded_value,
                    "turnover": turnover,
                    **trade_stats,
                },
                "orders": [_order_display_dict(o) for o in reversed(orders[-200:])],
                "trade_pnls": trade_stats["trade_pnls"],
            }
    except Exception as e:
        log.warning("run_performance_report failed: {}", e)
        return {}


def _equity_stats(snapshots: list[AccountSnapshot]) -> dict:
    if not snapshots:
        return {
            "initial_assets": 0.0,
            "final_assets": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "peak_assets": 0.0,
            "avg_equity": 0.0,
            "avg_exposure": 0.0,
            "snapshot_count": 0,
            "first_ts": None,
            "last_ts": None,
        }

    values = [float(s.total_assets) for s in snapshots]
    market_values = [float(s.market_value) for s in snapshots]
    initial = values[0]
    final = values[-1]
    peak = max(values)
    total_return = final / initial - 1 if initial > 0 else 0.0

    running_peak = values[0] if values else 0.0
    max_drawdown = 0.0
    for value in values:
        running_peak = max(running_peak, value)
        drawdown = value / running_peak - 1 if running_peak > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
    current_drawdown = final / peak - 1 if peak > 0 else 0.0
    avg_equity = sum(values) / len(values) if values else 0.0
    exposures = [
        market_value / total_assets
        for market_value, total_assets in zip(market_values, values, strict=False)
        if total_assets > 0
    ]
    avg_exposure = sum(exposures) / len(exposures) if exposures else 0.0
    return {
        "initial_assets": initial,
        "final_assets": final,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "current_drawdown": current_drawdown,
        "peak_assets": peak,
        "avg_equity": avg_equity,
        "avg_exposure": avg_exposure,
        "snapshot_count": len(snapshots),
        "first_ts": utc_naive_to_market_time(snapshots[0].ts),
        "last_ts": utc_naive_to_market_time(snapshots[-1].ts),
    }


def _fifo_trade_stats(orders: list[OrderRecord]) -> dict:
    lots_by_symbol: dict[str, list[_Lot]] = {}
    sell_trade_pnls: list[float] = []

    for order in orders:
        qty = int(order.filled_quantity or order.quantity or 0)
        price = float(order.avg_fill_price or 0)
        if qty <= 0 or price <= 0:
            continue
        fees = float(order.commission or 0) + float(order.stamp_tax or 0)
        if order.side.value == "BUY":
            cost_per_share = price + fees / qty
            lots_by_symbol.setdefault(order.symbol, []).append(_Lot(qty, cost_per_share))
            continue

        proceeds_per_share = price - fees / qty
        remaining = qty
        pnl = 0.0
        lots = lots_by_symbol.setdefault(order.symbol, [])
        while remaining > 0 and lots:
            lot = lots[0]
            matched = min(remaining, lot.quantity)
            pnl += (proceeds_per_share - lot.cost_per_share) * matched
            lot.quantity -= matched
            remaining -= matched
            if lot.quantity <= 0:
                lots.pop(0)
        if qty != remaining:
            sell_trade_pnls.append(pnl)

    wins = [p for p in sell_trade_pnls if p > 0]
    losses = [p for p in sell_trade_pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "completed_trade_count": len(sell_trade_pnls),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": len(wins) / len(sell_trade_pnls) if sell_trade_pnls else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": abs(avg_win / avg_loss) if avg_loss else 0.0,
        "best_trade_pnl": max(sell_trade_pnls) if sell_trade_pnls else 0.0,
        "worst_trade_pnl": min(sell_trade_pnls) if sell_trade_pnls else 0.0,
        "total_trade_pnl": sum(sell_trade_pnls),
        "trade_pnls": sell_trade_pnls,
    }


def _order_notional(order: OrderRecord) -> float:
    qty = int(order.filled_quantity or order.quantity or 0)
    price = float(order.avg_fill_price or order.limit_price or 0)
    return abs(qty * price)


def _order_display_dict(order: OrderRecord) -> dict:
    return {
        "id": order.id,
        "ts": utc_naive_to_market_time(order.created_at),
        "symbol": order.symbol,
        "side": order.side.value,
        "order_type": order.order_type.value,
        "quantity": order.quantity,
        "filled_quantity": order.filled_quantity,
        "avg_fill_price": float(order.avg_fill_price) if order.avg_fill_price else None,
        "limit_price": float(order.limit_price) if order.limit_price else None,
        "status": order.status.value,
        "commission": float(order.commission),
        "stamp_tax": float(order.stamp_tax),
        "reject_reason": order.reject_reason,
    }


def _initial_assets(run_id: int) -> float:
    """run 起始时刻的总资产（第一条 snapshot 或 100k 兜底）。"""
    try:
        with session_scope() as s:
            first = s.scalars(
                select(AccountSnapshot)
                .where(AccountSnapshot.run_id == run_id)
                .order_by(AccountSnapshot.ts.asc())
            ).first()
            return float(first.total_assets) if first else 100_000.0
    except Exception:
        return 100_000.0


def latest_account_snapshot(run_id: int) -> dict:
    try:
        with session_scope() as s:
            snap = s.scalars(
                select(AccountSnapshot)
                .where(AccountSnapshot.run_id == run_id)
                .order_by(desc(AccountSnapshot.ts))
            ).first()
            if not snap:
                return {}
            return {
                "ts": utc_naive_to_market_time(snap.ts),
                "cash": float(snap.cash),
                "frozen_cash": float(snap.frozen_cash),
                "market_value": float(snap.market_value),
                "total_assets": float(snap.total_assets),
                "realized_pnl": float(snap.realized_pnl),
                "unrealized_pnl": float(snap.unrealized_pnl),
            }
    except Exception as e:
        log.warning("latest_account_snapshot failed: {}", e)
        return {}


def latest_positions(run_id: int) -> list[dict]:
    """每个 symbol 的最新持仓快照（quantity > 0）。"""
    try:
        with session_scope() as s:
            symbols = [
                row[0] for row in s.execute(
                    select(PositionSnapshot.symbol)
                    .where(PositionSnapshot.run_id == run_id)
                    .distinct()
                )
            ]
            out: list[dict] = []
            for sym in symbols:
                p = s.scalars(
                    select(PositionSnapshot)
                    .where(PositionSnapshot.run_id == run_id, PositionSnapshot.symbol == sym)
                    .order_by(desc(PositionSnapshot.ts))
                ).first()
                if not p or p.quantity <= 0:
                    continue
                out.append({
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "available": p.available_quantity,
                    "avg_cost": float(p.avg_cost),
                    "market_price": float(p.market_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pnl),
                    "ts": utc_naive_to_market_time(p.ts),
                })
            return out
    except Exception as e:
        log.warning("latest_positions failed: {}", e)
        return []


def equity_curve(run_id: int, *, limit: int = 500) -> list[dict]:
    """按时间顺序的权益曲线点（最近 N 个）。"""
    try:
        with session_scope() as s:
            snaps = s.scalars(
                select(AccountSnapshot)
                .where(AccountSnapshot.run_id == run_id)
                .order_by(desc(AccountSnapshot.ts))
                .limit(limit)
            ).all()
            return [
                {"ts": utc_naive_to_market_time(x.ts), "total_assets": float(x.total_assets)}
                for x in reversed(snaps)
            ]
    except Exception as e:
        log.warning("equity_curve failed: {}", e)
        return []


def recent_orders(run_id: int | None = None, *, limit: int = 100) -> list[dict]:
    try:
        with session_scope() as s:
            stmt = select(OrderRecord).order_by(desc(OrderRecord.created_at)).limit(limit)
            if run_id is not None:
                stmt = stmt.where(OrderRecord.run_id == run_id)
            orders = s.scalars(stmt).all()
            return [
                {
                    "id": o.id,
                    "ts": utc_naive_to_market_time(o.created_at),
                    "client_order_id": o.client_order_id,
                    "broker_order_id": o.broker_order_id,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "order_type": o.order_type.value,
                    "quantity": o.quantity,
                    "filled_quantity": o.filled_quantity,
                    "avg_fill_price": float(o.avg_fill_price) if o.avg_fill_price else None,
                    "limit_price": float(o.limit_price) if o.limit_price else None,
                    "status": o.status.value,
                    "commission": float(o.commission),
                    "stamp_tax": float(o.stamp_tax),
                    "reject_reason": o.reject_reason,
                }
                for o in orders
            ]
    except Exception as e:
        log.warning("recent_orders failed: {}", e)
        return []


def recent_decisions(run_id: int | None = None, *, limit: int = 50) -> list[dict]:
    try:
        with session_scope() as s:
            stmt = select(AgentDecision).order_by(desc(AgentDecision.ts)).limit(limit)
            if run_id is not None:
                stmt = stmt.where(AgentDecision.run_id == run_id)
            rows = s.scalars(stmt).all()
            orders_by_decision: dict[int, list[OrderRecord]] = {}
            decision_ids = [d.id for d in rows]
            if decision_ids:
                order_rows = s.scalars(
                    select(OrderRecord)
                    .where(OrderRecord.decision_id.in_(decision_ids))
                    .order_by(desc(OrderRecord.created_at))
                ).all()
                for order in order_rows:
                    if order.decision_id is not None:
                        orders_by_decision.setdefault(order.decision_id, []).append(order)
            return [
                {
                    "id": d.id,
                    "ts": utc_naive_to_market_time(d.ts),
                    "strategy_id": d.strategy_id,
                    "symbol": d.symbol,
                    "action": d.action.value,
                    "confidence": float(d.confidence),
                    "target_pct": float(d.target_position_pct or 0),
                    "reasoning": d.reasoning,
                    "total_tokens": d.total_tokens,
                    "total_cost_usd": float(d.total_cost_usd),
                    "risk_passed": bool(d.risk_passed),
                    "risk_reject_reason": d.risk_reject_reason,
                    **_decision_execution_summary(d, orders_by_decision.get(d.id, [])),
                    "trace": d.trace_json,
                }
                for d in rows
            ]
    except Exception as e:
        log.warning("recent_decisions failed: {}", e)
        return []


def llm_usage_today() -> dict:
    """今日 LLM 决策预算使用情况。"""
    try:
        from qtrade.core.config import get_settings
        from qtrade.llm.budget import daily_usage

        settings = get_settings()
        usage = daily_usage()
        return {
            "day": usage.day,
            "decision_count": usage.decision_count,
            "total_tokens": usage.total_tokens,
            "total_cost_usd": usage.total_cost_usd,
            "daily_max_decisions": settings.llm.daily_max_decisions,
            "daily_max_tokens": settings.llm.daily_max_tokens,
            "daily_max_cost_usd": settings.llm.daily_max_cost_usd,
        }
    except Exception as e:
        log.warning("llm_usage_today failed: {}", e)
        return {}


def _decision_execution_summary(decision: AgentDecision, orders: list[OrderRecord]) -> dict:
    if orders:
        filled_qty = sum(int(o.filled_quantity or 0) for o in orders)
        latest = orders[0]
        statuses = {o.status.value for o in orders}
        if filled_qty > 0:
            return {
                "execution_status": "FILLED",
                "execution_summary": f"已成交 {filled_qty} 股",
                "order_count": len(orders),
                "latest_order_status": latest.status.value,
            }
        if "SUBMITTED" in statuses or "PENDING" in statuses or "PARTIAL" in statuses:
            qty = sum(int(o.quantity or 0) for o in orders)
            return {
                "execution_status": "SUBMITTED",
                "execution_summary": f"已挂单 {qty} 股",
                "order_count": len(orders),
                "latest_order_status": latest.status.value,
            }
        return {
            "execution_status": latest.status.value,
            "execution_summary": latest.reject_reason or latest.status.value,
            "order_count": len(orders),
            "latest_order_status": latest.status.value,
        }

    if not bool(decision.risk_passed):
        reason = decision.risk_reject_reason or "blocked"
        return {
            "execution_status": "BLOCKED",
            "execution_summary": _compact_reason(f"已拦截：{reason}", limit=80),
            "order_count": 0,
            "latest_order_status": None,
        }

    return {
        "execution_status": "NO_ORDER",
        "execution_summary": "未触发订单",
        "order_count": 0,
        "latest_order_status": None,
    }


def recent_tick_logs(run_id: int | None = None, *, limit: int = 100) -> list[dict]:
    """最近 tick 日志，用于 UI 观察模拟盘是否稳定运行。"""
    try:
        with session_scope() as s:
            stmt = select(TickLog).order_by(desc(TickLog.ts)).limit(limit)
            if run_id is not None:
                stmt = stmt.where(TickLog.run_id == run_id)
            rows = s.scalars(stmt).all()
            out = []
            for row in rows:
                per_symbol = row.per_symbol_json or []
                errors = row.errors_json or []
                out.append(
                    {
                        "id": row.id,
                        "run_id": row.run_id,
                        "strategy_id": row.strategy_id,
                        "ts": utc_naive_to_market_time(row.ts),
                        "quotes_count": row.quotes_count,
                        "tradeable_quotes": row.tradeable_quotes,
                        "fallback_quotes": row.fallback_quotes,
                        "signals_generated": row.signals_generated,
                        "strategy_skipped": row.strategy_skipped,
                        "orders_placed": row.orders_placed,
                        "risk_rejected": row.risk_rejected,
                        "limit_orders_submitted": row.limit_orders_submitted,
                        "limit_orders_matched": row.limit_orders_matched,
                        "symbols": row.symbols_json or [],
                        "per_symbol": per_symbol,
                        "errors": errors,
                        "latest_reason": _latest_tick_reason(per_symbol, errors),
                    }
                )
            return out
    except Exception as e:
        log.warning("recent_tick_logs failed: {}", e)
        return []


def _latest_tick_reason(per_symbol: list, errors: list) -> str:
    """提取 tick 表格用的一句话原因，完整细节仍保留在详情 JSON。"""
    for item in per_symbol:
        if not isinstance(item, dict):
            continue
        for issue_key in ("order_rejections", "order_skips"):
            for issue in item.get(issue_key) or []:
                reason = issue.get("reason") if isinstance(issue, dict) else str(issue)
                if reason:
                    return _compact_reason(str(reason))
        reason = item.get("skip_reason")
        if reason:
            return _compact_reason(str(reason))

    if errors:
        return _compact_reason(str(errors[0]))
    return "-"


def _compact_reason(reason: str, *, limit: int = 72) -> str:
    reason = " ".join(reason.split())
    if len(reason) <= limit:
        return reason
    return reason[: limit - 1] + "…"


def paper_run_health(
    run_id: int | None = None,
    *,
    tick_limit: int = 20,
    order_limit: int = 50,
) -> dict:
    """模拟盘健康摘要。

    只读 SQLite 已落库的 run / tick / order / snapshot，不拉实时行情，也不写入数据。
    Dashboard 用它判断：进程是否仍在 tick、行情是否可用于实时模拟、近期是否有拒单。
    """
    try:
        with session_scope() as s:
            run = _load_run_for_health(s, run_id)
            if not run:
                return {
                    "status": "NO_RUN",
                    "status_text": "无运行记录",
                    "severity": "muted",
                    "run": {},
                }

            ticks = s.scalars(
                select(TickLog)
                .where(TickLog.run_id == run.id)
                .order_by(desc(TickLog.ts))
                .limit(tick_limit)
            ).all()
            orders = s.scalars(
                select(OrderRecord)
                .where(OrderRecord.run_id == run.id)
                .order_by(desc(OrderRecord.created_at))
                .limit(order_limit)
            ).all()
            snap = s.scalars(
                select(AccountSnapshot)
                .where(AccountSnapshot.run_id == run.id)
                .order_by(desc(AccountSnapshot.ts))
            ).first()

            latest_tick = ticks[0] if ticks else None
            latest_tick_age = _age_from_utc(latest_tick.ts) if latest_tick else None
            stale_after = _estimated_tick_stale_threshold_seconds(ticks)
            order_counts = Counter(order.status.value for order in orders)
            risk_rejected_total = sum(t.risk_rejected for t in ticks)
            error_total = sum(len(t.errors_json or []) for t in ticks)
            status, status_text, severity = _paper_health_status(
                run=run,
                latest_tick=latest_tick,
                latest_tick_age=latest_tick_age,
                stale_after=stale_after,
                rejected_order_count=order_counts.get("REJECTED", 0),
                risk_rejected_total=risk_rejected_total,
                error_total=error_total,
            )

            return {
                "status": status,
                "status_text": status_text,
                "severity": severity,
                "run": {
                    "run_id": run.id,
                    "run_uid": run.run_uid,
                    "strategy_id": run.strategy_id,
                    "mode": run.mode,
                    "started_at": utc_naive_to_market_time(run.started_at),
                    "ended_at": utc_naive_to_market_time(run.ended_at),
                },
                "latest_tick_ts": utc_naive_to_market_time(latest_tick.ts) if latest_tick else None,
                "latest_tick_age_seconds": latest_tick_age,
                "stale_after_seconds": stale_after,
                "latest_quotes_count": latest_tick.quotes_count if latest_tick else 0,
                "latest_tradeable_quotes": latest_tick.tradeable_quotes if latest_tick else 0,
                "latest_fallback_quotes": latest_tick.fallback_quotes if latest_tick else 0,
                "latest_signals_generated": latest_tick.signals_generated if latest_tick else 0,
                "latest_strategy_skipped": latest_tick.strategy_skipped if latest_tick else 0,
                "latest_orders_placed": latest_tick.orders_placed if latest_tick else 0,
                "latest_risk_rejected": latest_tick.risk_rejected if latest_tick else 0,
                "latest_error_count": len(latest_tick.errors_json or []) if latest_tick else 0,
                "latest_quote_sources": _quote_sources_from_tick(latest_tick),
                "latest_reason": _latest_tick_reason(
                    latest_tick.per_symbol_json or [],
                    latest_tick.errors_json or [],
                ) if latest_tick else "-",
                "window_ticks": len(ticks),
                "risk_rejected_total": risk_rejected_total,
                "error_total": error_total,
                "orders_placed_total": sum(t.orders_placed for t in ticks),
                "strategy_skipped_total": sum(t.strategy_skipped for t in ticks),
                "recent_order_count": len(orders),
                "open_order_count": sum(order_counts.get(k, 0) for k in _OPEN_ORDER_STATUSES),
                "filled_order_count": order_counts.get("FILLED", 0),
                "rejected_order_count": order_counts.get("REJECTED", 0),
                "order_status_counts": dict(order_counts),
                "account": _account_snapshot_dict(snap),
            }
    except Exception as e:
        log.warning("paper_run_health failed: {}", e)
        return {}


def _load_run_for_health(s, run_id: int | None) -> StrategyRun | None:
    if run_id is not None:
        return s.get(StrategyRun, run_id)
    run = s.scalars(
        select(StrategyRun)
        .where(StrategyRun.ended_at.is_(None), StrategyRun.strategy_id != "demo_seed")
        .order_by(desc(StrategyRun.started_at))
    ).first()
    if run:
        return run
    return s.scalars(select(StrategyRun).order_by(desc(StrategyRun.started_at))).first()


def _age_from_utc(ts) -> float:
    return max((utcnow() - ts).total_seconds(), 0.0)


def _estimated_tick_stale_threshold_seconds(rows: list[TickLog]) -> float:
    gaps: list[float] = []
    ordered_ts = sorted((row.ts for row in rows if row.ts), reverse=True)
    for newer, older in zip(ordered_ts, ordered_ts[1:], strict=False):
        gap = (newer - older).total_seconds()
        if 0 < gap < 86_400:
            gaps.append(gap)
    if not gaps:
        return _DEFAULT_TICK_STALE_SECONDS
    return max(60.0, float(median(gaps)) * 3)


def _quote_sources_from_tick(tick: TickLog | None) -> list[str]:
    if tick is None:
        return []
    sources: list[str] = []
    for item in tick.per_symbol_json or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        if source and source not in sources:
            sources.append(str(source))
    return sources


def _account_snapshot_dict(snap: AccountSnapshot | None) -> dict:
    if snap is None:
        return {}
    return {
        "ts": utc_naive_to_market_time(snap.ts),
        "cash": float(snap.cash),
        "market_value": float(snap.market_value),
        "total_assets": float(snap.total_assets),
        "realized_pnl": float(snap.realized_pnl),
        "unrealized_pnl": float(snap.unrealized_pnl),
    }


def _paper_health_status(
    *,
    run: StrategyRun,
    latest_tick: TickLog | None,
    latest_tick_age: float | None,
    stale_after: float,
    rejected_order_count: int,
    risk_rejected_total: int,
    error_total: int,
) -> tuple[str, str, str]:
    if run.ended_at is not None:
        return "ENDED", "已停止", "muted"
    if latest_tick is None:
        return "WAITING_TICK", "等待 tick", "warning"
    if latest_tick_age is not None and latest_tick_age > stale_after:
        return "STALE_TICK", "tick 停滞", "warning"
    if error_total > 0:
        return "ERROR", "tick 有错误", "error"
    if latest_tick.quotes_count <= 0:
        return "NO_QUOTES", "无行情", "warning"
    if latest_tick.tradeable_quotes <= 0:
        return "NO_TRADEABLE_QUOTES", "行情不可模拟", "warning"
    if latest_tick.fallback_quotes > 0 or latest_tick.tradeable_quotes < latest_tick.quotes_count:
        return "PARTIAL_QUOTES", "部分行情不可模拟", "warning"
    if risk_rejected_total > 0 or rejected_order_count > 0:
        return "RISK_REJECTED", "有风控/订单拦截", "warning"
    return "OK", "运行正常", "ok"


def market_data_quality(run_id: int | None = None, *, limit: int = 20) -> dict:
    """最近 tick 的行情质量摘要。

    这是 UI 总览页的轻量监控：不直接拉 AkShare，只读取已落库的 TickLog。
    """
    try:
        with session_scope() as s:
            stmt = select(TickLog).order_by(desc(TickLog.ts)).limit(limit)
            if run_id is not None:
                stmt = stmt.where(TickLog.run_id == run_id)
            rows = s.scalars(stmt).all()
            if not rows:
                return {}

            latest = rows[0]
            skip_counts: dict[str, int] = {}
            error_count = 0
            for row in rows:
                error_count += len(row.errors_json or [])
                for item in row.per_symbol_json or []:
                    reason = item.get("skip_reason") if isinstance(item, dict) else None
                    if reason:
                        skip_counts[reason] = skip_counts.get(reason, 0) + 1

            quotes_total = sum(row.quotes_count for row in rows)
            tradeable_total = sum(row.tradeable_quotes for row in rows)
            fallback_total = sum(row.fallback_quotes for row in rows)
            skipped_total = sum(row.strategy_skipped for row in rows)

            top_skip_reasons = [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    skip_counts.items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )[:5]
            ]

            return {
                "latest_ts": utc_naive_to_market_time(latest.ts),
                "latest_quotes_count": latest.quotes_count,
                "latest_tradeable_quotes": latest.tradeable_quotes,
                "latest_fallback_quotes": latest.fallback_quotes,
                "latest_strategy_skipped": latest.strategy_skipped,
                "latest_errors": len(latest.errors_json or []),
                "window_ticks": len(rows),
                "quotes_total": quotes_total,
                "tradeable_total": tradeable_total,
                "fallback_total": fallback_total,
                "skipped_total": skipped_total,
                "error_total": error_count,
                "tradeable_ratio": (tradeable_total / quotes_total) if quotes_total else 0.0,
                "top_skip_reasons": top_skip_reasons,
            }
    except Exception as e:
        log.warning("market_data_quality failed: {}", e)
        return {}
