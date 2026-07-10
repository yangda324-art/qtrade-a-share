# AGENTS.md

## Project

This is a China A-share quantitative trading and research platform.

The current default workflow is AkShare market data + SQLite persistence + PaperBroker
simulation + PySide6 UI + optional DeepSeek/LangGraph multi-agent decision logic.

The goal is to build reliable tools for data loading, signal research, backtesting,
risk analysis, paper trading, UI monitoring, and eventually carefully gated live trading.

## General Rules

- Make small, safe changes.
- Do not rewrite large parts of the project unless asked.
- Check existing code before inventing new APIs, config names, schemas, or UI patterns.
- Preserve user data, trading history, local config, and unrelated worktree changes.
- Do not modify, regenerate, truncate, or delete local SQLite databases, cached market data, exported reports, logs, or user-created config files unless explicitly requested.
- Add or update tests when behavior changes.
- After editing, run the relevant checks if available.
- If a command does not exist or the existing baseline fails, report that honestly.

## Project Architecture

- Keep data adapters and market data normalization under `src/qtrade/data`.
- Keep strategy logic under `src/qtrade/strategies`.
- Keep order management, accounts, positions, fees, and broker abstractions under `src/qtrade/execution`.
- Keep live/paper orchestration, scheduling, manual trading, and runtime state under `src/qtrade/runtime`.
- Keep risk rules under `src/qtrade/risk`; risk checks must remain a hard gate before orders.
- Keep SQLite models and database setup under `src/qtrade/persistence`.
- Keep PySide6 UI code under `src/qtrade/ui`.
- Do not mix UI, broker execution, strategy generation, persistence, and data-fetching logic in one layer.

## Quant Rules

- Avoid look-ahead bias.
- Do not leak future data into features, labels, filters, normalization, ranking, or portfolio construction.
- Use realistic fees, slippage, spread, liquidity, and turnover assumptions.
- Keep in-sample, validation, and out-of-sample periods separate.
- Do not claim a strategy works unless assumptions and backtest results are shown.
- Report drawdown, Sharpe, Sortino, volatility, turnover, win rate, exposure, and trade count when relevant.
- Be careful with overfitting.
- Keep random seeds fixed for reproducible experiments.
- Never silently change timestamps, timezone logic, corporate action handling, or historical data.

## A-Share Market Rules

- Respect Asia/Shanghai timezone and China A-share trading hours.
- Respect non-trading days and China holiday calendars.
- Respect T+1 selling constraints.
- A-share rules must be board-aware and instrument-aware.
- Do not hardcode one universal rule for all A-share instruments.
- Main Board stock buy orders are usually in 100-share lots.
- Odd-lot selling may be allowed when selling the full remaining position.
- STAR Market, ChiNext, ETFs, convertible bonds, and funds may have different lot sizes, price limits, tick sizes, trading mechanisms, and IPO-period rules.
- Price-limit logic must be board-aware and date-aware.
- Account for price-limit exceptions, suspensions, ST stocks, odd-lot rules, and rejected orders explicitly.
- Do not silently change qfq/hfq/no-adjust behavior.
- Cached daily close prices are not tradeable real-time prices.

## Market Data Rules

- Treat AkShare quotes as best-effort polling data, not exchange-grade real-time feeds.
- Every quote used for trading decisions must carry source, fetched_at, and stale/fallback status where possible.
- Historical backtests may use cached historical bars, but the simulation must label them as historical data.
- Cached daily close prices must not be treated as real-time executable prices.
- If quote data is stale, missing, cached, fallback, or ambiguous, do not place live orders or real-time paper orders.
- For real-time paper trading, stale or fallback quotes should lead to HOLD / skip, with the skip reason logged.
- Log fallback usage, skipped strategy decisions, and skip reasons in tick logs.
- Keep data-loading tests independent from live network availability whenever practical.
- Add a new data source through an adapter boundary instead of hardcoding it into strategies or UI.

## Paper And Live Trading Safety

- Default to paper trading.
- Implementing live trading code is not permission to send real orders.
- Never submit, cancel, or modify real broker orders unless the user explicitly asks to operate live trading now.
- Put real order execution behind explicit broker adapters and explicit live config flags.
- Never reuse PaperBroker assumptions for live broker behavior without an adapter boundary.
- Live broker adapters must support account sync, position sync, order submit, cancel, query, and reconciliation.
- All live order actions must require explicit live config flags, a real broker adapter, fresh market data, account sync, position sync, and final risk checks immediately before submission.
- Run risk checks and stale-price protection immediately before any live order submit.
- Fail closed if risk checks, broker responses, order state, or market data are unclear.
- Add logs for orders, fills, positions, errors, retries, rejected orders, and reconciliation differences.

## LLM Agent Rules

- LLM agents may propose intents, not executable orders.
- Normalize LLM output through structured schemas before use.
- All LLM outputs must be converted into validated order intents first.
- Validated intents must still pass deterministic risk, position, cash, T+1, stale-data, price-limit, and broker checks before any order action.
- Invalid, incomplete, low-confidence, or unparsable LLM decisions must default to HOLD / no order.
- LLMs must not be trusted for factual market data unless the data comes from project tools.
- Persist prompts, model names, confidence, rationale, agent traces, decisions, and downstream order links.
- Keep token/cost-sensitive paths testable without real LLM calls.
- Do not let LLM logic bypass risk, position, cash, T+1, stale-data, or broker checks.

## UI Rules

- Do not block the PySide6 UI thread with slow network, LLM, or database operations.
- Keep market-watch polling active only when the relevant UI page is visible.
- UI must clearly distinguish live quotes, stale quotes, fallback quotes, simulated orders, and real orders.
- Do not remove chart pan/zoom behavior unless explicitly requested.
- Preserve stable chart bounds so candles, wicks, moving averages, and volume bars remain visible.
- Avoid UI changes outside the requested surface.

## Persistence Rules

- Do not delete or rewrite user trading history unless explicitly requested.
- When tests need data, use fixtures, temporary directories, or mock adapters instead of changing real local data.
- Schema changes must preserve existing data or include a migration/backfill plan.
- Orders, fills, positions, account snapshots, tick logs, and agent decisions are audit records.
- Prefer append-only audit logs for trading events.
- Store enough context to reproduce decisions and explain rejected orders.

## Python Rules

- Use type hints for public functions.
- Keep data loading, signal generation, backtesting, risk, and execution logic separated.
- Prefer simple baselines before complex models.
- Do not add new dependencies unless clearly needed.
- Prefer existing project patterns over new abstractions.

## Test Guidance

- Runtime/execution/risk changes: run relevant live runner, manual trading, broker, scheduler, and risk tests.
- Data changes: run loader/cache/schema tests.
- UI data display changes: run UI display tests and avoid requiring live network.
- LLM changes: run schema, agent, and LLM tests with mocked clients.
- Full check before major handoff: `.venv/bin/python -m pytest`.

## Commands

Preferred commands:

- Run tests: `.venv/bin/python -m pytest`
- Run focused tests: `.venv/bin/python -m pytest tests/test_xxx.py`
- Run lint: `.venv/bin/ruff check .`
- Compile check: `.venv/bin/python -m compileall -q src/qtrade tests`

Currently non-gating unless explicitly requested:

- Format check: `.venv/bin/ruff format --check .`
- Type check: `.venv/bin/mypy .`

## Handoff Rules

When finishing a task, report:

- What changed.
- Which files were changed.
- Which checks were run.
- Which checks were not run and why.
- Any known risks, assumptions, or follow-up work.
