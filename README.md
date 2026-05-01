# qtrade — A 股 LLM 多 Agent 量化交易平台

基于 DeepSeek 和 LangGraph 的多 Agent 决策框架，目标市场为 A 股，决策频率为日级。

> **状态**：Phase 0 / 早期开发中。**严禁直接用真实资金运行**。先通过回测和模拟盘验证至少 1–3 个月。

## 设计要点

- **多 Agent 协作**：技术分析师、舆情分析师、多空研究员辩论、决策合成。
- **硬风控**：风控规则可否决任意 Agent 决策，硬编码上限不可被 LLM 绕过。
- **三段式工作流**：回测 → 模拟盘 → 小额实盘，每段需独立通过验收标准。
- **可观测性**：每次 Agent 决策、每笔订单都进 SQLite 审计日志，可完整复盘。

## 目录结构

```
src/qtrade/
├── core/         # 配置 / 日志 / 交易日历 / 异常
├── data/         # 数据适配器 + Parquet 缓存
├── persistence/  # SQLAlchemy 模型 + DB 初始化
├── strategies/   # 策略库（technical / ml / llm）
├── llm/          # DeepSeek 客户端 + 缓存 + 重试
├── execution/    # OMS / 持仓 / 资金 / Broker 适配
├── risk/         # 风控引擎（pre/in/post trade）
├── backtest/     # 回测引擎 + 指标
├── monitoring/   # 告警 + 审计日志
├── ui/           # PySide6 桌面应用
└── cli.py        # 命令行入口
scripts/          # 一次性脚本（init_db / download / demo）
config/           # YAML 配置
tests/            # 单元 / 集成测试
```

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. 安装
pip install -e ".[dev]"

# 3. 配置环境变量
cp .env.example .env
# 填入 DEEPSEEK_API_KEY 等

# 4. 初始化数据库
python scripts/init_db.py

# 5. 下载历史数据（默认沪深 300 成分股近 2 年日线）
python scripts/download_history.py --symbols 600519 000001 --start 2023-01-01

# 6. 跑一个回测 demo（双均线策略，不依赖 LLM）
python scripts/run_backtest_demo.py

# 7. 打开桌面 UI（可选）
python -m qtrade.ui.main
```

## 路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 工程基础（配置/日志/数据库/CLI） | 🟡 进行中 |
| 1 | 数据层 + 回测引擎 + 双均线基线 | 🟡 进行中 |
| 2 | 模拟盘对接（掘金 GM SDK）+ OMS + 基础风控 | ⚪ 待开始 |
| 3 | DeepSeek 多 Agent 决策图（4 Agent） | 🟡 骨架已有 |
| 4 | 完整风控 + 监控告警 + UI 完整化 | ⚪ 待开始 |
| 5 | 小额实盘（Live） | ⚪ 待开始 |
| 6 | ML 因子 / 多策略组合 / 舆情深度抽取 | ⚪ 待开始 |

## 重要警告

1. 量化策略历史回测好不等于未来赚钱，**不要依赖回测结果做投资决策**。
2. LLM Agent 可能产生幻觉，**所有"事实"必须来自工具调用，不能信赖模型内部记忆**。
3. A 股 T+1、涨跌停、停牌、退市规则已在回测中建模，但实盘仍可能与回测有偏差。
4. 本项目不构成任何投资建议，作者不对使用本项目造成的任何损失负责。
