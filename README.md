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
qtrade ui
```

## 用 LLM 多 Agent 决策

1. 注册 + 充值（最少 1 元）：https://platform.deepseek.com
2. 拿到 API key 填进 `.env`：
   ```
   DEEPSEEK_API_KEY=sk-xxxxx
   ```
3. 跑一次决策（不下单，纯调试 / 看 token 成本）：
   ```bash
   qtrade decide 600519 --lookback-days 120
   ```
4. 在桌面 UI 的「Agent 决策」tab 里点任意一行可看 5 个 Agent 的完整 trace。
5. 让 LLM 直接驱动模拟盘：
   ```bash
   qtrade run-paper --strategy llm --symbol 600519 --once
   qtrade run-paper --strategy llm --symbol 600519 --interval-seconds 10 --decision-interval-minutes 15
   ```

启动 `llm` 模拟盘前，CLI 会检查 `DEEPSEEK_API_KEY`、prompt 版本、辩论轮数和仓位参数。
LLM 只输出交易意图，真实执行前仍会经过行情新鲜度、价格偏离保护、现金/持仓/T+1 和硬风控检查。

**成本预估**：单次完整 5-Agent 决策约 ¥0.05–0.20。日级策略 10 只票一天 ¥0.5–2，月成本约 ¥10–60。

## 路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 工程基础（配置/日志/数据库/CLI） | ✅ 完成 |
| 1 | 实时盯盘 LiveRunner + Scheduler + UI 实时联动 | ✅ 完成 |
| 2 | K 线 / 回测 UI 表单化 + LLM 多 Agent 跑通 + 测试 | ✅ 完成 |
| 3 | 多策略并行（独立账户） + 业绩对比 + 告警 + 后台守护 | ✅ 完成 |
| 4 | LLM Agent 深度优化（research 数据 + 多轮辩论 + Prompt A/B） | ✅ 完成 |
| 5 | 接入真实券商 + 小额实盘 | ⚪ 待开始 |
| 6 | ML 因子 / 舆情深度抽取 | ⚪ 待开始 |

## 重要警告

1. 量化策略历史回测好不等于未来赚钱，**不要依赖回测结果做投资决策**。
2. LLM Agent 可能产生幻觉，**所有"事实"必须来自工具调用，不能信赖模型内部记忆**。
3. A 股 T+1、涨跌停、停牌、退市规则已在回测中建模，但实盘仍可能与回测有偏差。
4. 本项目不构成任何投资建议，作者不对使用本项目造成的任何损失负责。
