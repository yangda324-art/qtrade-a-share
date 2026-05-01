"""自定义异常体系。

设计原则：
- 所有业务异常都继承自 ``QtradeError``，方便统一捕获和告警。
- 异常按"模块"划分，定位问题时只看异常类型就能知道在哪一层。
- 风控触发用 ``RiskRejected`` —— 这个异常**不应被静默吞掉**，
  必须传到上层让监控记录、告警、并阻止后续动作。
"""
from __future__ import annotations


class QtradeError(Exception):
    """所有业务异常的根基类。"""


class ConfigError(QtradeError):
    """配置加载或校验失败。"""


class DataError(QtradeError):
    """数据缺失、格式错误、源不可达等。"""


class BrokerError(QtradeError):
    """券商接口报错（下单失败、查询失败等）。"""


class RiskRejected(QtradeError):
    """风控否决了某个订单或决策。

    Attributes:
        rule: 触发的风控规则名（如 ``max_order_value``）。
        detail: 人类可读的拒绝原因。
    """

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"[{rule}] {detail}")
        self.rule = rule
        self.detail = detail


class StrategyError(QtradeError):
    """策略内部错误（不应让单个策略错误打挂整个进程）。"""


class LLMError(QtradeError):
    """LLM 调用失败、超时、输出格式错误等。"""
