"""传统技术指标策略库。"""
from qtrade.strategies.technical.bollinger import BollingerStrategy
from qtrade.strategies.technical.ma_cross import MACrossStrategy
from qtrade.strategies.technical.macd import MACDStrategy
from qtrade.strategies.technical.momentum import MomentumStrategy
from qtrade.strategies.technical.turtle import TurtleStrategy

# 策略名 → 类的映射（CLI / config 引用用）
STRATEGY_REGISTRY = {
    "ma_cross":  MACrossStrategy,
    "macd":      MACDStrategy,
    "bollinger": BollingerStrategy,
    "momentum":  MomentumStrategy,
    "turtle":    TurtleStrategy,
}

__all__ = [
    "MACrossStrategy",
    "MACDStrategy",
    "BollingerStrategy",
    "MomentumStrategy",
    "TurtleStrategy",
    "STRATEGY_REGISTRY",
]
