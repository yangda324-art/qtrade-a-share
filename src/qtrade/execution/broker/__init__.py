"""Broker 适配层。"""
from qtrade.execution.broker.base import Broker, BrokerOrderResult
from qtrade.execution.broker.paper import PaperBroker

__all__ = ["Broker", "BrokerOrderResult", "PaperBroker"]
