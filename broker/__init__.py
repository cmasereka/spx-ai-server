"""
Broker Adapter package.

Abstracts order execution behind a common interface, enabling the same
strategy logic to run in simulation, paper trading, and live trading.

Available adapters:
  NullBrokerAdapter  — phantom fills at target price (simulation / backtesting)
  IBKRBrokerAdapter  — real orders via IBKR TWS/IB Gateway (paper / live trading)
"""

from .adapter import BrokerAdapter, OrderResult
from .null_adapter import NullBrokerAdapter

__all__ = [
    "BrokerAdapter",
    "OrderResult",
    "NullBrokerAdapter",
]

# IBKRBrokerAdapter is imported lazily (requires ib_insync) — do not import at package level
# Use: from broker.ibkr_adapter import IBKRBrokerAdapter
