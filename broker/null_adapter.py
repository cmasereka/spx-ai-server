"""
NullBrokerAdapter — phantom fills for simulation / backtesting.

Every order is instantly "filled" at exactly the target price with zero
slippage.  No network connection is required.  This is the default adapter
used by PaperTradingService in simulation mode.
"""

import uuid
from typing import List

from .adapter import BrokerAdapter, OrderResult


class NullBrokerAdapter(BrokerAdapter):
    """
    Phantom order execution adapter.

    Instantly fills every open/close at the requested target price.
    Used for:
      - Backtesting (no order submission needed, engine tracks P&L internally)
      - Simulation paper trading (replays historical data at full speed)
    """

    def open_position(self, strategy, quantity: int, timestamp: str,
                      target_credit: float) -> OrderResult:
        strategy_type = getattr(strategy, "strategy_type", "unknown")
        if hasattr(strategy_type, "value"):
            strategy_type = strategy_type.value
        return OrderResult(
            order_id=str(uuid.uuid4()),
            symbol="SPXW",
            fill_price=target_credit,
            limit_price=target_credit,
            quantity=quantity,
            strategy_type=str(strategy_type),
            is_entry=True,
            timestamp=timestamp,
        )

    def close_position(self, strategy, quantity: int, timestamp: str,
                       target_debit: float) -> OrderResult:
        strategy_type = getattr(strategy, "strategy_type", "unknown")
        if hasattr(strategy_type, "value"):
            strategy_type = strategy_type.value
        return OrderResult(
            order_id=str(uuid.uuid4()),
            symbol="SPXW",
            fill_price=target_debit,
            limit_price=target_debit,
            quantity=quantity,
            strategy_type=str(strategy_type),
            is_entry=False,
            timestamp=timestamp,
        )

    def close_all(self, open_strategies: List, timestamp: str) -> List[OrderResult]:
        return [
            self.close_position(s, getattr(s, "quantity", 1), timestamp, 0.0)
            for s in open_strategies
        ]

    @property
    def is_connected(self) -> bool:
        return True  # always "connected" — no real connection needed
