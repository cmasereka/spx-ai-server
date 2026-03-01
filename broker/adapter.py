"""
Broker Adapter abstract interface and OrderResult dataclass.

All broker backends (simulation, IBKR paper, IBKR live) implement the
BrokerAdapter interface.  The engine and service layers interact only with
this interface; swapping from paper to live trading is a single constructor
argument change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class OrderResult:
    """
    Result of a single broker order (open or close).

    Captures both the intended price (limit_price) and the actual fill
    (fill_price) so that slippage can be measured over time.
    """
    order_id: str
    symbol: str
    fill_price: float        # Actual execution price
    limit_price: float       # Price we requested
    quantity: int
    strategy_type: str       # e.g. "iron_condor", "put_spread", "call_spread"
    is_entry: bool           # True = opening trade, False = closing trade
    timestamp: str           # HH:MM:SS when the order was submitted
    success: bool = True
    error_message: Optional[str] = None
    broker_data: Dict[str, Any] = field(default_factory=dict)

    # Computed on construction
    slippage: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.slippage = self.fill_price - self.limit_price


class BrokerAdapter(ABC):
    """
    Abstract base class for all broker order-execution backends.

    Lifecycle
    ---------
    1. Call connect() before the trading session starts.
    2. Use open_position() / close_position() during the session.
    3. Call close_all() at end-of-day if any positions remain.
    4. Call disconnect() after the session ends.

    Implementations
    ---------------
    NullBrokerAdapter
        Phantom fills at the requested price.  Zero latency, no real money.
        Used for backtesting and simulation paper trading.

    IBKRBrokerAdapter  (Phase 2)
        Submits real orders to IBKR TWS / IB Gateway via ib_insync.
        Works for both IBKR paper (port 7497) and live (port 7496) accounts.
    """

    @abstractmethod
    def open_position(self, strategy, quantity: int, timestamp: str,
                      target_credit: float) -> OrderResult:
        """
        Open a new spread / IC position.

        Parameters
        ----------
        strategy:      The OptionsStrategy object to be opened.
        quantity:      Number of contracts.
        timestamp:     HH:MM:SS — current bar time.
        target_credit: Expected net credit per spread per share (mid-price model).

        Returns an OrderResult with the actual fill details.
        """

    @abstractmethod
    def close_position(self, strategy, quantity: int, timestamp: str,
                       target_debit: float) -> OrderResult:
        """
        Close an existing spread / IC position.

        Parameters
        ----------
        strategy:     The OptionsStrategy object to be closed.
        quantity:     Number of contracts.
        timestamp:    HH:MM:SS — current bar time.
        target_debit: Expected cost to close per spread per share (mid-price model).

        Returns an OrderResult with the actual fill details.
        """

    @abstractmethod
    def close_all(self, open_strategies: List, timestamp: str) -> List[OrderResult]:
        """
        Force-close every open position immediately.

        Called at end-of-day (15:45 ET) or when the session is stopped.
        Returns a list of OrderResults, one per strategy.
        """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if the broker connection is currently active."""

    # ------------------------------------------------------------------
    # Optional lifecycle hooks — override in subclasses that need them
    # ------------------------------------------------------------------

    def connect(self, **kwargs) -> bool:
        """
        Establish the broker connection.  Returns True on success.
        Default implementation is a no-op (NullBrokerAdapter is always connected).
        """
        return True

    def disconnect(self):
        """
        Close the broker connection gracefully.
        Default implementation is a no-op.
        """
