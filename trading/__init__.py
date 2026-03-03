"""
Trading package.

Contains the live trading session implementation that orchestrates
real-time market data, signal generation, strike selection, and
order execution for IBKR paper / live trading.
"""

from .session import LiveTradingSession

__all__ = ["LiveTradingSession"]
