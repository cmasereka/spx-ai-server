"""
Market Data Provider package.

Abstracts all market data sources behind a common interface, enabling
the trading engine to be data-source agnostic.

Available providers:
  ParquetMarketDataProvider  — historical Parquet files (backtesting / simulation)
  IBKRMarketDataProvider     — real-time data via IBKR TWS/IB Gateway (live trading)
  RealtimeMarketDataProvider — time-sync wrapper for IBKR provider
"""

from .provider import MarketDataProvider
from .parquet_provider import ParquetMarketDataProvider

__all__ = [
    "MarketDataProvider",
    "ParquetMarketDataProvider",
]

# IBKR providers are imported lazily (require ib_insync) — do not import at package level
# Use:
#   from market_data.ibkr_provider import IBKRMarketDataProvider
#   from market_data.realtime_provider import RealtimeMarketDataProvider
