"""
Parquet-backed Market Data Provider.

Thin facade over the existing EnhancedQueryEngineAdapter so that backtesting
and simulation paper trading can use the MarketDataProvider interface without
any changes to the underlying data-loading logic.
"""

from typing import Optional, List
import pandas as pd

from .provider import MarketDataProvider
from query_engine_adapter import EnhancedQueryEngineAdapter
from src.data.query_engine import BacktestQueryEngine


class ParquetMarketDataProvider(MarketDataProvider):
    """
    Historical market data provider backed by local Parquet files.

    Delegates every call to the existing EnhancedQueryEngineAdapter — zero
    logic duplication.  Can be constructed either by supplying a data path
    (the adapter is created internally) or by wrapping an already-initialised
    adapter (avoids double-loading Parquet files when the backtesting engine
    has already done so).
    """

    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        query_engine = BacktestQueryEngine(data_path)
        self._adapter = EnhancedQueryEngineAdapter(query_engine)

    @classmethod
    def from_adapter(cls, adapter: EnhancedQueryEngineAdapter) -> "ParquetMarketDataProvider":
        """
        Wrap an already-created EnhancedQueryEngineAdapter.

        Use this inside BacktestService / PaperTradingService to share the
        single adapter that was loaded at startup, rather than creating a
        new one and loading all the Parquet files again.
        """
        instance = cls.__new__(cls)
        instance._adapter = adapter
        return instance

    # ------------------------------------------------------------------
    # MarketDataProvider implementation
    # ------------------------------------------------------------------

    def get_fastest_spx_price(self, date: str, timestamp: str) -> Optional[float]:
        return self._adapter.get_fastest_spx_price(date, timestamp)

    def get_spx_data(self, date: str,
                     start_time: str = "09:30:00",
                     end_time: str = "16:00:00") -> Optional[pd.DataFrame]:
        return self._adapter.get_spx_data(date, start_time, end_time)

    def get_options_data(self, date: str, timestamp: str) -> Optional[pd.DataFrame]:
        return self._adapter.get_options_data(date, timestamp)

    def get_options_chain_at_time(self, date: str, timestamp: str,
                                  center_strike: float,
                                  strike_range: int = 300) -> Optional[pd.DataFrame]:
        """Delegate to the underlying query engine loader."""
        try:
            return self._adapter.query_engine.loader.get_options_chain_at_time(
                date, timestamp,
                center_strike=center_strike,
                strike_range=strike_range,
            )
        except Exception:
            return None

    @property
    def available_dates(self) -> List[str]:
        return self._adapter.query_engine.available_dates or []

    # ------------------------------------------------------------------
    # Transparent delegation for any method not on the ABC
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """
        Delegate any attribute or method not explicitly defined here to the
        underlying EnhancedQueryEngineAdapter.  This ensures full backward
        compatibility with any code that calls adapter-specific helpers
        (e.g. get_options_data_for_strategy via the strategy_builder).
        """
        return getattr(self._adapter, name)
