"""
Abstract Market Data Provider interface.

All data sources (Parquet, IBKR, etc.) implement this interface,
allowing the trading engine to be completely data-source agnostic.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd


class MarketDataProvider(ABC):
    """
    Abstract base class for all market data sources.

    Implementations must provide:
      - Historical or real-time SPX price data
      - Options chain snapshots with Greeks (or at minimum bid/ask for BS delta calc)
      - A list of available trading dates

    Implementations
    ---------------
    ParquetMarketDataProvider
        Historical backtesting via local Parquet files.  Zero latency; used for
        backtesting and simulation paper trading.

    IBKRMarketDataProvider  (Phase 2)
        Real-time bars and options snapshots via IBKR TWS / IB Gateway using
        ib_insync.  Used for live paper / live trading.

    RealtimeMarketDataProvider  (Phase 2)
        Wraps IBKRMarketDataProvider and adds wall-clock time synchronisation:
        calls to get_fastest_spx_price() block until the requested bar time
        has actually passed in the real world.
    """

    # ------------------------------------------------------------------
    # Core price methods
    # ------------------------------------------------------------------

    @abstractmethod
    def get_fastest_spx_price(self, date: str, timestamp: str) -> Optional[float]:
        """
        Return the SPX price at *timestamp* on *date*.

        For historical providers this is an instant Parquet lookup.
        For real-time providers this should return the most-recent bar price
        at or before *timestamp* that is currently available in the buffer.

        Parameters
        ----------
        date:      YYYY-MM-DD
        timestamp: HH:MM:SS

        Returns None if no price is available.
        """

    @abstractmethod
    def get_spx_data(self, date: str,
                     start_time: str = "09:30:00",
                     end_time: str = "16:00:00") -> Optional[pd.DataFrame]:
        """
        Return a DataFrame of SPX 1-minute bars for the given window.

        The returned DataFrame must have a 'close' column.  The index
        should be a DatetimeIndex (or at minimum contain timestamp info).
        Returns None if data is unavailable.
        """

    # ------------------------------------------------------------------
    # Options chain methods
    # ------------------------------------------------------------------

    @abstractmethod
    def get_options_data(self, date: str, timestamp: str) -> Optional[pd.DataFrame]:
        """
        Return a snapshot of the SPX / SPXW options chain at *timestamp*.

        Required columns:
          strike, option_type ('call' or 'put'), bid, ask, delta

        Optional but useful columns:
          gamma, theta, vega, volume, open_interest, implied_volatility

        Returns None if data is unavailable.
        """

    @abstractmethod
    def get_options_chain_at_time(self, date: str, timestamp: str,
                                  center_strike: float,
                                  strike_range: int = 300) -> Optional[pd.DataFrame]:
        """
        Return the raw options chain centred around *center_strike*, spanning
        [center_strike - strike_range, center_strike + strike_range].

        This is the low-level method used by DeltaStrikeSelector for strike
        selection; it returns data in the raw Parquet / IBKR format rather
        than the normalised format produced by get_options_data().

        Returns None if data is unavailable.
        """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def available_dates(self) -> List[str]:
        """
        Return a list of YYYY-MM-DD strings for which data is available.

        For historical providers this is pre-computed from the Parquet files.
        For real-time providers this typically returns [today] or [] when the
        market is closed.
        """
