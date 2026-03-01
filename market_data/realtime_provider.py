"""
RealtimeMarketDataProvider — wall-clock time-sync wrapper.

Wraps any MarketDataProvider (typically IBKRMarketDataProvider) and adds
time synchronisation: when the engine requests a bar at time T, this wrapper
blocks (sleeps) until that wall-clock time has actually been reached.

This turns the instant-replay loop inside backtest_day_intraday() into a
real-time loop that processes one bar per minute as the market advances.

Usage
-----
from market_data.ibkr_provider import IBKRMarketDataProvider
from market_data.realtime_provider import RealtimeMarketDataProvider

ibkr = IBKRMarketDataProvider(port=7497)
ibkr.connect()
provider = RealtimeMarketDataProvider(ibkr, trade_date="2026-02-27")

# Now pass provider to LiveTradingSession — it will run at real-time speed.
"""

import time
from datetime import datetime
from typing import Optional, List
import pandas as pd
from loguru import logger

from .provider import MarketDataProvider


class RealtimeMarketDataProvider(MarketDataProvider):
    """
    Time-synchronised wrapper around a live MarketDataProvider.

    Any call to get_fastest_spx_price() or get_options_data() that requests
    data for a timestamp that has *not yet arrived* in wall-clock time will
    block until that minute has passed (plus a small grace period to allow
    the bar to propagate from IBKR to the buffer).

    Calls for historical timestamps (already passed) return immediately.

    Parameters
    ----------
    inner:       The underlying provider (IBKRMarketDataProvider).
    trade_date:  YYYY-MM-DD — the date being traded.
    grace_secs:  Extra seconds to wait after the bar minute before reading
                 (default 3 — allows IBKR to process and push the last bar).
    """

    def __init__(self,
                 inner: MarketDataProvider,
                 trade_date: str,
                 grace_secs: float = 3.0):
        self._inner = inner
        self._trade_date = trade_date
        self._grace_secs = grace_secs

    # ------------------------------------------------------------------
    # Time-sync helper
    # ------------------------------------------------------------------

    def _wait_for_bar(self, timestamp: str):
        """
        Block until wall-clock time >= *timestamp* + grace_secs.

        If the timestamp is already in the past, returns immediately.
        """
        try:
            bar_dt = datetime.strptime(
                f"{self._trade_date} {timestamp}", "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            return

        target = bar_dt.timestamp() + self._grace_secs
        now = time.time()
        if now < target:
            sleep_secs = target - now
            logger.debug(
                f"RealtimeProvider: waiting {sleep_secs:.1f}s for bar at {timestamp}"
            )
            time.sleep(sleep_secs)

    # ------------------------------------------------------------------
    # MarketDataProvider implementation — delegates with time-sync
    # ------------------------------------------------------------------

    def get_fastest_spx_price(self, date: str, timestamp: str) -> Optional[float]:
        if date == self._trade_date:
            self._wait_for_bar(timestamp)
        return self._inner.get_fastest_spx_price(date, timestamp)

    def get_spx_data(self, date: str,
                     start_time: str = "09:30:00",
                     end_time: str = "16:00:00") -> Optional[pd.DataFrame]:
        # For historical windows (start to now), no need to wait.
        if date == self._trade_date:
            self._wait_for_bar(end_time)
        return self._inner.get_spx_data(date, start_time, end_time)

    def get_options_data(self, date: str, timestamp: str) -> Optional[pd.DataFrame]:
        if date == self._trade_date:
            self._wait_for_bar(timestamp)
        return self._inner.get_options_data(date, timestamp)

    def get_options_chain_at_time(self, date: str, timestamp: str,
                                  center_strike: float,
                                  strike_range: int = 300) -> Optional[pd.DataFrame]:
        if date == self._trade_date:
            self._wait_for_bar(timestamp)
        return self._inner.get_options_chain_at_time(
            date, timestamp, center_strike, strike_range
        )

    @property
    def available_dates(self) -> List[str]:
        return self._inner.available_dates

    # ------------------------------------------------------------------
    # Transparent delegation for provider-specific helpers
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
