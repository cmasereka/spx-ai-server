"""
IBKR Real-Time Market Data Provider.

Provides live SPX price bars and SPXW options chain snapshots via IBKR
TWS / IB Gateway using the ib_insync library.

Usage
-----
from market_data.ibkr_provider import IBKRMarketDataProvider

provider = IBKRMarketDataProvider(host="127.0.0.1", port=7497, client_id=1)
provider.connect()
price = provider.get_fastest_spx_price("2026-02-27", "10:00:00")
chain  = provider.get_options_data("2026-02-27", "10:00:00")
provider.disconnect()

IBKR Port Reference
-------------------
7497 — TWS paper trading
7496 — TWS live trading
4002 — IB Gateway paper trading
4001 — IB Gateway live trading
"""

import time
import threading
from collections import defaultdict
from datetime import datetime, date as date_type
from typing import Optional, List, Dict, Deque
from collections import deque

import pandas as pd
import numpy as np
from loguru import logger

from .provider import MarketDataProvider
from query_engine_adapter import (
    _time_to_expiry_years, _compute_bs_delta, RISK_FREE_RATE
)

try:
    from ib_insync import IB, Stock, Index, Option, Contract
    from ib_insync import util as ib_util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning(
        "ib_insync is not installed. IBKRMarketDataProvider will not be available. "
        "Install it with: pip install ib_insync"
    )


class IBKRMarketDataProvider(MarketDataProvider):
    """
    Real-time market data provider backed by IBKR TWS / IB Gateway.

    Architecture
    ------------
    • A background thread runs the ib_insync event loop.
    • 5-second real-time bars for SPX are requested on connect() and
      pushed into a thread-safe ring buffer keyed by minute (HH:MM:SS).
    • get_fastest_spx_price() reads from this buffer — no blocking.
    • get_options_data() fires synchronous reqMktData snapshot requests
      for the full 0DTE SPXW chain around the current SPX price.

    Thread Safety
    -------------
    All buffer access is protected by a threading.Lock.
    """

    # Number of minutes of SPX price history to keep in the rolling buffer
    _PRICE_BUFFER_MINUTES = 120

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 7497,
                 client_id: int = 1,
                 account: str = ""):
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync must be installed to use IBKRMarketDataProvider. "
                "Run: pip install ib_insync"
            )
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account

        self._ib: Optional["IB"] = None
        self._lock = threading.Lock()
        self._connected = False

        # price_buffer[date_str][time_str] = float SPX price
        self._price_buffer: Dict[str, Dict[str, float]] = defaultdict(dict)
        # rolling deque of (date_str, time_str) for buffer eviction
        self._price_keys: Deque = deque(maxlen=self._PRICE_BUFFER_MINUTES)

        self._today: str = ""   # YYYY-MM-DD of the current trading session
        self._spx_contract = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, timeout: int = 15) -> bool:
        """Connect to IBKR TWS / IB Gateway. Returns True on success."""
        if not IB_AVAILABLE:
            return False
        # ib_insync's synchronous connect() calls asyncio.get_event_loop() internally.
        # When running inside a ThreadPoolExecutor worker there is no event loop yet,
        # so we create and register one before touching any ib_insync code.
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id,
                             timeout=timeout)
            self._connected = self._ib.isConnected()
            if self._connected:
                self._today = datetime.now().strftime("%Y-%m-%d")
                self._spx_contract = Index("SPX", "CBOE", "USD")
                self._ib.qualifyContracts(self._spx_contract)
                self._subscribe_spx_bars()
                logger.info(
                    f"IBKRMarketDataProvider connected to {self.host}:{self.port} "
                    f"(clientId={self.client_id})"
                )
            return self._connected
        except Exception as exc:
            logger.error(f"IBKR connection failed: {exc}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("IBKRMarketDataProvider disconnected")

    # ------------------------------------------------------------------
    # Real-time bar subscription
    # ------------------------------------------------------------------

    def _subscribe_spx_bars(self):
        """Subscribe to 5-second real-time bars for SPX."""
        if not self._ib or not self._spx_contract:
            return
        try:
            bars = self._ib.reqRealTimeBars(
                contract=self._spx_contract,
                barSize=5,
                whatToShow="MIDPOINT",   # SPX is a cash index — TRADES is not supported
                useRTH=True,
            )
            bars.updateEvent += self._on_rtbar
            logger.info("Subscribed to SPX real-time bars (5s, MIDPOINT)")
        except Exception as exc:
            logger.warning(f"Failed to subscribe to SPX real-time bars: {exc}")

    def _on_rtbar(self, bars, has_new_bar: bool):
        """Callback invoked by ib_insync on each new 5-second bar."""
        if not has_new_bar or not bars:
            return
        bar = bars[-1]
        bar_dt = datetime.fromtimestamp(bar.time)
        date_str = bar_dt.strftime("%Y-%m-%d")
        # Snap to the nearest minute
        time_str = bar_dt.strftime("%H:%M") + ":00"
        close_price = float(bar.close)
        if close_price <= 0:
            return
        with self._lock:
            self._price_buffer[date_str][time_str] = close_price
            self._price_keys.append((date_str, time_str))

    # ------------------------------------------------------------------
    # MarketDataProvider — price methods
    # ------------------------------------------------------------------

    def get_fastest_spx_price(self, date: str, timestamp: str) -> Optional[float]:
        """
        Return the most recent SPX price at or before *timestamp* on *date*.

        Searches the buffer backwards from *timestamp* up to 5 minutes to find
        the closest available reading.
        """
        with self._lock:
            day_buf = self._price_buffer.get(date, {})
        if not day_buf:
            return None

        # Try exact match first
        if timestamp in day_buf:
            return day_buf[timestamp]

        # Walk backwards up to 5 minutes
        try:
            target_dt = datetime.strptime(f"{date} {timestamp}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

        for delta_min in range(1, 6):
            candidate = (target_dt - pd.Timedelta(minutes=delta_min)).strftime("%H:%M:%S")
            if candidate in day_buf:
                return day_buf[candidate]

        return None

    def get_spx_data(self, date: str,
                     start_time: str = "09:30:00",
                     end_time: str = "16:00:00") -> Optional[pd.DataFrame]:
        """
        Return the buffered 1-minute SPX bar series for the given window.
        """
        with self._lock:
            day_buf = dict(self._price_buffer.get(date, {}))
        if not day_buf:
            return None

        rows = []
        try:
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
            end_dt   = datetime.strptime(f"{date} {end_time}",   "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

        for t_str, price in day_buf.items():
            try:
                t_dt = datetime.strptime(f"{date} {t_str}", "%Y-%m-%d %H:%M:%S")
                if start_dt <= t_dt <= end_dt:
                    rows.append({"timestamp": t_dt, "close": price})
            except ValueError:
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows).sort_values("timestamp").set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # MarketDataProvider — options methods
    # ------------------------------------------------------------------

    def get_options_data(self, date: str, timestamp: str) -> Optional[pd.DataFrame]:
        """
        Fetch a live options chain snapshot for today's 0DTE SPXW options.

        Requests market data for all SPXW strikes within ±150 points of the
        current SPX price.  Each request uses a frozen snapshot (reqMktData with
        snapshot=True) so that no ongoing subscription is created.
        """
        if not self._ib or not self._connected:
            return None

        spx_price = self.get_fastest_spx_price(date, timestamp)
        if not spx_price:
            return None

        try:
            # Build list of strikes to query (±150 pts, 5-pt intervals for SPXW)
            low  = int(spx_price - 150)
            high = int(spx_price + 155)
            strikes = list(range(low - (low % 5), high, 5))

            expiry = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
            T = _time_to_expiry_years(timestamp, date)

            rows = []
            for strike in strikes:
                for right in ("P", "C"):
                    contract = Option("SPXW", expiry, strike, right, "SMART", "USD")
                    try:
                        ticker = self._ib.reqMktData(contract, "", True, False)
                        self._ib.sleep(0)  # allow event loop to process
                        bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
                        ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
                        if ask <= 0:
                            continue
                        mid_price = (bid + ask) / 2.0
                        is_call = right == "C"
                        delta = _compute_bs_delta(
                            S=spx_price,
                            K=float(strike),
                            T=T,
                            r=RISK_FREE_RATE,
                            mid_price=mid_price,
                            is_call=is_call,
                        )
                        rows.append({
                            "strike": float(strike),
                            "option_type": "call" if is_call else "put",
                            "expiration": date,
                            "bid": bid,
                            "ask": ask,
                            "delta": delta,
                            "gamma": float(ticker.modelGreeks.gamma) if ticker.modelGreeks else 0.0,
                            "theta": float(ticker.modelGreeks.theta) if ticker.modelGreeks else 0.0,
                            "vega":  float(ticker.modelGreeks.vega)  if ticker.modelGreeks else 0.0,
                            "volume": int(ticker.volume) if ticker.volume else 0,
                        })
                        self._ib.cancelMktData(contract)
                    except Exception as opt_exc:
                        logger.debug(f"Options snapshot failed for {strike}{right}: {opt_exc}")
                        continue

            return pd.DataFrame(rows) if rows else None

        except Exception as exc:
            logger.warning(f"get_options_data IBKR failed at {date} {timestamp}: {exc}")
            return None

    def get_options_chain_at_time(self, date: str, timestamp: str,
                                  center_strike: float,
                                  strike_range: int = 300) -> Optional[pd.DataFrame]:
        """Return the options chain centred on center_strike (delegates to get_options_data)."""
        df = self.get_options_data(date, timestamp)
        if df is None:
            return None
        filtered = df[
            (df["strike"] >= center_strike - strike_range) &
            (df["strike"] <= center_strike + strike_range)
        ]
        return filtered if len(filtered) > 0 else None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def available_dates(self) -> List[str]:
        """Return the session's trade date when connected.

        Uses self._today (set in connect(), overrideable by callers) rather than
        recomputing datetime.now() so that sessions targeting a future date
        (e.g. started on Sunday for Monday's trading) are not rejected.
        """
        if self._connected and self._today:
            return [self._today]
        return []

    @property
    def is_connected(self) -> bool:
        return self._connected and bool(self._ib) and self._ib.isConnected()
