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
import pytz
from collections import defaultdict
from datetime import datetime, date as date_type
from typing import Optional, List, Dict, Deque

_ET = pytz.timezone("America/New_York")
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
        # The caller (_run_session_in_thread) sets a fresh event loop on this
        # thread before calling connect(), so we just verify one exists.
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            self._ib = IB()
            # Retry up to 3 times — previous session's clientId slot may still
            # be held by the Gateway for a few seconds after disconnect.
            last_exc = None
            for attempt in range(1, 4):
                try:
                    self._ib.connect(self.host, self.port, clientId=self.client_id,
                                     timeout=timeout)
                    if self._ib.isConnected():
                        break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        f"IBKR connect attempt {attempt}/3 failed: {exc} — "
                        f"waiting 5s before retry"
                    )
                    time.sleep(5)
            else:
                raise last_exc or ConnectionError("IBKR connect failed after 3 attempts")
            self._connected = self._ib.isConnected()
            if self._connected:
                # Request live market data (requires active CBOE options subscription).
                self._ib.reqMarketDataType(1)
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
        """Subscribe to 5-second real-time bars for SPX and backfill today's 1-min history."""
        if not self._ib or not self._spx_contract:
            logger.error("_subscribe_spx_bars: ib or spx_contract not set")
            return
        logger.info("Requesting SPX real-time bars subscription...")
        try:
            bars = self._ib.reqRealTimeBars(
                contract=self._spx_contract,
                barSize=5,
                whatToShow="TRADES",   # SPX Index uses TRADES (= calculated index value)
                useRTH=True,
            )
            bars.updateEvent += self._on_rtbar
            logger.info("Subscribed to SPX real-time bars (5s, MIDPOINT)")
        except Exception as exc:
            logger.error(f"Failed to subscribe to SPX real-time bars: {exc}")

        # Backfill today's 1-minute bars so that opening price and drift guards
        # work correctly when connecting mid-session.
        logger.info("Requesting historical 1-min SPX bars for backfill...")
        try:
            # Small pause to avoid IBKR pacing violation after reqRealTimeBars
            self._ib.sleep(1)
            hist_bars = self._ib.reqHistoricalData(
                contract=self._spx_contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",   # SPX Index uses TRADES for historical data
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
                timeout=30,
            )
            logger.info(f"reqHistoricalData returned {len(hist_bars) if hist_bars else 0} raw bars")
            count = 0
            for bar in hist_bars:
                # bar.date can be a datetime object or a string depending on formatDate.
                # Convert to ET so buffer keys match _wait_for_bar.
                if isinstance(bar.date, datetime):
                    if bar.date.tzinfo is not None:
                        bar_dt = bar.date.astimezone(_ET).replace(tzinfo=None)
                    else:
                        # Assume IBKR returned local (server) time — convert via local tz
                        bar_dt = pytz.timezone("America/Chicago").localize(bar.date).astimezone(_ET).replace(tzinfo=None)
                else:
                    try:
                        bar_dt = datetime.strptime(bar.date, "%Y%m%d %H:%M:%S")
                    except ValueError:
                        try:
                            bar_dt = datetime.strptime(bar.date, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            logger.warning(f"Unrecognised bar date format: {bar.date!r}")
                            continue
                date_str = bar_dt.strftime("%Y-%m-%d")
                time_str = bar_dt.strftime("%H:%M:%S")
                price = float(bar.close)
                if price > 0:
                    with self._lock:
                        self._price_buffer[date_str][time_str] = price
                        self._price_keys.append((date_str, time_str))
                    count += 1
            logger.info(f"Backfilled {count} historical 1-min SPX bars into buffer")
            if count > 0:
                # Log a sample so we can confirm dates/times are correct
                sample_date = list(self._price_buffer.keys())[-1]
                sample_times = sorted(self._price_buffer[sample_date].keys())
                logger.info(
                    f"Buffer sample — date={sample_date} "
                    f"first={sample_times[0]} last={sample_times[-1]} "
                    f"bars={len(sample_times)}"
                )
        except Exception as exc:
            logger.error(f"Historical backfill failed: {exc}")

    def _on_rtbar(self, bars, has_new_bar: bool):
        """Callback invoked by ib_insync on each new 5-second bar."""
        if not has_new_bar or not bars:
            return
        bar = bars[-1]
        # bar.time is a datetime when using TRADES, or a unix int with MIDPOINT.
        # Convert to ET so buffer keys match _wait_for_bar which uses ET.
        if isinstance(bar.time, datetime):
            bar_dt = bar.time.astimezone(_ET).replace(tzinfo=None)
        else:
            bar_dt = datetime.fromtimestamp(bar.time, tz=_ET).replace(tzinfo=None)
        date_str = bar_dt.strftime("%Y-%m-%d")
        # Snap to the nearest minute
        time_str = bar_dt.strftime("%H:%M") + ":00"
        close_price = float(bar.close)
        if close_price <= 0:
            return
        with self._lock:
            self._price_buffer[date_str][time_str] = close_price
            self._price_keys.append((date_str, time_str))
        logger.debug(f"SPX bar: {date_str} {time_str} close={close_price:.2f}")

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
            logger.warning(f"get_options_data: no SPX price in buffer for {date} {timestamp}")
            return None

        try:
            # Build list of strikes to query (±150 pts, 5-pt intervals for SPXW)
            low  = int(spx_price - 150)
            high = int(spx_price + 155)
            strikes = list(range(low - (low % 5), high, 5))

            expiry = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
            T = _time_to_expiry_years(timestamp, date)

            logger.debug(
                f"IBKR options fetch: SPX={spx_price:.2f} center={round(spx_price/5)*5} "
                f"strikes={len(strikes)} expiry={expiry} @ {timestamp}"
            )

            # Submit all snapshot requests up-front, then wait once for IBKR to respond.
            # snapshot=True with a single ib.sleep(0) per contract is unreliable —
            # IBKR needs ~2-3 s to push all responses after the batch is submitted.
            pending: list = []
            for strike in strikes:
                for right in ("P", "C"):
                    contract = Option("SPX", expiry, strike, right,
                                      exchange="CBOE", currency="USD",
                                      multiplier="100",
                                      tradingClass="SPXW")
                    try:
                        ticker = self._ib.reqMktData(contract, "", True, False)
                        pending.append((strike, right, contract, ticker))
                    except Exception as opt_exc:
                        logger.debug(f"reqMktData failed for {strike}{right}: {opt_exc}")

            # Wait for IBKR to push all snapshot responses
            self._ib.sleep(3)

            rows = []
            for strike, right, contract, ticker in pending:
                try:
                    bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
                    ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
                    # snapshot=True subscriptions are auto-cancelled by IBKR once the
                    # snapshot arrives — do NOT call cancelMktData() or we get Error 300.
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
                        "volume": int(ticker.volume) if ticker.volume and ticker.volume == ticker.volume else 0,
                    })
                except Exception as opt_exc:
                    logger.debug(f"Options snapshot failed for {strike}{right}: {opt_exc}")
                    continue

            logger.debug(
                f"IBKR options fetch complete: {len(rows)}/{len(pending)} contracts had quotes"
            )
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
