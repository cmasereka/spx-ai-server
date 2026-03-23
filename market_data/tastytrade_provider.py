"""
TastyTrade Real-Time Market Data Provider.

Provides live SPX price bars and SPXW 0DTE options chain snapshots via the
TastyTrade API using DXLink streaming.

Requires tastytrade >= 12.0 (OAuth2 Session API).

Architecture
------------
• A background thread runs a dedicated asyncio event loop.
• The DXLinkStreamer subscribes to:
    - Candle events on '$SPX' (1-minute interval) → fills _price_buffer ring buffer
    - Quote events on SPXW option streamer symbols → fills _options_cache bid/ask
    - Greeks events on SPXW option streamer symbols → fills _options_cache delta/greeks
• get_fastest_spx_price() reads from the price buffer (non-blocking).
• get_options_data() returns a snapshot DataFrame built from _options_cache.

Paper vs Live
-------------
Both paper and live accounts use the production session (api.tastyworks.com).
DXLink streaming requires a production session — the cert sandbox has no live
market data feed.  Paper trading is handled by using a paper account number.

Authentication
--------------
tastytrade v12+ uses OAuth2: provider_secret + refresh_token (no username/password).

Usage
-----
from market_data.tastytrade_provider import TastyTradeMarketDataProvider

provider = TastyTradeMarketDataProvider(
    provider_secret="your_provider_secret",
    refresh_token="your_refresh_token",
)
provider.connect()
price = provider.get_fastest_spx_price("2026-03-03", "10:00:00")
chain = provider.get_options_data("2026-03-03", "10:00:00")
provider.disconnect()
"""

import asyncio
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, date as date_type
from typing import Dict, List, Optional, Deque

import pandas as pd
import pytz
from loguru import logger

from .provider import MarketDataProvider

_ET = pytz.timezone("America/New_York")

try:
    from tastytrade import Session
    from tastytrade.instruments import NestedOptionChain
    from tastytrade.market_data import get_market_data_by_type
    from tastytrade.streamer import DXLinkStreamer
    from tastytrade.dxfeed import Quote, Greeks
    TT_AVAILABLE = True
except ImportError:
    TT_AVAILABLE = False
    logger.warning(
        "tastytrade is not installed. TastyTradeMarketDataProvider will not be available. "
        "Install it with: pip install tastytrade"
    )

# Seconds to wait for the first SPX price to arrive after subscribing
_CONNECT_WAIT_SECS = 60
# Number of minutes of SPX history to keep in the rolling buffer
_PRICE_BUFFER_MINUTES = 120
# Re-subscribe to options if SPX moves more than this many points from the
# initial subscription centre.
_OPTION_RESUBSCRIBE_THRESHOLD = 100


class TastyTradeMarketDataProvider(MarketDataProvider):
    """
    Real-time market data provider backed by the TastyTrade API (DXLink streaming).

    Thread Safety
    -------------
    All buffer reads/writes are protected by a threading.Lock.
    The DXLink async loop runs on its own background thread.
    """

    def __init__(self,
                 provider_secret: str,
                 refresh_token: str):
        if not TT_AVAILABLE:
            raise ImportError(
                "tastytrade must be installed to use TastyTradeMarketDataProvider. "
                "Run: pip install tastytrade"
            )
        self._provider_secret = provider_secret
        self._refresh_token = refresh_token

        self._session: Optional["Session"] = None
        self._connected = False
        self._today: str = ""

        # price_buffer[date_str][time_str] = float SPX close price
        self._price_buffer: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._price_keys: Deque = deque(maxlen=_PRICE_BUFFER_MINUTES)

        # options_cache[streamer_symbol] = dict with bid, ask, delta, gamma, theta, vega, iv
        self._options_cache: Dict[str, dict] = {}
        # Map streamer_symbol → {strike, option_type, expiration} for DataFrame construction
        self._symbol_meta: Dict[str, dict] = {}

        # SPX price at connect time — used to decide when to re-subscribe to options
        self._options_center_price: Optional[float] = None
        # Flag set by candle handler when options chain should be reloaded
        self._needs_chain_reload: bool = False

        self._lock = threading.Lock()
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Set once subscriptions are live (option chain loaded + DXLink channels open)
        self._stream_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Set by _stream_async if session creation or chain loading fails
        self._connect_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Start the DXLink streaming background thread (Session creation, option chain
        load, and subscriptions all happen inside the thread on one event loop).

        Returns True once DXLink subscriptions are live.  SPX price bars may not
        be present immediately if connecting outside market hours — they will
        arrive automatically when the market opens.
        """
        try:
            self._today = datetime.now(_ET).strftime("%Y-%m-%d")
            self._connect_error = None
            self._stream_ready.clear()
            logger.info("TastyTradeMarketDataProvider authenticating…")

            # Start background streaming thread.
            # Session creation and option chain loading happen inside _stream_async
            # so that everything runs on the same asyncio event loop.
            self._stop_event.clear()
            self._stream_thread = threading.Thread(
                target=self._run_stream_thread,
                daemon=True,
                name="tt-dxlink-stream",
            )
            self._stream_thread.start()

            # Wait up to _CONNECT_WAIT_SECS for the streaming subscriptions to open.
            # We do NOT require a price bar here — after-hours the SPX feed is silent
            # until market open.
            if not self._stream_ready.wait(timeout=_CONNECT_WAIT_SECS):
                if self._connect_error:
                    logger.error(
                        f"TastyTradeMarketDataProvider connection failed: "
                        f"{self._connect_error}"
                    )
                else:
                    logger.error(
                        "TastyTradeMarketDataProvider: DXLink subscriptions did not "
                        f"open within {_CONNECT_WAIT_SECS}s"
                    )
                return False

            if self._connect_error:
                logger.error(
                    f"TastyTradeMarketDataProvider connection failed: "
                    f"{self._connect_error}"
                )
                return False

            self._connected = True
            with self._lock:
                bar_count = len(self._price_buffer.get(self._today, {}))
            logger.info(
                f"TastyTradeMarketDataProvider connected "
                f"trade_date={self._today} "
                f"spx_bars={bar_count} "
                f"option_symbols={len(self._symbol_meta)}"
                + (" (no SPX bars yet — market closed)" if bar_count == 0 else "")
            )
            return True

        except Exception as exc:
            logger.error(f"TastyTradeMarketDataProvider connection failed: {exc}")
            return False

    def disconnect(self):
        """Stop the background stream thread and close the session."""
        self._stop_event.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=10)
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        self._session = None
        self._connected = False
        logger.info("TastyTradeMarketDataProvider disconnected")

    # ------------------------------------------------------------------
    # Background streaming thread
    # ------------------------------------------------------------------

    def _run_stream_thread(self):
        """Entry point for the background streaming thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._stream_async())
        except Exception as exc:
            logger.error(f"TastyTrade DXLink stream error: {exc}")
        finally:
            loop.close()

    async def _stream_async(self):
        """
        Main async streaming coroutine.  Creates the Session, loads the option
        chain, subscribes to SPXW 0DTE option quotes/greeks via DXLink, and
        polls the SPX spot price via REST every 30 seconds.

        DXLink candle subscriptions are not used — TastyTrade's server does not
        deliver candle events via the streaming API.  SPX price comes from the
        REST endpoint: GET /market-data/by-type?index=SPX.
        """
        try:
            # Always use the production session (is_test=False).
            # TastyTrade paper trading uses a paper account number on the production
            # API — the cert environment (is_test=True) is a developer sandbox with
            # no live market data feed, so DXLink streaming requires production.
            self._session = Session(
                self._provider_secret,
                self._refresh_token,
                is_test=False,
            )
            await self._load_option_chain_async()
        except Exception as exc:
            self._connect_error = str(exc)
            logger.error(f"TastyTrade session/chain init failed: {exc}")
            return

        with self._lock:
            option_symbols = list(self._symbol_meta.keys())

        try:
            async with DXLinkStreamer(self._session) as streamer:
                # Subscribe to option quotes and greeks
                if option_symbols:
                    logger.info(
                        f"TastyTrade: subscribing to {len(option_symbols)} option symbols"
                    )
                    await streamer.subscribe(Quote, option_symbols)
                    await streamer.subscribe(Greeks, option_symbols)

                # Signal connect() that subscriptions are live
                self._stream_ready.set()

                # Fan out to concurrent listeners
                tasks = [
                    asyncio.create_task(self._poll_spx_price()),
                    asyncio.create_task(self._listen_quotes(streamer)),
                    asyncio.create_task(self._listen_greeks(streamer)),
                    asyncio.create_task(self._chain_reload_monitor(streamer)),
                    asyncio.create_task(self._stop_watcher()),
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    if not self._stop_event.is_set():
                        name = getattr(t, 'get_name', lambda: repr(t))()
                        exc = t.exception() if not t.cancelled() else None
                        logger.warning(
                            f"TastyTrade: stream task completed unexpectedly — "
                            f"task={name} exc={exc}"
                        )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error(f"TastyTrade DXLink stream exception: {exc}")

    async def _stop_watcher(self):
        """Coroutine that returns once the stop event is set."""
        while not self._stop_event.is_set():
            await asyncio.sleep(1)

    async def _chain_reload_monitor(self, streamer: "DXLinkStreamer"):
        """
        Periodically checks the _needs_chain_reload flag set by _handle_candle.
        When set, reloads the option chain and resubscribes the streamer.
        """
        while not self._stop_event.is_set():
            await asyncio.sleep(5)
            if self._needs_chain_reload:
                self._needs_chain_reload = False
                try:
                    old_symbols = set(self._symbol_meta.keys())
                    await self._load_option_chain_async()
                    new_symbols = set(self._symbol_meta.keys())
                    added = new_symbols - old_symbols
                    removed = old_symbols - new_symbols
                    if removed:
                        await streamer.unsubscribe(Quote, list(removed))
                        await streamer.unsubscribe(Greeks, list(removed))
                    if added:
                        await streamer.subscribe(Quote, list(added))
                        await streamer.subscribe(Greeks, list(added))
                    logger.info(
                        f"TastyTrade: option chain reloaded "
                        f"(+{len(added)} / -{len(removed)} symbols)"
                    )
                except Exception as exc:
                    logger.error(f"TastyTrade chain reload failed: {exc}")

    async def _poll_spx_price(self):
        """
        Poll the SPX spot price via the TastyTrade REST API every 30 seconds
        and write it into _price_buffer[date][timestamp].

        Uses GET /market-data/by-type?index=SPX (symbol "SPX", not the DXFeed
        streaming symbol "$SPX.X").  This is the only reliable way to get the
        SPX cash index price from TastyTrade — DXLink candle subscriptions are
        silently ignored by the server.
        """
        logger.info("TastyTrade: SPX price poll started (REST, every 30s)")
        count = 0
        while not self._stop_event.is_set():
            try:
                results = await get_market_data_by_type(self._session, indices=["SPX"])
                spx_data = results[0] if results else None
                spx = float(spx_data.mark) if spx_data and float(spx_data.mark) > 0 else 0.0

                if spx > 0:
                    now_et = datetime.now(_ET)
                    date_str = now_et.strftime("%Y-%m-%d")
                    time_str = now_et.strftime("%H:%M") + ":00"

                    with self._lock:
                        self._price_buffer[date_str][time_str] = spx
                        self._price_keys.append((date_str, time_str))

                    count += 1
                    logger.info(f"TT SPX price: {date_str} {time_str} spx={spx:.2f}")

                    # Set centre price on first poll (for drift monitoring)
                    if self._options_center_price is None:
                        self._options_center_price = spx

                    # Check if options chain needs resubscription due to SPX drift
                    if abs(spx - self._options_center_price) > _OPTION_RESUBSCRIBE_THRESHOLD:
                        logger.info(
                            f"SPX moved {spx - self._options_center_price:+.0f} pts from "
                            f"options subscription centre — scheduling option chain reload"
                        )
                        self._options_center_price = spx
                        self._needs_chain_reload = True
                else:
                    logger.warning("TastyTrade: SPX REST poll returned 0 or unavailable")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.warning(f"TastyTrade SPX poll error: {exc}")

            await asyncio.sleep(30)

        logger.info(f"TastyTrade: SPX price poll stopped (polled {count} times)")

    async def _listen_quotes(self, streamer: "DXLinkStreamer"):
        """Process option Quote events → update _options_cache bid/ask."""
        try:
            async for event in streamer.listen(Quote):
                if self._stop_event.is_set():
                    break
                self._handle_quote(event)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error(f"TastyTrade quote listener error: {exc}")

    async def _listen_greeks(self, streamer: "DXLinkStreamer"):
        """Process option Greeks events → update _options_cache delta/greeks."""
        try:
            async for event in streamer.listen(Greeks):
                if self._stop_event.is_set():
                    break
                self._handle_greeks(event)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error(f"TastyTrade greeks listener error: {exc}")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_quote(self, event):
        """Handle a DXLink Quote event for an option symbol."""
        try:
            sym = getattr(event, "event_symbol", None)
            if sym is None or sym not in self._symbol_meta:
                return
            bid = float(getattr(event, "bid_price", 0) or 0)
            ask = float(getattr(event, "ask_price", 0) or 0)
            with self._lock:
                entry = self._options_cache.setdefault(sym, {})
                entry["bid"] = bid
                entry["ask"] = ask
        except Exception as exc:
            logger.debug(f"TT _handle_quote error: {exc}")

    def _handle_greeks(self, event):
        """Handle a DXLink Greeks event for an option symbol."""
        try:
            sym = getattr(event, "event_symbol", None)
            if sym is None or sym not in self._symbol_meta:
                return
            with self._lock:
                entry = self._options_cache.setdefault(sym, {})
                for field in ("delta", "gamma", "theta", "vega", "rho", "volatility"):
                    val = getattr(event, field, None)
                    if val is not None:
                        entry[field] = float(val)
        except Exception as exc:
            logger.debug(f"TT _handle_greeks error: {exc}")

    # ------------------------------------------------------------------
    # Option chain helpers
    # ------------------------------------------------------------------

    async def _load_option_chain_async(self):
        """
        Fetch today's SPXW 0DTE option chain via REST and populate _symbol_meta.
        Uses NestedOptionChain to get streamer symbols without fetching individual options.
        """
        if self._session is None:
            return
        try:
            today = date_type.today()
            chains = await NestedOptionChain.get(self._session, "SPXW")

            # Find today's expiry across all chains
            today_expir = None
            for chain in chains:
                for expir in chain.expirations:
                    if expir.expiration_date == today:
                        today_expir = expir
                        break
                if today_expir:
                    break

            if today_expir is None:
                logger.warning(
                    f"TastyTrade: no SPXW 0DTE options found for {today} — "
                    "market may be closed or it is not a trading day"
                )
                return

            with self._lock:
                self._symbol_meta.clear()
                for strike_entry in today_expir.strikes:
                    strike_price = float(strike_entry.strike_price)
                    exp_str = str(today)
                    if strike_entry.call_streamer_symbol:
                        self._symbol_meta[strike_entry.call_streamer_symbol] = {
                            "strike": strike_price,
                            "option_type": "call",
                            "expiration": exp_str,
                        }
                    if strike_entry.put_streamer_symbol:
                        self._symbol_meta[strike_entry.put_streamer_symbol] = {
                            "strike": strike_price,
                            "option_type": "put",
                            "expiration": exp_str,
                        }

            logger.info(
                f"TastyTrade: loaded {len(self._symbol_meta)} SPXW 0DTE option symbols "
                f"for {today}"
            )
        except Exception as exc:
            logger.error(f"TastyTrade _load_option_chain_async failed: {exc}")

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
            day_buf = dict(self._price_buffer.get(date, {}))
        if not day_buf:
            return None

        if timestamp in day_buf:
            return day_buf[timestamp]

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
        Return a DataFrame of 1-minute SPX bars for the given window.
        Only the 'close' column is guaranteed; matches the IBKRMarketDataProvider output.
        """
        with self._lock:
            day_buf = dict(self._price_buffer.get(date, {}))
        if not day_buf:
            return None

        rows = []
        try:
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M:%S")
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

        return pd.DataFrame(rows).sort_values("timestamp").set_index("timestamp")

    # ------------------------------------------------------------------
    # MarketDataProvider — options methods
    # ------------------------------------------------------------------

    def get_options_data(self, date: str, timestamp: str) -> Optional[pd.DataFrame]:
        """
        Return a snapshot of the SPXW 0DTE options chain built from the
        DXLink streaming cache.

        Required output columns: strike, option_type, bid, ask, delta
        """
        with self._lock:
            cache_copy = dict(self._options_cache)
            meta_copy = dict(self._symbol_meta)

        if not cache_copy:
            logger.warning(
                f"TastyTrade get_options_data: options cache empty for {date} {timestamp}"
            )
            return None

        rows = []
        for sym, meta in meta_copy.items():
            data = cache_copy.get(sym, {})
            bid = data.get("bid", 0.0)
            ask = data.get("ask", 0.0)
            if ask <= 0:
                continue
            rows.append({
                "strike": meta["strike"],
                "option_type": meta["option_type"],
                "expiration": meta["expiration"],
                "bid": bid,
                "ask": ask,
                "delta": data.get("delta", 0.0),
                "gamma": data.get("gamma", 0.0),
                "theta": data.get("theta", 0.0),
                "vega": data.get("vega", 0.0),
                "implied_volatility": data.get("volatility", 0.0),
            })

        if not rows:
            return None

        return pd.DataFrame(rows)

    def get_options_chain_at_time(self, date: str, timestamp: str,
                                  center_strike: float,
                                  strike_range: int = 300) -> Optional[pd.DataFrame]:
        """Return the options chain filtered to [center_strike ± strike_range]."""
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
        """Return today's date when connected; empty list otherwise."""
        if self._connected and self._today:
            return [self._today]
        return []

    @property
    def is_connected(self) -> bool:
        return self._connected
