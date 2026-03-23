"""
TastyTrade Broker Adapter — real order execution via TastyTrade REST API.

Submits SPXW credit spread and iron condor orders to TastyTrade as multi-leg
orders.  Supports both TastyTrade paper trading (production API with a paper
account number) and live trading (production API with a live account number).

Both paper and live accounts use the production TastyTrade API
(api.tastyworks.com). The cert sandbox (api.cert.tastyworks.com) is a developer
environment unrelated to paper trading and does not carry live positions.

Requires tastytrade >= 12.0 (OAuth2 Session API).

Order execution strategy
------------------------
1. Calculate per-share price from target_credit / target_debit.
2. Submit a limit order at that price.
3. Wait up to INITIAL_WAIT_SECS for a fill by polling order status.
4. If no fill, improve the price by IMPROVE_STEP toward the market and replace.
5. After MAX_IMPROVE_ATTEMPTS, switch to a market order as a last resort.
6. Time out after TOTAL_TIMEOUT_SECS — treat as failed fill.

Concurrency model
-----------------
All async operations (session creation, instrument lookup, order placement)
run on a single persistent background event loop started in connect().  This
avoids the "Event loop is closed" error that occurs when an httpx.AsyncClient
is reused across multiple short-lived asyncio.run() calls.

OCC symbol format for SPXW options
------------------------------------
Root (6 chars, space-padded) + YYMMDD + C/P + 8-digit strike (× 1000)
Example: "SPXW  260321P04800000"  →  SPXW put expiring 2026-03-21, strike 4800
"""

import asyncio
import os
import threading
import time
import uuid
from datetime import date as date_type
from decimal import Decimal
from typing import List, Optional, Dict

from loguru import logger

from .adapter import BrokerAdapter, OrderResult

try:
    from tastytrade import Session
    from tastytrade.account import Account
    from tastytrade.instruments import Option, InstrumentType
    from tastytrade.order import (
        NewOrder, OrderAction, OrderTimeInForce, OrderType, Leg, OrderStatus,
    )
    TT_AVAILABLE = True
except ImportError:
    TT_AVAILABLE = False
    logger.warning(
        "tastytrade is not installed. TastyTradeBrokerAdapter will not be available. "
        "Install it with: pip install tastytrade"
    )

# Order fill timing constants — mirrors ibkr_adapter.py for consistency
INITIAL_WAIT_SECS    = 30    # Wait this long before first price improvement
IMPROVE_STEP         = 0.05  # How much to move the limit price per improvement
MAX_IMPROVE_ATTEMPTS = 3     # Improvements before going to market
IMPROVE_WAIT_SECS    = 15    # Wait between improvement attempts
TOTAL_TIMEOUT_SECS   = 120   # Absolute timeout
POLL_INTERVAL_SECS   = 2     # How often to check order status


def _round_to_increment(price: float, increment: float = 0.05) -> float:
    """Round price to the nearest valid TastyTrade tick increment ($0.05)."""
    return round(round(price / increment) * increment, 2)


def _occ_symbol(strike: float, option_type: str, expiration) -> str:
    """
    Build the OCC option symbol used by TastyTrade's REST API.

    Format: {root:<6}{YYMMDD}{C|P}{strike*1000:08d}
    Example: strike=4800.0, put, 2026-03-21  →  "SPXW  260321P04800000"

    Accepts both date and datetime for expiration.
    """
    root = "SPXW"
    exp = expiration.strftime("%y%m%d")
    right = "C" if "call" in str(option_type).lower() else "P"
    strike_int = int(round(strike * 1000))
    return f"{root:<6}{exp}{right}{strike_int:08d}"


class TastyTradeBrokerAdapter(BrokerAdapter):
    """
    Real order execution adapter for TastyTrade paper / live trading.

    Parameters
    ----------
    provider_secret:  OAuth2 provider secret (client secret).
    refresh_token:    OAuth2 refresh token for the user.
    account_number:   TastyTrade account number (e.g. '5WT00000').
                      Both paper and live accounts use the production API.
    """

    def __init__(self,
                 provider_secret: str,
                 refresh_token: str,
                 account_number: str,
                 session_data: Optional[str] = None):
        if not TT_AVAILABLE:
            raise ImportError(
                "tastytrade must be installed to use TastyTradeBrokerAdapter. "
                "Run: pip install tastytrade"
            )
        self._provider_secret = provider_secret
        self._refresh_token = refresh_token
        self._account_number = account_number

        # Serialized Session JSON from the market data provider, if available.
        # TastyTrade rotates refresh tokens on first use, so two independent
        # Session() calls with the same token cause the second to fail.  Instead
        # we serialize the already-authenticated Session (which carries a valid
        # access_token), then deserialize it inside _connect_async on our own
        # event loop — giving us a fresh httpx.AsyncClient and asyncio.Lock bound
        # to the correct loop while skipping the refresh-token exchange entirely.
        self._session_data: Optional[str] = session_data
        self._session: Optional["Session"] = None
        self._account: Optional["Account"] = None
        self._connected = False

        # Persistent background event loop — all async ops run here
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        # Cache of OCC symbol → Option instrument to avoid repeated REST calls
        self._instrument_cache: Dict[str, "Option"] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, **kwargs) -> bool:
        """
        Authenticate with TastyTrade, resolve the trading account, and start
        the persistent background event loop used for all async operations.
        """
        logger.info(
            f"TastyTradeBrokerAdapter.connect() called "
            f"account={self._account_number!r} "
            f"has_session_data={self._session_data is not None}"
        )
        try:
            # Start a persistent background event loop thread.
            # Session and Account are created inside this loop so that the
            # httpx.AsyncClient is bound to the correct loop for all subsequent calls.
            self._loop = asyncio.new_event_loop()
            ready_event = threading.Event()

            def _run_loop():
                asyncio.set_event_loop(self._loop)
                self._loop.call_soon(ready_event.set)
                self._loop.run_forever()

            self._loop_thread = threading.Thread(
                target=_run_loop, daemon=True, name="tt-broker-loop"
            )
            self._loop_thread.start()
            ready_event.wait(timeout=10)

            # Create Session + resolve Account on the persistent loop
            future = asyncio.run_coroutine_threadsafe(
                self._connect_async(), self._loop
            )
            future.result(timeout=30)

            logger.info(
                f"TastyTradeBrokerAdapter connected "
                f"account={self._account.account_number}"
            )
            return True

        except Exception as exc:
            logger.error(f"TastyTradeBrokerAdapter connection failed: {exc}")
            self._stop_loop()
            return False

    async def _connect_async(self):
        """
        Create (or restore) the TastyTrade Session and resolve the Account object.
        Must run on the persistent background loop.

        If session_data was provided (serialized from the market data provider after
        it already authenticated), we deserialize it here — this creates a fresh
        httpx.AsyncClient and asyncio.Lock on the current loop while reusing the
        still-valid access_token, so no refresh-token exchange is needed.

        Otherwise a new Session is created, which will exchange the refresh_token
        for an access_token on the first REST call.
        """
        if self._session_data is not None:
            logger.info("TastyTradeBrokerAdapter: deserializing shared session...")
            self._session = Session.deserialize(self._session_data)
            logger.info("TastyTradeBrokerAdapter: session deserialized, fetching account...")
        else:
            logger.info("TastyTradeBrokerAdapter: creating new session (no shared session data)...")
            self._session = Session(
                self._provider_secret,
                self._refresh_token,
                is_test=False,
            )
            logger.info("TastyTradeBrokerAdapter: new session created, fetching account...")
        self._account = await Account.get(self._session, self._account_number)
        logger.info(f"TastyTradeBrokerAdapter: account resolved: {self._account_number}")
        self._connected = True

    def disconnect(self):
        """Invalidate the session and stop the background event loop."""
        self._connected = False
        self._session = None
        self._stop_loop()
        logger.info("TastyTradeBrokerAdapter disconnected")

    def _stop_loop(self):
        """Stop the persistent background event loop (if running)."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
        self._loop = None
        self._loop_thread = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Internal helper: submit a coroutine to the persistent loop
    # ------------------------------------------------------------------

    def _run(self, coro, timeout: float = TOTAL_TIMEOUT_SECS + 30):
        """
        Submit an async coroutine to the persistent background loop and block
        until it completes.  All Session / Account usage runs on the same loop
        the objects were created on, preventing 'Event loop is closed' errors.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(
                "TastyTradeBrokerAdapter: background event loop is not running. "
                "Did you call connect()?"
            )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # BrokerAdapter interface
    # ------------------------------------------------------------------

    def open_position(self, strategy, quantity: int, timestamp: str,
                      target_credit: float) -> OrderResult:
        """Submit a credit-collecting opening multi-leg order."""
        return self._execute_order(
            strategy=strategy,
            quantity=quantity,
            timestamp=timestamp,
            is_entry=True,
            target_price=target_credit,
        )

    def close_position(self, strategy, quantity: int, timestamp: str,
                       target_debit: float) -> OrderResult:
        """Submit a debit-paying closing multi-leg order."""
        return self._execute_order(
            strategy=strategy,
            quantity=quantity,
            timestamp=timestamp,
            is_entry=False,
            target_price=target_debit,
        )

    def close_all(self, open_strategies: List, timestamp: str) -> List[OrderResult]:
        """Force-close all open positions at market price."""
        results = []
        for strat in open_strategies:
            try:
                result = self._execute_order(
                    strategy=strat,
                    quantity=getattr(strat, "quantity", 1),
                    timestamp=timestamp,
                    is_entry=False,
                    target_price=0.0,
                    use_market=True,
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"TastyTradeBrokerAdapter.close_all failed: {exc}")
                results.append(OrderResult(
                    order_id=str(uuid.uuid4()),
                    symbol="SPXW",
                    fill_price=0.0,
                    limit_price=0.0,
                    quantity=getattr(strat, "quantity", 1),
                    strategy_type=str(getattr(strat, "strategy_type", "unknown")),
                    is_entry=False,
                    timestamp=timestamp,
                    success=False,
                    error_message=str(exc),
                ))
        return results

    # ------------------------------------------------------------------
    # Internal order construction and submission
    # ------------------------------------------------------------------

    def _execute_order(self, strategy, quantity: int, timestamp: str,
                       is_entry: bool, target_price: float,
                       use_market: bool = False) -> OrderResult:
        """
        Build and submit a multi-leg limit (or market) order for the given strategy.

        target_price is the total dollar amount (entry_credit or exit_cost × 100 × qty).
        We convert to per-share price before submission.
        """
        strategy_type = getattr(strategy, "strategy_type", "unknown")
        if hasattr(strategy_type, "value"):
            strategy_type = strategy_type.value
        strategy_type = str(strategy_type)

        try:
            legs = self._strategy_to_legs(strategy, is_entry)
            if not legs:
                raise ValueError("Could not build order legs from strategy")

            # entry_credit / exit_cost are TOTAL dollars (price × 100 × qty, minus commissions).
            # Divide back to per-share, then round to the $0.05 tick required by TastyTrade.
            per_share_price = _round_to_increment(abs(target_price) / (100.0 * max(quantity, 1)))
            fill_price = self._run(
                self._submit_with_retry_async(
                    legs=legs,
                    per_share_price=per_share_price,
                    is_entry=is_entry,
                    use_market=use_market,
                )
            )

            return OrderResult(
                order_id=str(uuid.uuid4()),
                symbol="SPXW",
                fill_price=fill_price if fill_price is not None else per_share_price,
                limit_price=per_share_price,
                quantity=quantity,
                strategy_type=strategy_type,
                is_entry=is_entry,
                timestamp=timestamp,
                success=fill_price is not None,
                error_message=None if fill_price is not None else "Order timed out without fill",
            )

        except Exception as exc:
            logger.error(f"TastyTradeBrokerAdapter order failed: {exc}")
            fallback = _round_to_increment(abs(target_price) / (100.0 * max(quantity, 1)))
            return OrderResult(
                order_id=str(uuid.uuid4()),
                symbol="SPXW",
                fill_price=fallback,
                limit_price=fallback,
                quantity=quantity,
                strategy_type=strategy_type,
                is_entry=is_entry,
                timestamp=timestamp,
                success=False,
                error_message=str(exc),
            )

    def _strategy_to_legs(self, strategy, is_entry: bool) -> List["Leg"]:
        """Convert an OptionsStrategy object into a list of TastyTrade Leg objects."""
        legs_raw = getattr(strategy, "legs", None)
        if not legs_raw:
            return []

        expiration = getattr(strategy, "expiration", date_type.today())
        # Normalise datetime → date so _occ_symbol always gets a date object
        if hasattr(expiration, "date") and callable(expiration.date):
            expiration = expiration.date()

        result_legs = []
        for leg in legs_raw:
            strike = float(leg.strike)
            option_type = str(getattr(leg, "option_type", "put")).lower()
            position_side = str(getattr(leg, "position_side", "LONG")).upper()

            instrument = self._get_option_instrument(strike, option_type, expiration)
            if instrument is None:
                raise ValueError(
                    f"Could not resolve TastyTrade instrument for "
                    f"SPXW {expiration} {strike} {option_type}"
                )

            is_short = "SHORT" in position_side
            if is_entry:
                action = OrderAction.SELL_TO_OPEN if is_short else OrderAction.BUY_TO_OPEN
            else:
                action = OrderAction.BUY_TO_CLOSE if is_short else OrderAction.SELL_TO_CLOSE

            result_legs.append(instrument.build_leg(quantity=1, action=action))

        return result_legs

    def _get_option_instrument(self, strike: float, option_type: str,
                               expiration) -> Optional["Option"]:
        """
        Resolve a TastyTrade Option instrument for the given strike/type/expiry.
        Results are cached to avoid repeated REST calls during a session.
        Uses the persistent event loop so the Session's httpx client is consistent.
        """
        occ = _occ_symbol(strike, option_type, expiration)
        if occ in self._instrument_cache:
            return self._instrument_cache[occ]

        try:
            result = self._run(Option.get(self._session, occ))
            instrument = result[0] if isinstance(result, list) else result
            if instrument is not None:
                self._instrument_cache[occ] = instrument
            return instrument
        except Exception as exc:
            logger.warning(f"Could not resolve instrument for {occ}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Async order submission (runs on the persistent background loop)
    # ------------------------------------------------------------------

    async def _submit_with_retry_async(self, legs: List["Leg"], per_share_price: float,
                                       is_entry: bool,
                                       use_market: bool = False) -> Optional[float]:
        """
        Submit a multi-leg order and retry with price improvements if not filled.
        Returns the fill price on success, or None if the order timed out.
        """
        if use_market:
            return await self._place_aggressive_limit_async(legs, is_entry)

        current_price = per_share_price
        deadline = time.time() + TOTAL_TIMEOUT_SECS
        current_order_id: Optional[int] = None

        for attempt in range(MAX_IMPROVE_ATTEMPTS + 1):
            wait_secs = INITIAL_WAIT_SECS if attempt == 0 else IMPROVE_WAIT_SECS
            fill, current_order_id = await self._place_and_poll_async(
                legs=legs,
                price=current_price,
                timeout=min(wait_secs, max(0, deadline - time.time())),
                existing_order_id=current_order_id if attempt > 0 else None,
            )
            if fill is not None:
                return fill

            if time.time() >= deadline:
                break

            if attempt < MAX_IMPROVE_ATTEMPTS:
                # Move price toward market (keep on $0.05 grid)
                if is_entry:
                    current_price = _round_to_increment(max(0.05, current_price - IMPROVE_STEP))
                else:
                    current_price = _round_to_increment(current_price + IMPROVE_STEP)
                logger.info(
                    f"TastyTradeBrokerAdapter: no fill after {wait_secs}s, "
                    f"improving price to {current_price:.2f} (attempt {attempt + 1})"
                )

        # Last resort: cancel any open order, then use aggressive limit
        # (TastyTrade does not support Market orders for multi-leg strategies)
        if current_order_id is not None:
            try:
                await self._account.delete_order(self._session, current_order_id)
            except Exception:
                pass

        logger.warning("TastyTradeBrokerAdapter: falling back to aggressive limit order")
        return await self._place_aggressive_limit_async(legs, is_entry)

    async def _place_and_poll_async(
        self, legs: List["Leg"], price: float, timeout: float,
        existing_order_id: Optional[int] = None,
    ):
        """
        Place (or replace) a limit order and poll until filled or timeout.

        Returns (fill_price, order_id).  fill_price is None if not filled within timeout.
        """
        order_id: Optional[int] = None
        try:
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.LIMIT,
                price=Decimal(str(round(price, 2))),
                legs=legs,
            )

            if existing_order_id is not None:
                # Price improvement: replace the existing open order
                placed = await self._account.replace_order(
                    self._session, existing_order_id, order
                )
                order_id = placed.id
            else:
                response = await self._account.place_order(
                    self._session, order, dry_run=False
                )
                order_id = response.order.id

            deadline = time.time() + timeout
            while time.time() < deadline:
                await asyncio.sleep(POLL_INTERVAL_SECS)
                fetched = await self._account.get_order(self._session, order_id)
                if fetched.status == OrderStatus.FILLED:
                    fill = float(fetched.price or price)
                    logger.info(f"TastyTrade order {order_id} filled at {fill:.4f}")
                    return fill, order_id
                if fetched.status in (
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REMOVED,
                ):
                    logger.warning(
                        f"TastyTrade order {order_id} terminal status: {fetched.status}"
                    )
                    return None, None

        except Exception as exc:
            logger.error(f"TastyTrade _place_and_poll_async failed: {exc}")

        # Timed out — return the open order_id so caller can replace it
        return None, order_id

    async def _place_aggressive_limit_async(self, legs: List["Leg"],
                                             is_entry: bool) -> Optional[float]:
        """
        Place an aggressively-priced limit order as a last resort.

        TastyTrade does not allow Market orders for multi-leg strategies.
        Instead we use a limit price that is almost certain to fill immediately:
          - Entry (selling for credit): $0.05 minimum credit
          - Close  (buying for debit): $9.95 maximum debit
        """
        aggressive_price = Decimal("0.05") if is_entry else Decimal("9.95")
        try:
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.LIMIT,
                price=aggressive_price,
                legs=legs,
            )
            response = await self._account.place_order(
                self._session, order, dry_run=False
            )
            order_id = response.order.id

            deadline = time.time() + TOTAL_TIMEOUT_SECS
            while time.time() < deadline:
                await asyncio.sleep(POLL_INTERVAL_SECS)
                fetched = await self._account.get_order(self._session, order_id)
                if fetched.status == OrderStatus.FILLED:
                    fill = float(fetched.price or float(aggressive_price))
                    logger.info(
                        f"TastyTrade aggressive limit order {order_id} filled at {fill:.4f}"
                    )
                    return fill
                if fetched.status in (
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REMOVED,
                ):
                    logger.warning(
                        f"TastyTrade aggressive limit order {order_id} status: {fetched.status}"
                    )
                    break

        except Exception as exc:
            logger.error(f"TastyTrade aggressive limit order failed: {exc}")

        return None
