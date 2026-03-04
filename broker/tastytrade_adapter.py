"""
TastyTrade Broker Adapter — real order execution via TastyTrade REST API.

Submits SPXW credit spread and iron condor orders to TastyTrade as multi-leg
orders.  Supports both TastyTrade paper trading (certification environment at
api.cert.tastyworks.com) and live trading (api.tastyworks.com).

Requires tastytrade >= 12.0 (OAuth2 Session API).

Order execution strategy
------------------------
1. Calculate per-share price from target_credit / target_debit.
2. Submit a limit order at that price.
3. Wait up to INITIAL_WAIT_SECS for a fill by polling order status.
4. If no fill, improve the price by IMPROVE_STEP toward the market and replace.
5. After MAX_IMPROVE_ATTEMPTS, switch to a market order as a last resort.
6. Time out after TOTAL_TIMEOUT_SECS — treat as failed fill.

Paper vs Live
-------------
is_paper=True  → Session(is_test=True)   (api.cert.tastyworks.com)
is_paper=False → Session(is_test=False)  (api.tastyworks.com)

Authentication
--------------
tastytrade v12+ uses OAuth2: provider_secret + refresh_token (no username/password).

OCC symbol format for SPXW options
------------------------------------
Root (6 chars, space-padded) + YYMMDD + C/P + 8-digit strike (× 1000)
Example: "SPXW  260321P04800000"  →  SPXW put expiring 2026-03-21, strike 4800
"""

import asyncio
import concurrent.futures
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


def _occ_symbol(strike: float, option_type: str, expiration: date_type) -> str:
    """
    Build the OCC option symbol used by TastyTrade's REST API.

    Format: {root:<6}{YYMMDD}{C|P}{strike*1000:08d}
    Example: strike=4800.0, put, 2026-03-21  →  "SPXW  260321P04800000"
    """
    root = "SPXW"
    exp = expiration.strftime("%y%m%d")
    right = "C" if "call" in str(option_type).lower() else "P"
    strike_int = int(round(strike * 1000))
    return f"{root:<6}{exp}{right}{strike_int:08d}"


def _run_in_new_loop(coro):
    """
    Run an async coroutine in a brand-new thread with its own event loop.

    This is safe to call from any context — including threads that already
    have a running event loop — because the coroutine runs on a completely
    separate thread/loop.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


class TastyTradeBrokerAdapter(BrokerAdapter):
    """
    Real order execution adapter for TastyTrade paper / live trading.

    Parameters
    ----------
    provider_secret:  OAuth2 provider secret (client secret).
    refresh_token:    OAuth2 refresh token for the user.
    account_number:   TastyTrade account number (e.g. '5WT00000').
    is_paper:         True  → certification API (paper money).
                      False → production API (real money).
    """

    def __init__(self,
                 provider_secret: str,
                 refresh_token: str,
                 account_number: str,
                 is_paper: bool = True):
        if not TT_AVAILABLE:
            raise ImportError(
                "tastytrade must be installed to use TastyTradeBrokerAdapter. "
                "Run: pip install tastytrade"
            )
        self._provider_secret = provider_secret
        self._refresh_token = refresh_token
        self._account_number = account_number
        self._is_paper = is_paper

        self._session: Optional["Session"] = None
        self._account: Optional["Account"] = None
        self._connected = False

        # Cache of OCC symbol → Option instrument to avoid repeated REST round-trips
        self._instrument_cache: Dict[str, "Option"] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, **kwargs) -> bool:
        """Authenticate with TastyTrade and resolve the trading account."""
        try:
            env = "paper (cert)" if self._is_paper else "live (prod)"
            self._session = Session(
                self._provider_secret,
                self._refresh_token,
                is_test=self._is_paper,
            )
            account = _run_in_new_loop(
                Account.get(self._session, self._account_number)
            )
            self._account = account
            self._connected = True
            logger.info(
                f"TastyTradeBrokerAdapter connected [{env}] "
                f"account={self._account.account_number}"
            )
            return True

        except Exception as exc:
            logger.error(f"TastyTradeBrokerAdapter connection failed: {exc}")
            return False

    def disconnect(self):
        """Invalidate the session (best-effort)."""
        try:
            self._session = None
        except Exception:
            pass
        self._connected = False
        logger.info("TastyTradeBrokerAdapter disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

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

            # Convert total dollar amount → per-share price (matches ibkr_adapter convention)
            per_share_price = abs(target_price) / (100.0 * max(quantity, 1))
            fill_price = _run_in_new_loop(
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
            fallback = abs(target_price) / (100.0 * max(quantity, 1))
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
                               expiration: date_type) -> Optional["Option"]:
        """
        Resolve a TastyTrade Option instrument for the given strike/type/expiry.
        Results are cached to avoid repeated REST calls during a session.
        """
        occ = _occ_symbol(strike, option_type, expiration)
        if occ in self._instrument_cache:
            return self._instrument_cache[occ]

        try:
            result = _run_in_new_loop(Option.get(self._session, occ))
            instrument = result[0] if isinstance(result, list) else result
            if instrument is not None:
                self._instrument_cache[occ] = instrument
            return instrument
        except Exception as exc:
            logger.warning(f"Could not resolve instrument for {occ}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Async order submission (runs inside _run_in_new_loop)
    # ------------------------------------------------------------------

    async def _submit_with_retry_async(self, legs: List["Leg"], per_share_price: float,
                                       is_entry: bool,
                                       use_market: bool = False) -> Optional[float]:
        """
        Submit a multi-leg order and retry with price improvements if not filled.
        Returns the fill price on success, or None if the order timed out.
        """
        if use_market:
            return await self._place_market_order_async(legs)

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
                # Move price toward market
                if is_entry:
                    current_price = max(0.01, current_price - IMPROVE_STEP)
                else:
                    current_price = current_price + IMPROVE_STEP
                logger.info(
                    f"TastyTradeBrokerAdapter: no fill after {wait_secs}s, "
                    f"improving price to {current_price:.2f} (attempt {attempt + 1})"
                )

        # Last resort: cancel any open order, then go market
        if current_order_id is not None:
            try:
                await self._account.delete_order(self._session, current_order_id)
            except Exception:
                pass

        logger.warning("TastyTradeBrokerAdapter: falling back to market order")
        return await self._place_market_order_async(legs)

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

    async def _place_market_order_async(self, legs: List["Leg"]) -> Optional[float]:
        """Place a market order and return the fill price."""
        try:
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.MARKET,
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
                    fill = float(fetched.price or 0.0)
                    logger.info(f"TastyTrade market order {order_id} filled at {fill:.4f}")
                    return fill
                if fetched.status in (
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REMOVED,
                ):
                    logger.warning(
                        f"TastyTrade market order {order_id} status: {fetched.status}"
                    )
                    break

        except Exception as exc:
            logger.error(f"TastyTrade market order failed: {exc}")

        return None
