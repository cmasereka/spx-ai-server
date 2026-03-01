"""
IBKR Broker Adapter — real order execution via TWS / IB Gateway.

Submits SPXW credit spread and iron condor orders to IBKR as combo (BAG)
contracts.  Works for both IBKR paper trading accounts (port 7497) and live
accounts (port 7496) — the only difference is the port number.

Order execution strategy
------------------------
1. Calculate mid-price of the combo from current bid/ask.
2. Submit a limit order at mid-price.
3. Wait up to INITIAL_WAIT_SECS for a fill.
4. If no fill, improve the price by IMPROVE_STEP toward the market and resubmit.
5. After MAX_IMPROVE_ATTEMPTS, switch to a market order as a last resort.
6. Time out after TOTAL_TIMEOUT_SECS — treat as failed fill.

The IBKR connection is shared with IBKRMarketDataProvider (same IB instance).
Pass the IB instance to the constructor to avoid creating two connections.
"""

import time
import uuid
from typing import List, Optional

from loguru import logger

from .adapter import BrokerAdapter, OrderResult

try:
    from ib_insync import (
        IB, Contract, Order, ComboLeg, TagValue,
        LimitOrder, MarketOrder,
    )
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

# Order fill timing constants
INITIAL_WAIT_SECS   = 20   # Wait this long for initial limit order fill
IMPROVE_STEP        = 0.05  # How much to move the limit price per improvement step
MAX_IMPROVE_ATTEMPTS = 3   # Number of price improvements before going to market
IMPROVE_WAIT_SECS   = 10   # Wait between each improvement attempt
TOTAL_TIMEOUT_SECS  = 90   # Give up after this many seconds


class IBKRBrokerAdapter(BrokerAdapter):
    """
    Real order execution adapter for IBKR paper / live trading.

    Parameters
    ----------
    ib:      A connected ib_insync.IB instance (shared with IBKRMarketDataProvider).
    account: IBKR account string (e.g. 'DU123456' for paper, 'U123456' for live).
    """

    def __init__(self, ib: "IB", account: str = ""):
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync must be installed to use IBKRBrokerAdapter. "
                "Run: pip install ib_insync"
            )
        self._ib = ib
        self._account = account

    # ------------------------------------------------------------------
    # BrokerAdapter implementation
    # ------------------------------------------------------------------

    def open_position(self, strategy, quantity: int, timestamp: str,
                      target_credit: float) -> OrderResult:
        """Submit a credit-collecting opening combo order."""
        return self._execute_combo_order(
            strategy=strategy,
            quantity=quantity,
            timestamp=timestamp,
            is_entry=True,
            target_price=target_credit,
        )

    def close_position(self, strategy, quantity: int, timestamp: str,
                       target_debit: float) -> OrderResult:
        """Submit a debit-paying closing combo order."""
        return self._execute_combo_order(
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
                result = self._execute_combo_order(
                    strategy=strat,
                    quantity=getattr(strat, "quantity", 1),
                    timestamp=timestamp,
                    is_entry=False,
                    target_price=0.0,
                    use_market=True,
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"close_all failed for strategy: {exc}")
                results.append(OrderResult(
                    order_id=str(uuid.uuid4()),
                    symbol="SPXW",
                    fill_price=0.0,
                    limit_price=0.0,
                    quantity=getattr(strat, "quantity", 1),
                    strategy_type="unknown",
                    is_entry=False,
                    timestamp=timestamp,
                    success=False,
                    error_message=str(exc),
                ))
        return results

    @property
    def is_connected(self) -> bool:
        return bool(self._ib) and self._ib.isConnected()

    # ------------------------------------------------------------------
    # Internal order construction and submission
    # ------------------------------------------------------------------

    def _execute_combo_order(self, strategy, quantity: int, timestamp: str,
                              is_entry: bool, target_price: float,
                              use_market: bool = False) -> OrderResult:
        """
        Build and submit a BAG (combo) order for the given OptionsStrategy.

        For an opening (credit) order: action = "SELL" (collecting credit).
        For a closing (debit) order:   action = "BUY"  (paying debit).
        """
        strategy_type = getattr(strategy, "strategy_type", "unknown")
        if hasattr(strategy_type, "value"):
            strategy_type = strategy_type.value
        strategy_type = str(strategy_type)

        try:
            combo = self._build_combo_contract(strategy)
            if combo is None:
                raise ValueError("Could not build combo contract from strategy legs")

            action = "SELL" if is_entry else "BUY"
            fill_price = self._submit_with_retry(
                combo=combo,
                action=action,
                quantity=quantity,
                target_price=abs(target_price),
                use_market=use_market,
            )

            return OrderResult(
                order_id=str(uuid.uuid4()),
                symbol="SPXW",
                fill_price=fill_price if fill_price is not None else target_price,
                limit_price=target_price,
                quantity=quantity,
                strategy_type=strategy_type,
                is_entry=is_entry,
                timestamp=timestamp,
                success=fill_price is not None,
                error_message=None if fill_price is not None else "Order timed out without fill",
            )

        except Exception as exc:
            logger.error(f"IBKRBrokerAdapter order failed: {exc}")
            return OrderResult(
                order_id=str(uuid.uuid4()),
                symbol="SPXW",
                fill_price=target_price,
                limit_price=target_price,
                quantity=quantity,
                strategy_type=strategy_type,
                is_entry=is_entry,
                timestamp=timestamp,
                success=False,
                error_message=str(exc),
            )

    def _build_combo_contract(self, strategy) -> Optional["Contract"]:
        """
        Build an IBKR BAG (combo) contract from the strategy's legs.

        Each leg maps to a ComboLeg:
          - Short legs → action "SELL"
          - Long legs  → action "BUY"
        """
        try:
            legs = strategy.legs
            if not legs:
                return None

            combo_legs = []
            for leg in legs:
                strike = float(leg.strike)
                right  = "C" if "call" in str(leg.option_type).lower() else "P"
                expiry = strategy.expiration.strftime("%Y%m%d") if hasattr(strategy, "expiration") else ""

                # Qualify the individual option contract to get conId
                opt_contract = Contract(
                    secType="OPT",
                    symbol="SPXW",
                    lastTradeDateOrContractMonth=expiry,
                    strike=strike,
                    right=right,
                    exchange="SMART",
                    currency="USD",
                    multiplier="100",
                )
                qualified = self._ib.qualifyContracts(opt_contract)
                if not qualified:
                    logger.warning(f"Could not qualify contract: SPXW {expiry} {strike}{right}")
                    return None

                is_short = "SHORT" in str(getattr(leg, "position_side", "")).upper()
                action = "SELL" if is_short else "BUY"
                combo_legs.append(
                    ComboLeg(
                        conId=qualified[0].conId,
                        ratio=1,
                        action=action,
                        exchange="SMART",
                    )
                )

            bag = Contract()
            bag.symbol   = "SPXW"
            bag.secType  = "BAG"
            bag.currency = "USD"
            bag.exchange = "SMART"
            bag.comboLegs = combo_legs
            return bag

        except Exception as exc:
            logger.error(f"_build_combo_contract failed: {exc}")
            return None

    def _submit_with_retry(self, combo: "Contract", action: str,
                           quantity: int, target_price: float,
                           use_market: bool = False) -> Optional[float]:
        """
        Submit a limit order and retry with price improvements.
        Returns the fill price, or None if it timed out.
        """
        if use_market:
            order = MarketOrder(action, quantity)
            if self._account:
                order.account = self._account
            trade = self._ib.placeOrder(combo, order)
            return self._wait_for_fill(trade)

        limit_price = target_price
        for attempt in range(MAX_IMPROVE_ATTEMPTS + 1):
            order = LimitOrder(action, quantity, round(limit_price, 2))
            if self._account:
                order.account = self._account
            trade = self._ib.placeOrder(combo, order)
            fill = self._wait_for_fill(trade, timeout=INITIAL_WAIT_SECS)
            if fill is not None:
                return fill

            # Improve the limit price toward the market
            if action == "SELL":
                limit_price -= IMPROVE_STEP   # Accept a lower credit
            else:
                limit_price += IMPROVE_STEP   # Offer a higher debit
            logger.debug(
                f"No fill after {INITIAL_WAIT_SECS}s, improving to {limit_price:.2f} "
                f"(attempt {attempt + 1}/{MAX_IMPROVE_ATTEMPTS})"
            )
            # Cancel the unfilled order before retrying
            try:
                self._ib.cancelOrder(order)
                self._ib.sleep(1)
            except Exception:
                pass

        # Final attempt: market order
        logger.warning("All limit attempts failed, submitting market order")
        market_order = MarketOrder(action, quantity)
        if self._account:
            market_order.account = self._account
        trade = self._ib.placeOrder(combo, market_order)
        return self._wait_for_fill(trade, timeout=IMPROVE_WAIT_SECS)

    def _wait_for_fill(self, trade, timeout: int = INITIAL_WAIT_SECS) -> Optional[float]:
        """Poll until the trade is filled or timeout expires. Returns fill price or None."""
        start = time.time()
        while time.time() - start < timeout:
            self._ib.sleep(0.5)
            if trade.isDone():
                if trade.fills:
                    fill_price = sum(f.execution.price * f.execution.shares
                                     for f in trade.fills) / sum(f.execution.shares
                                                                  for f in trade.fills)
                    logger.info(f"Order filled at {fill_price:.2f}")
                    return float(fill_price)
                break
        return None
