"""
IBKR Broker Adapter — real order execution via TWS / IB Gateway.

Submits SPXW credit spread and iron condor orders to IBKR as combo (BAG)
contracts.  Works for both IBKR paper trading accounts (port 7497) and live
accounts (port 7496) — the only difference is the port number.

Order execution strategy
------------------------
1. Calculate the mid-price of the combo from the current bid/ask.
2. Submit a single limit order at the mid-price.
3. Wait up to ENTRY_FILL_WAIT_SECS (entries) or EXIT_FILL_WAIT_SECS (exits).
4. If no fill within the timeout, return OrderResult(success=False).
   No price chasing, no market orders.

Connection model
----------------
IBKRBrokerAdapter uses its OWN dedicated IB connection (separate clientId from
the market data provider) so that order submission is never blocked by the
snapshot data flood on the market data connection.

Pass host/port/client_id to the constructor and call connect() before trading.
"""

import time
import uuid
from typing import List, Optional

from loguru import logger

from .adapter import BrokerAdapter, OrderResult

try:
    from ib_insync import (
        IB, Contract, Order, ComboLeg, TagValue, LimitOrder,
    )
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

# Order fill timing constants
ENTRY_FILL_WAIT_SECS = 60   # How long to wait for an opening limit order
EXIT_FILL_WAIT_SECS  = 45   # How long to wait for a closing limit order


class IBKRBrokerAdapter(BrokerAdapter):
    """
    Real order execution adapter for IBKR paper / live trading.

    Uses a dedicated IB connection (separate from IBKRMarketDataProvider) so
    that order submission is never blocked by the concurrent snapshot data flood
    on the market data connection.

    Parameters
    ----------
    host:       TWS / IB Gateway host (e.g. '127.0.0.1')
    port:       TWS / IB Gateway port (e.g. 7497 for paper TWS)
    client_id:  Must be DIFFERENT from the market data provider clientId.
                Convention: market data uses client_id N, broker uses N+1.
    account:    IBKR account string (e.g. 'DU123456' for paper, 'U123456' live).
    """

    def __init__(self, host: str, port: int, client_id: int, account: str = ""):
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_insync must be installed to use IBKRBrokerAdapter. "
                "Run: pip install ib_insync"
            )
        self._host = host
        self._port = port
        self._client_id = client_id
        self._account = account
        self._ib: Optional["IB"] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout: int = 15) -> bool:
        """Open a dedicated IB connection for order execution."""
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        try:
            self._ib = IB()
            last_exc = None
            for attempt in range(1, 4):
                try:
                    self._ib.connect(self._host, self._port,
                                     clientId=self._client_id, timeout=timeout)
                    if self._ib.isConnected():
                        break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        f"IBKRBrokerAdapter connect attempt {attempt}/3 failed: {exc} "
                        "— waiting 3s before retry"
                    )
                    time.sleep(3)
            else:
                raise last_exc or ConnectionError(
                    f"IBKRBrokerAdapter could not connect after 3 attempts"
                )

            if self._ib.isConnected():
                logger.info(
                    f"IBKRBrokerAdapter connected to {self._host}:{self._port} "
                    f"(clientId={self._client_id})"
                )
                return True
            return False
        except Exception as exc:
            logger.error(f"IBKRBrokerAdapter connection failed: {exc}")
            return False

    def disconnect(self):
        """Close the dedicated IB connection."""
        if self._ib and self._ib.isConnected():
            try:
                self._ib.disconnect()
            except Exception:
                pass
        logger.info("IBKRBrokerAdapter disconnected")

    @property
    def is_connected(self) -> bool:
        return bool(self._ib) and self._ib.isConnected()

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
        """Force-close all open positions using aggressive limit orders."""
        results = []
        for strat in open_strategies:
            try:
                result = self._execute_combo_order(
                    strategy=strat,
                    quantity=getattr(strat, "quantity", 1),
                    timestamp=timestamp,
                    is_entry=False,
                    target_price=0.0,
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"close_all failed for strategy: {exc}")
                results.append(OrderResult(
                    order_id=str(uuid.uuid4()),
                    symbol="SPX",
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

    # ------------------------------------------------------------------
    # Internal order construction and submission
    # ------------------------------------------------------------------

    def _execute_combo_order(self, strategy, quantity: int, timestamp: str,
                              is_entry: bool, target_price: float) -> OrderResult:
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
            # target_price is the total dollar amount (entry_credit / exit_cost × 100 × qty).
            # IBKR combo lmtPrice is expressed as per-share price (before the 100 multiplier).
            per_share_price = abs(target_price) / (100.0 * max(quantity, 1))
            wait_secs = ENTRY_FILL_WAIT_SECS if is_entry else EXIT_FILL_WAIT_SECS
            fill_price = self._submit_limit_order(
                combo=combo,
                action=action,
                quantity=quantity,
                limit_price=per_share_price,
                wait_secs=wait_secs,
            )

            return OrderResult(
                order_id=str(uuid.uuid4()),
                symbol="SPX",
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
            logger.error(f"IBKRBrokerAdapter order failed: {exc}")
            fallback_price = abs(target_price) / (100.0 * max(quantity, 1))
            return OrderResult(
                order_id=str(uuid.uuid4()),
                symbol="SPX",
                fill_price=fallback_price,
                limit_price=fallback_price,
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
                    symbol="SPX",
                    lastTradeDateOrContractMonth=expiry,
                    strike=strike,
                    right=right,
                    exchange="CBOE",
                    currency="USD",
                    multiplier="100",
                    tradingClass="SPXW",
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
                        exchange="CBOE",
                    )
                )

            bag = Contract()
            bag.symbol   = "SPX"
            bag.secType  = "BAG"
            bag.currency = "USD"
            bag.exchange = "SMART"
            bag.comboLegs = combo_legs
            return bag

        except Exception as exc:
            logger.error(f"_build_combo_contract failed: {exc}")
            return None

    def _submit_limit_order(self, combo: "Contract", action: str,
                            quantity: int, limit_price: float,
                            wait_secs: int) -> Optional[float]:
        """
        Submit a single limit order at the given price and wait up to wait_secs.
        Returns the fill price, or None if the order did not fill in time.
        """
        order = LimitOrder(action, quantity, round(limit_price, 2))
        if self._account:
            order.account = self._account
        trade = self._ib.placeOrder(combo, order)
        logger.debug(f"Placed {action} limit @ {limit_price:.2f} — waiting up to {wait_secs}s")
        fill = self._wait_for_fill(trade, timeout=wait_secs)
        if fill is None:
            logger.warning(
                f"No fill after {wait_secs}s for {action} limit @ {limit_price:.2f} — cancelling"
            )
            try:
                self._ib.cancelOrder(order)
                self._ib.sleep(1)
            except Exception:
                pass
        return fill

    def _wait_for_fill(self, trade, timeout: int = ENTRY_FILL_WAIT_SECS) -> Optional[float]:
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
