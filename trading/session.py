"""
LiveTradingSession — standalone real-time trading session.

Orchestrates a single trading day using:
  - A MarketDataProvider  (Parquet for simulation, IBKR for live)
  - A BrokerAdapter       (Null for simulation, IBKR for live)
  - The same TechnicalAnalyzer and StrategySelector used in backtesting
  - The same DeltaStrikeSelector for strike selection
  - The same entry guards (drift, RSI, reversal) from enhanced_multi_strategy

This design keeps the engine logic identical across simulation, paper, and live
trading.  The only difference is which provider/adapter is injected.

Concurrency model
-----------------
This class is designed to run inside a ThreadPoolExecutor.  It is BLOCKING —
it does not use asyncio internally.  The async service wrapper
(api/live_trading_service.py) uses loop.run_in_executor() to run it in a
thread pool without blocking the FastAPI event loop.
"""

import sys
sys.path.append('.')

import threading
from datetime import datetime
from typing import List, Optional, Callable, Dict, Any

from loguru import logger

from market_data.provider import MarketDataProvider
from broker.adapter import BrokerAdapter, OrderResult
from broker.null_adapter import NullBrokerAdapter

# Reuse the same building blocks as the backtest engine
from enhanced_backtest import (
    StrategyType, TechnicalAnalyzer, StrategySelector,
    EnhancedBacktestResult, IronCondorLegStatus, DayBacktestResult,
)
from delta_strike_selector import (
    DeltaStrikeSelector, IntradayPositionMonitor,
    IronCondorStrikeSelection,
)
from enhanced_multi_strategy import (
    EnhancedBacktestingEngine,
    _build_minute_grid,
    ENTRY_SCAN_START, LAST_ENTRY_TIME, FINAL_EXIT_TIME,
    MIN_DISTANCE_IC, MIN_DISTANCE_SPREAD,
    DRIFT_BLOCK_POINTS, DRIFT_IC_BLOCK_POINTS,
    PUT_SPREAD_MAX_RSI_ON_NEG_DRIFT, PUT_SPREAD_MAX_RSI_ON_POS_DRIFT,
    INTRADAY_CALL_REVERSAL_POINTS, INTRADAY_PUT_REVERSAL_POINTS,
    CALL_SPREAD_MAX_ENTRY_RSI,
    STRATEGY_IRON_CONDOR, STRATEGY_CREDIT_SPREADS, STRATEGY_IC_CREDIT_SPREADS,
)


class LiveTradingSession:
    """
    A single real-time trading day session.

    Works in three modes depending on the injected providers:

      Simulation (default)
          market_data_provider = ParquetMarketDataProvider
          broker_adapter       = NullBrokerAdapter
          → Full-speed historical replay; same as PaperTradingService simulation.

      IBKR Paper Trading
          market_data_provider = RealtimeMarketDataProvider(IBKRMarketDataProvider(...))
          broker_adapter       = IBKRBrokerAdapter(ib)
          → Real-time bars from IBKR, real orders to IBKR paper account.

      IBKR Live Trading
          Same as paper, but with IBKR live port (7496) and live account.
          The code is identical — only the connection config differs.
    """

    def __init__(
        self,
        engine: EnhancedBacktestingEngine,
        market_data_provider: MarketDataProvider,
        broker_adapter: Optional[BrokerAdapter] = None,
    ):
        self._engine = engine
        self._provider = market_data_provider
        self._broker = broker_adapter or NullBrokerAdapter()

        # Reuse the engine's sub-components (same objects, same logic)
        self._technical_analyzer: TechnicalAnalyzer = engine.technical_analyzer
        self._strategy_selector: StrategySelector  = engine.strategy_selector
        self._delta_selector: DeltaStrikeSelector  = engine.delta_selector
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        date: str,
        take_profit: float = 0.10,
        stop_loss: float = 2.0,
        monitor_interval: int = 1,
        min_spread_width: int = 10,
        target_credit: float = 0.50,
        strategy_mode: str = STRATEGY_IRON_CONDOR,
        quantity: int = 1,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        entry_start_time: str = "10:00:00",
        last_entry_time: str = "14:00:00",
    ) -> DayBacktestResult:
        """
        Run the full trading day and return a DayBacktestResult.

        For simulation providers this completes instantly.
        For real-time providers this blocks until market close (or session stop).

        Parameters mirror backtest_day_intraday() so callers can swap between
        the engine method and this method without changing arguments.
        """
        logger.info(
            f"LiveTradingSession.run: {date} | mode={strategy_mode} "
            f"credit={target_credit} qty={quantity} tp={take_profit} sl={stop_loss}"
        )

        def _fire(event: Dict[str, Any]):
            if progress_callback:
                try:
                    progress_callback(event)
                except Exception as _e:
                    logger.debug(f"progress_callback error (ignored): {_e}")

        # Delegate to the engine's intraday scan loop, but inject our provider.
        # We swap the engine's enhanced_query_engine temporarily so that all
        # sub-components (delta_selector, intraday_monitor) also use the
        # injected provider.
        original_provider = self._engine.enhanced_query_engine
        try:
            self._engine.enhanced_query_engine = self._provider
            # Re-wire sub-components to the new provider
            self._engine.delta_selector.query_engine  = self._provider
            self._engine.intraday_monitor.query_engine = self._provider

            result = self._engine.backtest_day_intraday(
                date=date,
                take_profit=take_profit,
                stop_loss=stop_loss,
                monitor_interval=monitor_interval,
                min_spread_width=min_spread_width,
                target_credit=target_credit,
                strategy_mode=strategy_mode,
                quantity=quantity,
                progress_callback=self._wrap_callback_with_broker(
                    _fire, quantity
                ),
                entry_start_time=entry_start_time,
                last_entry_time=last_entry_time,
            )
        finally:
            # Always restore the original provider
            self._engine.enhanced_query_engine  = original_provider
            self._engine.delta_selector.query_engine   = original_provider
            self._engine.intraday_monitor.query_engine = original_provider

        return result

    # ------------------------------------------------------------------
    # Broker integration
    # ------------------------------------------------------------------

    def _wrap_callback_with_broker(
        self,
        user_callback: Callable,
        quantity: int,
    ) -> Callable:
        """
        Return a progress_callback that:
          1. Calls the user's callback (for WebSocket updates)
          2. Submits IBKR orders via the broker adapter on open/close events
        """
        broker = self._broker

        def wrapped_callback(event: Dict[str, Any]):
            ev = event.get("event")
            timestamp = event.get("entry_time") or event.get("exit_time") or "00:00:00"

            if ev == "position_opened":
                strategy = event.get("strategy_obj")
                credit   = float(event.get("entry_credit", 0))
                if strategy is not None:
                    order = broker.open_position(strategy, quantity, timestamp, credit)
                    event["order_result"] = order
                    if not order.success:
                        logger.warning(
                            f"Broker open_position failed: {order.error_message}"
                        )
                    else:
                        logger.info(
                            f"Broker filled OPEN {order.strategy_type} "
                            f"@ {order.fill_price:.2f} (target {order.limit_price:.2f}, "
                            f"slippage {order.slippage:+.2f})"
                        )

            elif ev == "position_closed":
                result = event.get("result")
                if result is not None:
                    debit = result.exit_cost or 0.0
                    # We don't have the original strategy object here, so use
                    # close_position with a stub (NullAdapter handles this fine;
                    # IBKRAdapter would need the original contract — tracked in
                    # LiveTradingService via the open_position order_id).
                    # For now log the event; full IBKR close wiring is in
                    # LiveTradingService._on_position_closed().
                    logger.info(
                        f"Broker position_closed event: {result.exit_reason} "
                        f"pnl={result.pnl:.2f}"
                    )

            # Always invoke the user's callback last
            user_callback(event)

        return wrapped_callback
