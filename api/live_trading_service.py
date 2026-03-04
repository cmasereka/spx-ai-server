"""
Live Trading Service — multi-broker paper / live trading sessions.

Manages live trading sessions that execute real orders via a selected broker
(IBKR or TastyTrade).  Each session uses a LiveTradingSession (trading/session.py)
running in a thread pool, with a broker-appropriate MarketDataProvider wrapped by
RealtimeMarketDataProvider for real-time bars and a matching BrokerAdapter for
order submission.

Session lifecycle:  pending → connecting → running → completed | stopped | failed
"""

import asyncio
import threading
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from .models import (
    LiveTradingRequest, LiveTradingStatus, BrokerEnum,
    PaperPosition, BacktestResult, StrategyDetails, OrderSlippage,
)
from .websocket_manager import WebSocketManager
from enhanced_multi_strategy import EnhancedBacktestingEngine
from enhanced_backtest import EnhancedBacktestResult, StrategyType as ST
from delta_strike_selector import IronCondorStrikeSelection
from market_data.ibkr_provider import IBKRMarketDataProvider
from market_data.realtime_provider import RealtimeMarketDataProvider
from broker.null_adapter import NullBrokerAdapter
from trading.session import LiveTradingSession
from src.database.connection import db_manager
from src.database.models import PaperTradingRun, BrokerOrder, Trade


class _LiveSessionState:
    """Thread-safe mutable state for a live trading session."""

    def __init__(self, session_id: str, trade_date: str, request: LiveTradingRequest):
        self.session_id = session_id
        self.mode = "live"
        self.trade_date = trade_date
        self.request = request
        self.status = "pending"
        self.broker_type: str = request.broker.value if request else "ibkr"
        self.broker_connected = False
        self.open_positions: List[PaperPosition] = []
        self.completed_trades: List[BacktestResult] = []
        self.day_pnl: float = 0.0
        self.trade_count: int = 0
        self.orders: List[OrderSlippage] = []
        self.created_at: datetime = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self._lock = threading.Lock()

    def to_status(self) -> LiveTradingStatus:
        with self._lock:
            return LiveTradingStatus(
                session_id=self.session_id,
                mode=self.mode,
                trade_date=self.trade_date,
                status=self.status,
                broker_type=self.broker_type,
                broker_connected=self.broker_connected,
                open_positions=list(self.open_positions),
                completed_trades=list(self.completed_trades),
                day_pnl=round(self.day_pnl, 2),
                trade_count=self.trade_count,
                orders=list(self.orders),
                total_slippage=round(sum(o.slippage for o in self.orders), 4),
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                error_message=self.error_message,
            )


class LiveTradingService:
    """Manages IBKR paper / live trading sessions."""

    def __init__(self):
        self._sessions: Dict[str, _LiveSessionState] = {}
        self.engine: Optional[EnhancedBacktestingEngine] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self, engine: EnhancedBacktestingEngine):
        """Called from FastAPI lifespan — shares the engine already loaded by BacktestService."""
        self.engine = engine
        self._load_sessions_from_db()
        logger.info("LiveTradingService initialised")

    def _load_sessions_from_db(self):
        """Restore completed/stopped/failed sessions from the database into memory."""
        try:
            with db_manager.get_session() as db_session:
                runs = db_session.query(PaperTradingRun).order_by(
                    PaperTradingRun.created_at.asc()
                ).all()

                for run in runs:
                    # Reconstruct a minimal LiveTradingRequest from stored parameters
                    params = run.parameters or {}
                    broker_str = run.broker_type or params.get("broker_type", "ibkr")
                    from .models import IBKRConnectionConfig, BacktestStrategyEnum, TastyTradeConfig
                    try:
                        if broker_str == "tastytrade":
                            request = LiveTradingRequest(
                                broker=BrokerEnum.TASTYTRADE,
                                tastytrade=TastyTradeConfig(
                                    provider_secret="",   # credentials not stored in DB
                                    refresh_token="",     # credentials not stored in DB
                                    account_number=params.get("tt_account", ""),
                                    is_paper=params.get("tt_is_paper", True),
                                ),
                                trade_date=run.trade_date,
                                strategy=params.get("strategy", BacktestStrategyEnum.CREDIT_SPREADS),
                                target_credit=params.get("target_credit", 0.35),
                                spread_width=params.get("spread_width", 10),
                                contracts=params.get("contracts", 1),
                                take_profit=params.get("take_profit", 0.05),
                                stop_loss=params.get("stop_loss", 3.0),
                                monitor_interval=params.get("monitor_interval", 5),
                                entry_start_time=params.get("entry_start_time", "10:00:00"),
                                last_entry_time=params.get("last_entry_time", "14:00:00"),
                                stale_loss_minutes=params.get("stale_loss_minutes", 120),
                                stale_loss_threshold=params.get("stale_loss_threshold", 1.5),
                                stagnation_window=params.get("stagnation_window", 30),
                                min_improvement=params.get("min_improvement", 0.05),
                                enable_stale_loss_exit=params.get("enable_stale_loss_exit", False),
                            )
                        else:
                            request = LiveTradingRequest(
                                ibkr=IBKRConnectionConfig(
                                    host=params.get("ibkr_host", "127.0.0.1"),
                                    port=params.get("ibkr_port", 7497),
                                    client_id=params.get("ibkr_client_id", 1),
                                    account=params.get("ibkr_account", ""),
                                ),
                                trade_date=run.trade_date,
                                strategy=params.get("strategy", BacktestStrategyEnum.CREDIT_SPREADS),
                                target_credit=params.get("target_credit", 0.35),
                                spread_width=params.get("spread_width", 10),
                                contracts=params.get("contracts", 1),
                                take_profit=params.get("take_profit", 0.05),
                                stop_loss=params.get("stop_loss", 3.0),
                                monitor_interval=params.get("monitor_interval", 5),
                                entry_start_time=params.get("entry_start_time", "10:00:00"),
                                last_entry_time=params.get("last_entry_time", "14:00:00"),
                                stale_loss_minutes=params.get("stale_loss_minutes", 120),
                                stale_loss_threshold=params.get("stale_loss_threshold", 1.5),
                                stagnation_window=params.get("stagnation_window", 30),
                                min_improvement=params.get("min_improvement", 0.05),
                                enable_stale_loss_exit=params.get("enable_stale_loss_exit", False),
                            )
                    except Exception as exc:
                        logger.warning(
                            f"Could not reconstruct request for session {run.session_id}: {exc}"
                        )
                        continue

                    state = _LiveSessionState(
                        session_id=run.session_id,
                        trade_date=str(run.trade_date),
                        request=request,
                    )
                    state.broker_type = run.broker_type or broker_str
                    state.status = run.status
                    state.created_at = run.created_at
                    state.started_at = run.started_at
                    state.completed_at = run.completed_at
                    state.error_message = run.error_message
                    state.trade_count = run.total_trades or 0
                    state.day_pnl = run.total_pnl or 0.0

                    # Load completed trades from the trades table
                    from src.database.models import Trade as TradeModel
                    db_trades = db_session.query(TradeModel).filter_by(
                        backtest_run_id=run.id
                    ).order_by(TradeModel.created_at.asc()).all()

                    for t in db_trades:
                        from .models import StrategyDetails
                        sd = t.strategy_details or {}
                        strategy = StrategyDetails(
                            strategy_type=t.strategy_type,
                            strikes=t.strikes or {},
                            entry_credit=t.entry_credit,
                            max_profit=t.max_profit or 0.0,
                            max_loss=t.max_loss or 0.0,
                            breakeven_points=sd.get("breakeven_points", []),
                        )
                        state.completed_trades.append(BacktestResult(
                            trade_id=t.trade_id,
                            trade_date=t.trade_date,
                            entry_time=t.entry_time,
                            exit_time=t.exit_time,
                            entry_spx_price=t.entry_spx_price,
                            exit_spx_price=t.exit_spx_price,
                            strategy=strategy,
                            entry_credit=t.entry_credit,
                            exit_cost=t.exit_cost,
                            pnl=t.pnl,
                            pnl_percentage=t.pnl_percentage,
                            exit_reason=t.exit_reason,
                            is_winner=t.is_winner,
                            monitoring_points=t.monitoring_data or [],
                            entry_rationale=sd.get("entry_rationale"),
                            exit_rationale=sd.get("exit_rationale"),
                        ))

                    self._sessions[run.session_id] = state

                logger.info(f"Loaded {len(runs)} live session(s) from database")

        except Exception as exc:
            logger.warning(f"Could not load live sessions from DB (may not exist yet): {exc}")

    async def cleanup(self):
        for sid, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled live trading session: {sid}")
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sessions(self) -> List[LiveTradingStatus]:
        return [s.to_status() for s in self._sessions.values()]

    def get_session(self, session_id: str) -> Optional[LiveTradingStatus]:
        s = self._sessions.get(session_id)
        return s.to_status() if s else None

    async def stop_session(self, session_id: str) -> bool:
        task = self._running_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        s = self._sessions.get(session_id)
        if not s:
            return False
        with s._lock:
            if s.status not in ("completed", "stopped", "failed"):
                s.status = "stopped"
                s.completed_at = datetime.now()
        # Persist immediately so a server restart doesn't revert the status
        await self._update_session_db_status(s)
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Stop the session if active, then remove it from memory and the database."""
        await self.stop_session(session_id)
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        self._running_tasks.pop(session_id, None)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._delete_session_sync, session_id)
        except Exception as exc:
            logger.error(f"Failed to delete live session {session_id} from DB: {exc}")
        return True

    def _delete_session_sync(self, session_id: str):
        with db_manager.get_session() as session:
            run = session.query(PaperTradingRun).filter_by(session_id=session_id).first()
            if run:
                session.delete(run)
                session.commit()

    async def start_session(
        self,
        request: LiveTradingRequest,
        websocket_manager: WebSocketManager,
    ) -> str:
        """Create and start a live trading session. Returns session_id."""
        if not self.engine:
            raise RuntimeError("LiveTradingService not initialised")

        trade_date = request.trade_date or date.today()
        session_id = str(uuid.uuid4())
        state = _LiveSessionState(
            session_id=session_id,
            trade_date=str(trade_date),
            request=request,
        )
        self._sessions[session_id] = state

        await self._save_session_to_db(state)

        task = asyncio.create_task(
            self._run_session(state, websocket_manager)
        )
        self._running_tasks[session_id] = task

        logger.info(
            f"Live trading session started: {session_id} date={trade_date} "
            f"broker={request.broker.value}"
        )
        return session_id

    # ------------------------------------------------------------------
    # Session runner
    # ------------------------------------------------------------------

    async def _run_session(
        self,
        state: _LiveSessionState,
        websocket_manager: WebSocketManager,
    ):
        """Async wrapper — runs the blocking session in the thread pool."""
        loop = asyncio.get_event_loop()
        backtest_id = state.session_id
        req = state.request

        # --- Create market data provider (broker-specific) ---
        with state._lock:
            state.status = "connecting"
        await websocket_manager.send_backtest_update(
            backtest_id, "status_change",
            {"status": "connecting", "session_id": backtest_id}
        )

        provider = _create_market_data_provider(req)

        # Shared container so the worker thread can report back connect result
        connect_result: dict = {}

        def _run_session_in_thread():
            """
            Connect to the broker and run the full trading session in a single thread.

            For IBKR: ib_insync's IB object binds its asyncio event loop to the thread
            that calls connect().  All subsequent reqMktData / sleep / cancelMktData
            calls must happen in that same thread.

            For TastyTrade: the DXLink streamer runs its own background thread internally,
            but order submission is synchronous REST — no special event-loop binding needed.

            A fresh event loop is created at the top of this function and set as the
            current thread's loop so that ib_insync always finds a clean, running loop.
            """
            import asyncio as _asyncio
            _thread_loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(_thread_loop)

            broker_label = req.broker.value
            logger.info(f"Session {backtest_id}: connecting to {broker_label}")
            connected = provider.connect()
            connect_result["connected"] = connected
            if not connected:
                connect_result["error"] = (
                    f"Could not connect to {broker_label}. "
                    "Check credentials / connection settings and try again."
                )
                return None

            # For IBKR, override trade date (connect() sets _today = datetime.now())
            if hasattr(provider, "_today"):
                provider._today = state.trade_date
            logger.info(
                f"Session {backtest_id}: {broker_label} market data connected — "
                f"trade_date={state.trade_date} "
                f"provider dates: {provider.available_dates}"
            )

            broker_adapter = _create_broker_adapter(req)
            broker_connected = broker_adapter.connect()
            if not broker_connected:
                logger.critical(
                    f"╔══ BROKER CONNECTION FAILED ══╗\n"
                    f"  Session  : {backtest_id}\n"
                    f"  Broker   : {broker_label}\n"
                    f"  ORDERS WILL NOT BE SUBMITTED — falling back to NullAdapter\n"
                    f"╚══════════════════════════════╝"
                )
                from broker.null_adapter import NullBrokerAdapter
                broker_adapter = NullBrokerAdapter()
            else:
                logger.info(
                    f"Session {backtest_id}: broker adapter connected "
                    f"({broker_label}) — orders WILL be submitted"
                )

            rt_provider = RealtimeMarketDataProvider(
                provider,
                trade_date=state.trade_date,
            )

            logger.info(
                f"Session {backtest_id}: scanning date={state.trade_date} "
                f"entry_window={req.entry_start_time}–{req.last_entry_time} "
                f"strategy={req.strategy.value} credit={req.target_credit} "
                f"tp={req.take_profit} sl={req.stop_loss}"
            )

            session = LiveTradingSession(
                engine=self.engine,
                market_data_provider=rt_provider,
                broker_adapter=broker_adapter,
            )

            callback = self._make_callback(state, loop, websocket_manager, backtest_id)

            try:
                return session.run(
                    date=state.trade_date,
                    take_profit=req.take_profit,
                    stop_loss=req.stop_loss,
                    monitor_interval=req.monitor_interval,
                    min_spread_width=req.spread_width,
                    target_credit=req.target_credit,
                    strategy_mode=req.strategy.value,
                    quantity=req.contracts,
                    progress_callback=callback,
                    entry_start_time=req.entry_start_time,
                    last_entry_time=req.last_entry_time,
                    stale_loss_minutes=req.stale_loss_minutes,
                    stale_loss_threshold=req.stale_loss_threshold,
                    stagnation_window=req.stagnation_window,
                    min_improvement=req.min_improvement,
                    enable_stale_loss_exit=req.enable_stale_loss_exit,
                    skip_indicators=True,  # Live sessions use drift-only guards until RSI/BB warmup is resolved
                )
            finally:
                try:
                    broker_adapter.disconnect()
                except Exception:
                    pass

        day_result = await loop.run_in_executor(self._executor, _run_session_in_thread)
        # Handle connect failure reported from the worker thread
        if not connect_result.get("connected", False):
            with state._lock:
                state.status = "failed"
                state.error_message = connect_result.get("error", "Broker connection failed")
                state.completed_at = datetime.now()
            await self._update_session_db_status(state)
            await websocket_manager.send_backtest_error(backtest_id, state.error_message)
            return

        with state._lock:
            state.status = "running"
            state.started_at = datetime.now()
            state.broker_connected = True

        await self._update_session_db_status(state)
        await websocket_manager.send_backtest_update(
            backtest_id, "status_change",
            {"status": "running", "session_id": backtest_id}
        )

        try:
            if day_result is None:
                # connect_result already handled above; nothing more to do
                return

            logger.info(
                f"Session {backtest_id}: scan complete — "
                f"trades={day_result.trade_count} pnl={day_result.total_pnl:.2f} "
                f"bars_checked={day_result.scan_minutes_checked}"
            )

            with state._lock:
                state.status = "completed"
                state.completed_at = datetime.now()
                state.open_positions = []
                seen = {t.entry_time for t in state.completed_trades}
                for trade in day_result.trades:
                    if trade.entry_time not in seen:
                        state.completed_trades.append(
                            _engine_result_to_api(trade, state.session_id)
                        )
                        state.day_pnl += trade.pnl
                        state.trade_count += 1
                state.day_pnl = round(day_result.total_pnl, 2)
                state.trade_count = day_result.trade_count

            await self._finalise_session_db(state)
            await websocket_manager.send_backtest_completed(
                backtest_id,
                {
                    "session_id": state.session_id,
                    "total_trades": state.trade_count,
                    "total_pnl": state.day_pnl,
                    "total_slippage": round(
                        sum(o.slippage for o in state.orders), 4
                    ),
                },
            )

        except asyncio.CancelledError:
            with state._lock:
                if state.status not in ("completed", "stopped", "failed"):
                    state.status = "stopped"
                    state.completed_at = datetime.now()
            await self._finalise_session_db(state)
            await websocket_manager.send_backtest_update(
                backtest_id, "status_change",
                {"status": "stopped", "session_id": backtest_id},
            )

        except Exception as exc:
            logger.error(f"Live trading session {state.session_id} failed: {exc}")
            with state._lock:
                state.status = "failed"
                state.error_message = str(exc)
                state.completed_at = datetime.now()
            await self._update_session_db_status(state)
            await websocket_manager.send_backtest_error(backtest_id, str(exc))

        finally:
            self._running_tasks.pop(state.session_id, None)
            try:
                provider.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Progress callback
    # ------------------------------------------------------------------

    def _make_callback(
        self,
        state: _LiveSessionState,
        loop: asyncio.AbstractEventLoop,
        websocket_manager: WebSocketManager,
        backtest_id: str,
    ):
        """Return a thread-safe progress callback that also persists IBKR orders."""

        def callback(event: dict):
            ev = event.get("event")
            with state._lock:
                if ev == "position_opened":
                    pos = PaperPosition(
                        position_id=str(uuid.uuid4()),
                        strategy_type=event.get("strategy_type", ""),
                        entry_time=event.get("entry_time", ""),
                        entry_spx_price=float(event.get("entry_spx", 0)),
                        entry_credit=float(event.get("entry_credit", 0)),
                        strikes=_strikes_from_selection(event.get("strikes")),
                        entry_rationale=event.get("entry_rationale"),
                    )
                    state.open_positions.append(pos)

                    # Record IBKR order slippage if available
                    order_result = event.get("order_result")
                    if order_result:
                        slippage = OrderSlippage(
                            order_id=order_result.order_id,
                            strategy_type=order_result.strategy_type,
                            is_entry=True,
                            limit_price=order_result.limit_price,
                            fill_price=order_result.fill_price,
                            slippage=order_result.slippage,
                            timestamp=order_result.timestamp,
                            success=order_result.success,
                        )
                        state.orders.append(slippage)

                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        websocket_manager.send_backtest_update(
                            backtest_id, "position_opened",
                            {"session_id": state.session_id, **pos.dict()}
                        )
                    )

                elif ev == "monitor_tick":
                    # Find the matching open position and broadcast its live PnL
                    strategy_type = event.get("strategy_type", "")
                    entry_time = event.get("entry_time", "")
                    pos = next(
                        (p for p in state.open_positions
                         if p.strategy_type == strategy_type and p.entry_time == entry_time),
                        None,
                    )
                    if pos:
                        loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            websocket_manager.send_backtest_update(
                                backtest_id, "position_update",
                                {
                                    "session_id": state.session_id,
                                    "position_id": pos.position_id,
                                    "strategy_type": strategy_type,
                                    "entry_time": entry_time,
                                    "time": event.get("time"),
                                    "spx": event.get("spx"),
                                    "pnl_per_share": event.get("pnl_per_share"),
                                    "entry_credit_per_share": event.get("entry_credit_per_share"),
                                }
                            )
                        )

                elif ev == "position_closed":
                    result: EnhancedBacktestResult = event.get("result")
                    if result:
                        api_result = _engine_result_to_api(result, state.session_id)
                        state.open_positions = [
                            p for p in state.open_positions
                            if not (
                                p.strategy_type == result.strategy_type.value
                                and p.entry_time == result.entry_time
                            )
                        ]
                        state.completed_trades.append(api_result)
                        state.day_pnl += result.pnl
                        state.trade_count += 1

                        # Record IBKR close order slippage if available
                        order_result = event.get("order_result")
                        if order_result:
                            slippage = OrderSlippage(
                                order_id=order_result.order_id,
                                strategy_type=order_result.strategy_type,
                                is_entry=False,
                                limit_price=order_result.limit_price,
                                fill_price=order_result.fill_price,
                                slippage=order_result.slippage,
                                timestamp=order_result.timestamp,
                                success=order_result.success,
                            )
                            state.orders.append(slippage)

                        # Persist trade to DB and broadcast via WebSocket
                        loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            self._save_and_send_live_trade(
                                state.session_id, api_result, websocket_manager, backtest_id
                            )
                        )

        return callback

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    async def _save_and_send_live_trade(
        self,
        session_id: str,
        api_result: BacktestResult,
        websocket_manager: WebSocketManager,
        backtest_id: str,
    ):
        """Persist a single live trade to the trades table and broadcast via WebSocket."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor, self._save_live_trade_sync, session_id, api_result
            )
        except Exception as exc:
            logger.error(f"Failed to persist live trade {api_result.trade_id}: {exc}")
        await websocket_manager.send_trade_result(backtest_id, api_result.dict())

    def _save_live_trade_sync(self, session_id: str, result: BacktestResult):
        """Write one completed trade to the trades table, linked to the live session."""
        with db_manager.get_session() as db_session:
            run = db_session.query(PaperTradingRun).filter_by(
                session_id=session_id
            ).first()
            if not run:
                logger.error(
                    f"PaperTradingRun {session_id} not found — cannot save trade {result.trade_id}"
                )
                return
            trade = Trade(
                trade_id=result.trade_id,
                backtest_run_id=run.id,
                trade_date=result.trade_date,
                entry_time=result.entry_time,
                exit_time=result.exit_time,
                entry_spx_price=result.entry_spx_price,
                exit_spx_price=result.exit_spx_price,
                strategy_type=result.strategy.strategy_type,
                strikes=result.strategy.strikes,
                entry_credit=result.entry_credit,
                exit_cost=result.exit_cost,
                pnl=result.pnl,
                pnl_percentage=result.pnl_percentage,
                exit_reason=result.exit_reason,
                is_winner=result.is_winner,
                max_profit=result.strategy.max_profit,
                max_loss=result.strategy.max_loss,
                strategy_details={
                    "strategy_type": result.strategy.strategy_type,
                    "strikes": result.strategy.strikes,
                    "entry_credit": result.strategy.entry_credit,
                    "max_profit": result.strategy.max_profit,
                    "max_loss": result.strategy.max_loss,
                    "breakeven_points": result.strategy.breakeven_points,
                    "entry_rationale": result.entry_rationale,
                    "exit_rationale": result.exit_rationale,
                },
                monitoring_data=result.monitoring_points,
            )
            db_session.add(trade)
            db_session.commit()
            logger.debug(f"Persisted live trade {result.trade_id} for session {session_id}")

    async def _save_session_to_db(self, state: _LiveSessionState):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_session_sync, state)
        except Exception as exc:
            logger.error(f"Failed to save live session to DB: {exc}")

    def _save_session_sync(self, state: _LiveSessionState):
        with db_manager.get_session() as session:
            req = state.request

            # Build broker-specific connection parameters (no passwords stored)
            if req.broker == BrokerEnum.TASTYTRADE and req.tastytrade:
                broker_params = {
                    # Do not store provider_secret or refresh_token in DB
                    "tt_account": req.tastytrade.account_number,
                    "tt_is_paper": req.tastytrade.is_paper,
                }
            else:
                broker_params = {
                    "ibkr_host": req.ibkr.host,
                    "ibkr_port": req.ibkr.port,
                    "ibkr_client_id": req.ibkr.client_id,
                    "ibkr_account": req.ibkr.account,
                }

            run = PaperTradingRun(
                session_id=state.session_id,
                mode="live",
                trade_date=datetime.strptime(state.trade_date, "%Y-%m-%d").date(),
                strategy_type=req.strategy.value,
                broker_type=req.broker.value,
                parameters={
                    "broker_type": req.broker.value,
                    "strategy": req.strategy.value,
                    "target_credit": req.target_credit,
                    "spread_width": req.spread_width,
                    "contracts": req.contracts,
                    "take_profit": req.take_profit,
                    "stop_loss": req.stop_loss,
                    "monitor_interval": req.monitor_interval,
                    "entry_start_time": req.entry_start_time,
                    "last_entry_time": req.last_entry_time,
                    "stale_loss_minutes": req.stale_loss_minutes,
                    "stale_loss_threshold": req.stale_loss_threshold,
                    "stagnation_window": req.stagnation_window,
                    "min_improvement": req.min_improvement,
                    "enable_stale_loss_exit": req.enable_stale_loss_exit,
                    **broker_params,
                },
                status=state.status,
                created_at=state.created_at,
            )
            session.add(run)
            session.commit()

    async def _update_session_db_status(self, state: _LiveSessionState):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._update_status_sync, state)
        except Exception as exc:
            logger.error(f"Failed to update live session status: {exc}")

    def _update_status_sync(self, state: _LiveSessionState):
        with db_manager.get_session() as session:
            run = session.query(PaperTradingRun).filter_by(
                session_id=state.session_id
            ).first()
            if run:
                run.status = state.status
                run.started_at = state.started_at
                run.completed_at = state.completed_at
                run.error_message = state.error_message
                session.commit()

    async def _finalise_session_db(self, state: _LiveSessionState):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._finalise_sync, state)
        except Exception as exc:
            logger.error(f"Failed to finalise live session in DB: {exc}")

    def _finalise_sync(self, state: _LiveSessionState):
        with db_manager.get_session() as session:
            run = session.query(PaperTradingRun).filter_by(
                session_id=state.session_id
            ).first()
            if run:
                run.status = state.status
                run.completed_at = state.completed_at
                run.total_trades = state.trade_count
                run.successful_trades = sum(1 for t in state.completed_trades if t.is_winner)
                run.total_pnl = state.day_pnl
                session.commit()

                # Upsert every completed trade — this catches any that were
                # missed by the async position_closed callback (e.g. session
                # crashed or was stopped before the coroutine executed).
                existing_ids = {
                    row[0] for row in
                    session.query(Trade.trade_id).filter_by(backtest_run_id=run.id).all()
                }
                for t in state.completed_trades:
                    if t.trade_id in existing_ids:
                        continue
                    sd = t.strategy.__dict__ if hasattr(t.strategy, "__dict__") else {}
                    trade = Trade(
                        trade_id=t.trade_id,
                        backtest_run_id=run.id,
                        trade_date=t.trade_date,
                        entry_time=t.entry_time,
                        exit_time=t.exit_time,
                        entry_spx_price=t.entry_spx_price,
                        exit_spx_price=t.exit_spx_price,
                        strategy_type=t.strategy.strategy_type,
                        strikes=t.strategy.strikes,
                        entry_credit=t.entry_credit,
                        exit_cost=t.exit_cost,
                        pnl=t.pnl,
                        pnl_percentage=t.pnl_percentage,
                        exit_reason=t.exit_reason,
                        is_winner=t.is_winner,
                        max_profit=t.strategy.max_profit,
                        max_loss=t.strategy.max_loss,
                        strategy_details={
                            "strategy_type": t.strategy.strategy_type,
                            "strikes": t.strategy.strikes,
                            "entry_credit": t.strategy.entry_credit,
                            "max_profit": t.strategy.max_profit,
                            "max_loss": t.strategy.max_loss,
                            "breakeven_points": t.strategy.breakeven_points,
                            "entry_rationale": t.entry_rationale,
                            "exit_rationale": t.exit_rationale,
                        },
                        monitoring_data=t.monitoring_points,
                    )
                    session.add(trade)
                session.commit()
            for order in state.orders:
                broker_rec = BrokerOrder(
                    order_id=order.order_id,
                    session_id=state.session_id,
                    broker_type=state.broker_type,
                    symbol="SPXW",
                    strategy_type=order.strategy_type,
                    is_entry=order.is_entry,
                    limit_price=order.limit_price,
                    fill_price=order.fill_price,
                    slippage=order.slippage,
                    quantity=1,  # stored per-contract in OrderSlippage
                    success=order.success,
                    timestamp=order.timestamp,
                )
                session.merge(broker_rec)
            session.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_market_data_provider(req: LiveTradingRequest):
    """
    Instantiate the appropriate MarketDataProvider for the requested broker.
    connect() is called separately (inside _run_session_in_thread) so that
    ib_insync's event loop binds to the correct thread.
    """
    if req.broker == BrokerEnum.TASTYTRADE:
        from market_data.tastytrade_provider import TastyTradeMarketDataProvider
        cfg = req.tastytrade
        return TastyTradeMarketDataProvider(
            provider_secret=cfg.provider_secret,
            refresh_token=cfg.refresh_token,
            is_paper=cfg.is_paper,
        )
    # Default: IBKR
    return IBKRMarketDataProvider(
        host=req.ibkr.host,
        port=req.ibkr.port,
        client_id=req.ibkr.client_id,
        account=req.ibkr.account,
    )


def _create_broker_adapter(req: LiveTradingRequest):
    """
    Instantiate the appropriate BrokerAdapter for the requested broker.
    For IBKR, uses client_id + 1 to avoid conflicting with the market data connection.
    """
    if req.broker == BrokerEnum.TASTYTRADE:
        from broker.tastytrade_adapter import TastyTradeBrokerAdapter
        cfg = req.tastytrade
        return TastyTradeBrokerAdapter(
            provider_secret=cfg.provider_secret,
            refresh_token=cfg.refresh_token,
            account_number=cfg.account_number,
            is_paper=cfg.is_paper,
        )
    # Default: IBKR
    from broker.ibkr_adapter import IBKRBrokerAdapter
    return IBKRBrokerAdapter(
        host=req.ibkr.host,
        port=req.ibkr.port,
        # Use client_id + 1 so order submission never shares a connection
        # with the market data provider and avoids the 10197 flood conflict.
        client_id=req.ibkr.client_id + 1,
        account=req.ibkr.account,
    )


def _strikes_from_selection(sel) -> dict:
    """Convert a StrikeSelection or IronCondorStrikeSelection to a plain dict."""
    if sel is None:
        return {}
    if isinstance(sel, IronCondorStrikeSelection):
        return {
            "put_long":   sel.put_long_strike,
            "put_short":  sel.put_short_strike,
            "call_short": sel.call_short_strike,
            "call_long":  sel.call_long_strike,
        }
    return {
        "short_strike": getattr(sel, "short_strike", 0),
        "long_strike":  getattr(sel, "long_strike",  0),
    }


def _engine_result_to_api(result: EnhancedBacktestResult, session_id: str) -> BacktestResult:
    """Convert engine EnhancedBacktestResult → API BacktestResult."""
    ss = result.strike_selection
    is_ic = result.strategy_type == ST.IRON_CONDOR

    if is_ic and isinstance(ss, IronCondorStrikeSelection):
        strikes = {
            "put_long":   ss.put_long_strike,
            "put_short":  ss.put_short_strike,
            "call_short": ss.call_short_strike,
            "call_long":  ss.call_long_strike,
        }
    elif result.strategy_type == ST.PUT_SPREAD:
        strikes = {
            "put_long":   getattr(ss, "long_strike",  0) if ss else 0,
            "put_short":  getattr(ss, "short_strike", 0) if ss else 0,
            "call_short": 0, "call_long": 0,
        }
    else:  # CALL_SPREAD
        strikes = {
            "put_long": 0, "put_short": 0,
            "call_short": getattr(ss, "short_strike", 0) if ss else 0,
            "call_long":  getattr(ss, "long_strike",  0) if ss else 0,
        }

    strategy_details = StrategyDetails(
        strategy_type=result.strategy_type.value if result.strategy_type else "unknown",
        strikes=strikes,
        entry_credit=result.entry_credit or 0,
        max_profit=result.max_profit or 0,
        max_loss=result.max_loss or 0,
        breakeven_points=[],
    )

    return BacktestResult(
        trade_id=str(uuid.uuid4()),
        trade_date=result.date,
        entry_time=result.entry_time or "10:00:00",
        exit_time=result.exit_time,
        entry_spx_price=result.entry_spx_price or 0,
        exit_spx_price=result.exit_spx_price,
        strategy=strategy_details,
        entry_credit=result.entry_credit or 0,
        exit_cost=result.exit_cost or 0,
        pnl=result.pnl or 0,
        pnl_percentage=result.pnl_pct or 0,
        exit_reason=result.exit_reason or "Unknown",
        is_winner=(result.pnl or 0) > 0,
        monitoring_points=result.monitoring_points or [],
        entry_rationale=result.entry_rationale,
        exit_rationale=result.exit_rationale,
    )
