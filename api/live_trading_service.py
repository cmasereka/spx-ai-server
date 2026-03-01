"""
Live Trading Service for IBKR paper / live trading sessions.

Manages live trading sessions that execute real orders via IBKR.
Each session uses a LiveTradingSession (trading/session.py) running in a
thread pool, with the IBKRMarketDataProvider + RealtimeMarketDataProvider
for real-time bars and IBKRBrokerAdapter for order submission.

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
    LiveTradingRequest, LiveTradingStatus,
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
from src.database.models import PaperTradingRun, IBKROrder


class _LiveSessionState:
    """Thread-safe mutable state for a live trading session."""

    def __init__(self, session_id: str, trade_date: str, request: LiveTradingRequest):
        self.session_id = session_id
        self.mode = "live"
        self.trade_date = trade_date
        self.request = request
        self.status = "pending"
        self.ibkr_connected = False
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
                ibkr_connected=self.ibkr_connected,
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
        logger.info("LiveTradingService initialised")

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
            if s.status in ("running", "connecting"):
                s.status = "stopped"
                s.completed_at = datetime.now()
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
            f"ibkr={request.ibkr.host}:{request.ibkr.port}"
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

        # --- Connect to IBKR ---
        with state._lock:
            state.status = "connecting"
        await websocket_manager.send_backtest_update(
            backtest_id, "status_change",
            {"status": "connecting", "session_id": backtest_id}
        )

        ibkr_provider = IBKRMarketDataProvider(
            host=req.ibkr.host,
            port=req.ibkr.port,
            client_id=req.ibkr.client_id,
            account=req.ibkr.account,
        )
        connected = await loop.run_in_executor(None, ibkr_provider.connect)
        if not connected:
            with state._lock:
                state.status = "failed"
                state.error_message = (
                    f"Could not connect to IBKR at {req.ibkr.host}:{req.ibkr.port}. "
                    "Ensure TWS / IB Gateway is running and API connections are enabled."
                )
                state.completed_at = datetime.now()
            await self._update_session_db_status(state)
            await websocket_manager.send_backtest_error(backtest_id, state.error_message)
            return

        with state._lock:
            state.status = "running"
            state.started_at = datetime.now()
            state.ibkr_connected = True

        await self._update_session_db_status(state)
        await websocket_manager.send_backtest_update(
            backtest_id, "status_change",
            {"status": "running", "session_id": backtest_id}
        )

        try:
            from broker.ibkr_adapter import IBKRBrokerAdapter
            broker_adapter = IBKRBrokerAdapter(ibkr_provider._ib, account=req.ibkr.account)

            rt_provider = RealtimeMarketDataProvider(
                ibkr_provider,
                trade_date=state.trade_date,
            )

            session = LiveTradingSession(
                engine=self.engine,
                market_data_provider=rt_provider,
                broker_adapter=broker_adapter,
            )

            callback = self._make_callback(state, loop, websocket_manager, backtest_id)

            day_result = await loop.run_in_executor(
                self._executor,
                lambda: session.run(
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
                ),
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
                if state.status == "running":
                    state.status = "stopped"
                    state.completed_at = datetime.now()
            await self._update_session_db_status(state)

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
                ibkr_provider.disconnect()
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

                        loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            websocket_manager.send_trade_result(
                                backtest_id, api_result.dict()
                            )
                        )

        return callback

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    async def _save_session_to_db(self, state: _LiveSessionState):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_session_sync, state)
        except Exception as exc:
            logger.error(f"Failed to save live session to DB: {exc}")

    def _save_session_sync(self, state: _LiveSessionState):
        with db_manager.get_session() as session:
            req = state.request
            run = PaperTradingRun(
                session_id=state.session_id,
                mode="live",
                trade_date=datetime.strptime(state.trade_date, "%Y-%m-%d").date(),
                strategy_type=req.strategy.value,
                parameters={
                    "strategy": req.strategy.value,
                    "target_credit": req.target_credit,
                    "spread_width": req.spread_width,
                    "contracts": req.contracts,
                    "take_profit": req.take_profit,
                    "stop_loss": req.stop_loss,
                    "monitor_interval": req.monitor_interval,
                    "ibkr_host": req.ibkr.host,
                    "ibkr_port": req.ibkr.port,
                    "ibkr_client_id": req.ibkr.client_id,
                    "ibkr_account": req.ibkr.account,
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

            # Persist IBKR order records
            for order in state.orders:
                ibkr_rec = IBKROrder(
                    order_id=order.order_id,
                    session_id=state.session_id,
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
                session.merge(ibkr_rec)
            session.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
