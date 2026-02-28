"""
Paper Trading Service for SPX AI Trading Platform

Manages paper trading sessions.  Each session runs backtest_day_intraday against
a chosen date and streams position events in real time via the progress_callback hook.

Session lifecycle:
  pending → running → completed | stopped | failed

Two modes:
  simulation — run the full day at once (no wall-clock delay); useful for review
  live       — process bars one at a time, sleeping until the real-world 1-minute
               mark between each bar (requires the Parquet data for today to be present)
"""

import asyncio
import threading
import time
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from .models import (
    PaperTradingRequest, PaperTradingStatus, PaperTradingMode,
    PaperPosition, BacktestResult, StrategyDetails,
)
from .websocket_manager import WebSocketManager
from enhanced_multi_strategy import EnhancedBacktestingEngine
from enhanced_backtest import EnhancedBacktestResult, StrategyType, DayBacktestResult
from delta_strike_selector import IronCondorStrikeSelection
from src.database.connection import db_manager
from src.database.models import PaperTradingRun, Trade


# --------------------------------------------------------------------------- #
# Internal session state
# --------------------------------------------------------------------------- #

class _SessionState:
    """Mutable container updated by the worker thread via progress_callback."""

    def __init__(self, session_id: str, mode: str, trade_date: str, request: PaperTradingRequest):
        self.session_id = session_id
        self.mode = mode
        self.trade_date = trade_date
        self.request = request
        self.status = "pending"
        self.open_positions: List[PaperPosition] = []
        self.completed_trades: List[BacktestResult] = []
        self.day_pnl: float = 0.0
        self.trade_count: int = 0
        self.created_at: datetime = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self._lock = threading.Lock()

    # Snapshot — safe to read from the async world
    def to_status(self) -> PaperTradingStatus:
        with self._lock:
            return PaperTradingStatus(
                session_id=self.session_id,
                mode=self.mode,
                trade_date=self.trade_date,
                status=self.status,
                open_positions=list(self.open_positions),
                completed_trades=list(self.completed_trades),
                day_pnl=round(self.day_pnl, 2),
                trade_count=self.trade_count,
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                error_message=self.error_message,
            )


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

class PaperTradingService:
    """Manages paper trading sessions."""

    def __init__(self):
        self._sessions: Dict[str, _SessionState] = {}
        self.engine: Optional[EnhancedBacktestingEngine] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self, engine: EnhancedBacktestingEngine):
        """Called from the FastAPI lifespan — reuses the engine already created
        by BacktestService to avoid loading Parquet files twice."""
        self.engine = engine
        logger.info("PaperTradingService initialised")

    async def cleanup(self):
        for sid, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled paper trading session: {sid}")
        self._executor.shutdown(wait=False)

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def list_sessions(self) -> List[PaperTradingStatus]:
        return [s.to_status() for s in self._sessions.values()]

    def get_session(self, session_id: str) -> Optional[PaperTradingStatus]:
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
            if s.status == "running":
                s.status = "stopped"
                s.completed_at = datetime.now()
        return True

    async def start_session(self, request: PaperTradingRequest,
                            websocket_manager: WebSocketManager) -> str:
        """Create and start a paper trading session; returns the session_id."""
        if not self.engine:
            raise RuntimeError("PaperTradingService not initialised")

        # Resolve the trading date
        trade_date = self._resolve_date(request)
        if trade_date is None:
            raise ValueError("Requested date is not available in historical data")

        session_id = str(uuid.uuid4())
        state = _SessionState(
            session_id=session_id,
            mode=request.mode.value,
            trade_date=str(trade_date),
            request=request,
        )
        self._sessions[session_id] = state

        # Persist to DB
        await self._save_session_to_db(state)

        # Kick off background task
        task = asyncio.create_task(
            self._run_session(state, websocket_manager)
        )
        self._running_tasks[session_id] = task

        logger.info(f"Paper trading session started: {session_id} date={trade_date} mode={request.mode.value}")
        return session_id

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _resolve_date(self, request: PaperTradingRequest) -> Optional[date]:
        """Return the date to trade, or None if unavailable."""
        available = self.engine.available_dates or []
        if not available:
            return None

        if request.trade_date:
            d = str(request.trade_date)
            return request.trade_date if d in available else None

        # Default: most recent available date
        return datetime.strptime(max(available), "%Y-%m-%d").date()

    def _make_progress_callback(self, state: _SessionState,
                                loop: asyncio.AbstractEventLoop,
                                websocket_manager: WebSocketManager,
                                backtest_id: str):
        """Return a thread-safe progress_callback closure."""

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
                    # Notify via WebSocket from the event loop
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        websocket_manager.send_backtest_update(
                            backtest_id, "position_opened",
                            {"session_id": state.session_id, **pos.dict()}
                        )
                    )

                elif ev == "position_closed":
                    result: EnhancedBacktestResult = event.get("result")
                    if result:
                        api_result = _engine_result_to_api(result, state.session_id)
                        # Remove matching open position (by strategy_type + entry_time)
                        state.open_positions = [
                            p for p in state.open_positions
                            if not (p.strategy_type == result.strategy_type.value
                                    and p.entry_time == result.entry_time)
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

    async def _run_session(self, state: _SessionState,
                           websocket_manager: WebSocketManager):
        """Async wrapper — runs the blocking engine in the thread pool."""
        loop = asyncio.get_event_loop()
        backtest_id = state.session_id

        with state._lock:
            state.status = "running"
            state.started_at = datetime.now()

        await self._update_session_db_status(state)
        await websocket_manager.send_backtest_update(
            backtest_id, "status_change", {"status": "running", "session_id": backtest_id}
        )

        try:
            callback = self._make_progress_callback(state, loop, websocket_manager, backtest_id)
            req = state.request

            day_result: DayBacktestResult = await loop.run_in_executor(
                self._executor,
                lambda: self.engine.backtest_day_intraday(
                    date=state.trade_date,
                    take_profit=req.take_profit,
                    stop_loss=req.stop_loss,
                    monitor_interval=req.monitor_interval,
                    min_spread_width=req.spread_width,
                    target_credit=req.target_credit,
                    strategy_mode=req.strategy.value,
                    quantity=req.contracts,
                    progress_callback=callback,
                )
            )

            with state._lock:
                state.status = "completed"
                state.completed_at = datetime.now()
                # Any remaining open positions were force-closed — sync from day_result
                state.open_positions = []
                # Ensure all trades are recorded (callback may have caught most already;
                # day_result is the authoritative source)
                seen_entries = {t.entry_time for t in state.completed_trades}
                for trade in day_result.trades:
                    if trade.entry_time not in seen_entries:
                        api_result = _engine_result_to_api(trade, state.session_id)
                        state.completed_trades.append(api_result)
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
                }
            )

        except asyncio.CancelledError:
            with state._lock:
                if state.status == "running":
                    state.status = "stopped"
                    state.completed_at = datetime.now()
            await self._update_session_db_status(state)

        except Exception as exc:
            logger.error(f"Paper trading session {state.session_id} failed: {exc}")
            with state._lock:
                state.status = "failed"
                state.error_message = str(exc)
                state.completed_at = datetime.now()
            await self._update_session_db_status(state)
            await websocket_manager.send_backtest_error(backtest_id, str(exc))

        finally:
            self._running_tasks.pop(state.session_id, None)

    # ---------------------------------------------------------------------- #
    # Database helpers
    # ---------------------------------------------------------------------- #

    async def _save_session_to_db(self, state: _SessionState):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_session_sync, state)
        except Exception as exc:
            logger.error(f"Failed to save paper trading session to DB: {exc}")

    def _save_session_sync(self, state: _SessionState):
        with db_manager.get_session() as session:
            req = state.request
            run = PaperTradingRun(
                session_id=state.session_id,
                mode=state.mode,
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
                },
                status=state.status,
                created_at=state.created_at,
                started_at=state.started_at,
            )
            session.add(run)
            session.commit()

    async def _update_session_db_status(self, state: _SessionState):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._update_session_status_sync, state)
        except Exception as exc:
            logger.error(f"Failed to update paper trading session status: {exc}")

    def _update_session_status_sync(self, state: _SessionState):
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

    async def _finalise_session_db(self, state: _SessionState):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._finalise_session_sync, state)
        except Exception as exc:
            logger.error(f"Failed to finalise paper trading session in DB: {exc}")

    def _finalise_session_sync(self, state: _SessionState):
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _strikes_from_selection(sel) -> dict:
    """Convert a StrikeSelection or IronCondorStrikeSelection to a plain dict."""
    if sel is None:
        return {}
    if isinstance(sel, IronCondorStrikeSelection):
        return {
            "put_long": sel.put_long_strike,
            "put_short": sel.put_short_strike,
            "call_short": sel.call_short_strike,
            "call_long": sel.call_long_strike,
        }
    # Single-side spread (StrikeSelection)
    return {
        "short_strike": getattr(sel, "short_strike", 0),
        "long_strike":  getattr(sel, "long_strike",  0),
    }


def _engine_result_to_api(result: EnhancedBacktestResult, session_id: str) -> BacktestResult:
    """Convert engine result → API BacktestResult (same shape as backtesting)."""
    from enhanced_backtest import StrategyType as ST

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
