"""
Backtest Service for SPX AI Trading Platform
Handles async backtesting operations and state management.
"""

import asyncio
import time
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

from loguru import logger

from .models import (
    BacktestRequest, BacktestStatus, BacktestResult, SystemStatus,
    BacktestStatusEnum, ProgressInfo, StrategyDetails
)
from .websocket_manager import WebSocketManager

# Import existing backtesting components
from enhanced_multi_strategy import EnhancedBacktestingEngine
from enhanced_backtest import StrategyType, EnhancedBacktestResult, DayBacktestResult

# Import database components
from src.database.connection import db_manager
from src.database.models import BacktestRun, Trade


class BacktestService:
    """Service for managing async backtesting operations"""
    
    def __init__(self):
        self.backtests: Dict[str, BacktestStatus] = {}
        self.backtest_results: Dict[str, List[BacktestResult]] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent backtests
        self.engine: Optional[EnhancedBacktestingEngine] = None
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize the backtest service"""
        try:
            # Initialize the enhanced backtesting engine
            logger.info("Initializing Enhanced Backtesting Engine...")
            self.engine = EnhancedBacktestingEngine("data/processed/parquet_1m")
            
            logger.info("✅ Backtest service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backtest service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled running backtest: {task_id}")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        logger.info("Backtest service cleanup complete")
    
    async def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        if not self.engine:
            raise RuntimeError("Backtest engine not initialized")
        
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get available dates from engine
        available_dates = self.engine.available_dates or []
        
        return SystemStatus(
            status="online",
            version="1.0.0",
            uptime_seconds=time.time() - self.start_time,
            available_dates=available_dates,
            date_range={
                "start": min(available_dates) if available_dates else date.today(),
                "end": max(available_dates) if available_dates else date.today()
            },
            total_dates=len(available_dates),
            memory_usage_mb=memory_mb,
            active_backtests=len([b for b in self.backtests.values() 
                               if b.status == BacktestStatusEnum.RUNNING])
        )
    
    def get_backtest_status(self, backtest_id: str) -> Optional[BacktestStatus]:
        """Get status of a specific backtest"""
        return self.backtests.get(backtest_id)
    
    def get_backtest_results(self, backtest_id: str) -> Optional[List[BacktestResult]]:
        """Get results of a completed backtest"""
        return self.backtest_results.get(backtest_id)
    
    def list_backtests(self) -> List[BacktestStatus]:
        """List all backtests sorted by most recent first"""
        backtests = list(self.backtests.values())
        # Sort by created_at timestamp, most recent first
        backtests.sort(key=lambda x: x.created_at, reverse=True)
        return backtests
    
    async def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel a running backtest"""
        if backtest_id in self.running_tasks:
            task = self.running_tasks[backtest_id]
            if not task.done():
                task.cancel()
                
                # Update status
                if backtest_id in self.backtests:
                    self.backtests[backtest_id].status = BacktestStatusEnum.CANCELLED
                    self.backtests[backtest_id].completed_at = datetime.now()
                
                logger.info(f"Cancelled backtest: {backtest_id}")
                return True
        
        return False
    
    async def run_backtest(self, backtest_id: str, request: BacktestRequest, 
                          websocket_manager: WebSocketManager):
        """Run backtest asynchronously"""
        
        # Create initial status
        status = BacktestStatus(
            backtest_id=backtest_id,
            status=BacktestStatusEnum.PENDING,
            mode=request.mode,
            created_at=datetime.now(),
            total_trades=0,
            successful_trades=0
        )
        
        self.backtests[backtest_id] = status
        
        # Save backtest run to database
        await self._save_backtest_run_to_db(backtest_id, request, status)
        
        try:
            # Update status to running
            status.status = BacktestStatusEnum.RUNNING
            status.started_at = datetime.now()
            
            # Update database with running status
            await self._update_backtest_run_status(backtest_id, status)
            
            await websocket_manager.send_backtest_update(
                backtest_id, "status_change", {"status": "running"}
            )
            
            # Send initial progress
            await websocket_manager.send_backtest_progress(backtest_id, 0, 1, "Starting backtest...")
            
            # Run the actual backtest in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Create a wrapper that will handle progress updates
            results = await self._run_backtest_with_progress(
                backtest_id, request, websocket_manager, loop
            )
            
            # Store results in memory (trades already persisted per-trade during the run)
            self.backtest_results[backtest_id] = results

            # Calculate final statistics
            total_pnl = sum(r.pnl for r in results)
            max_drawdown = self._calculate_max_drawdown(results)
            win_rate = (status.successful_trades / status.total_trades * 100) if status.total_trades > 0 else 0
            
            # Update final status
            status.status = BacktestStatusEnum.COMPLETED
            status.completed_at = datetime.now()
            status.total_trades = len(results)
            status.successful_trades = len([r for r in results if r.is_winner])
            
            # Update database with final results
            await self._update_backtest_run_final_results(backtest_id, status, total_pnl, max_drawdown, win_rate)
            
            # Send completion notification
            await websocket_manager.send_backtest_completed(
                backtest_id, 
                {
                    "total_trades": status.total_trades,
                    "successful_trades": status.successful_trades,
                    "win_rate": win_rate,
                    "total_pnl": total_pnl,
                    "avg_pnl": total_pnl / len(results) if results else 0,
                    "max_drawdown": max_drawdown
                }
            )
            
            logger.info(f"Backtest {backtest_id} completed successfully with {len(results)} trades")
            
        except asyncio.CancelledError:
            status.status = BacktestStatusEnum.CANCELLED
            status.completed_at = datetime.now()
            
            # Update database with cancelled status
            await self._update_backtest_run_status(backtest_id, status)
            logger.info(f"Backtest {backtest_id} was cancelled")
            
        except Exception as e:
            status.status = BacktestStatusEnum.FAILED
            status.completed_at = datetime.now()
            status.error_message = str(e)
            
            # Update database with failed status
            await self._update_backtest_run_status(backtest_id, status)
            
            await websocket_manager.send_backtest_error(backtest_id, str(e))
            logger.error(f"Backtest {backtest_id} failed: {e}")
        
        finally:
            # Clean up running task
            if backtest_id in self.running_tasks:
                del self.running_tasks[backtest_id]
    
    async def _run_backtest_with_progress(self, backtest_id: str, request: BacktestRequest,
                                        websocket_manager: WebSocketManager, loop) -> List[BacktestResult]:
        """Run backtest with progress updates"""

        if not self.engine:
            raise RuntimeError("Backtest engine not initialized")

        results = []

        # Determine dates to test
        if request.mode.value == "single_day":
            test_dates = [request.single_date] if request.single_date else []
        elif request.mode.value == "specific_dates":
            test_dates = sorted(request.specific_dates) if request.specific_dates else []
        else:
            # Filter available dates by range
            available_dates = self.engine.available_dates or []

            # Convert string dates to date objects for comparison
            from datetime import datetime

            test_dates = []
            for date_str in available_dates:
                try:
                    # Convert string date to date object
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()

                    # Check if date falls within requested range
                    if (not request.start_date or date_obj >= request.start_date) and \
                       (not request.end_date or date_obj <= request.end_date):
                        test_dates.append(date_obj)

                except ValueError:
                    logger.warning(f"Invalid date format in available_dates: {date_str}")
                    continue

        total_dates = len(test_dates)

        # Track open positions keyed by (strategy_type, entry_time) so we can
        # match them to subsequent monitor_tick and position_closed events.
        open_position_ids: dict = {}
        day_results: list = []  # populated by callback's position_closed handler

        def make_progress_callback(day_backtest_id: str):
            """Return a thread-safe callback for one day's backtest run."""
            import uuid as _uuid

            def callback(event: dict):
                ev = event.get("event")
                if ev == "position_opened":
                    key = (event.get("strategy_type", ""), event.get("entry_time", ""))
                    position_id = str(_uuid.uuid4())
                    open_position_ids[key] = position_id
                    # Convert strike selection object to plain dict
                    raw_strikes = event.get("strikes")
                    if hasattr(raw_strikes, "__dict__"):
                        strikes = {k: v for k, v in vars(raw_strikes).items() if not k.startswith("_")}
                    elif hasattr(raw_strikes, "_asdict"):
                        strikes = raw_strikes._asdict()
                    else:
                        strikes = raw_strikes or {}
                    payload = {
                        "position_id":     position_id,
                        "strategy_type":   event.get("strategy_type", ""),
                        "entry_time":      event.get("entry_time", ""),
                        "entry_spx_price": float(event.get("entry_spx", 0)),
                        "entry_credit":    float(event.get("entry_credit", 0)),
                        "strikes":         strikes,
                        "entry_rationale": event.get("entry_rationale"),
                    }
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        websocket_manager.send_backtest_update(day_backtest_id, "position_opened", payload)
                    )

                elif ev == "monitor_tick":
                    key = (event.get("strategy_type", ""), event.get("entry_time", ""))
                    position_id = open_position_ids.get(key)
                    if position_id:
                        payload = {
                            "position_id":            position_id,
                            "strategy_type":          event.get("strategy_type"),
                            "entry_time":             event.get("entry_time"),
                            "time":                   event.get("time"),
                            "spx":                    event.get("spx"),
                            "pnl_per_share":          event.get("pnl_per_share"),
                            "entry_credit_per_share": event.get("entry_credit_per_share"),
                        }
                        loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            websocket_manager.send_backtest_update(day_backtest_id, "position_update", payload)
                        )

                elif ev == "position_closed":
                    strategy_type = event.get("strategy_type", "")
                    result = event.get("result")
                    entry_time = getattr(result, "entry_time", event.get("entry_time", ""))
                    key = (strategy_type, entry_time)
                    open_position_ids.pop(key, None)
                    if result:
                        api_result = self._convert_single_trade(result, day_backtest_id)
                        # Schedule DB save + trade_result WS message from the thread
                        loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            self._save_and_send_trade(day_backtest_id, api_result, websocket_manager)
                        )
                        day_results.append(api_result)

            return callback

        for i, test_date in enumerate(test_dates, 1):
            try:
                # Send progress update
                await websocket_manager.send_backtest_progress(
                    backtest_id, i, total_dates, str(test_date)
                )

                callback = make_progress_callback(backtest_id)
                day_results.clear()

                # Run single day intraday scan in thread pool.
                # Trades are sent via callback as they close; day_results is populated there.
                await loop.run_in_executor(
                    self.executor,
                    self._run_single_day_backtest,
                    test_date,
                    request,
                    callback,
                )

                # Wait briefly for all queued WS sends to flush before moving to the next day
                await asyncio.sleep(0.05)

                # Accumulate into overall results (already persisted + sent by callback)
                results.extend(day_results)

            except Exception as e:
                logger.error(f"Failed to process date {test_date}: {e}")
                continue

        return results

    def _run_single_day_backtest(self, test_date, request: BacktestRequest, progress_callback=None) -> Optional[DayBacktestResult]:
        """Run a single day intraday backtest — safe for thread pool execution."""
        try:
            date_str = test_date.strftime('%Y-%m-%d') if hasattr(test_date, 'strftime') else str(test_date)
            return self.engine.backtest_day_intraday(
                date=date_str,
                take_profit=request.take_profit,
                stop_loss=request.stop_loss,
                monitor_interval=request.monitor_interval,
                min_spread_width=request.spread_width,
                target_credit=request.target_credit,
                strategy_mode=request.strategy.value,
                quantity=request.contracts,
                entry_start_time=request.entry_start_time,
                last_entry_time=request.last_entry_time,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.error(f"Single day backtest failed for {test_date}: {e}")
            return None

    async def _save_and_send_trade(
        self,
        backtest_id: str,
        api_result: BacktestResult,
        websocket_manager: WebSocketManager,
    ):
        """Persist a single trade to DB and broadcast it via WebSocket."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._save_single_trade_sync,
                backtest_id,
                api_result,
            )
        except Exception as exc:
            logger.error(f"Failed to persist trade for backtest {backtest_id}: {exc}")
        await websocket_manager.send_trade_result(backtest_id, api_result.model_dump(mode="json"))
    
    def _convert_single_trade(self, result: EnhancedBacktestResult,
                              backtest_id: str) -> BacktestResult:
        """Convert engine result to API format"""
        from enhanced_backtest import StrategyType as ST
        from delta_strike_selector import IronCondorStrikeSelection

        ss = result.strike_selection
        is_ic = (result.strategy_type == ST.IRON_CONDOR)

        if is_ic and isinstance(ss, IronCondorStrikeSelection):
            strikes = {
                "put_long": ss.put_long_strike,
                "put_short": ss.put_short_strike,
                "call_short": ss.call_short_strike,
                "call_long": ss.call_long_strike,
            }
        elif is_ic:
            # IC but stored as single StrikeSelection (legacy / fallback path)
            strikes = {
                "put_long": getattr(ss, 'long_strike', 0) if ss else 0,
                "put_short": getattr(ss, 'short_strike', 0) if ss else 0,
                "call_short": getattr(ss, 'short_strike', 0) if ss else 0,
                "call_long": getattr(ss, 'long_strike', 0) if ss else 0,
            }
        elif result.strategy_type == ST.PUT_SPREAD:
            strikes = {
                "put_long": getattr(ss, 'long_strike', 0) if ss else 0,
                "put_short": getattr(ss, 'short_strike', 0) if ss else 0,
                "call_short": 0,
                "call_long": 0,
            }
        else:  # CALL_SPREAD
            strikes = {
                "put_long": 0,
                "put_short": 0,
                "call_short": getattr(ss, 'short_strike', 0) if ss else 0,
                "call_long": getattr(ss, 'long_strike', 0) if ss else 0,
            }

        # Create strategy details
        strategy_details = StrategyDetails(
            strategy_type=result.strategy_type.value if result.strategy_type else "unknown",
            strikes=strikes,
            entry_credit=result.entry_credit or 0,
            max_profit=result.max_profit or 0,
            max_loss=result.max_loss or 0,
            breakeven_points=[]  # Can be calculated later if needed
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

    # Backward-compatible alias
    _convert_result_to_api_format = _convert_single_trade

    # Database methods
    async def _save_backtest_run_to_db(self, backtest_id: str, request: BacktestRequest, status: BacktestStatus):
        """Save backtest run to database"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_backtest_run_sync, backtest_id, request, status)
        except Exception as e:
            logger.error(f"Failed to save backtest run to database: {e}")
    
    def _save_backtest_run_sync(self, backtest_id: str, request: BacktestRequest, status: BacktestStatus):
        """Synchronous database save for backtest run"""
        with db_manager.get_session() as session:
            backtest_run = BacktestRun(
                backtest_id=backtest_id,
                mode=request.mode.value,
                strategy_type=request.strategy.value,

                # Date configuration
                start_date=request.start_date,
                end_date=request.end_date,
                single_date=request.single_date,

                # Strategy parameters
                target_delta=0.0,
                put_distance=0,
                call_distance=0,
                spread_width=request.spread_width,

                # Risk management
                decay_threshold=0.0,

                # Execution parameters
                entry_time="09:35:00",
                monitor_interval=request.monitor_interval,

                # Status
                status=status.status.value,
                created_at=status.created_at,
                started_at=status.started_at,

                # Active parameters
                parameters={
                    "strategy": request.strategy.value,
                    "contracts": request.contracts,
                    "target_credit": request.target_credit,
                    "spread_width": request.spread_width,
                    "take_profit": request.take_profit,
                    "stop_loss": request.stop_loss,
                    "monitor_interval": request.monitor_interval,
                    "specific_dates": [str(d) for d in request.specific_dates] if request.specific_dates else None,
                }
            )
            
            session.add(backtest_run)
            session.commit()
            logger.info(f"Saved backtest run {backtest_id} to database")
    
    async def _update_backtest_run_status(self, backtest_id: str, status: BacktestStatus):
        """Update backtest run status in database"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._update_backtest_run_status_sync, backtest_id, status)
        except Exception as e:
            logger.error(f"Failed to update backtest run status: {e}")
    
    def _update_backtest_run_status_sync(self, backtest_id: str, status: BacktestStatus):
        """Synchronous database update for backtest run status"""
        with db_manager.get_session() as session:
            backtest_run = session.query(BacktestRun).filter_by(backtest_id=backtest_id).first()
            if backtest_run:
                backtest_run.status = status.status.value
                backtest_run.started_at = status.started_at
                backtest_run.completed_at = status.completed_at
                backtest_run.error_message = status.error_message
                session.commit()
    
    async def _update_backtest_run_final_results(self, backtest_id: str, status: BacktestStatus, 
                                               total_pnl: float, max_drawdown: float, win_rate: float):
        """Update backtest run with final results"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._update_backtest_run_final_results_sync, 
                                     backtest_id, status, total_pnl, max_drawdown, win_rate)
        except Exception as e:
            logger.error(f"Failed to update backtest run final results: {e}")
    
    def _update_backtest_run_final_results_sync(self, backtest_id: str, status: BacktestStatus,
                                              total_pnl: float, max_drawdown: float, win_rate: float):
        """Synchronous database update for final results"""
        with db_manager.get_session() as session:
            backtest_run = session.query(BacktestRun).filter_by(backtest_id=backtest_id).first()
            if backtest_run:
                backtest_run.status = status.status.value
                backtest_run.completed_at = status.completed_at
                backtest_run.total_trades = status.total_trades
                backtest_run.successful_trades = status.successful_trades
                backtest_run.total_pnl = total_pnl
                backtest_run.max_drawdown = max_drawdown
                backtest_run.win_rate = win_rate
                session.commit()
    
    def _save_single_trade_sync(self, backtest_id: str, result: BacktestResult):
        """Persist one trade to the database immediately after it closes."""
        with db_manager.get_session() as session:
            backtest_run = session.query(BacktestRun).filter_by(backtest_id=backtest_id).first()
            if not backtest_run:
                logger.error(f"Backtest run {backtest_id} not found — cannot save trade {result.trade_id}")
                return

            trade = Trade(
                trade_id=result.trade_id,
                backtest_run_id=backtest_run.id,
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
            session.add(trade)
            session.commit()
            logger.debug(f"Persisted trade {result.trade_id} for backtest {backtest_id}")

    async def _save_trades_to_db(self, backtest_id: str, results: List[BacktestResult]):
        """Save individual trades to database"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_trades_to_db_sync, backtest_id, results)
        except Exception as e:
            logger.error(f"Failed to save trades to database: {e}")
    
    def _save_trades_to_db_sync(self, backtest_id: str, results: List[BacktestResult]):
        """Synchronous database save for trades"""
        with db_manager.get_session() as session:
            # Get the backtest run ID
            backtest_run = session.query(BacktestRun).filter_by(backtest_id=backtest_id).first()
            if not backtest_run:
                logger.error(f"Backtest run {backtest_id} not found in database")
                return
            
            for result in results:
                trade = Trade(
                    trade_id=result.trade_id,
                    backtest_run_id=backtest_run.id,
                    
                    # Trade timing
                    trade_date=result.trade_date,
                    entry_time=result.entry_time,
                    exit_time=result.exit_time,
                    
                    # Market data
                    entry_spx_price=result.entry_spx_price,
                    exit_spx_price=result.exit_spx_price,
                    
                    # Strategy details
                    strategy_type=result.strategy.strategy_type,
                    strikes=result.strategy.strikes,
                    
                    # Trade performance
                    entry_credit=result.entry_credit,
                    exit_cost=result.exit_cost,
                    pnl=result.pnl,
                    pnl_percentage=result.pnl_percentage,
                    
                    # Trade metadata
                    exit_reason=result.exit_reason,
                    is_winner=result.is_winner,
                    max_profit=result.strategy.max_profit,
                    max_loss=result.strategy.max_loss,
                    
                    # Strategy and monitoring data
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
                    monitoring_data=result.monitoring_points
                )
                
                session.add(trade)
            
            session.commit()
            logger.info(f"Saved {len(results)} trades for backtest {backtest_id} to database")
    
    def _calculate_max_drawdown(self, results: List[BacktestResult]) -> float:
        """Calculate maximum drawdown from results"""
        if not results:
            return 0.0
        
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        
        for result in results:
            cumulative_pnl += result.pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown