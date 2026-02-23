"""
Database query endpoints for retrieving backtest data
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc, func

from src.database.connection import db_manager
from src.database.models import BacktestRun, Trade
from .models import BacktestStatus, BacktestResult, TradeCheckpoint, TradeCheckpointsResponse

router = APIRouter(prefix="/api/v1/database", tags=["database"])

@router.get("/backtests", response_model=List[Dict[str, Any]])
async def list_backtests(
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """List all backtest runs from database"""
    try:
        with db_manager.get_session() as session:
            backtests = (
                session.query(BacktestRun)
                .order_by(desc(BacktestRun.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            results = []
            for backtest in backtests:
                results.append({
                    "backtest_id": backtest.backtest_id,
                    "mode": backtest.mode,
                    "strategy_type": backtest.strategy_type,
                    "start_date": backtest.start_date,
                    "end_date": backtest.end_date,
                    "single_date": backtest.single_date,
                    "target_delta": backtest.target_delta,
                    "decay_threshold": backtest.decay_threshold,
                    "status": backtest.status,
                    "created_at": backtest.created_at,
                    "started_at": backtest.started_at,
                    "completed_at": backtest.completed_at,
                    "total_trades": backtest.total_trades,
                    "successful_trades": backtest.successful_trades,
                    "total_pnl": backtest.total_pnl,
                    "max_drawdown": backtest.max_drawdown,
                    "win_rate": backtest.win_rate,
                    "error_message": backtest.error_message
                })
            
            return results
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/backtests/{backtest_id}")
async def get_backtest(backtest_id: str):
    """Get specific backtest run details"""
    try:
        with db_manager.get_session() as session:
            backtest = session.query(BacktestRun).filter_by(backtest_id=backtest_id).first()
            
            if not backtest:
                raise HTTPException(status_code=404, detail="Backtest not found")
            
            return {
                "backtest_id": backtest.backtest_id,
                "mode": backtest.mode,
                "strategy_type": backtest.strategy_type,
                "start_date": backtest.start_date,
                "end_date": backtest.end_date,
                "single_date": backtest.single_date,
                "target_delta": backtest.target_delta,
                "decay_threshold": backtest.decay_threshold,
                "status": backtest.status,
                "created_at": backtest.created_at,
                "started_at": backtest.started_at,
                "completed_at": backtest.completed_at,
                "total_trades": backtest.total_trades,
                "successful_trades": backtest.successful_trades,
                "total_pnl": backtest.total_pnl,
                "max_drawdown": backtest.max_drawdown,
                "win_rate": backtest.win_rate,
                "error_message": backtest.error_message,
                "parameters": backtest.parameters
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/backtests/{backtest_id}/trades")
async def get_backtest_trades(
    backtest_id: str,
    limit: int = Query(100, ge=1, le=500, description="Number of trades to return"),
    offset: int = Query(0, ge=0, description="Number of trades to skip")
):
    """Get trades for a specific backtest"""
    try:
        with db_manager.get_session() as session:
            # First check if backtest exists
            backtest = session.query(BacktestRun).filter_by(backtest_id=backtest_id).first()
            if not backtest:
                raise HTTPException(status_code=404, detail="Backtest not found")
            
            # Get trades for this backtest
            trades = (
                session.query(Trade)
                .filter_by(backtest_run_id=backtest.id)
                .order_by(Trade.trade_date, Trade.entry_time)
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            results = []
            for trade in trades:
                results.append({
                    "trade_id": trade.trade_id,
                    "trade_date": trade.trade_date,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_spx_price": trade.entry_spx_price,
                    "exit_spx_price": trade.exit_spx_price,
                    "strategy_type": trade.strategy_type,
                    "strikes": trade.strikes,
                    "entry_credit": trade.entry_credit,
                    "exit_cost": trade.exit_cost,
                    "pnl": trade.pnl,
                    "pnl_percentage": trade.pnl_percentage,
                    "exit_reason": trade.exit_reason,
                    "is_winner": trade.is_winner,
                    "max_profit": trade.max_profit,
                    "max_loss": trade.max_loss,
                    "strategy_details": trade.strategy_details,
                    "monitoring_data": trade.monitoring_data,
                    "created_at": trade.created_at
                })
            
            return {
                "backtest_id": backtest_id,
                "total_trades": len(trades),
                "trades": results
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/backtests/{backtest_id}/trades/{trade_id}/checkpoints",
            response_model=TradeCheckpointsResponse)
async def get_trade_checkpoints(backtest_id: str, trade_id: str):
    """
    Return the per-bar P&L and cost-to-close snapshots for a single trade.
    Use these to plot drawdown curves and model alternative stop-loss levels.
    """
    try:
        with db_manager.get_session() as session:
            backtest = session.query(BacktestRun).filter_by(backtest_id=backtest_id).first()
            if not backtest:
                raise HTTPException(status_code=404, detail="Backtest not found")

            trade = (
                session.query(Trade)
                .filter_by(trade_id=trade_id, backtest_run_id=backtest.id)
                .first()
            )
            if not trade:
                raise HTTPException(status_code=404, detail="Trade not found")

            raw_checkpoints: list = trade.monitoring_data or []
            checkpoints = [
                TradeCheckpoint(
                    time=cp.get("time", ""),
                    spx=cp.get("spx", 0.0),
                    cost_per_share=cp.get("cost_per_share", 0.0),
                    pnl_per_share=cp.get("pnl_per_share", 0.0),
                    put_cost_per_share=cp.get("put_cost_per_share"),
                    call_cost_per_share=cp.get("call_cost_per_share"),
                )
                for cp in raw_checkpoints
            ]

            params: dict = backtest.parameters or {}
            entry_credit_ps = round(trade.entry_credit / 100, 4) if trade.entry_credit else 0.0

            return TradeCheckpointsResponse(
                trade_id=trade.trade_id,
                trade_date=str(trade.trade_date),
                strategy_type=trade.strategy_type,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                entry_credit_per_share=entry_credit_ps,
                take_profit=params.get("take_profit", 0.10),
                stop_loss=params.get("stop_loss", 2.0),
                checkpoints=checkpoints,
                checkpoint_count=len(checkpoints),
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")



async def get_stats_summary():
    """Get summary statistics across all backtests"""
    try:
        with db_manager.get_session() as session:
            # Count backtests by status
            status_counts = (
                session.query(BacktestRun.status, func.count(BacktestRun.id))
                .group_by(BacktestRun.status)
                .all()
            )
            
            # Get total trades and P&L
            total_stats = (
                session.query(
                    func.sum(BacktestRun.total_trades),
                    func.sum(BacktestRun.successful_trades),
                    func.sum(BacktestRun.total_pnl),
                    func.avg(BacktestRun.win_rate),
                    func.count(BacktestRun.id)
                )
                .filter(BacktestRun.status == 'completed')
                .first()
            )
            
            # Recent backtests
            recent_backtests = (
                session.query(BacktestRun)
                .filter(BacktestRun.status == 'completed')
                .order_by(desc(BacktestRun.completed_at))
                .limit(10)
                .all()
            )
            
            return {
                "status_counts": dict(status_counts),
                "total_backtests": total_stats[4] or 0,
                "total_trades": total_stats[0] or 0,
                "total_successful_trades": total_stats[1] or 0,
                "total_pnl": float(total_stats[2] or 0),
                "average_win_rate": float(total_stats[3] or 0),
                "recent_backtests": [
                    {
                        "backtest_id": bt.backtest_id,
                        "completed_at": bt.completed_at,
                        "total_trades": bt.total_trades,
                        "total_pnl": bt.total_pnl,
                        "win_rate": bt.win_rate
                    }
                    for bt in recent_backtests
                ]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.delete("/backtests/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest and all its trades"""
    try:
        with db_manager.get_session() as session:
            # Find the backtest
            backtest = session.query(BacktestRun).filter_by(backtest_id=backtest_id).first()
            if not backtest:
                raise HTTPException(status_code=404, detail="Backtest not found")
            
            # Delete associated trades first (due to foreign key constraint)
            trades_deleted = session.query(Trade).filter_by(backtest_run_id=backtest.id).delete()
            
            # Delete the backtest run
            session.delete(backtest)
            session.commit()
            
            return {
                "message": f"Deleted backtest {backtest_id}",
                "trades_deleted": trades_deleted
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")