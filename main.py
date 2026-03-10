"""
SPX AI Trading Platform - FastAPI Service
Main application entry point for the web service.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, date
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.models import (
    BacktestRequest, BacktestResponse, BacktestStatus,
    BacktestResult, SystemStatus
)
from api.backtest_service import BacktestService
from api.websocket_manager import WebSocketManager
from api import database_routes
from api.live_trading_service import LiveTradingService
import api.live_trading_routes as live_trading_routes
from api.auth import get_approved_user
from api.auth_routes import router as auth_router
from api.user_routes import router as user_router
from api.admin_routes import router as admin_router

# Global services
backtest_service = BacktestService()
live_trading_svc = LiveTradingService()
websocket_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 SPX AI Trading Platform starting up...")

    # Initialize services
    await backtest_service.initialize()
    logger.info("✅ Backtest service initialized")

    # Live trading service (IBKR) — reuses the same engine
    await live_trading_svc.initialize(backtest_service.engine)
    live_trading_routes.init_router(live_trading_svc, websocket_manager)
    logger.info("✅ Live trading service initialized")

    yield

    # Cleanup
    await live_trading_svc.cleanup()
    await backtest_service.cleanup()
    logger.info("🛑 SPX AI Trading Platform shutting down")

# Create FastAPI application
app = FastAPI(
    title="SPX AI Trading Platform",
    description="Real-time SPX options backtesting and monitoring platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth routes (no authentication required)
app.include_router(auth_router)

# User / admin routes
app.include_router(user_router)
app.include_router(admin_router)

# Include database routes (protected)
app.include_router(database_routes.router, dependencies=[Depends(get_approved_user)])

# Include live (IBKR) trading routes (protected)
app.include_router(live_trading_routes.router, dependencies=[Depends(get_approved_user)])

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "SPX AI Trading Platform API",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/status", response_model=SystemStatus, tags=["System"])
async def get_system_status(_: None = Depends(get_approved_user)):
    """Get system status and available data"""
    try:
        status = await backtest_service.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/backtest/start", response_model=BacktestResponse, tags=["Backtesting"])
async def start_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(get_approved_user),
):
    """Start a new backtest"""
    try:
        # Generate unique backtest ID
        backtest_id = str(uuid.uuid4())
        
        # Start backtest in background
        background_tasks.add_task(
            backtest_service.run_backtest,
            backtest_id,
            request,
            websocket_manager
        )
        
        return BacktestResponse(
            backtest_id=backtest_id,
            status="started",
            message="Backtest started successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/backtest/{backtest_id}/status", response_model=BacktestStatus, tags=["Backtesting"])
async def get_backtest_status(backtest_id: str, _: None = Depends(get_approved_user)):
    """Get backtest status"""
    try:
        status = backtest_service.get_backtest_status(backtest_id)
        if not status:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/backtest/{backtest_id}/results", response_model=List[BacktestResult], tags=["Backtesting"])
async def get_backtest_results(backtest_id: str, _: None = Depends(get_approved_user)):
    """Get backtest results"""
    try:
        results = backtest_service.get_backtest_results(backtest_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/backtest/{backtest_id}", tags=["Backtesting"])
async def cancel_backtest(backtest_id: str, _: None = Depends(get_approved_user)):
    """Cancel a running backtest"""
    try:
        success = await backtest_service.cancel_backtest(backtest_id)
        if not success:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return {"message": "Backtest cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/backtest", tags=["Backtesting"])
async def list_backtests(_: None = Depends(get_approved_user)):
    """List all backtests (combines database and in-memory data, sorted by most recent)"""
    try:
        # Get in-memory backtests (includes active/running ones)
        memory_backtests = backtest_service.list_backtests()
        
        # Convert to dict format for easy merging
        result_backtests = []
        
        for backtest in memory_backtests:
            result_backtests.append({
                "backtest_id": backtest.backtest_id,
                "mode": backtest.mode.value,
                "status": backtest.status.value,
                "created_at": backtest.created_at,
                "started_at": backtest.started_at,
                "completed_at": backtest.completed_at,
                "total_trades": backtest.total_trades,
                "successful_trades": backtest.successful_trades,
                "error_message": backtest.error_message,
                "source": "memory"  # Indicates this is from in-memory store
            })
        
        return {"backtests": result_backtests}
        
    except Exception as e:
        logger.error(f"Failed to list backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message from {client_id}: {data}")
            
            # Echo back for now (can add more sophisticated handling later)
            await websocket_manager.send_personal_message(
                f"Received: {data}", 
                client_id
            )
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting SPX AI Trading Platform...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )