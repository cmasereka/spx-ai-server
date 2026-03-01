"""
Live Trading API Routes (IBKR paper / live)

Endpoints:
  POST   /api/v1/trading/sessions          — start a new IBKR session
  GET    /api/v1/trading/sessions          — list all sessions
  GET    /api/v1/trading/sessions/{id}     — get session status
  DELETE /api/v1/trading/sessions/{id}     — stop a running session
  POST   /api/v1/trading/test-connection   — test IBKR connectivity
"""

import asyncio
from fastapi import APIRouter, HTTPException
from loguru import logger

from .models import LiveTradingRequest, LiveTradingStatus, IBKRConnectionConfig
from .live_trading_service import LiveTradingService
from .websocket_manager import WebSocketManager

router = APIRouter(prefix="/api/v1/trading", tags=["Live Trading"])

# Injected by main.py after services are initialised
live_trading_service: LiveTradingService = None   # type: ignore[assignment]
websocket_manager: WebSocketManager = None        # type: ignore[assignment]


def init_router(service: LiveTradingService, ws_manager: WebSocketManager):
    """Call this from main.py lifespan once services are ready."""
    global live_trading_service, websocket_manager
    live_trading_service = service
    websocket_manager = ws_manager


@router.post("/sessions", response_model=dict, status_code=201)
async def start_live_trading_session(request: LiveTradingRequest):
    """
    Start a new IBKR paper / live trading session.

    Set **ibkr.port** to:
    - `7497` for TWS paper trading (default)
    - `7496` for TWS live trading
    - `4002` for IB Gateway paper trading
    - `4001` for IB Gateway live trading

    The session will attempt to connect to IBKR immediately.  If the
    connection fails, the session status will be set to `failed`.
    """
    try:
        session_id = await live_trading_service.start_session(request, websocket_manager)
        return {
            "session_id": session_id,
            "status": "connecting",
            "message": "Live trading session initiated — connecting to IBKR",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to start live trading session: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions", response_model=list)
async def list_live_trading_sessions():
    """List all live trading sessions (most recent first)."""
    try:
        sessions = live_trading_service.list_sessions()
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return [s.dict() for s in sessions]
    except Exception as exc:
        logger.error(f"Failed to list live trading sessions: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions/{session_id}", response_model=LiveTradingStatus)
async def get_live_trading_session(session_id: str):
    """Get the current status of a live trading session."""
    status = live_trading_service.get_session(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return status


@router.delete("/sessions/{session_id}")
async def stop_live_trading_session(session_id: str):
    """Stop a running live trading session and close all open IBKR positions."""
    try:
        stopped = await live_trading_service.stop_session(session_id)
        if not stopped:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session stopped", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to stop live session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/test-connection", response_model=dict)
async def test_ibkr_connection(config: IBKRConnectionConfig):
    """
    Test IBKR TWS / IB Gateway connectivity without starting a trading session.

    Attempts to connect, retrieves the account summary, then disconnects.
    Returns connection status, account info, and the next valid order ID
    so you can confirm the API is fully functional.

    Use this before starting a live session to diagnose connection issues.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _test_ibkr_connection_sync, config
        )
        return result
    except Exception as exc:
        logger.error(f"IBKR connection test failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _test_ibkr_connection_sync(config: IBKRConnectionConfig) -> dict:
    """Blocking IBKR connection test — runs in a thread pool."""
    import asyncio
    import traceback

    # ib_insync's synchronous helpers call asyncio.get_event_loop() internally.
    # ThreadPoolExecutor worker threads have no event loop by default, so we
    # must create and set one before touching any ib_insync code.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        from ib_insync import IB
    except ImportError:
        return {
            "connected": False,
            "error": "ib_insync is not installed. Run: pip install ib_insync",
        }

    ib = IB()
    try:
        ib.connect(config.host, config.port, clientId=config.client_id, timeout=10)

        if not ib.isConnected():
            return {
                "connected": False,
                "host": config.host,
                "port": config.port,
                "error": "Connection returned but isConnected() is False",
            }

        # Fetch account summary to confirm API is fully operational
        account = config.account or (ib.managedAccounts()[0] if ib.managedAccounts() else "")
        summary = ib.accountSummary(account)
        next_order_id = ib.client.getReqId()

        fields = {}
        for item in summary:
            if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower", "AccountType"):
                fields[item.tag] = item.value

        return {
            "connected": True,
            "host": config.host,
            "port": config.port,
            "client_id": config.client_id,
            "account": account,
            "next_order_id": next_order_id,
            "account_summary": fields,
        }

    except Exception as exc:
        error_type = type(exc).__name__
        error_detail = str(exc)
        tb = traceback.format_exc()
        logger.error(f"IBKR connection test [{error_type}]: {error_detail}")
        return {
            "connected": False,
            "host": config.host,
            "port": config.port,
            "error_type": error_type,
            "error": error_detail,
            "traceback": tb,
        }
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
        loop.close()
