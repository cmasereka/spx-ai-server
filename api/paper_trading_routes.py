"""
Paper Trading API Routes

Endpoints:
  POST   /api/v1/paper-trading/sessions          — start a new session
  GET    /api/v1/paper-trading/sessions          — list all sessions
  GET    /api/v1/paper-trading/sessions/{id}     — get session status
  DELETE /api/v1/paper-trading/sessions/{id}     — stop a running session
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from .models import PaperTradingRequest, PaperTradingStatus
from .paper_trading_service import PaperTradingService
from .websocket_manager import WebSocketManager

router = APIRouter(prefix="/api/v1/paper-trading", tags=["Paper Trading"])

# Injected by main.py after services are initialised
paper_trading_service: PaperTradingService = None   # type: ignore[assignment]
websocket_manager: WebSocketManager = None          # type: ignore[assignment]


def init_router(service: PaperTradingService, ws_manager: WebSocketManager):
    """Call this from main.py lifespan once services are ready."""
    global paper_trading_service, websocket_manager
    paper_trading_service = service
    websocket_manager = ws_manager


@router.post("/sessions", response_model=dict, status_code=201)
async def start_paper_trading_session(request: PaperTradingRequest):
    """
    Start a new paper trading session.

    **simulation** mode replays the chosen date (defaults to the most recent
    available date) through the full strategy engine immediately.

    **live** mode processes today's data as bars become available in real time
    (requires the Parquet data for today to be present on disk).
    """
    try:
        session_id = await paper_trading_service.start_session(request, websocket_manager)
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Paper trading session started",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to start paper trading session: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions", response_model=list)
async def list_paper_trading_sessions():
    """List all paper trading sessions (most recent first)."""
    try:
        sessions = paper_trading_service.list_sessions()
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return [s.dict() for s in sessions]
    except Exception as exc:
        logger.error(f"Failed to list paper trading sessions: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions/{session_id}", response_model=PaperTradingStatus)
async def get_paper_trading_session(session_id: str):
    """Get the current status of a paper trading session."""
    status = paper_trading_service.get_session(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return status


@router.delete("/sessions/{session_id}")
async def stop_paper_trading_session(session_id: str):
    """Stop a running paper trading session."""
    try:
        stopped = await paper_trading_service.stop_session(session_id)
        if not stopped:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session stopped", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to stop session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
