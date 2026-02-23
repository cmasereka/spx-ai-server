"""
WebSocket Manager for SPX AI Trading Platform
Handles real-time communication with frontend clients.
"""

import json
from typing import Dict, List
from fastapi import WebSocket
from loguru import logger


class WebSocketManager:
    """Manages WebSocket connections and real-time messaging"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_backtests: Dict[str, List[str]] = {}  # client_id -> [backtest_ids]
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_backtests[client_id] = []
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connection",
                "message": "Connected to SPX AI Trading Platform",
                "client_id": client_id
            },
            client_id
        )
    
    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_backtests:
            del self.client_backtests[client_id]
        
        logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to a specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                
                # Ensure message is properly formatted
                if isinstance(message, str):
                    await websocket.send_text(message)
                else:
                    await websocket.send_text(json.dumps(message, default=str))
                    
            except Exception as e:
                logger.error(f"Failed to send message to client {client_id}: {e}")
                # Remove dead connection
                self.disconnect(client_id)
    
    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        dead_connections = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                if isinstance(message, str):
                    await websocket.send_text(message)
                else:
                    await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to broadcast to client {client_id}: {e}")
                dead_connections.append(client_id)
        
        # Clean up dead connections
        for client_id in dead_connections:
            self.disconnect(client_id)
    
    async def send_backtest_update(self, backtest_id: str, update_type: str, data: dict):
        """Send backtest-specific update to interested clients"""
        message = {
            "type": update_type,
            "backtest_id": backtest_id,
            "data": data,
            "timestamp": str(data.get("timestamp", ""))
        }
        
        # Send to all clients for now (can be more targeted later)
        await self.broadcast_message(message)
        
        logger.debug(f"Sent {update_type} update for backtest {backtest_id}")
    
    async def send_backtest_progress(self, backtest_id: str, current_step: int, 
                                   total_steps: int, current_date: str = None):
        """Send progress update for a backtest"""
        progress_data = {
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_percentage": round((current_step / total_steps) * 100, 1),
            "current_date": current_date
        }
        
        await self.send_backtest_update(backtest_id, "progress", progress_data)
    
    async def send_trade_result(self, backtest_id: str, trade_result: dict):
        """Send individual trade result"""
        await self.send_backtest_update(backtest_id, "trade_result", trade_result)
    
    async def send_backtest_completed(self, backtest_id: str, summary: dict):
        """Send backtest completion notification"""
        await self.send_backtest_update(backtest_id, "backtest_completed", summary)
    
    async def send_backtest_error(self, backtest_id: str, error_message: str):
        """Send backtest error notification"""
        error_data = {
            "error": error_message,
            "backtest_id": backtest_id
        }
        await self.send_backtest_update(backtest_id, "error", error_data)
    
    def get_connected_clients(self) -> List[str]:
        """Get list of connected client IDs"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """Get number of connected clients"""
        return len(self.active_connections)