"""
API Models for SPX AI Trading Platform
Pydantic models for request/response validation and serialization.
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator


class BacktestModeEnum(str, Enum):
    """Backtest execution modes"""
    SINGLE_DAY = "single_day"
    DATE_RANGE = "date_range"
    LIVE_SIMULATION = "live_simulation"


class BacktestStatusEnum(str, Enum):
    """Backtest status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BacktestRequest(BaseModel):
    """Request model for starting a backtest"""
    mode: BacktestModeEnum = Field(..., description="Backtest execution mode")

    # Date configuration
    start_date: Optional[date] = Field(None, description="Start date for backtesting")
    end_date: Optional[date] = Field(None, description="End date for backtesting")
    single_date: Optional[date] = Field(None, description="Single date for backtesting")

    # Strike selection
    target_credit: float = Field(0.50, ge=0.05, le=5.0, description="Target net credit per spread per share. IC total will be 2x this value.")
    spread_width: int = Field(10, ge=5, le=100, description="Width of each spread in strike points")

    # Risk management
    take_profit: float = Field(0.10, ge=0.01, le=10.0, description="Take profit: exit when cost-to-close per spread per share drops to or below this absolute value. E.g. 0.10 exits when closing costs $0.10/share.")
    stop_loss: float = Field(2.0, ge=0.10, le=20.0, description="Stop loss: exit when cost-to-close per spread per share reaches this absolute value. E.g. 2.0 exits when closing costs $2.00/share.")
    monitor_interval: int = Field(1, ge=1, le=15, description="Minutes between position checks. 1 = every bar, 5 = every 5 minutes.")

    @field_validator('start_date', 'end_date', 'single_date')
    @classmethod
    def validate_dates(cls, v):
        if v and v > date.today():
            raise ValueError("Date cannot be in the future")
        return v

    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        if v and 'start_date' in info.data and info.data['start_date']:
            if v < info.data['start_date']:
                raise ValueError("End date must be after start date")
        return v


class BacktestResponse(BaseModel):
    """Response model for backtest operations"""
    backtest_id: str = Field(..., description="Unique identifier for the backtest")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ProgressInfo(BaseModel):
    """Progress information for running backtests"""
    current_step: int = Field(..., description="Current step number")
    total_steps: int = Field(..., description="Total number of steps")
    current_date: Optional[date] = Field(None, description="Currently processing date")
    elapsed_seconds: float = Field(..., description="Elapsed time in seconds")
    estimated_remaining_seconds: Optional[float] = Field(None, description="Estimated remaining time")


class BacktestStatus(BaseModel):
    """Status model for backtest execution"""
    backtest_id: str = Field(..., description="Unique identifier")
    status: BacktestStatusEnum = Field(..., description="Current status")
    mode: BacktestModeEnum = Field(..., description="Execution mode")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    progress: Optional[ProgressInfo] = Field(None, description="Progress information")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    total_trades: Optional[int] = Field(None, description="Total number of trades executed")
    successful_trades: Optional[int] = Field(None, description="Number of successful trades")


class StrategyDetails(BaseModel):
    """Details about the options strategy executed"""
    strategy_type: str = Field(..., description="Strategy type")
    strikes: Dict[str, float] = Field(..., description="Strike prices by position")
    entry_credit: float = Field(..., description="Credit received at entry")
    max_profit: float = Field(..., description="Maximum possible profit")
    max_loss: float = Field(..., description="Maximum possible loss")
    breakeven_points: List[float] = Field(..., description="Breakeven prices")


class BacktestResult(BaseModel):
    """Individual trade result from backtesting"""
    trade_id: str = Field(..., description="Unique trade identifier")
    trade_date: date = Field(..., description="Trade date")
    entry_time: str = Field(..., description="Entry time")
    exit_time: Optional[str] = Field(None, description="Exit time")
    
    # Market data
    entry_spx_price: float = Field(..., description="SPX price at entry")
    exit_spx_price: Optional[float] = Field(None, description="SPX price at exit")
    
    # Strategy details
    strategy: StrategyDetails = Field(..., description="Strategy details")
    
    # Trade performance
    entry_credit: float = Field(..., description="Credit received")
    exit_cost: float = Field(..., description="Cost to exit")
    pnl: float = Field(..., description="Profit/Loss")
    pnl_percentage: float = Field(..., description="P&L as percentage of credit")
    
    # Trade metadata
    exit_reason: str = Field(..., description="Reason for exit")
    is_winner: bool = Field(..., description="Whether trade was profitable")
    monitoring_points: List[Dict[str, Any]] = Field(default_factory=list, description="Monitoring data points")


class SystemStatus(BaseModel):
    """System status information"""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    
    # Data availability
    available_dates: List[date] = Field(..., description="Available trading dates")
    date_range: Dict[str, date] = Field(..., description="Date range of available data")
    total_dates: int = Field(..., description="Total number of available dates")
    
    # System resources
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    active_backtests: int = Field(..., description="Number of active backtests")


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    backtest_id: Optional[str] = Field(None, description="Related backtest ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    data: Dict[str, Any] = Field(..., description="Message payload")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")