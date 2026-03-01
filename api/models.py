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
    SPECIFIC_DATES = "specific_dates"
    LIVE_SIMULATION = "live_simulation"


class BacktestStrategyEnum(str, Enum):
    """Strategy mode controlling which trade types are considered"""
    IRON_CONDOR     = "iron_condor"       # IC entries only
    CREDIT_SPREADS  = "credit_spreads"    # Put / call spreads only, no IC
    IC_CREDIT_SPREADS = "ic_credit_spreads"  # All entry types (current behaviour)


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

    # Strategy mode
    strategy: BacktestStrategyEnum = Field(
        BacktestStrategyEnum.IRON_CONDOR,
        description="Which trade types to consider: iron_condor, credit_spreads, or ic_credit_spreads"
    )

    # Date configuration
    start_date: Optional[date] = Field(None, description="Start date for backtesting")
    end_date: Optional[date] = Field(None, description="End date for backtesting")
    single_date: Optional[date] = Field(None, description="Single date for backtesting")
    specific_dates: Optional[List[date]] = Field(None, description="List of specific dates for backtesting")

    # Strike selection
    target_credit: float = Field(0.40, ge=0.05, le=5.0, description="Target net credit per spread per share. IC total will be 2x this value.")
    spread_width: int = Field(10, ge=5, le=100, description="Width of each spread in strike points")

    # Position sizing
    contracts: int = Field(1, ge=1, le=100, description="Number of contracts per position")

    # Risk management
    take_profit: float = Field(0.05, ge=0.01, le=10.0, description="Take profit: exit when cost-to-close per spread per share drops to or below this absolute value. E.g. 0.10 exits when closing costs $0.10/share.")
    stop_loss: float = Field(6.0, ge=0.10, le=20.0, description="Stop loss: exit when cost-to-close per spread per share reaches this absolute value. E.g. 2.0 exits when closing costs $2.00/share.")
    monitor_interval: int = Field(5, ge=1, le=15, description="Minutes between position checks. 1 = every bar, 5 = every 5 minutes.")

    # Entry / exit time window
    entry_start_time: str = Field(
        "10:00:00",
        description="Time to begin scanning for entry signals (HH:MM:SS, 24-hour). Default 10:00:00."
    )
    last_entry_time: str = Field(
        "14:00:00",
        description="No new entries at or after this time (HH:MM:SS, 24-hour). Default 14:00:00."
    )

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

    # Decision audit trail
    entry_rationale: Optional[Dict[str, Any]] = Field(None, description="All factors that led to this trade being opened")
    exit_rationale: Optional[Dict[str, Any]] = Field(None, description="All factors that triggered the close")


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


class TradeCheckpoint(BaseModel):
    """Single monitoring snapshot recorded at each position check."""
    time: str = Field(..., description="Bar time HH:MM:SS")
    spx: float = Field(..., description="SPX price at this bar")
    cost_per_share: float = Field(..., description="Current cost to close the full position per share")
    pnl_per_share: float = Field(..., description="Unrealised P&L per share (entry_credit - cost_per_share)")
    # IC-only fields — absent for single spreads
    put_cost_per_share: Optional[float] = Field(None, description="Cost to close put side per share (IC only)")
    call_cost_per_share: Optional[float] = Field(None, description="Cost to close call side per share (IC only)")


class TradeCheckpointsResponse(BaseModel):
    """Checkpoint series for one trade, ready for charting."""
    trade_id: str
    trade_date: str
    strategy_type: str
    entry_time: str
    exit_time: Optional[str]
    entry_credit_per_share: float = Field(..., description="Credit collected at entry per share")
    take_profit: float = Field(..., description="Take-profit threshold used ($/share)")
    stop_loss: float = Field(..., description="Stop-loss threshold used ($/share)")
    checkpoints: List[TradeCheckpoint]
    checkpoint_count: int


class WebSocketMessage(BaseModel):
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


class PaperPosition(BaseModel):
    """An open (not yet closed) trading position"""
    position_id: str = Field(..., description="Unique identifier for this position")
    strategy_type: str = Field(..., description="IC / Put Spread / Call Spread")
    entry_time: str = Field(..., description="Entry time HH:MM:SS")
    entry_spx_price: float = Field(..., description="SPX price at entry")
    entry_credit: float = Field(..., description="Total credit received")
    strikes: Dict[str, Any] = Field(default_factory=dict, description="Strike prices")
    entry_rationale: Optional[Dict[str, Any]] = Field(None, description="Why this trade was entered")


# ---------------------------------------------------------------------------
# Live / IBKR Trading models
# ---------------------------------------------------------------------------

class IBKRConnectionConfig(BaseModel):
    """IBKR TWS / IB Gateway connection parameters."""
    host: str = Field("127.0.0.1", description="TWS / Gateway host")
    port: int = Field(7497, ge=1, le=65535,
                      description="7497 = TWS paper, 7496 = TWS live, "
                                  "4002 = Gateway paper, 4001 = Gateway live")
    client_id: int = Field(1, ge=1, le=999, description="IBKR client ID (must be unique per connection)")
    account: str = Field("", description="IBKR account string (e.g. DU123456 for paper)")


class LiveTradingRequest(BaseModel):
    """Request model for starting a live / IBKR paper trading session."""

    # IBKR connection
    ibkr: IBKRConnectionConfig = Field(
        default_factory=IBKRConnectionConfig,
        description="IBKR connection settings"
    )

    # Optional date: defaults to today when omitted
    trade_date: Optional[date] = Field(
        None,
        description="Trading date (YYYY-MM-DD). Defaults to today."
    )

    # Strategy — same knobs as backtesting
    strategy: BacktestStrategyEnum = Field(
        BacktestStrategyEnum.CREDIT_SPREADS,
        description="Which trade types to consider"
    )
    target_credit: float = Field(0.35, ge=0.05, le=5.0,
                                 description="Target net credit per spread per share")
    spread_width: int = Field(10, ge=5, le=100,
                              description="Spread width in strike points")
    contracts: int = Field(1, ge=1, le=100, description="Number of contracts per position")
    take_profit: float = Field(0.05, ge=0.01, le=10.0,
                               description="Take-profit threshold ($/share cost-to-close)")
    stop_loss: float = Field(3.0, ge=0.10, le=20.0,
                             description="Stop-loss threshold ($/share cost-to-close)")
    monitor_interval: int = Field(5, ge=1, le=15,
                                  description="Minutes between position checks")

    # Entry / exit time window
    entry_start_time: str = Field(
        "10:00:00",
        description="Time to begin scanning for entry signals (HH:MM:SS, 24-hour). Default 10:00:00."
    )
    last_entry_time: str = Field(
        "14:00:00",
        description="No new entries at or after this time (HH:MM:SS, 24-hour). Default 14:00:00."
    )

    @field_validator("trade_date")
    @classmethod
    def validate_trade_date(cls, v):
        # Live trading can target today or a future date — no restriction needed
        return v


class OrderSlippage(BaseModel):
    """Slippage record for a single IBKR order."""
    order_id: str
    strategy_type: str
    is_entry: bool
    limit_price: float
    fill_price: float
    slippage: float
    timestamp: str
    success: bool


class LiveTradingStatus(BaseModel):
    """Full status of a live / IBKR paper trading session."""
    session_id: str = Field(..., description="Unique session identifier")
    mode: str = Field(..., description="live")
    trade_date: str = Field(..., description="Date being traded YYYY-MM-DD")
    status: str = Field(..., description="pending | connecting | running | completed | stopped | failed")
    ibkr_connected: bool = Field(False, description="Whether IBKR connection is active")

    # Live position state
    open_positions: List[PaperPosition] = Field(default_factory=list)
    completed_trades: List[BacktestResult] = Field(default_factory=list)
    day_pnl: float = Field(0.0)
    trade_count: int = Field(0)

    # Slippage tracking
    orders: List[OrderSlippage] = Field(default_factory=list,
                                        description="All IBKR orders with fill details")
    total_slippage: float = Field(0.0, description="Cumulative slippage across all orders")

    # Timing
    created_at: datetime = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)