"""
API Models for SPX AI Trading Platform
Pydantic models for request/response validation and serialization.
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


class BacktestModeEnum(str, Enum):
    """Backtest execution modes"""
    SINGLE_DAY = "single_day"
    DATE_RANGE = "date_range"
    SPECIFIC_DATES = "specific_dates"
    LIVE_SIMULATION = "live_simulation"


class BacktestStrategyEnum(str, Enum):
    """Strategy mode controlling which trade types are considered"""
    IRON_CONDOR       = "iron_condor"         # IC entries only
    CREDIT_SPREADS    = "credit_spreads"      # Put / call spreads only, no IC
    IC_CREDIT_SPREADS = "ic_credit_spreads"   # All entry types (current behaviour)
    DEBIT_SPREADS     = "debit_spreads"       # Directional debit spreads only


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

    # Stale-loss exit parameters
    enable_stale_loss_exit: bool = Field(
        False,
        description="Enable the stale-loss exit feature. When False (default) the position is "
                    "held until take_profit, stop_loss, or expiry regardless of duration."
    )
    stale_loss_minutes: int = Field(
        120, ge=10, le=360,
        description="Close a position that has been continuously losing for this many consecutive "
                    "minutes (cost > entry_credit × stale_loss_threshold). Default 120 (2 hours)."
    )
    stale_loss_threshold: float = Field(
        1.5, ge=1.0, le=5.0,
        description="Cost-to-close must exceed entry_credit × this value to qualify as 'in the red' "
                    "for stale-loss detection. Default 1.5 (50% above original credit collected)."
    )
    stagnation_window: int = Field(
        30, ge=5, le=120,
        description="Number of recent 1-min bars to examine for improvement. "
                    "If the cost has not dropped by min_improvement in this window the position "
                    "is considered stagnant. Default 30 bars."
    )
    min_improvement: float = Field(
        0.05, ge=0.01, le=1.0,
        description="Minimum cost improvement ($/share) required in the stagnation window to "
                    "keep the position open. Default $0.05/share."
    )

    # ── Debit-spread specific parameters (only used when strategy='debit_spreads') ──
    target_debit: float = Field(
        1.00, ge=0.10, le=5.0,
        description="Target net debit per spread per share when entering debit spreads. "
                    "E.g. 1.00 = aim to pay ~$1.00/share ($100/contract)."
    )
    debit_take_profit_pct: float = Field(
        0.60, ge=0.10, le=0.99,
        description="Close the debit spread when its mark-to-market value reaches this fraction "
                    "of the theoretical max profit. E.g. 0.60 = exit at 60% of max profit."
    )
    debit_stop_loss_pct: float = Field(
        0.50, ge=0.05, le=0.99,
        description="Close the debit spread when it has lost this fraction of the debit paid. "
                    "E.g. 0.50 = exit when the spread has lost 50% of its entry debit."
    )
    debit_last_entry_time: str = Field(
        "14:00:00",
        description="No new debit spread entries at or after this time (HH:MM:SS)."
    )
    debit_time_stop: str = Field(
        "15:30:00",
        description="Hard close all open debit positions by this time (HH:MM:SS) regardless of P&L. "
                    "Protects against extreme afternoon theta decay."
    )
    debit_min_trend_points: float = Field(
        10.0, ge=5.0, le=100.0,
        description="Minimum directional SPX move (points from open) required before a debit "
                    "spread entry is allowed. Ensures a trend is already established."
    )
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


class LivePosition(BaseModel):
    """An open (not yet closed) trading position"""
    position_id: str = Field(..., description="Unique identifier for this position")
    strategy_type: str = Field(..., description="IC / Put Spread / Call Spread")
    entry_time: str = Field(..., description="Entry time HH:MM:SS")
    entry_spx_price: float = Field(..., description="SPX price at entry")
    entry_credit: float = Field(..., description="Total credit received")
    strikes: Dict[str, Any] = Field(default_factory=dict, description="Strike prices")
    entry_rationale: Optional[Dict[str, Any]] = Field(None, description="Why this trade was entered")


# Backward-compat alias
PaperPosition = LivePosition


# ---------------------------------------------------------------------------
# Live / IBKR Trading models
# ---------------------------------------------------------------------------

class BrokerEnum(str, Enum):
    """Supported order-execution brokers."""
    IBKR = "ibkr"
    TASTYTRADE = "tastytrade"


class IBKRConnectionConfig(BaseModel):
    """
    IBKR TWS / IB Gateway connection parameters.

    All fields are optional in the request body — when omitted (or left as
    empty / zero) the server falls back to environment variables:
      IBKR_HOST        (default: 127.0.0.1)
      IBKR_PORT        (default: 7497)
      IBKR_CLIENT_ID   (default: 1)
      IBKR_ACCOUNT     (default: "")
    """
    host: str = Field("", description="TWS / Gateway host — omit to use IBKR_HOST env var")
    port: int = Field(0, ge=0, le=65535,
                      description="7497 = TWS paper, 7496 = TWS live, "
                                  "4002 = Gateway paper, 4001 = Gateway live — "
                                  "omit to use IBKR_PORT env var")
    client_id: int = Field(0, ge=0, le=999,
                           description="IBKR client ID — omit to use IBKR_CLIENT_ID env var")
    account: str = Field("", description="IBKR account string (e.g. DU123456) — omit to use IBKR_ACCOUNT env var")

    @model_validator(mode="before")
    @classmethod
    def fill_from_env(cls, data):
        import os
        if isinstance(data, dict):
            if not data.get("host"):
                data["host"] = os.getenv("IBKR_HOST", "127.0.0.1")
            if not data.get("port"):
                data["port"] = int(os.getenv("IBKR_PORT", "7497"))
            if not data.get("client_id"):
                data["client_id"] = int(os.getenv("IBKR_CLIENT_ID", "1"))
            if not data.get("account"):
                data["account"] = os.getenv("IBKR_ACCOUNT", "")
        return data


class TastyTradeConfig(BaseModel):
    """
    TastyTrade OAuth2 credentials and account settings (tastytrade v12+).

    All three credential fields are optional in the request body — when omitted
    (or left as empty strings) the server falls back to the environment variables
    TASTYTRADE_PROVIDER_SECRET, TASTYTRADE_REFRESH_TOKEN, and
    TASTYTRADE_ACCOUNT_NUMBER respectively.
    """
    provider_secret: str = Field(
        "", description="OAuth2 provider secret — omit to use TASTYTRADE_PROVIDER_SECRET env var"
    )
    refresh_token: str = Field(
        "", description="OAuth2 refresh token — omit to use TASTYTRADE_REFRESH_TOKEN env var"
    )
    account_number: str = Field(
        "", description="Account number (e.g. 5WT00000) — omit to use TASTYTRADE_ACCOUNT_NUMBER env var"
    )


class IBKRDiagnosticRequest(BaseModel):
    """
    Request body for POST /api/v1/trading/diagnose-market-data.

    Tests the full data path (SPX price + options chain) without starting a
    trading session.  Uses a separate clientId so it never interferes with a
    running session.
    """
    ibkr: IBKRConnectionConfig = Field(default_factory=IBKRConnectionConfig)
    trade_date: Optional[str] = Field(
        None,
        description="Date to use for options expiry (YYYY-MM-DD). Defaults to today."
    )
    spx_price_hint: float = Field(
        5800.0,
        description="Approximate SPX level to centre the options chain query around "
                    "when live price is unavailable (e.g. outside market hours)."
    )
    diagnostic_client_id: int = Field(
        50,
        ge=1, le=999,
        description="clientId used for the diagnostic connection. "
                    "Must differ from any running session clientId."
    )
    num_strikes: int = Field(
        5,
        ge=1, le=20,
        description="Number of strikes on each side (put/call) to sample from the chain."
    )


class DiagnosticRequest(BaseModel):
    """
    Broker-agnostic request body for POST /api/v1/trading/diagnose-market-data.

    Set broker='ibkr' to test an IBKR connection (ibkr fields required).
    Set broker='tastytrade' to test a TastyTrade connection.
      - Either pass broker_config_id (UUID of a saved UserBrokerConfig) to load
        credentials from the database, OR supply inline tastytrade credentials.
    """
    broker: BrokerEnum = Field(BrokerEnum.IBKR, description="Broker to diagnose")

    # Use a saved broker config from the database (preferred)
    broker_config_id: Optional[uuid.UUID] = Field(
        None,
        description="UUID of an approved UserBrokerConfig. When set, credentials are "
                    "loaded from the database and tastytrade fields are ignored."
    )

    # IBKR fields (used when broker=ibkr)
    ibkr: IBKRConnectionConfig = Field(default_factory=IBKRConnectionConfig)
    diagnostic_client_id: int = Field(
        50, ge=1, le=999,
        description="IBKR clientId for the diagnostic connection"
    )

    # TastyTrade fields (used when broker=tastytrade and broker_config_id is absent)
    tastytrade: Optional[TastyTradeConfig] = Field(
        None, description="TastyTrade credentials (used when broker_config_id is not set)"
    )

    # Shared fields
    trade_date: Optional[str] = Field(
        None, description="Date for options expiry (YYYY-MM-DD). Defaults to today."
    )
    spx_price_hint: float = Field(
        5800.0,
        description="Fallback SPX level for options chain centre when live price unavailable."
    )
    num_strikes: int = Field(
        10, ge=1, le=50,
        description="Number of strikes on each side (put/call) to sample."
    )

    @model_validator(mode="after")
    def validate_broker_config(self) -> "DiagnosticRequest":
        import os
        if self.broker == BrokerEnum.TASTYTRADE and self.broker_config_id is None:
            # No saved config — fall back to inline credentials / env vars
            if self.tastytrade is None:
                self.tastytrade = TastyTradeConfig()
            cfg = self.tastytrade
            if not cfg.provider_secret:
                cfg.provider_secret = os.getenv("TASTYTRADE_PROVIDER_SECRET", "")
            if not cfg.refresh_token:
                cfg.refresh_token = os.getenv("TASTYTRADE_REFRESH_TOKEN", "")
            if not cfg.account_number:
                cfg.account_number = os.getenv("TASTYTRADE_ACCOUNT_NUMBER", "")
            missing = [k for k, v in [
                ("provider_secret", cfg.provider_secret),
                ("refresh_token",   cfg.refresh_token),
                ("account_number",  cfg.account_number),
            ] if not v]
            if missing:
                raise ValueError(
                    f"TastyTrade credentials missing: {missing}. "
                    "Provide broker_config_id or set TASTYTRADE_PROVIDER_SECRET / "
                    "TASTYTRADE_REFRESH_TOKEN / TASTYTRADE_ACCOUNT_NUMBER environment variables."
                )
        return self


class LiveTradingRequest(BaseModel):
    """Request model for starting a live / paper trading session."""

    # Broker selection
    broker: BrokerEnum = Field(
        BrokerEnum.IBKR,
        description="Which broker to use for order execution and market data"
    )

    # IBKR connection (used when broker=ibkr)
    ibkr: IBKRConnectionConfig = Field(
        default_factory=IBKRConnectionConfig,
        description="IBKR connection settings (only used when broker=ibkr)"
    )

    # TastyTrade credentials (required when broker=tastytrade)
    tastytrade: Optional[TastyTradeConfig] = Field(
        None,
        description="TastyTrade credentials (required when broker=tastytrade)"
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

    # Stale-loss exit parameters
    enable_stale_loss_exit: bool = Field(
        False,
        description="Enable the stale-loss exit feature. When False (default) the position is "
                    "held until take_profit, stop_loss, or expiry regardless of duration."
    )
    stale_loss_minutes: int = Field(
        120, ge=10, le=360,
        description="Close a position that has been continuously losing for this many consecutive "
                    "minutes (cost > entry_credit × stale_loss_threshold). Default 120 (2 hours)."
    )
    stale_loss_threshold: float = Field(
        1.5, ge=1.0, le=5.0,
        description="Cost-to-close must exceed entry_credit × this value to qualify as 'in the red' "
                    "for stale-loss detection. Default 1.5 (50% above original credit collected)."
    )
    stagnation_window: int = Field(
        30, ge=5, le=120,
        description="Number of recent 1-min bars to examine for improvement. "
                    "If the cost has not dropped by min_improvement in this window the position "
                    "is considered stagnant. Default 30 bars."
    )
    min_improvement: float = Field(
        0.05, ge=0.01, le=1.0,
        description="Minimum cost improvement ($/share) required in the stagnation window to "
                    "keep the position open. Default $0.05/share."
    )

    skip_indicators: bool = Field(
        True,
        description="When True, skip RSI/MACD/Bollinger warmup and use drift-only entry guards. "
                    "Useful for debugging or when indicator data is unavailable at session start."
    )

    # ── Debit-spread specific parameters (only used when strategy='debit_spreads') ──
    target_debit: float = Field(
        1.00, ge=0.10, le=5.0,
        description="Target net debit per spread per share when entering debit spreads."
    )
    debit_take_profit_pct: float = Field(
        0.60, ge=0.10, le=0.99,
        description="Close debit spread when its value reaches this fraction of theoretical max profit."
    )
    debit_stop_loss_pct: float = Field(
        0.50, ge=0.05, le=0.99,
        description="Close debit spread when it has lost this fraction of the debit paid."
    )
    debit_last_entry_time: str = Field(
        "14:00:00",
        description="No new debit spread entries at or after this time (HH:MM:SS)."
    )
    debit_time_stop: str = Field(
        "15:30:00",
        description="Hard close all open debit positions by this time (HH:MM:SS) regardless of P&L."
    )
    debit_min_trend_points: float = Field(
        10.0, ge=5.0, le=100.0,
        description="Minimum directional SPX move (points from open) required for a debit entry."
    )

    broker_config_id: Optional[uuid.UUID] = Field(
        None,
        description="UUID of an approved UserBrokerConfig to use for credentials. "
                    "When set, overrides any tastytrade fields and env-var fallback."
    )

    @field_validator("trade_date")
    @classmethod
    def validate_trade_date(cls, v):
        # Live trading can target today or a future date — no restriction needed
        return v

    @model_validator(mode="after")
    def validate_broker_config(self) -> "LiveTradingRequest":
        import os
        if self.broker == BrokerEnum.TASTYTRADE:
            if self.tastytrade is None:
                # Auto-build from env vars if the field was omitted entirely
                self.tastytrade = TastyTradeConfig()
            cfg = self.tastytrade
            # Fill empty credential fields from env vars
            if not cfg.provider_secret:
                cfg.provider_secret = os.getenv("TASTYTRADE_PROVIDER_SECRET", "")
            if not cfg.refresh_token:
                cfg.refresh_token = os.getenv("TASTYTRADE_REFRESH_TOKEN", "")
            if not cfg.account_number:
                cfg.account_number = os.getenv("TASTYTRADE_ACCOUNT_NUMBER", "")
            # Final check — at least the essential credentials must be present
            missing = [k for k, v in [
                ("provider_secret", cfg.provider_secret),
                ("refresh_token",   cfg.refresh_token),
                ("account_number",  cfg.account_number),
            ] if not v]
            if missing:
                raise ValueError(
                    f"TastyTrade credentials missing: {missing}. "
                    "Provide them in the request body or set "
                    "TASTYTRADE_PROVIDER_SECRET / TASTYTRADE_REFRESH_TOKEN / "
                    "TASTYTRADE_ACCOUNT_NUMBER environment variables."
                )
        return self


class OrderSlippage(BaseModel):
    """Slippage record for a single broker order."""
    order_id: str
    strategy_type: str
    is_entry: bool
    limit_price: float
    fill_price: float
    slippage: float
    timestamp: str
    success: bool


class LiveTradingStatus(BaseModel):
    """Full status of a live trading session."""
    session_id: str = Field(..., description="Unique session identifier")
    mode: str = Field(..., description="live")
    trade_date: str = Field(..., description="Date being traded YYYY-MM-DD")
    status: str = Field(..., description="pending | connecting | running | completed | stopped | failed")
    broker_type: str = Field("ibkr", description="Broker used: ibkr | tastytrade")
    broker_connected: bool = Field(False, description="Whether the broker connection is active")

    # Live position state
    open_positions: List[LivePosition] = Field(default_factory=list)
    completed_trades: List[BacktestResult] = Field(default_factory=list)
    day_pnl: float = Field(0.0)
    trade_count: int = Field(0)

    # Slippage tracking
    orders: List[OrderSlippage] = Field(default_factory=list,
                                        description="All broker orders with fill details")
    total_slippage: float = Field(0.0, description="Cumulative slippage across all orders")

    # Timing
    created_at: datetime = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)