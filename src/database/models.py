"""
SQLAlchemy models for SPX trading data
"""

from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Boolean, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from .connection import Base


class User(Base):
    __tablename__ = 'users'

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    full_name       = Column(String(100), nullable=False)
    phone           = Column(String(30), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    role            = Column(String(10),  nullable=False, default='user')  # 'admin'|'user'
    status          = Column(String(20),  nullable=False, default='pending_approval')
    # 'pending_approval' | 'approved' | 'suspended'
    invited_by      = Column(UUID(as_uuid=True), nullable=True)
    created_at      = Column(DateTime, default=func.now())
    last_login_at   = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(email={self.email}, role={self.role}, status={self.status})>"


class Invitation(Base):
    __tablename__ = 'invitations'

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code           = Column(String(64), unique=True, nullable=False, index=True)
    created_by     = Column(UUID(as_uuid=True), nullable=False)
    note           = Column(String(200), nullable=True)
    invited_email  = Column(String(255), nullable=True)
    is_used        = Column(Boolean, nullable=False, default=False)
    used_by        = Column(UUID(as_uuid=True), nullable=True)
    used_at        = Column(DateTime, nullable=True)
    cancelled_at   = Column(DateTime, nullable=True)
    expires_at     = Column(DateTime, nullable=False)
    created_at     = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Invitation(code={self.code[:8]}…, used={self.is_used})>"


class UserBrokerConfig(Base):
    __tablename__ = 'user_broker_configs'

    id                    = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id               = Column(UUID(as_uuid=True), nullable=False, index=True)
    broker_type           = Column(String(20), nullable=False)   # 'tastytrade'
    label                 = Column(String(100), nullable=True)
    account_number        = Column(String(50),  nullable=False)
    encrypted_credentials = Column(Text, nullable=False)         # Fernet JSON blob
    status                = Column(String(20), nullable=False, default='pending_approval')
    # 'pending_approval' | 'approved' | 'rejected'
    approved_by           = Column(UUID(as_uuid=True), nullable=True)
    approved_at           = Column(DateTime, nullable=True)
    created_at            = Column(DateTime, default=func.now())
    updated_at            = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<UserBrokerConfig(user={self.user_id}, broker={self.broker_type}, status={self.status})>"

class BacktestRun(Base):
    __tablename__ = 'backtest_runs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id = Column(String(50), unique=True, nullable=False, index=True)
    mode = Column(String(20), nullable=False)
    strategy_type = Column(String(20), nullable=False)
    
    # Date configuration
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    single_date = Column(Date, nullable=True)
    
    # Strategy parameters
    put_distance = Column(Integer, nullable=False)
    call_distance = Column(Integer, nullable=False)
    spread_width = Column(Integer, nullable=False)
    
    # Risk management
    decay_threshold = Column(Float, nullable=False)
    profit_target = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    
    # Execution parameters
    entry_time = Column(String(8), nullable=False)
    monitor_interval = Column(Integer, nullable=False)
    
    # Status and timing
    status = Column(String(20), nullable=False, default='pending')
    created_at = Column(DateTime, default=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Results
    total_trades = Column(Integer, nullable=True)
    successful_trades = Column(Integer, nullable=True)
    total_pnl = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    
    # Additional metadata
    parameters = Column(JSONB, nullable=True)

    # Auth
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    def __repr__(self):
        return f"<BacktestRun(id={self.backtest_id}, status={self.status})>"

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    backtest_run_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Trade timing
    trade_date = Column(Date, nullable=False, index=True)
    entry_time = Column(String(8), nullable=False)
    exit_time = Column(String(8), nullable=True)
    
    # Market data
    entry_spx_price = Column(Float, nullable=False)
    exit_spx_price = Column(Float, nullable=True)
    
    # Strategy details
    strategy_type = Column(String(20), nullable=False)
    strikes = Column(JSONB, nullable=False)
    
    # Trade performance
    entry_credit = Column(Float, nullable=False)
    exit_cost = Column(Float, nullable=False, default=0.0)
    pnl = Column(Float, nullable=False)
    pnl_percentage = Column(Float, nullable=False)
    
    # Trade metadata
    exit_reason = Column(String(200), nullable=False)  # Increased from 50 to 200
    is_winner = Column(Boolean, nullable=False)
    max_profit = Column(Float, nullable=True)
    max_loss = Column(Float, nullable=True)
    
    # Strategy-specific data
    strategy_details = Column(JSONB, nullable=True)
    monitoring_data = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Trade(id={self.trade_id}, date={self.trade_date}, pnl={self.pnl})>"

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Price data
    price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)
    
    # OHLC data if available
    open_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    
    # Data source
    data_source = Column(String(20), nullable=False, default='theta')
    
    # Additional metadata
    extra_metadata = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<MarketData(symbol={self.symbol}, date={self.date}, price={self.price})>"

class OptionData(Base):
    __tablename__ = 'option_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    underlying_symbol = Column(String(10), nullable=False, index=True)
    
    # Option details
    strike = Column(Float, nullable=False, index=True)
    expiration_date = Column(Date, nullable=False, index=True)
    option_type = Column(String(4), nullable=False)  # 'call' or 'put'
    
    # Market data
    date = Column(Date, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Option pricing
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    mid_price = Column(Float, nullable=True)
    last_price = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    open_interest = Column(Integer, nullable=True)
    
    # Greeks
    delta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)
    vega = Column(Float, nullable=True)
    rho = Column(Float, nullable=True)
    implied_volatility = Column(Float, nullable=True)
    
    # Underlying data
    underlying_price = Column(Float, nullable=True)
    
    # Data source
    data_source = Column(String(20), nullable=False, default='theta')
    
    # Additional metadata
    extra_metadata = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<OptionData(symbol={self.symbol}, strike={self.strike}, exp={self.expiration_date})>"

class LiveTradingRun(Base):
    __tablename__ = 'live_trading_runs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(50), unique=True, nullable=False, index=True)
    mode = Column(String(20), nullable=False)        # 'live'
    trade_date = Column(Date, nullable=False)
    strategy_type = Column(String(20), nullable=False)

    # Broker used for this session
    broker_type = Column(String(20), nullable=True)  # 'ibkr' | 'tastytrade'

    # Strategy parameters
    parameters = Column(JSONB, nullable=True)

    # Status and timing
    status = Column(String(20), nullable=False, default='pending')
    created_at = Column(DateTime, default=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Results summary
    total_trades = Column(Integer, nullable=True)
    successful_trades = Column(Integer, nullable=True)
    total_pnl = Column(Float, nullable=True)

    # Auth
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    def __repr__(self):
        return f"<LiveTradingRun(session_id={self.session_id}, date={self.trade_date}, status={self.status}, broker={self.broker_type})>"


# Backward-compat alias so any remaining import of PaperTradingRun still resolves.
PaperTradingRun = LiveTradingRun


class BrokerOrder(Base):
    """
    Records every order submitted to a broker during a live / paper trading session.

    Captures both the intended price (from the model) and the actual fill so
    that slippage can be tracked and compared across brokers.

    Previously named IBKROrder; the table name 'ibkr_orders' is kept unchanged
    to avoid a data migration.
    """
    __tablename__ = 'ibkr_orders'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(String(50), unique=True, nullable=False, index=True)

    # Session linkage
    session_id = Column(String(50), nullable=False, index=True)

    # Which broker submitted this order
    broker_type = Column(String(20), nullable=False, server_default='ibkr')  # 'ibkr' | 'tastytrade'

    # Order details
    symbol = Column(String(20), nullable=False, default='SPXW')
    strategy_type = Column(String(30), nullable=False)
    is_entry = Column(Boolean, nullable=False)   # True = opening, False = closing

    # Pricing
    limit_price = Column(Float, nullable=False)   # Price we requested (model mid)
    fill_price = Column(Float, nullable=False)     # Actual fill
    slippage = Column(Float, nullable=False)       # fill_price - limit_price
    quantity = Column(Integer, nullable=False)

    # Status
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)

    # Timing
    timestamp = Column(String(8), nullable=False)  # HH:MM:SS bar time
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Raw broker response (optional JSON blob)
    broker_data = Column(JSONB, nullable=True)

    def __repr__(self):
        direction = "OPEN" if self.is_entry else "CLOSE"
        return (
            f"<BrokerOrder(id={self.order_id}, broker={self.broker_type}, {direction} {self.strategy_type}, "
            f"fill={self.fill_price:.2f}, slippage={self.slippage:+.2f})>"
        )


# Backward-compat alias so any remaining references to IBKROrder still resolve.
IBKROrder = BrokerOrder


class SystemLog(Base):
    __tablename__ = 'system_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    level = Column(String(10), nullable=False, index=True)
    logger = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    
    # Context
    module = Column(String(50), nullable=True)
    function = Column(String(50), nullable=True)
    line_number = Column(Integer, nullable=True)
    
    # Associated records
    backtest_run_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    trade_id = Column(String(50), nullable=True, index=True)
    
    # Additional context
    context = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<SystemLog(level={self.level}, message={self.message[:50]}...)>"