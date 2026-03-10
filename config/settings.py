import os
from dotenv import load_dotenv

load_dotenv()

# ThetaData API Configuration
THETA_USERNAME = os.getenv("THETA_USERNAME")
THETA_PASSWORD = os.getenv("THETA_PASSWORD")
THETA_TERMINAL_PORT = int(os.getenv("THETA_TERMINAL_PORT", "25503"))  # Default port 25503

# Trading Symbol Configuration
SPX_SYMBOL = os.getenv("SPX_SYMBOL", "SPXW")  # Use SPXW for 0DTE options, SPX for monthly

# Data Storage Configuration
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Database Configuration
DATABASE_PATH = os.path.join(PROCESSED_DATA_DIR, "spx_options.db")

# PostgreSQL Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://{os.getenv('USER', 'postgres')}@/spx_ai?gssencmode=disable")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "spx_ai")
POSTGRES_USER = os.getenv("POSTGRES_USER", os.getenv("USER", "postgres"))
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# Trading Configuration
SYMBOL = "SPX"
DEFAULT_DTE = 0  # 0 Days to Expiration

# Strategy Parameters
IRON_CONDOR_PARAMS = {
    "put_strike_distance": 50,
    "call_strike_distance": 50,
    "profit_target": 0.5,
    "stop_loss": 2.0
}

SPREAD_PARAMS = {
    "strike_distance": 25,
    "profit_target": 0.6,
    "stop_loss": 2.5
}

# Technical Indicator Parameters
MACD_PARAMS = {
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9
}

RSI_PARAMS = {
    "period": 14,
    "overbought": 70,
    "oversold": 30
}

BOLLINGER_PARAMS = {
    "period": 20,
    "std_dev": 2
}

# Backtesting Parameters
INITIAL_CAPITAL = 100000
COMMISSION_PER_CONTRACT = 0.65