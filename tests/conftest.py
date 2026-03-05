"""Test configuration and common utilities for the test suite."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test data configuration
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "fixtures"
SAMPLE_DATE = "2026-02-09"
SAMPLE_SPX_PRICE = 6925.0

class TestDataGenerator:
    """Generate realistic test data for backtesting."""
    
    @staticmethod
    def create_sample_spx_data(date: str = SAMPLE_DATE, 
                              base_price: float = SAMPLE_SPX_PRICE,
                              num_points: int = 390) -> pd.DataFrame:
        """Create sample SPX price data for a trading day."""
        start_time = pd.Timestamp(f"{date} 09:30:00")
        timestamps = pd.date_range(start_time, periods=num_points, freq='1min')
        
        # Generate realistic price movements
        np.random.seed(42)  # Deterministic for testing
        returns = np.random.normal(0, 0.0005, num_points)  # 0.05% std dev
        returns[0] = 0  # Start at base price
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'close': prices,
            'volume': np.random.randint(1000, 10000, num_points)
        })
    
    @staticmethod
    def create_sample_options_data(date: str = SAMPLE_DATE,
                                  spx_price: float = SAMPLE_SPX_PRICE,
                                  timestamp: str = "10:00:00") -> pd.DataFrame:
        """Create sample options data around SPX price."""
        
        # Generate strikes around SPX price
        strikes = []
        for distance in range(-200, 201, 5):  # -200 to +200 in 5pt increments
            strikes.append(spx_price + distance)
        
        options_data = []
        
        for strike in strikes:
            # Calculate approximate delta and prices
            moneyness = (spx_price - strike) / spx_price
            
            # Put option
            put_delta = min(max(-abs(moneyness * 2), -0.99), -0.01)
            put_price = max(0.05, abs(moneyness) * spx_price * 0.01)
            put_bid = put_price * 0.95
            put_ask = put_price * 1.05
            
            options_data.append({
                'timestamp': pd.Timestamp(f"{date} {timestamp}"),
                'strike': strike,
                'option_type': 'put',
                'expiration': date,
                'bid': put_bid,
                'ask': put_ask,
                'delta': put_delta,
                'gamma': 0.01,
                'theta': -0.05,
                'vega': 0.1,
                'implied_volatility': 0.15
            })
            
            # Call option
            call_delta = min(max(abs(moneyness * 2), 0.01), 0.99)
            call_price = max(0.05, abs(moneyness) * spx_price * 0.01)
            call_bid = call_price * 0.95
            call_ask = call_price * 1.05
            
            options_data.append({
                'timestamp': pd.Timestamp(f"{date} {timestamp}"),
                'strike': strike,
                'option_type': 'call',
                'expiration': date,
                'bid': call_bid,
                'ask': call_ask,
                'delta': call_delta,
                'gamma': 0.01,
                'theta': -0.05,
                'vega': 0.1,
                'implied_volatility': 0.15
            })
        
        return pd.DataFrame(options_data)

class MockQueryEngine:
    """Mock query engine for testing."""
    
    def __init__(self):
        self.spx_data = TestDataGenerator.create_sample_spx_data()
        self.options_data = TestDataGenerator.create_sample_options_data()
    
    def get_fastest_spx_price(self, date: str, time: str) -> Optional[float]:
        """Mock SPX price retrieval."""
        try:
            timestamp = pd.Timestamp(f"{date} {time}")
            closest_row = self.spx_data.iloc[
                (self.spx_data['timestamp'] - timestamp).abs().argsort()[:1]
            ]
            return float(closest_row['close'].iloc[0])
        except:
            return SAMPLE_SPX_PRICE
    
    def get_options_data(self, date: str, timestamp: str) -> Optional[pd.DataFrame]:
        """Mock options data retrieval."""
        return self.options_data.copy()
    
    def get_trading_session_data(self, date: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """Mock trading session data."""
        return {
            'spx_prices': self.spx_data,
            'options_data': self.options_data
        }

# Test assertions and utilities
def assert_valid_backtest_result(result, expected_success: bool = True):
    """Assert that a backtest result has valid structure."""
    assert hasattr(result, 'success')
    assert hasattr(result, 'date')
    assert hasattr(result, 'pnl')
    assert hasattr(result, 'entry_credit')
    assert hasattr(result, 'exit_cost')
    
    if expected_success:
        assert result.success is True
        assert result.entry_credit > 0
        assert result.pnl is not None
    
    # Validate data types
    assert isinstance(result.date, str)
    assert isinstance(result.pnl, (int, float))
    assert isinstance(result.entry_credit, (int, float))
    assert isinstance(result.exit_cost, (int, float))

def assert_valid_technical_indicators(indicators):
    """Assert technical indicators have valid values."""
    assert 0 <= indicators.rsi <= 100
    assert 0 <= indicators.bb_position <= 1
    assert indicators.bb_upper > indicators.bb_middle > indicators.bb_lower
    assert isinstance(indicators.macd_line, (int, float))
    assert isinstance(indicators.macd_signal, (int, float))
    assert isinstance(indicators.macd_histogram, (int, float))

def assert_valid_strike_selection(strike_selection, spx_price: float):
    """Assert strike selection is valid."""
    assert strike_selection.short_strike > 0
    assert strike_selection.long_strike > 0
    assert strike_selection.spread_width > 0

    # Strikes should be reasonable relative to SPX price
    assert abs(strike_selection.short_strike - spx_price) < spx_price * 0.2  # Within 20%
    assert abs(strike_selection.long_strike - spx_price) < spx_price * 0.2