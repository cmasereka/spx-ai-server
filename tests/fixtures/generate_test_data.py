"""Sample test data generation and fixture creation."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Test data parameters
SAMPLE_DATES = ["2026-02-09", "2026-02-10", "2026-02-11"]
SAMPLE_SPX_PRICES = [6925.0, 6978.0, 6961.0]


def create_sample_parquet_data():
    """Create sample parquet files for testing."""
    
    fixtures_dir = Path(__file__).parent
    data_dir = fixtures_dir / "sample_data"
    data_dir.mkdir(exist_ok=True)
    
    for i, (date, spx_price) in enumerate(zip(SAMPLE_DATES, SAMPLE_SPX_PRICES)):
        # Create SPX price data
        spx_data = create_spx_data_for_date(date, spx_price)
        spx_file = data_dir / f"SPX_index_price_1m_{date}.parquet"
        spx_data.to_parquet(spx_file, index=False)
        
        # Create options data
        options_data = create_options_data_for_date(date, spx_price)
        options_file = data_dir / f"SPXW_option_quotes_1m_{date}_exp{date}_sr200.parquet"
        options_data.to_parquet(options_file, index=False)
        
        print(f"Created test data for {date}: SPX={spx_price}")
    
    print(f"Sample parquet files created in: {data_dir}")
    return data_dir


def create_spx_data_for_date(date: str, base_price: float, num_points: int = 390):
    """Create realistic SPX price data for a full trading day."""
    
    # Generate timestamps (9:30 AM to 4:00 PM ET)
    start_time = pd.Timestamp(f"{date} 09:30:00")
    timestamps = pd.date_range(start_time, periods=num_points, freq='1min')
    
    # Generate realistic intraday price movements
    np.random.seed(hash(date) % 2**32)  # Deterministic based on date
    
    # Create intraday pattern (opening gap, midday lull, closing move)
    time_weights = np.concatenate([
        np.linspace(1.0, 0.3, 120),  # Morning volatility decline
        np.ones(150) * 0.3,          # Midday low volatility  
        np.linspace(0.3, 0.8, 120)   # Afternoon pickup
    ])
    
    # Generate returns with time-varying volatility
    base_vol = 0.0005  # Base 0.05% per minute volatility
    returns = np.random.normal(0, base_vol * time_weights)
    returns[0] = 0  # Start at exact base price
    
    # Calculate prices
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Generate volume (higher at open/close)
    volume_pattern = np.concatenate([
        np.linspace(8000, 3000, 60),    # Opening volume
        np.ones(270) * 2000,            # Steady midday volume
        np.linspace(3000, 6000, 60)     # Closing volume
    ])
    volume_noise = np.random.uniform(0.8, 1.2, num_points)
    volumes = (volume_pattern * volume_noise).astype(int)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,  # Simplified - using same as close
        'high': [p * np.random.uniform(1.0, 1.002) for p in prices],
        'low': [p * np.random.uniform(0.998, 1.0) for p in prices],
        'close': prices,
        'volume': volumes
    })


def create_options_data_for_date(date: str, spx_price: float, timestamp: str = "10:00:00"):
    """Create realistic options chain data."""
    
    np.random.seed(hash(f"{date}_{timestamp}") % 2**32)
    
    # Generate strike prices around SPX (every $5 from -$200 to +$200)
    strike_range = range(int(spx_price - 200), int(spx_price + 201), 5)
    
    options_data = []
    
    for strike in strike_range:
        moneyness = (spx_price - strike) / spx_price
        time_to_expiry = 0.25  # Assume 6 hours to expiration (0DTE)
        
        # Calculate realistic Greeks and prices using Black-Scholes approximation
        for option_type in ['put', 'call']:
            
            if option_type == 'put':
                # Put delta (negative, more negative as strike increases above spot)
                delta = min(max(-0.99, -0.5 - moneyness * 2), -0.01)
                # Put price increases as strike increases above spot
                intrinsic = max(0, strike - spx_price)
                time_value = max(0.05, abs(moneyness) * spx_price * 0.01 * time_to_expiry)
                
            else:  # call
                # Call delta (positive, more positive as strike decreases below spot)
                delta = min(max(0.01, 0.5 + moneyness * 2), 0.99)
                # Call price increases as strike decreases below spot
                intrinsic = max(0, spx_price - strike)
                time_value = max(0.05, abs(moneyness) * spx_price * 0.01 * time_to_expiry)
            
            theoretical_price = intrinsic + time_value
            
            # Add bid-ask spread (wider for OTM options)
            spread_width = max(0.05, abs(moneyness) * 0.5 + 0.05)
            bid = max(0.05, theoretical_price - spread_width/2)
            ask = theoretical_price + spread_width/2
            
            # Other Greeks (simplified)
            gamma = max(0.001, 0.01 * (1 - abs(moneyness) * 3))  # Higher ATM
            theta = -max(0.01, theoretical_price * 0.1)  # Time decay
            vega = max(0.05, abs(moneyness) * 0.2 + 0.1)  # Vol sensitivity
            iv = max(0.10, 0.15 + abs(moneyness) * 0.1)  # Vol smile
            
            # Volume (higher for ATM options)
            volume = int(max(10, 1000 * (1 - abs(moneyness) * 2)) * np.random.uniform(0.5, 2.0))
            open_interest = volume * np.random.randint(5, 50)
            
            options_data.append({
                'timestamp': pd.Timestamp(f"{date} {timestamp}"),
                'strike': float(strike),
                'option_type': option_type,
                'expiration': date,
                'bid': round(bid, 2),
                'ask': round(ask, 2),
                'last': round((bid + ask) / 2, 2),
                'volume': volume,
                'open_interest': open_interest,
                'delta': round(delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'implied_volatility': round(iv, 4)
            })
    
    return pd.DataFrame(options_data)


def create_test_configuration():
    """Create test configuration files."""
    
    fixtures_dir = Path(__file__).parent
    
    # Test settings
    test_config = {
        "TEST_DATA_DIR": "tests/fixtures/sample_data",
        "TEST_DATES": SAMPLE_DATES,
        "TEST_SPX_PRICES": SAMPLE_SPX_PRICES,
        "DEFAULT_STRATEGY_PARAMS": {
            "target_delta": 0.15,
            "target_prob_itm": 0.15,
            "min_spread_width": 10,
            "decay_threshold": 0.1,
            "entry_time": "10:00:00",
            "exit_time": "15:45:00"
        },
        "MOCK_SETTINGS": {
            "use_mock_data": True,
            "enable_logging": False,
            "fast_mode": True
        }
    }
    
    config_file = fixtures_dir / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"Test configuration created: {config_file}")
    return config_file


def create_performance_benchmark_data():
    """Create data for performance benchmarking tests."""
    
    fixtures_dir = Path(__file__).parent
    benchmark_dir = fixtures_dir / "benchmark_data"
    benchmark_dir.mkdir(exist_ok=True)
    
    # Create larger dataset for performance testing
    large_date_range = pd.date_range("2026-01-01", "2026-02-28", freq='D')
    trading_days = [d for d in large_date_range if d.weekday() < 5]  # Weekdays only
    
    benchmark_data = {
        'dates': [d.strftime('%Y-%m-%d') for d in trading_days],
        'expected_runtimes': {
            'single_day_backtest': 2.0,  # seconds
            'technical_analysis': 0.1,
            'strike_selection': 0.5,
            'position_monitoring': 1.0
        },
        'memory_usage_mb': {
            'base_system': 50,
            'per_day_data': 5,
            'max_acceptable': 500
        }
    }
    
    benchmark_file = benchmark_dir / "performance_benchmarks.json"
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"Performance benchmark data created: {benchmark_file}")
    return benchmark_file


if __name__ == "__main__":
    """Generate all test fixtures."""
    print("Generating comprehensive test fixtures...")
    
    # Create sample data
    data_dir = create_sample_parquet_data()
    
    # Create test configuration
    config_file = create_test_configuration()
    
    # Create benchmark data
    benchmark_file = create_performance_benchmark_data()
    
    print(f"\n✅ Test fixtures created successfully!")
    print(f"📁 Sample data: {data_dir}")
    print(f"⚙️ Test config: {config_file}")
    print(f"🚀 Benchmarks: {benchmark_file}")
    print(f"\nRun tests with: pytest tests/ -v")