#!/usr/bin/env python3
"""
Advanced query engine with pre-built indexes for ultra-fast backtesting queries.
Optimized for 0DTE SPX options trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, time
from functools import lru_cache
import bisect
from loguru import logger

from .parquet_loader import ParquetDataLoader


class BacktestQueryEngine:
    """
    High-performance query engine with pre-built indexes for backtesting.
    Designed for sub-millisecond option chain lookups during strategy execution.
    """
    
    def __init__(self, data_loader: ParquetDataLoader):
        self.loader = data_loader
        self._time_index = {}
        self._strike_index = {}
        self._atm_index = {}
        self._volume_index = {}
        
        # Build indexes on initialization
        self._build_indexes()
    
    def _build_indexes(self):
        """Build optimized indexes for fast querying"""
        logger.info("Building query indexes for fast backtesting...")
        
        for date in self.loader.available_dates:
            date_str = date.strftime('%Y-%m-%d')
            logger.debug(f"Indexing data for {date_str}")
            
            # Load data for this date
            spx_data = self.loader.load_spx_data(date)
            options_data = self.loader.load_options_data(date)
            
            if not spx_data.empty and not options_data.empty:
                self._build_time_index(date_str, spx_data, options_data)
                self._build_strike_index(date_str, options_data)
                self._build_atm_index(date_str, spx_data, options_data)
        
        logger.info(f"Completed indexing for {len(self.loader.available_dates)} dates")
    
    def _build_time_index(self, date_str: str, spx_data: pd.DataFrame, options_data: pd.DataFrame):
        """Build time-based index for fast timestamp lookups"""
        self._time_index[date_str] = {
            'spx_times': sorted(spx_data.index.tolist()),
            'option_times': sorted(options_data.index.get_level_values('timestamp').unique().tolist())
        }
    
    def _build_strike_index(self, date_str: str, options_data: pd.DataFrame):
        """Build strike-based index for fast strike range queries"""
        strikes = options_data.index.get_level_values('strike').unique()
        self._strike_index[date_str] = {
            'all_strikes': sorted(strikes.tolist()),
            'call_strikes': sorted(strikes.tolist()),  # Both same for SPXW
            'put_strikes': sorted(strikes.tolist())
        }
    
    def _build_atm_index(self, date_str: str, spx_data: pd.DataFrame, options_data: pd.DataFrame):
        """Build ATM (at-the-money) index for quick ATM option lookups"""
        atm_data = {}
        
        # Sample every 5 minutes to build ATM reference
        sample_times = spx_data.index[::5]  # Every 5th minute
        
        for timestamp in sample_times:
            if timestamp in spx_data.index:
                spx_price = spx_data.loc[timestamp, 'price']
                
                # Find closest strikes to ATM
                strikes = self._strike_index[date_str]['all_strikes']
                atm_strike_idx = bisect.bisect_left(strikes, spx_price)
                
                # Get ATM and nearby strikes
                atm_strikes = []
                for i in range(max(0, atm_strike_idx-2), min(len(strikes), atm_strike_idx+3)):
                    atm_strikes.append(strikes[i])
                
                atm_data[timestamp] = {
                    'spx_price': spx_price,
                    'atm_strikes': atm_strikes,
                    'closest_strike': strikes[min(atm_strike_idx, len(strikes)-1)]
                }
        
        self._atm_index[date_str] = atm_data
    
    def get_fastest_spx_price(self, date: Union[str, datetime], 
                            target_time: Union[str, datetime, time]) -> Optional[float]:
        """
        Ultra-fast SPX price lookup using pre-built indexes.
        
        Args:
            date: Trading date
            target_time: Target time for price lookup
            
        Returns:
            SPX price or None if not found
        """
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        if date_str not in self._time_index:
            return None
        
        # Convert target_time to datetime for comparison
        if isinstance(target_time, str):
            target_dt = pd.to_datetime(f"{date_str} {target_time}")
        elif isinstance(target_time, time):
            target_dt = pd.to_datetime(f"{date_str} {target_time.strftime('%H:%M:%S')}")
        else:
            target_dt = target_time
        
        # Binary search in time index
        spx_times = self._time_index[date_str]['spx_times']
        idx = bisect.bisect_right(spx_times, target_dt)
        
        if idx > 0:
            # Get actual data (this should be cached)
            spx_data = self.loader.load_spx_data(date)
            nearest_time = spx_times[idx - 1]
            return float(spx_data.loc[nearest_time, 'price'])
        
        return None
    
    def get_atm_options_fast(self, date: Union[str, datetime],
                           target_time: Union[str, datetime, time],
                           strike_count: int = 5) -> pd.DataFrame:
        """
        Ultra-fast ATM options lookup using pre-built ATM index.
        
        Args:
            date: Trading date
            target_time: Target time
            strike_count: Number of strikes around ATM to return
            
        Returns:
            DataFrame with ATM options
        """
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        if date_str not in self._atm_index:
            return pd.DataFrame()
        
        # Convert target_time to datetime
        if isinstance(target_time, str):
            target_dt = pd.to_datetime(f"{date_str} {target_time}")
        elif isinstance(target_time, time):
            target_dt = pd.to_datetime(f"{date_str} {target_time.strftime('%H:%M:%S')}")
        else:
            target_dt = target_time
        
        # Find nearest ATM index entry — sort keys so bisect works correctly
        atm_times = sorted(self._atm_index[date_str].keys())
        idx = bisect.bisect_right(atm_times, target_dt)
        
        if idx > 0:
            nearest_time = atm_times[idx - 1]
            atm_info = self._atm_index[date_str][nearest_time]
            
            # Get options data around ATM strikes
            center_strike = atm_info['closest_strike']
            return self.loader.get_options_chain_at_time(
                date, target_time, center_strike, strike_count * 25
            )
        
        return pd.DataFrame()
    
    def find_liquid_options(self, date: Union[str, datetime],
                          target_time: Union[str, datetime, time],
                          min_bid: float = 0.05,
                          max_spread_pct: float = 50.0) -> pd.DataFrame:
        """
        Find liquid options based on bid/ask criteria.
        
        Args:
            date: Trading date
            target_time: Target time
            min_bid: Minimum bid price
            max_spread_pct: Maximum bid/ask spread percentage
            
        Returns:
            DataFrame with liquid options only
        """
        options_chain = self.loader.get_options_chain_at_time(date, target_time)
        
        if options_chain.empty:
            return pd.DataFrame()
        
        # Apply liquidity filters
        liquid_mask = (
            (options_chain['bid'] >= min_bid) &
            (options_chain['ask'] > options_chain['bid']) &
            (((options_chain['ask'] - options_chain['bid']) / options_chain['bid'] * 100) <= max_spread_pct)
        )
        
        liquid_options = options_chain[liquid_mask].copy()
        
        # Add spread metrics
        liquid_options['spread_dollars'] = liquid_options['ask'] - liquid_options['bid']
        liquid_options['spread_pct'] = (liquid_options['spread_dollars'] / liquid_options['bid']) * 100
        
        return liquid_options.sort_values('spread_pct')
    
    def get_option_by_delta(self, date: Union[str, datetime],
                          target_time: Union[str, datetime, time],
                          target_delta: float,
                          option_type: str = 'call',
                          tolerance: float = 0.05) -> Optional[pd.Series]:
        """
        Find option closest to target delta.
        
        Args:
            date: Trading date
            target_time: Target time
            target_delta: Target delta value (0.0 to 1.0 for calls, -1.0 to 0.0 for puts)
            option_type: 'call' or 'put' (or 'C'/'P')
            tolerance: Maximum delta difference tolerance
            
        Returns:
            Option data as Series or None
        """
        # Normalize option type
        option_type = option_type.upper()
        if option_type in ['CALL', 'CALLS']:
            option_type = 'C'
        elif option_type in ['PUT', 'PUTS']:
            option_type = 'P'
        
        options_chain = self.loader.get_options_chain_at_time(date, target_time)
        
        if options_chain.empty:
            return None
        
        # Filter by option type
        type_filtered = options_chain[options_chain['right'] == option_type]
        
        if type_filtered.empty:
            return None
        
        # Find closest delta (if delta column exists)
        if 'delta' not in type_filtered.columns:
            logger.warning("Delta column not found in options data")
            return None
        
        # Calculate delta differences
        delta_diff = (type_filtered['delta'] - target_delta).abs()
        closest_idx = delta_diff.idxmin()
        
        # Check if within tolerance
        if delta_diff.loc[closest_idx] <= tolerance:
            return type_filtered.loc[closest_idx]
        
        return None
    
    def get_option_by_moneyness(self, date: Union[str, datetime],
                              target_time: Union[str, datetime, time],
                              moneyness: float,
                              option_type: str = 'call') -> Optional[pd.Series]:
        """
        Find option by moneyness (K/S ratio).
        
        Args:
            date: Trading date
            target_time: Target time
            moneyness: Strike/Spot ratio (1.0 = ATM, >1.0 = OTM calls/ITM puts)
            option_type: 'call' or 'put'
            
        Returns:
            Option data as Series or None
        """
        # Get current SPX price
        spx_price = self.get_fastest_spx_price(date, target_time)
        if spx_price is None:
            return None
        
        # Calculate target strike
        target_strike = spx_price * moneyness
        
        # Get options chain
        options_chain = self.loader.get_options_chain_at_time(date, target_time)
        if options_chain.empty:
            return None
        
        # Normalize option type
        option_type = option_type.upper()
        if option_type in ['CALL', 'CALLS']:
            option_type = 'C'
        elif option_type in ['PUT', 'PUTS']:
            option_type = 'P'
        
        # Filter by option type
        type_filtered = options_chain[options_chain['right'] == option_type]
        
        if type_filtered.empty:
            return None
        
        # Find closest strike to target
        strike_diff = (type_filtered['strike'] - target_strike).abs()
        closest_idx = strike_diff.idxmin()
        
        return type_filtered.loc[closest_idx]
    
    def get_trading_session_data(self, date: Union[str, datetime],
                               session_start: str = "09:30:00",
                               session_end: str = "16:00:00") -> Dict[str, pd.DataFrame]:
        """
        Get all data for a trading session with time filtering.
        
        Args:
            date: Trading date
            session_start: Session start time (HH:MM:SS)
            session_end: Session end time (HH:MM:SS)
            
        Returns:
            Dictionary with 'spx' and 'options' DataFrames for the session
        """
        # Load full day data
        spx_data = self.loader.load_spx_data(date)
        options_data = self.loader.load_options_data(date)
        
        result = {}
        
        # Filter SPX data by time
        if not spx_data.empty:
            session_mask = (
                (spx_data.index.time >= pd.to_datetime(session_start).time()) &
                (spx_data.index.time <= pd.to_datetime(session_end).time())
            )
            result['spx'] = spx_data[session_mask]
        
        # Filter options data by time
        if not options_data.empty:
            timestamps = options_data.index.get_level_values('timestamp')
            session_mask = (
                (timestamps.time >= pd.to_datetime(session_start).time()) &
                (timestamps.time <= pd.to_datetime(session_end).time())
            )
            result['options'] = options_data[session_mask]
        
        return result
    
    def clear_indexes(self):
        """Clear all indexes to free memory"""
        self._time_index.clear()
        self._strike_index.clear()
        self._atm_index.clear()
        self._volume_index.clear()
        logger.info("Cleared all query indexes")
    
    def rebuild_indexes(self):
        """Rebuild all indexes (useful after data updates)"""
        self.clear_indexes()
        self._build_indexes()
    
    def get_index_summary(self) -> Dict:
        """Get summary of built indexes"""
        return {
            'indexed_dates': list(self._time_index.keys()),
            'total_atm_entries': sum(len(v) for v in self._atm_index.values()),
            'memory_usage': {
                'time_index': len(str(self._time_index)),
                'strike_index': len(str(self._strike_index)),
                'atm_index': len(str(self._atm_index))
            }
        }


# Convenience function for quick setup
def create_fast_query_engine(data_path: str = "data/processed/parquet_1m") -> BacktestQueryEngine:
    """
    Create a fully indexed query engine ready for fast backtesting.
    
    Args:
        data_path: Path to parquet data directory
        
    Returns:
        Configured BacktestQueryEngine with pre-built indexes
    """
    loader = ParquetDataLoader(data_path)
    engine = BacktestQueryEngine(loader)
    return engine