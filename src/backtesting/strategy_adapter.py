#!/usr/bin/env python3
"""
Enhanced Strategy Adapter for integrating parquet data with existing strategy framework.
Converts DataFrame-based data to the dictionary format expected by existing strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, time
from functools import lru_cache
from loguru import logger

from ..data.parquet_loader import ParquetDataLoader
from ..data.query_engine import BacktestQueryEngine
from ..strategies.options_strategies import OptionsStrategy, IronCondor, VerticalSpread, StrategyBuilder


class ParquetDataAdapter:
    """
    Adapter that converts parquet DataFrame data to dictionary format 
    expected by existing strategy classes.
    """
    
    def __init__(self, query_engine: BacktestQueryEngine):
        self.query_engine = query_engine
        self._option_cache = {}
        self._spx_cache = {}
    
    def convert_options_dataframe_to_dict(self, options_df: pd.DataFrame, 
                                        spx_price: Optional[float] = None) -> Dict[str, Dict]:
        """
        Convert options DataFrame to strategy-expected dictionary format.
        
        Args:
            options_df: Options DataFrame from parquet loader
            spx_price: Current SPX price for calculations
            
        Returns:
            Dictionary in format: {"strike_optiontype": {"mid_price": ..., "delta": ..., ...}}
        """
        if options_df.empty:
            return {}
        
        options_dict = {}
        
        for _, row in options_df.iterrows():
            # Normalize option type
            option_type = str(row['right']).lower()
            if option_type in ['c', 'call']:
                option_type = 'call'
            elif option_type in ['p', 'put']:
                option_type = 'put'
            else:
                continue
            
            # Create key
            strike = float(row['strike'])
            key = f"{strike}_{option_type}"
            
            # Calculate mid price
            bid = float(row.get('bid', 0))
            ask = float(row.get('ask', 0))
            mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
            
            # Extract Greeks (may not be available in all data)
            delta = float(row.get('delta', 0))
            gamma = float(row.get('gamma', 0)) 
            theta = float(row.get('theta', 0))
            vega = float(row.get('vega', 0))
            iv = float(row.get('iv', 0))
            
            # Add calculated metrics
            bid_ask_spread = ask - bid if ask > bid else 0.0
            spread_pct = (bid_ask_spread / mid_price * 100) if mid_price > 0 else 0.0
            
            # Estimate moneyness if SPX price available
            moneyness = strike / spx_price if spx_price and spx_price > 0 else 1.0
            
            options_dict[key] = {
                'mid_price': mid_price,
                'bid': bid,
                'ask': ask,
                'strike': strike,
                'option_type': option_type,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'iv': iv,
                'bid_ask_spread': bid_ask_spread,
                'spread_pct': spread_pct,
                'moneyness': moneyness,
                'volume': int(row.get('volume', 0)),
                'open_interest': int(row.get('open_interest', 0))
            }
        
        return options_dict
    
    def diagnose_data_availability(self, date: Union[str, datetime], 
                                 timestamp: Union[str, datetime, time]) -> Dict[str, any]:
        """
        Diagnostic method to check data availability for troubleshooting.
        
        Args:
            date: Trading date to check
            timestamp: Time to check
            
        Returns:
            Dictionary with diagnostic information
        """
        diagnosis = {
            'date': date,
            'timestamp': timestamp,
            'spx_available': False,
            'options_available': False,
            'spx_price': None,
            'options_count': 0,
            'available_strikes': [],
            'time_range': None,
            'errors': []
        }
        
        try:
            # Check SPX data
            spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
            diagnosis['spx_available'] = spx_price is not None
            diagnosis['spx_price'] = spx_price
            
            # Check options data
            options_df = self.query_engine.loader.load_options_data(date)
            if not options_df.empty:
                diagnosis['options_available'] = True
                diagnosis['options_count'] = len(options_df)
                
                # Get time range
                times = options_df.index.get_level_values('timestamp').unique()
                diagnosis['time_range'] = (times.min(), times.max())
                
                # Get available strikes
                strikes = options_df.index.get_level_values('strike').unique()
                diagnosis['available_strikes'] = sorted(strikes.tolist())
                
                # Check specific timestamp
                target_dt = pd.to_datetime(f"{date} {timestamp}") if isinstance(timestamp, str) else timestamp
                near_times = times[abs(times - target_dt) <= pd.Timedelta(minutes=5)]
                diagnosis['timestamps_within_5min'] = len(near_times)
                
        except Exception as e:
            diagnosis['errors'].append(str(e))
        
        return diagnosis

    def get_options_data_for_strategy(self, date: Union[str, datetime], 
                                    timestamp: Union[str, datetime, time],
                                    center_strike: Optional[float] = None,
                                    strike_range: float = 200) -> Dict[str, Dict]:
        """
        Get options data formatted for strategy creation.
        
        Args:
            date: Trading date
            timestamp: Time for data lookup
            center_strike: Center strike (uses SPX price if None)
            strike_range: Strike range around center
            
        Returns:
            Dictionary formatted for strategy classes
        """
        # Enhanced debug logging for monitoring issues
        logger.debug(f"get_options_data_for_strategy called: date={date}, timestamp={timestamp}, "
                    f"center_strike={center_strike}, strike_range={strike_range}")
        
        # Get SPX price
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            logger.warning(f"No SPX price found for {date} {timestamp}")
            # Debug: Check if date exists in time index
            date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date)
            available_dates = list(self.query_engine._time_index.keys()) if hasattr(self.query_engine, '_time_index') else []
            logger.debug(f"Date {date_str} not found. Available indexed dates: {available_dates[:5]}{'...' if len(available_dates) > 5 else ''}")
            return {}
        
        logger.debug(f"Retrieved SPX price: {spx_price} for {date} {timestamp}")
        
        # Use SPX price as center if not provided
        if center_strike is None:
            center_strike = spx_price
            logger.debug(f"Using SPX price as center_strike: {center_strike}")
        
        # Get options chain with enhanced logging
        logger.debug(f"Requesting options chain: center_strike={center_strike}, strike_range={strike_range}")
        options_df = self.query_engine.loader.get_options_chain_at_time(
            date, timestamp, center_strike, strike_range
        )
        
        if options_df.empty:
            logger.warning(f"No options data found for {date} {timestamp}")
            # Debug: Check if options data is available for this date
            try:
                all_options = self.query_engine.loader.load_options_data(date)
                logger.debug(f"Total options data available for {date}: {len(all_options)} rows")
                if not all_options.empty:
                    available_times = all_options.index.get_level_values('timestamp').unique()
                    logger.debug(f"Available option timestamps: {len(available_times)} from "
                               f"{available_times.min()} to {available_times.max()}")
                    # Check for near timestamps
                    target_dt = pd.to_datetime(f"{date} {timestamp}") if isinstance(timestamp, str) else timestamp
                    near_times = available_times[abs(available_times - target_dt) <= pd.Timedelta(minutes=5)]
                    logger.debug(f"Options timestamps within 5 min of {target_dt}: {len(near_times)}")
            except Exception as e:
                logger.debug(f"Error checking options data availability: {e}")
            return {}
        
        logger.debug(f"Retrieved options chain: {len(options_df)} rows with strikes from "
                    f"{options_df['strike'].min()} to {options_df['strike'].max()}")
        
        # Convert to dictionary format with size logging
        options_dict = self.convert_options_dataframe_to_dict(options_df, spx_price)
        logger.debug(f"Converted to dictionary format: {len(options_dict)} options with keys: "
                    f"{list(options_dict.keys())[:5]}{'...' if len(options_dict) > 5 else ''}")
        
        return options_dict
    
    def get_liquid_options_for_strategy(self, date: Union[str, datetime],
                                      timestamp: Union[str, datetime, time],
                                      min_bid: float = 0.05,
                                      max_spread_pct: float = 30.0,
                                      center_strike: Optional[float] = None,
                                      strike_range: float = 150) -> Dict[str, Dict]:
        """
        Get liquid options data formatted for strategy creation.
        
        Args:
            date: Trading date
            timestamp: Time for data lookup
            min_bid: Minimum bid price for liquidity
            max_spread_pct: Maximum bid/ask spread percentage
            center_strike: Center strike for filtering
            strike_range: Strike range around center
            
        Returns:
            Dictionary with liquid options only
        """
        # Get liquid options using query engine
        liquid_df = self.query_engine.find_liquid_options(
            date, timestamp, min_bid, max_spread_pct
        )
        
        if liquid_df.empty:
            logger.warning(f"No liquid options found for {date} {timestamp}")
            return {}
        
        # Apply strike filtering if requested
        if center_strike is not None:
            strike_mask = (
                (liquid_df['strike'] >= center_strike - strike_range) &
                (liquid_df['strike'] <= center_strike + strike_range)
            )
            liquid_df = liquid_df[strike_mask]
        
        # Get SPX price for calculations
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        
        return self.convert_options_dataframe_to_dict(liquid_df, spx_price)


class EnhancedStrategyBuilder:
    """
    Enhanced strategy builder that uses parquet data for efficient strategy creation.
    """
    
    def __init__(self, query_engine: BacktestQueryEngine):
        self.query_engine = query_engine
        self.data_adapter = ParquetDataAdapter(query_engine)
        self._strategy_cache = {}
    
    def build_iron_condor_optimized(self, 
                                  date: Union[str, datetime],
                                  timestamp: Union[str, datetime, time],
                                  put_distance: int = 50,
                                  call_distance: int = 50,
                                  spread_width: int = 10,
                                  quantity: int = 1,
                                  use_liquid_options: bool = True,
                                  min_bid: float = 0.10,
                                  max_spread_pct: float = 20.0) -> Optional[IronCondor]:
        """
        Build Iron Condor using optimized parquet data access.
        
        Args:
            date: Trading date
            timestamp: Entry timestamp
            put_distance: Distance for put short strike from underlying
            call_distance: Distance for call short strike from underlying  
            spread_width: Width of each spread
            quantity: Number of contracts
            use_liquid_options: Only use liquid options
            min_bid: Minimum bid for liquidity filter
            max_spread_pct: Maximum spread % for liquidity filter
            
        Returns:
            IronCondor strategy or None if insufficient data
        """
        # Get SPX price
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            logger.error(f"No SPX price found for {date} {timestamp}")
            return None
        
        # Get options data
        if use_liquid_options:
            options_data = self.data_adapter.get_liquid_options_for_strategy(
                date, timestamp, min_bid, max_spread_pct, spx_price, 200
            )
        else:
            options_data = self.data_adapter.get_options_data_for_strategy(
                date, timestamp, spx_price, 200
            )
        
        if not options_data:
            logger.warning(f"No suitable options data found for Iron Condor at {date} {timestamp}")
            return None
        
        # Build strategy using existing builder
        try:
            iron_condor = StrategyBuilder.build_iron_condor(
                entry_date=pd.to_datetime(f"{date} {timestamp}") if isinstance(timestamp, str) else timestamp,
                underlying_price=spx_price,
                options_data=options_data,
                put_distance=put_distance,
                call_distance=call_distance,
                spread_width=spread_width,
                quantity=quantity,
                expiration=pd.to_datetime(date)  # 0DTE
            )
            
            # Add metadata
            iron_condor.entry_spx_price = spx_price
            iron_condor.entry_timestamp = timestamp
            iron_condor.liquidity_filtered = use_liquid_options
            
            return iron_condor
            
        except Exception as e:
            logger.error(f"Failed to build Iron Condor: {e}")
            return None
    
    def build_call_spread_optimized(self,
                                  date: Union[str, datetime],
                                  timestamp: Union[str, datetime, time],
                                  strike_distance: int = 10,
                                  spread_width: int = 10,
                                  is_debit: bool = True,
                                  quantity: int = 1,
                                  use_liquid_options: bool = True) -> Optional[VerticalSpread]:
        """Build Call Spread using optimized data access."""
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            return None
        
        if use_liquid_options:
            options_data = self.data_adapter.get_liquid_options_for_strategy(
                date, timestamp, 0.05, 30.0, spx_price, 150
            )
        else:
            options_data = self.data_adapter.get_options_data_for_strategy(
                date, timestamp, spx_price, 150
            )
        
        if not options_data:
            return None
        
        try:
            return StrategyBuilder.build_call_spread(
                entry_date=pd.to_datetime(f"{date} {timestamp}") if isinstance(timestamp, str) else timestamp,
                underlying_price=spx_price,
                options_data=options_data,
                strike_distance=strike_distance,
                spread_width=spread_width,
                is_debit=is_debit,
                quantity=quantity,
                expiration=pd.to_datetime(date)
            )
        except Exception as e:
            logger.error(f"Failed to build Call Spread: {e}")
            return None
    
    def build_put_spread_optimized(self,
                                 date: Union[str, datetime],
                                 timestamp: Union[str, datetime, time],
                                 strike_distance: int = 10,
                                 spread_width: int = 10,
                                 is_debit: bool = True,
                                 quantity: int = 1,
                                 use_liquid_options: bool = True) -> Optional[VerticalSpread]:
        """Build Put Spread using optimized data access."""
        spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
        if spx_price is None:
            return None
        
        if use_liquid_options:
            options_data = self.data_adapter.get_liquid_options_for_strategy(
                date, timestamp, 0.05, 30.0, spx_price, 150
            )
        else:
            options_data = self.data_adapter.get_options_data_for_strategy(
                date, timestamp, spx_price, 150
            )
        
        if not options_data:
            return None
        
        try:
            return StrategyBuilder.build_put_spread(
                entry_date=pd.to_datetime(f"{date} {timestamp}") if isinstance(timestamp, str) else timestamp,
                underlying_price=spx_price,
                options_data=options_data,
                strike_distance=strike_distance,
                spread_width=spread_width,
                is_debit=is_debit,
                quantity=quantity,
                expiration=pd.to_datetime(date)
            )
        except Exception as e:
            logger.error(f"Failed to build Put Spread: {e}")
            return None
    
    def update_strategy_prices_optimized(self, 
                                       strategy: OptionsStrategy,
                                       date: Union[str, datetime],
                                       timestamp: Union[str, datetime, time]) -> bool:
        """
        Update strategy prices using optimized data access.
        
        Args:
            strategy: Strategy to update
            date: Trading date
            timestamp: Current timestamp
            
        Returns:
            Success boolean
        """
        try:
            # Enhanced debug logging for monitoring periods
            logger.debug(f"update_strategy_prices_optimized called for {date} {timestamp}")
            
            # Get current options data for all legs
            spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
            if spx_price is None:
                logger.debug(f"No SPX price available for {date} {timestamp}")
                return False
            
            # Get strikes from strategy legs
            strikes = [leg.strike for leg in strategy.legs]
            if not strikes:
                logger.debug("Strategy has no legs with strikes")
                return False
            
            logger.debug(f"Strategy leg strikes: {strikes}, SPX price: {spx_price}")
            
            center_strike = spx_price
            strike_range = max(abs(max(strikes) - center_strike), 
                             abs(min(strikes) - center_strike)) + 50
            
            logger.debug(f"Calculated strike range: {center_strike} ± {strike_range}")
            
            # This is where get_options_data_for_strategy is called - lines 371-373 equivalent
            options_data = self.data_adapter.get_options_data_for_strategy(
                date, timestamp, center_strike, strike_range
            )
            
            if not options_data:
                logger.debug(f"No options data available for {date} {timestamp} around strike {center_strike}")
                # Additional diagnostic information
                diagnosis = self.data_adapter.diagnose_data_availability(date, timestamp)
                logger.debug(f"Data diagnosis: SPX={diagnosis['spx_available']}, "
                           f"Options={diagnosis['options_available']}, "
                           f"Options count={diagnosis['options_count']}")
                return False
            
            # Debug: Log what data was retrieved
            logger.debug(f"Retrieved options data for {len(options_data)} strikes at {timestamp}")
            available_strikes = [float(key.split('_')[0]) for key in options_data.keys()]
            logger.debug(f"Available option strikes: {sorted(set(available_strikes))}")
            
            # Check if required strikes are available
            missing_strikes = []
            for strike in strikes:
                call_key = f"{strike}_call"
                put_key = f"{strike}_put"
                if call_key not in options_data and put_key not in options_data:
                    missing_strikes.append(strike)
            
            if missing_strikes:
                logger.debug(f"Missing data for strikes: {missing_strikes}")
            
            # Update strategy with new prices
            strategy.update_prices(options_data)
            
            # Debug: Check if prices were actually updated
            updated_legs = sum(1 for leg in strategy.legs if leg.current_price > 0)
            logger.debug(f"Updated prices for {updated_legs}/{len(strategy.legs)} legs at {timestamp}")
            
            # Log individual leg updates
            for i, leg in enumerate(strategy.legs):
                option_key = f"{leg.strike}_{leg.option_type}"
                has_data = option_key in options_data
                logger.debug(f"Leg {i}: {option_key}, price={leg.current_price}, data_available={has_data}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update strategy prices: {e}")
            return False
    
    def clear_cache(self):
        """Clear strategy cache to free memory."""
        self._strategy_cache.clear()
        self.data_adapter._option_cache.clear()
        self.data_adapter._spx_cache.clear()
        logger.info("Cleared strategy adapter cache")


# Convenience functions for backtesting
def create_strategy_builder_for_backtest(data_path: str = "data/processed/parquet_1m") -> EnhancedStrategyBuilder:
    """
    Create strategy builder configured for backtesting.
    
    Args:
        data_path: Path to parquet data
        
    Returns:
        Configured EnhancedStrategyBuilder
    """
    from ..data.query_engine import create_fast_query_engine
    
    query_engine = create_fast_query_engine(data_path)
    return EnhancedStrategyBuilder(query_engine)


def quick_iron_condor_test(date: str, time_str: str, 
                          data_path: str = "data/processed/parquet_1m") -> Optional[IronCondor]:
    """
    Quick test function to create an Iron Condor.
    
    Args:
        date: Date string (YYYY-MM-DD)
        time_str: Time string (HH:MM:SS)
        data_path: Path to data
        
    Returns:
        IronCondor strategy or None
    """
    builder = create_strategy_builder_for_backtest(data_path)
    
    return builder.build_iron_condor_optimized(
        date=date,
        timestamp=time_str,
        put_distance=50,
        call_distance=50,
        spread_width=10,
        quantity=1,
        use_liquid_options=True
    )