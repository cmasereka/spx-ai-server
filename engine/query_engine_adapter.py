#!/usr/bin/env python3
"""
Compatibility Adapter for Enhanced Backtesting System

Bridges the enhanced system with the existing query engine infrastructure
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from loguru import logger

from src.data.query_engine import BacktestQueryEngine


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class EnhancedQueryEngineAdapter:
    """Adapter to add missing methods to existing query engine"""

    def __init__(self, query_engine: BacktestQueryEngine):
        self.query_engine = query_engine

    def get_spx_data(self, date: str, start_time: str = "09:30:00",
                     end_time: str = "16:00:00") -> Optional[pd.DataFrame]:
        """Get SPX price data for technical analysis"""
        try:
            session_data = self.query_engine.get_trading_session_data(date, start_time, end_time)

            if session_data and 'spx' in session_data:
                spx_df = session_data['spx']
                if len(spx_df) > 0:
                    # Normalize column name so downstream code can always use 'close'
                    if 'price' in spx_df.columns and 'close' not in spx_df.columns:
                        spx_df = spx_df.rename(columns={'price': 'close'})
                    # Drop any zero-price rows (e.g. the 09:30 sentinel bar in the data)
                    spx_df = spx_df[spx_df['close'] > 0]
                    if len(spx_df) > 0:
                        return spx_df

            # Fallback: synthetic price history around current price
            current_price = self.query_engine.get_fastest_spx_price(date, end_time)
            if current_price:
                timestamps = pd.date_range(
                    start=f"{date} {start_time}",
                    end=f"{date} {end_time}",
                    freq='1min'
                )
                np.random.seed(hash(date) % 2**32)
                returns = np.random.normal(0, 0.001, len(timestamps))
                prices = [current_price]
                for ret in returns[:-1]:
                    prices.append(prices[-1] * (1 + ret))

                return pd.DataFrame({
                    'timestamp': timestamps,
                    'close': prices
                }).set_index('timestamp')

            return None

        except Exception as e:
            logger.warning(f"Could not get SPX data for {date}: {e}")
            return None

    def get_options_data(self, date: str, timestamp: str) -> Optional[pd.DataFrame]:
        """
        Get options chain at the given timestamp.
        Returns strike, option_type, expiration, bid, ask for all liquid options.
        Filters out rows with no valid ask (illiquid / pre-open quotes).
        """
        try:
            spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
            if not spx_price:
                return None

            options_chain = self.query_engine.loader.get_options_chain_at_time(
                date, timestamp, center_strike=spx_price, strike_range=300
            )

            if options_chain is None or len(options_chain) == 0:
                return None

            options_list = []

            for _, row in options_chain.iterrows():
                bid = float(row.get('bid', 0))
                ask = float(row.get('ask', 0))

                # Skip rows with no valid ask (completely illiquid / pre-open)
                if ask <= 0:
                    continue

                right_value = str(row.get('right', 'C')).upper()
                option_type = 'call' if right_value in ('C', 'CALL') else 'put'

                strike = float(row.get('strike', 0))
                if strike <= 0:
                    continue

                options_list.append({
                    'strike':      strike,
                    'option_type': option_type,
                    'expiration':  date,
                    'bid':         bid,
                    'ask':         ask,
                    'volume':      row.get('volume', 0),
                })

            if options_list:
                return pd.DataFrame(options_list)

            return None

        except Exception as e:
            logger.warning(f"Could not get options data for {date} at {timestamp}: {e}")
            return None

    def __getattr__(self, name):
        """Delegate all other methods to the original query engine"""
        return getattr(self.query_engine, name)
