#!/usr/bin/env python3
"""
Compatibility Adapter for Enhanced Backtesting System

Bridges the enhanced system with the existing query engine infrastructure
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from loguru import logger

from src.data.query_engine import BacktestQueryEngine


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

RISK_FREE_RATE = 0.045  # 4.5% — approximate Fed funds / short-term Treasury rate


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))


def _bs_price(S: float, K: float, T: float, r: float, sigma: float,
              is_call: bool) -> float:
    """Black-Scholes price for a European option."""
    if T <= 0 or sigma <= 0:
        # At expiry: intrinsic value only
        if is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    return K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float,
              is_call: bool) -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)

    if is_call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0  # negative for puts


def _implied_vol(S: float, K: float, T: float, r: float,
                 market_price: float, is_call: bool,
                 tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
    """
    Solve for implied volatility using Newton-Raphson.
    Returns None if no solution is found (e.g. price below intrinsic).
    """
    # Check that market_price is above intrinsic — otherwise IV is undefined
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if market_price <= intrinsic + 1e-8:
        return None

    # Initial guess: simple Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * abs(np.log(S / K)) / T) if T > 0 else 0.2
    sigma = max(sigma, 0.01)

    for _ in range(max_iter):
        price = _bs_price(S, K, T, r, sigma, is_call)
        # Vega = S * sqrt(T) * N'(d1)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        vega = S * sqrt_T * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)

        if abs(vega) < 1e-10:
            break

        diff = price - market_price
        if abs(diff) < tol:
            break

        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-4  # clamp before next iteration

    if sigma <= 0 or sigma > 20:  # sanity check
        return None
    return sigma


def _compute_bs_delta(S: float, K: float, T: float, r: float,
                      mid_price: float, is_call: bool) -> float:
    """
    Compute Black-Scholes delta by first solving for IV, then computing delta.
    Falls back to a sign-correct zero if IV cannot be solved.
    """
    if mid_price <= 0 or S <= 0 or K <= 0 or T < 0:
        return 0.0

    iv = _implied_vol(S, K, T, r, mid_price, is_call)
    if iv is None:
        return 0.0

    return _bs_delta(S, K, T, r, iv, is_call)


def _time_to_expiry_years(timestamp_str: str, expiration_str: str) -> float:
    """
    Fraction of a year from timestamp to market close (4 PM ET) on expiration date.
    Minimum floor of 1 minute to avoid division-by-zero at the final bar.
    """
    try:
        ts = datetime.strptime(timestamp_str, "%H:%M:%S")
        market_close = datetime.strptime("16:00:00", "%H:%M:%S")
        minutes_remaining = max((market_close - ts).total_seconds() / 60.0, 1.0)
        # Trading year ≈ 252 days × 390 minutes/day
        return minutes_remaining / (252 * 390)
    except Exception:
        return 1.0 / 252  # fallback: ~1 trading day


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
        Get options data with Black-Scholes delta computed from bid/ask mid-price.
        Filters out zero-bid rows (illiquid / pre-open quotes).
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

            T = _time_to_expiry_years(timestamp, date)
            options_list = []

            for _, row in options_chain.iterrows():
                bid = float(row.get('bid', 0))
                ask = float(row.get('ask', 0))

                # Skip rows with no valid ask (completely illiquid / pre-open)
                if ask <= 0:
                    continue

                right_value = str(row.get('right', 'C')).upper()
                is_call = right_value in ('C', 'CALL')
                option_type = 'call' if is_call else 'put'

                strike = float(row.get('strike', 0))
                if strike <= 0:
                    continue

                mid_price = (bid + ask) / 2.0

                delta = _compute_bs_delta(
                    S=spx_price,
                    K=strike,
                    T=T,
                    r=RISK_FREE_RATE,
                    mid_price=mid_price,
                    is_call=is_call
                )

                options_list.append({
                    'strike': strike,
                    'option_type': option_type,
                    'expiration': date,
                    'bid': bid,
                    'ask': ask,
                    'delta': delta,
                    'gamma': row.get('gamma', 0),
                    'theta': row.get('theta', 0),
                    'vega': row.get('vega', 0),
                    'volume': row.get('volume', 0)
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
