#!/usr/bin/env python3
"""
Enhanced SPX 0DTE Multi-Strategy Backtesting System

Features:
1. Multi-strategy selection (IC, Put Spreads, Call Spreads)
2. Technical indicators (RSI, MACD, Bollinger Bands) for strategy selection
3. Delta/Probability ITM based strike selection
4. Dynamic position monitoring with 5-minute intervals
5. Decay-based exit rules (0.1 threshold)
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Literal
from dataclasses import dataclass
import argparse
from loguru import logger
from enum import Enum



class SimpleBacktester:
    """Minimal base class kept for backward compatibility with EnhancedMultiStrategyBacktester."""
    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        from src.data.query_engine import create_fast_query_engine
        from src.backtesting.iron_condor_loader import IronCondorDataLoader
        from src.backtesting.strategy_adapter import EnhancedStrategyBuilder
        self.data_path = data_path
        self.query_engine = create_fast_query_engine(data_path)
        self.ic_loader = IronCondorDataLoader(self.query_engine)
        self.strategy_builder = EnhancedStrategyBuilder(self.query_engine)
        self.available_dates = [
            d.strftime('%Y-%m-%d') for d in self.query_engine.loader.available_dates
        ]


class StrategyType(Enum):
    IRON_CONDOR = "Iron Condor"
    PUT_SPREAD = "Put Spread"
    CALL_SPREAD = "Call Spread"


class MarketSignal(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish" 
    NEUTRAL = "Neutral"


@dataclass
class TechnicalIndicators:
    """Technical indicator values"""
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float  # Where price is relative to bands (0-1)


@dataclass
class StrategySelection:
    """Strategy selection based on technical analysis"""
    strategy_type: StrategyType
    market_signal: MarketSignal
    confidence: float  # 0-1 confidence score
    reason: str


@dataclass
class StrikeSelection:
    """Credit-based strike selection"""
    short_strike: float
    long_strike: float
    spread_width: float


@dataclass
class EnhancedBacktestResult:
    """Enhanced backtest result with detailed tracking"""
    date: str
    strategy_type: StrategyType
    market_signal: MarketSignal
    entry_time: str
    exit_time: str
    exit_reason: str

    # SPX movement
    entry_spx_price: float
    exit_spx_price: float

    # Technical indicators
    technical_indicators: TechnicalIndicators

    # Strike details
    strike_selection: StrikeSelection

    # P&L details
    entry_credit: float
    exit_cost: float
    pnl: float
    pnl_pct: float
    max_profit: float
    max_loss: float

    # Monitoring
    monitoring_points: List[Dict[str, Any]]  # 5-min intervals
    success: bool
    confidence: float
    notes: str

    # Full decision audit trail
    # entry_rationale: snapshot of every factor that led to this trade being opened
    # exit_rationale:  snapshot of every factor that triggered the close
    entry_rationale: Optional[Dict[str, Any]] = None
    exit_rationale:  Optional[Dict[str, Any]] = None

    # IC independent leg tracking (only populated for IC trades)
    ic_leg_status: Optional['IronCondorLegStatus'] = None


@dataclass
class IronCondorLegStatus:
    """Tracks independent close status of each IC side."""
    put_side_closed: bool = False
    call_side_closed: bool = False
    put_side_exit_time: Optional[str] = None
    call_side_exit_time: Optional[str] = None
    put_side_exit_cost: float = 0.0
    call_side_exit_cost: float = 0.0
    put_side_exit_reason: str = ""
    call_side_exit_reason: str = ""


@dataclass
class DayBacktestResult:
    """All trades executed on a single day."""
    date: str
    trades: List[EnhancedBacktestResult]   # 0..N trades per day
    total_pnl: float
    trade_count: int
    scan_minutes_checked: int              # How many 1-min bars were scanned


class TechnicalAnalyzer:
    """Technical analysis for strategy selection"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return (
            macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0,
            macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0.0,
            macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0.0
        )
    
    @staticmethod 
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float, float]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        current_price = prices.iloc[-1]
        bb_upper = upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else current_price * 1.02
        bb_middle = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else current_price
        bb_lower = lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else current_price * 0.98
        
        # Position within bands (0 = lower band, 0.5 = middle, 1 = upper band)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        
        return bb_upper, bb_middle, bb_lower, bb_position
    
    def analyze_market_conditions(self, spx_prices: pd.Series) -> TechnicalIndicators:
        """Analyze market conditions using technical indicators"""
        
        # Ensure we have enough data
        if len(spx_prices) < 50:
            current_price = spx_prices.iloc[-1]
            return TechnicalIndicators(
                rsi=50.0,
                macd_line=0.0,
                macd_signal=0.0, 
                macd_histogram=0.0,
                bb_upper=current_price * 1.02,
                bb_middle=current_price,
                bb_lower=current_price * 0.98,
                bb_position=0.5
            )
        
        # Calculate indicators
        rsi = self.calculate_rsi(spx_prices)
        macd_line, macd_signal, macd_histogram = self.calculate_macd(spx_prices)
        bb_upper, bb_middle, bb_lower, bb_position = self.calculate_bollinger_bands(spx_prices)
        
        return TechnicalIndicators(
            rsi=rsi,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_position=bb_position
        )


class StrategySelector:
    """Select optimal strategy based on technical indicators.

    All strategies are credit spreads — we always sell premium:
      - Neutral market  → Iron Condor (sell both sides)
      - Bullish market  → Put Credit Spread  (sell OTM put spread; profits if market stays up)
      - Bearish market  → Call Credit Spread (sell OTM call spread; profits if market stays down)
    """

    def select_strategy(self, indicators: TechnicalIndicators) -> StrategySelection:
        """Select strategy based on technical analysis"""

        # Analyze market conditions
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0

        # RSI analysis
        if indicators.rsi > 70:
            bearish_signals += 1
        elif indicators.rsi < 30:
            bullish_signals += 1
        elif 40 <= indicators.rsi <= 60:
            neutral_signals += 1

        # MACD analysis
        if indicators.macd_histogram > 0 and indicators.macd_line > indicators.macd_signal:
            bullish_signals += 1
        elif indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
            bearish_signals += 1
        else:
            neutral_signals += 1

        # Bollinger Bands analysis
        if indicators.bb_position > 0.8:  # Near upper band — overbought
            bearish_signals += 1
        elif indicators.bb_position < 0.2:  # Near lower band — oversold
            bullish_signals += 1
        elif 0.3 <= indicators.bb_position <= 0.7:  # Middle range
            neutral_signals += 1

        if neutral_signals >= 2:  # At least 2 neutral signals → range-bound
            market_signal = MarketSignal.NEUTRAL
            strategy_type = StrategyType.IRON_CONDOR
            confidence = neutral_signals / 3
            reason = f"Neutral market (RSI:{indicators.rsi:.1f}, BB_pos:{indicators.bb_position:.2f}) → Iron Condor"

        elif bullish_signals > bearish_signals:
            # Bullish bias: sell a put credit spread (collect premium below the market)
            market_signal = MarketSignal.BULLISH
            strategy_type = StrategyType.PUT_SPREAD
            confidence = bullish_signals / 3
            reason = f"Bullish market (RSI:{indicators.rsi:.1f}, MACD_hist:{indicators.macd_histogram:.3f}) → Put Credit Spread"

        else:
            # Bearish bias: sell a call credit spread (collect premium above the market)
            market_signal = MarketSignal.BEARISH
            strategy_type = StrategyType.CALL_SPREAD
            confidence = bearish_signals / 3
            reason = f"Bearish market (RSI:{indicators.rsi:.1f}, BB_pos:{indicators.bb_position:.2f}) → Call Credit Spread"

        return StrategySelection(
            strategy_type=strategy_type,
            market_signal=market_signal,
            confidence=confidence,
            reason=reason
        )


class EnhancedMultiStrategyBacktester(SimpleBacktester):
    """Enhanced backtester with multi-strategy support and technical analysis"""
    
    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        super().__init__(data_path)
        self.technical_analyzer = TechnicalAnalyzer()
        self.strategy_selector = StrategySelector()
    
    def get_spx_price_history(self, date: str, end_time: str, lookback_minutes: int = 60) -> pd.Series:
        """Get SPX price history for technical analysis"""
        try:
            # Get SPX data for the date using enhanced adapter
            spx_data = self.enhanced_query_engine.get_spx_data(date, start_time="09:30:00", end_time=end_time)
            
            if spx_data is None or len(spx_data) == 0:
                # Fallback to single price point
                current_price = self.enhanced_query_engine.get_fastest_spx_price(date, end_time)
                if current_price:
                    return pd.Series([current_price] * 50)  # Create dummy series
                else:
                    return pd.Series([4500.0] * 50)  # Ultimate fallback
            
            # Return price series
            return spx_data['close'].tail(lookback_minutes)
            
        except Exception as e:
            logger.warning(f"Could not get SPX history for {date}: {e}")
            # Return fallback series
            current_price = self.enhanced_query_engine.get_fastest_spx_price(date, end_time) or 4500.0
            return pd.Series([current_price] * 50)


# Save progress and continue in next response due to length
print("Enhanced backtesting system framework created - Part 1/3")