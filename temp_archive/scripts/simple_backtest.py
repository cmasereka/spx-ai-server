#!/usr/bin/env python3
"""
Simple SPX 0DTE Backtesting Pipeline

Start with single-day backtests, easily expandable to date ranges.
Focus on Iron Condor strategies with clear, actionable results.
"""
import sys
sys.path.append('.')

import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import json
from loguru import logger

from src.data.query_engine import create_fast_query_engine
from src.backtesting.iron_condor_loader import IronCondorDataLoader
from src.backtesting.strategy_adapter import EnhancedStrategyBuilder


@dataclass
class SimpleBacktestResult:
    """Simple backtest result for a single day"""
    date: str
    strategy: str
    entry_time: str
    exit_time: str
    entry_spx_price: float
    exit_spx_price: float
    entry_credit: float
    exit_cost: float
    pnl: float
    pnl_pct: float
    max_profit: float
    max_loss: float
    success: bool
    notes: str


class SimpleBacktester:
    """
    Simple backtesting pipeline for SPX 0DTE Iron Condor strategies.
    
    Designed to be:
    - Easy to understand and modify
    - Fast execution
    - Clear results
    - Expandable to date ranges
    """
    
    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        self.data_path = data_path
        
        # Initialize components
        logger.info("Initializing simple backtester...")
        self.query_engine = create_fast_query_engine(data_path)
        self.ic_loader = IronCondorDataLoader(self.query_engine)
        self.strategy_builder = EnhancedStrategyBuilder(self.query_engine)
        
        # Get available dates
        self.available_dates = [d.strftime('%Y-%m-%d') for d in self.query_engine.loader.available_dates]
        if self.available_dates:
            logger.info(f"Available dates: {len(self.available_dates)} days from {self.available_dates[0]} to {self.available_dates[-1]}")
        else:
            logger.warning("No available dates found in data source")
    
    def backtest_single_day(self, 
                           date: str,
                           entry_time: str = "10:00:00",
                           exit_time: str = "15:45:00",
                           put_distance: int = 50,
                           call_distance: int = 50,
                           spread_width: int = 10,
                           min_credit: float = 0.50) -> SimpleBacktestResult:
        """
        Run a simple Iron Condor backtest for a single day.
        
        Args:
            date: Date to backtest (YYYY-MM-DD)
            entry_time: Entry time (HH:MM:SS)
            exit_time: Exit time (HH:MM:SS)  
            put_distance: Put short strike distance from SPX
            call_distance: Call short strike distance from SPX
            spread_width: Spread width for both sides
            min_credit: Minimum credit required
            
        Returns:
            SimpleBacktestResult with trade details
        """
        logger.info(f"Backtesting {date} - Iron Condor {put_distance}P/{call_distance}C, {spread_width}W")
        
        # Validate date
        if date not in self.available_dates:
            return SimpleBacktestResult(
                date=date, strategy="Iron Condor", entry_time=entry_time, exit_time=exit_time,
                entry_spx_price=0, exit_spx_price=0, entry_credit=0, exit_cost=0,
                pnl=0, pnl_pct=0, max_profit=0, max_loss=0,
                success=False, notes=f"No data available for {date}"
            )
        
        try:
            # Step 1: Get entry SPX price
            entry_spx_price = self.query_engine.get_fastest_spx_price(date, entry_time)
            if entry_spx_price is None:
                return SimpleBacktestResult(
                    date=date, strategy="Iron Condor", entry_time=entry_time, exit_time=exit_time,
                    entry_spx_price=0, exit_spx_price=0, entry_credit=0, exit_cost=0,
                    pnl=0, pnl_pct=0, max_profit=0, max_loss=0,
                    success=False, notes=f"No SPX price at {entry_time}"
                )
            
            # Step 2: Find best Iron Condor setup
            ic_setup = self.ic_loader.get_best_iron_condor_setup(
                date=date,
                timestamp=entry_time,
                put_distances=[put_distance],
                call_distances=[call_distance], 
                spread_widths=[spread_width],
                min_credit=min_credit,
                optimize_for='credit'
            )
            
            if not ic_setup or not ic_setup.is_valid:
                return SimpleBacktestResult(
                    date=date, strategy="Iron Condor", entry_time=entry_time, exit_time=exit_time,
                    entry_spx_price=entry_spx_price, exit_spx_price=0, entry_credit=0, exit_cost=0,
                    pnl=0, pnl_pct=0, max_profit=0, max_loss=0,
                    success=False, notes=f"No viable Iron Condor setup found"
                )
            
            # Step 3: Create Iron Condor strategy
            iron_condor = self.strategy_builder.build_iron_condor_optimized(
                date=date,
                timestamp=entry_time,
                put_distance=put_distance,
                call_distance=call_distance,
                spread_width=spread_width,
                quantity=1,
                use_liquid_options=True
            )
            
            if not iron_condor or not iron_condor.legs:
                return SimpleBacktestResult(
                    date=date, strategy="Iron Condor", entry_time=entry_time, exit_time=exit_time,
                    entry_spx_price=entry_spx_price, exit_spx_price=0, 
                    entry_credit=ic_setup.net_credit, exit_cost=0,
                    pnl=0, pnl_pct=0, max_profit=ic_setup.max_profit, max_loss=ic_setup.max_loss,
                    success=False, notes="Could not create Iron Condor strategy"
                )
            
            # Step 4: Get exit SPX price and update strategy
            exit_spx_price = self.query_engine.get_fastest_spx_price(date, exit_time)
            if exit_spx_price is None:
                exit_spx_price = entry_spx_price  # Fallback
            
            # Update strategy prices at exit
            self.strategy_builder.update_strategy_prices_optimized(iron_condor, date, exit_time)
            
            # Step 5: Calculate P&L
            entry_credit = iron_condor.entry_credit
            current_cost = 0.0
            
            # Calculate current cost to close the position
            for leg in iron_condor.legs:
                if leg.position_side.name == 'SHORT':
                    current_cost += leg.current_price * 100  # Cost to buy back short
                else:
                    current_cost -= leg.current_price * 100  # Credit from selling long
            
            # P&L = entry credit - exit cost
            pnl = entry_credit - current_cost
            pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0
            
            # Success criteria: trade was executed (regardless of profit/loss)
            success = True  # If we reach this point, the trade was successfully executed
            
            # Generate notes
            strikes_info = f"Put: {ic_setup.put_long_strike}/{ic_setup.put_short_strike}, Call: {ic_setup.call_short_strike}/{ic_setup.call_long_strike}"
            notes = f"Strikes: {strikes_info}, Credit: ${entry_credit:.2f}"
            
            return SimpleBacktestResult(
                date=date,
                strategy="Iron Condor",
                entry_time=entry_time,
                exit_time=exit_time,
                entry_spx_price=entry_spx_price,
                exit_spx_price=exit_spx_price,
                entry_credit=entry_credit,
                exit_cost=current_cost,
                pnl=pnl,
                pnl_pct=pnl_pct,
                max_profit=iron_condor.max_profit,
                max_loss=iron_condor.max_loss,
                success=success,
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"Backtest failed for {date}: {e}")
            return SimpleBacktestResult(
                date=date, strategy="Iron Condor", entry_time=entry_time, exit_time=exit_time,
                entry_spx_price=entry_spx_price if 'entry_spx_price' in locals() else 0,
                exit_spx_price=0, entry_credit=0, exit_cost=0,
                pnl=0, pnl_pct=0, max_profit=0, max_loss=0,
                success=False, notes=f"Error: {str(e)}"
            )
    
    def backtest_date_range(self, 
                           start_date: str,
                           end_date: str,
                           **backtest_params) -> List[SimpleBacktestResult]:
        """
        Run backtests for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **backtest_params: Parameters to pass to backtest_single_day
            
        Returns:
            List of SimpleBacktestResult for each day
        """
        logger.info(f"Backtesting date range: {start_date} to {end_date}")
        
        # Generate date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        
        # Filter to available dates
        test_dates = []
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in self.available_dates:
                test_dates.append(date_str)
        
        logger.info(f"Testing {len(test_dates)} available days in range")
        
        # Run backtests
        results = []
        for i, date in enumerate(test_dates, 1):
            logger.info(f"Testing {i}/{len(test_dates)}: {date}")
            result = self.backtest_single_day(date, **backtest_params)
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[SimpleBacktestResult]) -> Dict[str, Any]:
        """
        Analyze backtest results and provide summary statistics.
        
        Args:
            results: List of backtest results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {"error": "No results to analyze"}
        
        successful_trades = [r for r in results if r.success]
        failed_trades = [r for r in results if not r.success]
        profitable_trades = [r for r in successful_trades if r.pnl > 0]  # Only from successful trades
        losing_trades = [r for r in successful_trades if r.pnl < 0]      # Only from successful trades
        
        total_pnl = sum(r.pnl for r in successful_trades)  # Only successful trades contribute to P&L
        total_credits = sum(r.entry_credit for r in successful_trades if r.entry_credit > 0)
        
        analysis = {
            "summary": {
                "total_days": len(results),
                "successful_setups": len(successful_trades),
                "failed_setups": len(failed_trades),
                "setup_success_rate": len(successful_trades) / len(results) * 100 if results else 0
            },
            "trading": {
                "total_pnl": total_pnl,
                "avg_pnl_per_day": total_pnl / len(successful_trades) if successful_trades else 0,
                "profitable_days": len(profitable_trades),
                "losing_days": len(losing_trades),
                "win_rate": len(profitable_trades) / len(successful_trades) * 100 if successful_trades else 0,
                "total_credits_collected": total_credits
            },
            "performance": {
                "best_day": max(successful_trades, key=lambda r: r.pnl) if successful_trades else None,
                "worst_day": min(successful_trades, key=lambda r: r.pnl) if successful_trades else None,
                "avg_credit": total_credits / len(successful_trades) if successful_trades else 0,
                "avg_win": sum(r.pnl for r in profitable_trades) / len(profitable_trades) if profitable_trades else 0,
                "avg_loss": sum(r.pnl for r in losing_trades) / len(losing_trades) if losing_trades else 0
            }
        }
        
        return analysis
    
    def print_results(self, results: List[SimpleBacktestResult], show_details: bool = True, detailed_trades: bool = False):
        """Print formatted results to console."""
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*120}")
        print(f"DETAILED BACKTEST RESULTS - {len(results)} Days")
        print(f"{'='*120}")
        
        if detailed_trades:
            self._print_detailed_trades(results)
        elif show_details:
            print(f"{'Date':<12} {'Entry':<8} {'SPX':<8} {'Credit':<8} {'P&L':<10} {'%':<7} {'Status':<10}")
            print(f"{'-'*80}")
            
            for result in results:
                status = "✓ WIN" if result.pnl > 0 else "✗ LOSS" if result.success else "SKIP"
                status_color = status
                
                print(f"{result.date:<12} {result.entry_time:<8} "
                      f"{result.entry_spx_price:<8.0f} ${result.entry_credit:<7.2f} "
                      f"{result.pnl:<9.2f} {result.pnl_pct:<6.1f}% {status_color:<10}")
        
        # Summary statistics
        analysis = self.analyze_results(results)
        summary = analysis["summary"]
        trading = analysis["trading"]
        performance = analysis["performance"]
        
        print(f"\n{'-'*80}")
        print(f"SUMMARY STATISTICS")
        print(f"{'-'*80}")
        print(f"Setup Success Rate:  {summary['setup_success_rate']:.1f}% ({summary['successful_setups']}/{summary['total_days']})")
        print(f"Trading Win Rate:    {trading['win_rate']:.1f}% ({trading['profitable_days']}/{summary['successful_setups']})")
        print(f"Total P&L:           ${trading['total_pnl']:.2f}")
        print(f"Avg P&L per Day:     ${trading['avg_pnl_per_day']:.2f}")
        print(f"Total Credits:       ${trading['total_credits_collected']:.2f}")
        print(f"Avg Credit per Day:  ${performance['avg_credit']:.2f}")
        
        if trading['profitable_days'] > 0:
            print(f"Avg Win:             ${performance['avg_win']:.2f}")
        if trading['losing_days'] > 0:
            print(f"Avg Loss:            ${performance['avg_loss']:.2f}")
        
        if performance['best_day']:
            best = performance['best_day']
            print(f"Best Day:            {best.date} (${best.pnl:.2f})")
        
        if performance['worst_day']:
            worst = performance['worst_day']  
            print(f"Worst Day:           {worst.date} (${worst.pnl:.2f})")
        
        print(f"{'='*120}\n")
    
    def _print_detailed_trades(self, results: List[SimpleBacktestResult]):
        """Print detailed trade information."""
        for i, result in enumerate(results, 1):
            status = "✓ WIN" if result.pnl > 0 else "✗ LOSS" if result.success else "✗ SKIP"
            status_emoji = "🟢" if result.pnl > 0 else "🔴" if result.success else "⚪"
            
            print(f"\n{status_emoji} TRADE {i}: {result.date} - {status}")
            print(f"{'-'*60}")
            
            if result.success:
                # Parse strikes from notes
                strikes_part = result.notes.split("Strikes: ")[1].split(", Credit:")[0]
                put_part = strikes_part.split(", Call: ")[0].replace("Put: ", "")
                call_part = strikes_part.split(", Call: ")[1]
                
                put_long, put_short = put_part.split("/")
                call_short, call_long = call_part.split("/")
                
                print(f"Entry Time:     {result.entry_time}")
                print(f"SPX Price:      ${result.entry_spx_price:.0f} → ${result.exit_spx_price:.0f} (Δ{result.exit_spx_price - result.entry_spx_price:+.0f})")
                print(f"")
                print(f"IRON CONDOR STRIKES:")
                print(f"Put Spread:     {put_long}P / {put_short}P  (Buy {put_long} / Sell {put_short})")
                print(f"Call Spread:    {call_short}C / {call_long}C  (Sell {call_short} / Buy {call_long})")
                print(f"")
                print(f"TRADE FINANCIALS:")
                print(f"Entry Credit:   ${result.entry_credit:>8.2f}  (Premium received)")
                print(f"Exit Cost:      ${result.exit_cost:>8.2f}  (Cost to close)")
                print(f"Net P&L:        ${result.pnl:>8.2f}  ({result.pnl_pct:+.1f}%)")
                print(f"Max Profit:     ${result.max_profit:>8.2f}")
                print(f"Max Loss:       ${result.max_loss:>8.2f}")
            else:
                print(f"Entry Time:     {result.entry_time}")
                print(f"SPX Price:      ${result.entry_spx_price:.0f}")
                print(f"Status:         No viable setup found")
                print(f"Reason:         {result.notes}")
            
            if i < len(results):
                print(f"")


def run_single_day_example():
    """Example of running a single day backtest."""
    print("Single Day Backtest Example")
    print("=" * 40)
    
    backtester = SimpleBacktester()
    
    # Test a single day
    result = backtester.backtest_single_day(
        date="2026-02-09",
        entry_time="10:00:00",
        exit_time="15:45:00",
        put_distance=50,
        call_distance=50,
        spread_width=10,
        min_credit=0.50
    )
    
    # Print detailed result
    backtester.print_results([result], detailed_trades=True)


def run_date_range_example():
    """Example of running a date range backtest."""
    print("Date Range Backtest Example")
    print("=" * 40)
    
    backtester = SimpleBacktester()
    
    # Test all available days
    results = backtester.backtest_date_range(
        start_date="2026-02-09",
        end_date="2026-02-13",
        entry_time="10:00:00",
        exit_time="15:45:00",
        put_distance=50,
        call_distance=50,
        spread_width=10,
        min_credit=0.50
    )
    
    # Print results
    backtester.print_results(results, show_details=True)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    print("SPX 0DTE Simple Backtesting Pipeline")
    print("=" * 50)
    print()
    
    # Run examples
    try:
        run_single_day_example()
        print()
        run_date_range_example()
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"Error: {e}")
        print("Make sure your data is available in data/processed/parquet_1m/")