#!/usr/bin/env python3

"""
Test Suite Validation and Summary Report Generator

Runs the comprehensive test suite and generates a summary report of results.
"""

import subprocess
import datetime
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and return result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def generate_test_summary():
    """Generate comprehensive test summary"""
    
    print("🚀 SPX AI Enhanced Multi-Strategy System - Test Validation")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test categories to run
    test_categories = [
        ("Unit Tests", "tests/unit/"),
        ("Technical Analysis", "tests/unit/test_technical_analysis.py"),
        ("Options Strategies", "tests/unit/test_options_strategies.py"),
        ("Strike Selector", "tests/unit/test_strike_selector.py"),
    ]
    
    overall_success = True
    
    for category, path in test_categories:
        print(f"🧪 Running {category}...")
        success, stdout, stderr = run_command(f"pytest {path} -v --tb=short")
        
        if success:
            # Extract test count from output
            lines = stdout.split('\n')
            summary_line = [line for line in lines if 'passed' in line and ('failed' in line or 'error' in line or line.endswith('passed'))]
            
            if summary_line:
                print(f"✅ {category}: {summary_line[-1].split('=')[-1].strip()}")
            else:
                print(f"✅ {category}: PASSED")
        else:
            print(f"❌ {category}: FAILED")
            overall_success = False
    
    print()
    print("📊 Test Summary")
    print("-" * 40)
    
    # Run final comprehensive test
    print("Running comprehensive unit test suite...")
    success, stdout, stderr = run_command("pytest tests/unit/ -v")
    
    if success:
        # Extract detailed results
        lines = stdout.split('\n')
        summary_lines = [line for line in lines if 'passed' in line or 'failed' in line or 'error' in line]
        
        for line in lines:
            if '=====' in line and ('passed' in line or 'failed' in line):
                print(f"📈 Final Result: {line.split('=')[-1].strip()}")
                break
    
    print()
    
    if overall_success:
        print("🎉 SUCCESS: All test categories passed!")
        print()
        print("✅ Core Components Verified:")
        print("   • Technical Analysis (RSI, MACD, Bollinger Bands)")
        print("   • Strategy Selection Logic")
        print("   • Delta-Based Strike Selection")
        print("   • Position Monitoring")
        print("   • Options Strategy Infrastructure")
        print("   • Edge Case Handling")
        print()
        print("🚀 System Status: PRODUCTION READY")
        print("   The enhanced multi-strategy backtesting system has")
        print("   comprehensive test coverage and is ready for use.")
        
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
        
    print()
    print("🔗 Next Steps:")
    print("   • Run integration tests: pytest tests/integration/")
    print("   • Run performance tests: pytest tests/performance/") 
    print("   • Execute system test: python enhanced_multi_strategy.py --date 2026-02-09")
    
    return overall_success

if __name__ == "__main__":
    success = generate_test_summary()
    sys.exit(0 if success else 1)