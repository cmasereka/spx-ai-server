#!/bin/bash

# SPX AI Trading System - Test Runner Script
# Comprehensive test execution with reporting

# Resolve the project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🚀 SPX AI Trading System - Comprehensive Test Suite"
echo "=================================================="

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ Error: pytest not found. Installing..."
    pip install pytest pytest-cov pytest-html pytest-xdist psutil
fi

# Create test results directory
mkdir -p tests/results
TEST_RESULTS_DIR="tests/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_RESULTS_DIR"

echo "📋 Test results will be saved to: $TEST_RESULTS_DIR"
echo ""

# Generate test fixtures first
echo "🔧 Generating test fixtures..."
cd tests/fixtures
python generate_test_data.py
cd ../..
echo "✅ Test fixtures generated"
echo ""

# Function to run test suite with reporting
run_test_suite() {
    local suite_name="$1"
    local test_pattern="$2"
    local additional_args="$3"
    
    echo "🧪 Running $suite_name..."
    echo "----------------------------------------"
    
    local suite_lower=$(echo "$suite_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    pytest $test_pattern $additional_args \
        --html="$TEST_RESULTS_DIR/${suite_lower}_report.html" \
        --self-contained-html \
        --junit-xml="$TEST_RESULTS_DIR/${suite_lower}_junit.xml" \
        --cov-report=html:"$TEST_RESULTS_DIR/${suite_lower}_coverage" \
        | tee "$TEST_RESULTS_DIR/${suite_lower}_output.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $suite_name PASSED"
    else
        echo "❌ $suite_name FAILED (exit code: $exit_code)"
    fi
    echo ""
    
    return $exit_code
}

# Track overall results
overall_result=0
perf_result=0
system_result=0

# 1. Unit Tests - Fast execution
run_test_suite "Unit Tests" "tests/unit/" ""
unit_result=$?
if [ $unit_result -ne 0 ]; then overall_result=1; fi

# 2. Integration Tests
run_test_suite "Integration Tests" "tests/integration/" ""
integration_result=$?
if [ $integration_result -ne 0 ]; then overall_result=1; fi

# 3. Performance Tests (if requested)
if [ "$1" == "--performance" ] || [ "$1" == "--all" ]; then
    echo "🏃‍♂️ Running Performance Tests..."
    echo "This may take several minutes..."
    run_test_suite "Performance Tests" "tests/performance/" "-m performance"
    perf_result=$?
    if [ $perf_result -ne 0 ]; then overall_result=1; fi
fi

# 4. Edge Cases and Error Handling
run_test_suite "Edge Cases" "tests/" "-m edge_cases"
edge_result=$?
if [ $edge_result -ne 0 ]; then overall_result=1; fi

# 5. Full System Integration Test (if all basic tests pass)
if [ $overall_result -eq 0 ]; then
    echo "🔄 Running Full System Integration Test..."
    echo "----------------------------------------"
    
    # Test the actual enhanced multi-strategy system
    timeout 30 python engine/enhanced_multi_strategy.py --date 2026-02-09 > "$TEST_RESULTS_DIR/system_test_output.log" 2>&1
    system_result=$?
    
    if [ $system_result -eq 0 ]; then
        echo "✅ Full System Integration PASSED"
    else
        echo "❌ Full System Integration FAILED (exit code: $system_result)"
        overall_result=1
    fi
    echo ""
fi

# Generate comprehensive report
echo "📊 Generating Test Summary Report..."
cat > "$TEST_RESULTS_DIR/test_summary.md" << EOF
# SPX AI Trading System - Test Results Summary

**Test Execution Date:** $(date)
**Test Results Directory:** $TEST_RESULTS_DIR

## Test Suite Results

| Test Suite | Status | Report |
|------------|--------|---------|
| Unit Tests | $([ $unit_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") | [HTML Report](unit_tests_report.html) |
| Integration Tests | $([ $integration_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") | [HTML Report](integration_tests_report.html) |
| Edge Cases | $([ $edge_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") | [HTML Report](edge_cases_report.html) |
| System Integration | $([ $system_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") | [Output Log](system_test_output.log) |

## Coverage Reports

- [Unit Test Coverage](unit_tests_coverage/index.html)
- [Integration Test Coverage](integration_tests_coverage/index.html)

## Individual Test Outputs

- [Unit Test Output](unit_tests_output.log)
- [Integration Test Output](integration_tests_output.log)
- [Edge Cases Output](edge_cases_output.log)

## Overall Result

**$([ $overall_result -eq 0 ] && echo "🎉 ALL TESTS PASSED" || echo "💥 SOME TESTS FAILED")**

$([ $overall_result -ne 0 ] && echo "
⚠️  **Action Required:** Review failed tests and fix issues before deployment.
" || echo "
✨ **Ready for Production:** All tests passed successfully!
")
EOF

# Final summary
echo "=================================================="
echo "📋 TEST EXECUTION COMPLETE"
echo "=================================================="
echo ""
echo "📁 Results Location: $TEST_RESULTS_DIR"
echo "📄 Summary Report: $TEST_RESULTS_DIR/test_summary.md"
echo ""

if [ $overall_result -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED! System is ready for production."
    echo ""
    echo "Next steps:"
    echo "  • Review coverage reports for any gaps"
    echo "  • Run performance tests if not already done: $0 --performance"
    echo "  • Deploy with confidence!"
else
    echo "💥 SOME TESTS FAILED! Please review and fix issues."
    echo ""
    echo "Next steps:"
    echo "  • Check individual test reports in $TEST_RESULTS_DIR"
    echo "  • Fix failing tests"
    echo "  • Re-run test suite"
    echo ""
    echo "Quick test command: pytest tests/ -v --tb=short"
fi

echo ""
echo "🔍 Quick Commands:"
echo "  • Run specific tests: pytest tests/unit/test_technical_analysis.py -v"
echo "  • Run with coverage: pytest tests/ --cov=src --cov-report=html"
echo "  • Run performance tests: pytest tests/performance/ -m performance"

exit $overall_result