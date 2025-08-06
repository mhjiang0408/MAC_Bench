#!/bin/bash
# Test script for MAC_Bench CLI functionality

echo "🧪 Testing MAC_Bench CLI Installation and Basic Functionality"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_TOTAL=0

test_command() {
    local description="$1"
    local command="$2"
    local expected_success="$3"  # 0 for success expected, 1 for failure expected
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -n "Testing: $description ... "
    
    if eval "$command" >/dev/null 2>&1; then
        if [ "$expected_success" -eq 0 ]; then
            echo -e "${GREEN}PASS${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}FAIL${NC} (expected failure but command succeeded)"
        fi
    else
        if [ "$expected_success" -eq 1 ]; then
            echo -e "${GREEN}PASS${NC} (expected failure)"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}FAIL${NC}"
        fi
    fi
}

# Change to script directory
cd "$(dirname "$0")"

echo -e "${YELLOW}Phase 1: Basic CLI Functionality${NC}"
echo "--------------------------------"

# Test if CLI script exists and is executable
test_command "CLI script exists and executable" "test -x ./mac" 0

# Test help command
test_command "Help command works" "./mac --help" 0

# Test info command
test_command "Info command works" "./mac info" 0

echo -e "\n${YELLOW}Phase 2: Command Availability${NC}"
echo "-----------------------------"

# Test all main commands are available
test_command "Run command available" "./mac run --help" 0
test_command "Analyze command available" "./mac analyze --help" 0
test_command "Config command available" "./mac config --help" 0
test_command "Status command available" "./mac status --help" 0

echo -e "\n${YELLOW}Phase 3: Status and Environment Checks${NC}"
echo "--------------------------------------"

# Test status command
test_command "Status command basic functionality" "./mac status" 0

echo -e "\n${YELLOW}Phase 4: Configuration Management${NC}"
echo "--------------------------------"

# Test config template generation
test_command "Config template generation" "./mac config template --output test_template.yaml --type basic" 0

# Test config validation (should pass for generated template)
if [ -f "test_template.yaml" ]; then
    test_command "Config validation with generated template" "./mac config validate test_template.yaml" 0
    test_command "Config show command" "./mac config show test_template.yaml" 0
else
    echo -e "${YELLOW}Skipping config validation tests - template not generated${NC}"
fi

echo -e "\n${YELLOW}Phase 5: Error Handling${NC}"
echo "----------------------"

# Test error handling with invalid inputs
test_command "Invalid config file handling" "./mac config validate nonexistent_file.yaml" 1
test_command "Invalid analyze path handling" "./mac analyze nonexistent_path/" 1

echo -e "\n${YELLOW}Phase 6: Dry Run Tests${NC}"
echo "---------------------"

# Test dry run functionality (should not execute but should validate)
if [ -f "test_template.yaml" ]; then
    # First, we need to make the template more realistic for dry run
    # This is optional since the generated template might not have valid API keys
    echo "Skipping dry run test - requires valid configuration with API keys"
else
    echo "Skipping dry run test - no test configuration available"
fi

echo -e "\n${YELLOW}Phase 7: Cleanup${NC}"
echo "--------------"

# Clean up test files
if [ -f "test_template.yaml" ]; then
    rm test_template.yaml
    echo "Cleaned up test files"
fi

echo
echo "============================================================"
echo -e "🧪 Test Results: ${GREEN}$TESTS_PASSED${NC}/${TESTS_TOTAL} tests passed"

if [ $TESTS_PASSED -eq $TESTS_TOTAL ]; then
    echo -e "${GREEN}✅ All tests passed! CLI is working correctly.${NC}"
    echo
    echo "🚀 Next steps:"
    echo "1. Install CLI dependencies: pip install -r requirements-cli.txt"
    echo "2. Create your configuration: ./mac config template --output my_config.yaml"
    echo "3. Edit my_config.yaml with your API keys and settings"
    echo "4. Validate your setup: ./mac status --detailed"
    echo "5. Run your first experiment: ./mac run --config my_config.yaml --dry-run"
    echo
    echo "📖 For detailed instructions, see CLI_USER_GUIDE.md"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the CLI installation.${NC}"
    echo
    echo "🔧 Troubleshooting:"
    echo "1. Make sure you're in the MAC_Bench project directory"
    echo "2. Check that all Python dependencies are installed"
    echo "3. Verify the ./mac script has execute permissions: chmod +x mac"
    echo "4. Try running: python -m mac_cli.cli --help"
    exit 1
fi