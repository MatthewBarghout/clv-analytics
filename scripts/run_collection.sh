#!/bin/bash

# NBA Odds Collection Runner
# Wraps the Python collection script with logging

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create logs directory if it doesn't exist
LOGS_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOGS_DIR"

# Generate timestamped log filename
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")
LOG_FILE="$LOGS_DIR/collection_$TIMESTAMP.log"

# Change to project directory
cd "$PROJECT_DIR"

# Run collection with all arguments passed to this script
echo "Starting NBA odds collection at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Arguments: $@" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Run the Python script and capture output
# Use python3 directly if poetry not available
if command -v poetry &> /dev/null; then
    poetry run python -m scripts.collect_odds "$@" 2>&1 | tee -a "$LOG_FILE"
else
    echo "Poetry not found, using python3 directly" | tee -a "$LOG_FILE"
    python3 -m scripts.collect_odds "$@" 2>&1 | tee -a "$LOG_FILE"
fi

EXIT_CODE=${PIPESTATUS[0]}

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Collection finished at $(date) with exit code $EXIT_CODE" | tee -a "$LOG_FILE"

exit $EXIT_CODE
