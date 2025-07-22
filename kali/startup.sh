#!/bin/bash

# Source the virtual environment to ensure all commands run within it
. /opt/venv/bin/activate

# Explicitly set PYTHONPATH for this shell session
export PYTHONPATH="/app"

# Delay before starting Locust/shell (in seconds)
DELAY_SECONDS="${DELAY_SECONDS:-10}"

# Path to the Locust file
LOCUST_FILE="${LOCUST_FILE:-/app/locustfile.py}"

# Host for Locust to target
LOCUST_HOST="${LOCUST_HOST:-http://coraza-proxy:8080}"

# Number of virtual users to simulate
LOCUST_USERS="${LOCUST_USERS:-1}"

# Spawn rate (users per second)
LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"

# Maximum run time (e.g., 3m for 3 minutes, 1h for 1 hour, 30s for 30 seconds)
LOCUST_RUN_TIME="${LOCUST_RUN_TIME:-5h}"

# Run Locust in headless mode (true/false). Default to true.
# Set to "false" or "0" to enable the web UI.
LOCUST_HEADLESS="${LOCUST_HEADLESS:-true}"

# Path for the HTML report output
LOCUST_HTML_REPORT="${LOCUST_HTML_REPORT:-/app/reports/traffic_report.html}"

# Path for the CSV stats output
LOCUST_CSV_REPORT="${LOCUST_CSV_REPORT:-/app/reports/traffic_stats.csv}"




# --- Construct the Locust command based on configurations ---
LOCUST_CMD_BASE="locust -f ${LOCUST_FILE} --host=${LOCUST_HOST} --users ${LOCUST_USERS} --spawn-rate ${LOCUST_SPAWN_RATE} --run-time ${LOCUST_RUN_TIME}"

# Add headless flag if enabled
if [ "${LOCUST_HEADLESS}" = "true" ] || [ "${LOCUST_HEADLESS}" = "1" ]; then
    LOCUST_CMD_BASE+=" --headless"
fi

# Add report paths
LOCUST_CMD_BASE+=" --html ${LOCUST_HTML_REPORT} --csv ${LOCUST_CSV_REPORT}"





# Store the first argument to decide behavior
COMMAND_MODE="$1"

# Shift the arguments so that "$@" later refers to arguments *after* the mode
shift

if [ "$COMMAND_MODE" = "shell" ]; then
    exec /bin/bash
elif [ "$COMMAND_MODE" = "locust-then-shell" ]; then
    echo "Waiting for $DELAY_SECONDS seconds before starting Locust..."
    sleep $DELAY_SECONDS
    echo "Executing Locust: ${LOCUST_CMD_BASE} $*" 
    eval "${LOCUST_CMD_BASE} $*" 
    echo "Locust finished. Starting interactive shell..."
    exec /bin/bash
else # Default mode: run locust directly
    echo "Waiting for $DELAY_SECONDS seconds before starting Locust..."
    sleep $DELAY_SECONDS
    echo "Executing Locust: ${LOCUST_CMD_BASE} $*" # Print the full command for debugging
    eval "${LOCUST_CMD_BASE} $*" # Use eval to correctly interpret the command string and append remaining args
fi