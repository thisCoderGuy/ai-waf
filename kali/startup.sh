#!/bin/bash

# Source the virtual environment to ensure all commands run within it
. /opt/venv/bin/activate

# Explicitly set PYTHONPATH for this shell session
export PYTHONPATH="/app"

# Define the delay in seconds (useful if services need time to start)
DELAY_SECONDS=6

# Store the first argument to decide behavior
COMMAND_MODE="$1"

# Shift the arguments so that "$@" later refers to arguments *after* the mode
shift

if [ "$COMMAND_MODE" = "shell" ]; then
    exec /bin/bash
elif [ "$COMMAND_MODE" = "locust-then-shell" ]; then
    echo "Waiting for $DELAY_SECONDS seconds before starting Locust..."
    sleep $DELAY_SECONDS
    locust -f /app/locustfile.py \
           --host=http://coraza-proxy:8080 \
           --users 100 --spawn-rate 10 \
           --run-time 1m \
           --headless \
           --html /app/reports/traffic_report.html \
           --csv /app/reports/traffic_stats.csv "$@" # "$@" now contains only the remaining args
    echo "Locust finished. Starting interactive shell..."
    exec /bin/bash
else # Default mode: run locust directly
    echo "Waiting for $DELAY_SECONDS seconds before starting Locust..."
    sleep $DELAY_SECONDS
    locust -f /app/locustfile.py \
           --host=http://coraza-proxy:8080 \
           --users 100 --spawn-rate 10 \
           --run-time 1m \
           --headless \
           --html /app/reports/traffic_report.html \
           --csv /app/reports/traffic_stats.csv "$@" # "$@" now contains only the remaining args
fi