#!/bin/bash

# Ensure the Wazuh manager IP is set via environment variable
if [ -z "$WAZUH_MANAGER" ]; then
    echo "Error: WAZUH_MANAGER environment variable not set."
    exit 1
fi

# Configure and enroll Wazuh agent
echo "Configuring Wazuh agent to connect to manager: ${WAZUH_MANAGER}"
sed -i "s@<manager>.*</manager>@<manager><address>${WAZUH_MANAGER}</address></manager>@" /var/ossec/etc/ossec.conf
/var/ossec/bin/agent-auth -m "${WAZUH_MANAGER}" || { echo "Wazuh agent enrollment failed!"; }

# Start Wazuh agent in the background
echo "Starting Wazuh agent..."
/var/ossec/bin/wazuh-agent || { echo "Wazuh agent failed to start!"; }

# Start the original Juice Shop application
# This assumes the default command/entrypoint of bkimminich/juice-shop
# If the original image uses 'npm start', this should be 'npm start' or similar.
# Check the original image's Dockerfile for its CMD/ENTRYPOINT.
# For Juice Shop, it's typically 'npm start'.
echo "Starting Juice Shop application..."
npm start
