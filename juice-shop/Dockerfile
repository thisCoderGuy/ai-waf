# Start from the official Juice Shop image
FROM bkimminich/juice-shop:latest


# Set environment variables for non-interactive apt commands
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        net-tools \
        procps \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*



RUN curl -sSL -o /tmp/wazuh-agent.deb https://packages.wazuh.com/4.x/apt/pool/main/w/wazuh-agent/wazuh-agent_4.8.0-1_amd64.deb && \
    # Install the downloaded .deb package
    dpkg -i /tmp/wazuh-agent.deb && \
    # Resolve any missing dependencies that dpkg might have left (usually handled by apt-get -f install)
    apt-get update && \
    apt-get install -fy && \
    # Clean up the downloaded .deb file
    rm /tmp/wazuh-agent.deb


# Set the Wazuh Manager IP/hostname. This will be provided by Docker Compose.
# The WAZUH_MANAGER environment variable is read by the agent upon startup/enrollment.
# We'll set a placeholder here, and it will be overridden by docker-compose.yml.
ENV WAZUH_MANAGER="wazuh.manager"

# Configure the agent: set its name (optional, but good for identification)
# And make sure it's enabled to run.
RUN sed -i "s/^#agent.name=.*/agent.name=juice-shop/" /var/ossec/etc/ossec.conf && \
    sed -i "s/^enabled=.*/enabled=yes/" /var/ossec/etc/ossec.conf

# Enroll the agent with the manager using agent-auth.
# This assumes the Wazuh Manager is reachable at the hostname defined in WAZUH_MANAGER
# and is listening on port 1515/tcp for enrollment.
# The 'nohup ... &' is to ensure the Juice Shop app starts in the foreground.
# The agent needs to be started as part of the container's lifecycle.
# We'll ensure the agent starts in the background and the main Juice Shop process
# stays in the foreground.
RUN /var/ossec/bin/agent-auth -m ${WAZUH_MANAGER} && \
    /var/ossec/bin/wazuh-agent start

# The original Juice Shop image already has its entrypoint/CMD to start the application.
# We need to make sure the Wazuh agent starts in the background and the Juice Shop
# application remains the foreground process.
# This is typically handled by modifying the ENTRYPOINT or CMD to run both.
# Let's adjust the entrypoint to start the agent and then the original Juice Shop app.
#COPY --from=bkimminich/juice-shop:latest /usr/src/app/package.json /usr/src/app/
#COPY --from=bkimminich/juice-shop:latest /usr/src/app/app.js /usr/src/app/
#COPY --from=bkimminich/juice-shop:latest /usr/src/app/server.js /usr/src/app/
# ... You would need to copy all relevant Juice Shop app files here if you want to rebuild its context.
# A simpler way is to wrap its existing CMD/ENTRYPOINT.

# Alternative: Create an entrypoint script to run both
COPY start_juice_shop_with_wazuh.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start_juice_shop_with_wazuh.sh

ENTRYPOINT ["/usr/local/bin/start_juice_shop_with_wazuh.sh"]

# Expose Juice Shop default port
EXPOSE 3000
