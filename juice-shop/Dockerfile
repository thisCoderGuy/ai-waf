FROM debian:bullseye-slim

# Environment
ENV DEBIAN_FRONTEND=noninteractive

# Install OS and Node.js dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        ca-certificates \
        git \
        procps \
        net-tools \
        lsb-release \
        wget \
        gnupg2 \
        python3 \
        make \
        build-essential \
        nodejs \
        npm && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js (if not installed via apt)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Create app directory
WORKDIR /app

# Clone Juice Shop
RUN git clone https://github.com/juice-shop/juice-shop.git . && \
    npm install --legacy-peer-deps

# Install Wazuh agent
RUN curl -sSL -o /tmp/wazuh-agent.deb https://packages.wazuh.com/4.x/apt/pool/main/w/wazuh-agent/wazuh-agent_4.8.0-1_amd64.deb && \
    dpkg -i /tmp/wazuh-agent.deb || apt-get install -fy && \
    rm /tmp/wazuh-agent.deb

# Copy custom Wazuh agent configuration
COPY wazuh_agent_config/ossec.conf /var/ossec/etc/ossec.conf


# Expose Juice Shop port
EXPOSE 3000

# Start Wazuh Agent and Juice Shop
CMD ["/bin/bash", "-c", "/var/ossec/bin/wazuh-agentd -d && npm start"]
