FROM python:3.9


WORKDIR /app


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py ./
COPY *.joblib ./

EXPOSE 5000


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


# Copy custom Wazuh agent configuration
COPY wazuh_agent_config/ossec.conf /var/ossec/etc/ossec.conf


# Start Wazuh agent first, then the Python app
CMD ["/bin/bash", "-c", "/var/ossec/bin/wazuh-agentd -d && python app.py"]