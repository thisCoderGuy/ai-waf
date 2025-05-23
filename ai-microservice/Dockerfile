FROM python:3.9-slim-buster


WORKDIR /app


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py ./


EXPOSE 5000


# Install Wazuh agent
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        net-tools \
        procps \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*


# Add Wazuh GPG key and repository
RUN curl -sS https://packages.wazuh.com/key/GPG-KEY-WAZUH | gpg --dearmor > /usr/share/keyrings/wazuh-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/wazuh-archive-keyring.gpg] https://packages.wazuh.com/4.x/apt/ stable main" | tee /etc/apt/sources.list.d/wazuh.list


# Install Wazuh agent
RUN apt-get update && \
    apt-get install -y wazuh-agent


# Copy custom Wazuh agent configuration
COPY wazuh_agent_config/ossec.conf /var/ossec/etc/ossec.conf


# Start Wazuh agent first, then the Python app
CMD ["/bin/bash", "-c", "/var/ossec/bin/wazuh-agentd -d && python app.py"]
