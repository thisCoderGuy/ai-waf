# Stage 1: Build the Go application
FROM golang:1.22 AS builder


WORKDIR /app


# Copy go mod and sum files
COPY go.mod go.sum ./
# Download all dependencies. Dependencies will be cached if the go.mod and go.sum files are unchanged
RUN go mod download


# Copy the rest of the application source code
COPY . .


# Build the Go application
RUN CGO_ENABLED=0 GOOS=linux go build -o /coraza-proxy ./main.go


# Stage 2: Create the final image with Wazuh agent
FROM debian:bookworm-slim


# Install necessary packages for Wazuh agent and curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        net-tools \
        procps \
        lsb-release \
        apt-transport-https \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# Add Wazuh GPG key and repository
RUN curl -sS https://packages.wazuh.com/key/GPG-KEY-WAZUH | gpg --dearmor > /usr/share/keyrings/wazuh-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/wazuh-archive-keyring.gpg] https://packages.wazuh.com/4.x/apt/ stable main" | tee /etc/apt/sources.list.d/wazuh.list


# Install Wazuh agent
RUN apt-get update && \
    apt-get install -y wazuh-agent


# Copy the built Go application from the builder stage
COPY --from=builder /coraza-proxy /usr/local/bin/coraza-proxy


# Copy Coraza configuration
COPY coraza.conf /etc/coraza/coraza.conf


# Copy custom Wazuh agent configuration
COPY wazuh_agent_config/ossec.conf /var/ossec/etc/ossec.conf


# Expose the proxy port
EXPOSE 8080


# Command to run the Wazuh agent and then the Coraza proxy
CMD ["/bin/bash", "-c", "/var/ossec/bin/wazuh-agentd -d && /usr/local/bin/coraza-proxy"]
