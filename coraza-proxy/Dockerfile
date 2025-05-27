# Stage 1: Build the Go application
FROM golang:1.23-alpine AS builder


WORKDIR /app


# Copy the go.mod and go.sum files to leverage Docker's build cache
# This ensures that dependencies are only re-downloaded if go.mod/go.sum change
COPY go.mod ./
COPY go.sum ./

# Initialize go modules if not already done (optional, but good for fresh builds)
# If your go.mod already exists, this command will effectively do nothing.
RUN go mod init coraza-proxy || true

# Get the specific Coraza v3 module and other dependencies
# This command fetches the module and updates go.mod/go.sum
RUN go get github.com/corazawaf/coraza/v3
RUN go get github.com/corazawaf/coraza/v3/types
RUN go get rsc.io/sampler@v1.3.0

# Clean up unnecessary dependencies and ensure consistent module state
RUN go mod tidy

# Copy the rest of your application source code
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



RUN curl -sSL -o /tmp/wazuh-agent.deb https://packages.wazuh.com/4.x/apt/pool/main/w/wazuh-agent/wazuh-agent_4.8.0-1_amd64.deb && \
    # Install the downloaded .deb package
    dpkg -i /tmp/wazuh-agent.deb && \
    # Resolve any missing dependencies that dpkg might have left (usually handled by apt-get -f install)
    apt-get update && \
    apt-get install -fy && \
    # Clean up the downloaded .deb file
    rm /tmp/wazuh-agent.deb


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
