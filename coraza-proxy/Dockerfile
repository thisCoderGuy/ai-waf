# Stage 1: Build the Go application
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Copy the go.mod and go.sum files to leverage Docker's build cache
# This ensures that dependencies are only re-downloaded if go.mod/go.sum change
COPY go.mod .
COPY go.sum .


# Download dependencies based on existing go.mod
# Remove the go mod init line as it overwrites your existing module
RUN go mod download && go mod verify

# Copy the rest of your application source code
COPY . .


# Build the Go application
# CGO_ENABLED=0 creates a statically linked binary, which is ideal for a small final image.
# Output the compiled binary to /coraza-proxy in this build stage.
RUN CGO_ENABLED=0 GOOS=linux go build -o /coraza-proxy .


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




# Copy the built Go application from the builder stage
COPY --from=builder /coraza-proxy /usr/local/bin/coraza-proxy


# Copy Coraza configuration
COPY coraza.conf /etc/coraza/coraza.conf



# Expose the proxy port
EXPOSE 8080


# Command to run the Coraza proxy
CMD ["/bin/bash", "-c", "/usr/local/bin/coraza-proxy"]
