# Use the official Kali Linux rolling release as the base image
FROM kalilinux/kali-rolling

# Update package lists
RUN apt update

# Install necessary system tools:
RUN apt -y install sqlmap python3-pip python3-venv

# Create a Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Locust into the virtual environment
RUN pip install locust

# Create directories for your test files and reports
RUN mkdir -p /app/locust_tests
RUN mkdir -p /app/reports

# Copy your 'locust_tests' package into the container
COPY locust_tests /app/locust_tests

# Copy locustfile.py to the /app/ directory (project root in container)
# This assumes you've moved locustfile.py to your local project root
COPY locustfile.py /app/locustfile.py

# *** NEW: Copy the external startup script and make it executable ***
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Set the working directory for the container
WORKDIR /app/

# PYTHONPATH is still needed for 'locust_tests' package to be found by absolute imports
ENV PYTHONPATH=/app

# Set the default command to run your startup script
CMD ["/app/startup.sh"]

# Expose the Locust web UI port (default 8089) - only needed if not using --headless
EXPOSE 8089