# Use the official Kali Linux rolling release as the base image
FROM kalilinux/kali-rolling

# Update package lists
RUN apt update

# Install necessary tools:
# - locust: For traffic generation
# - sqlmap: As previously requested
# - python3-pip: To manage Python packages, though locust is installed via apt here
# - python3-venv: Good practice for isolated Python environments, though not strictly needed for this direct run
RUN apt -y install locust sqlmap python3-pip

# Create a directory for your scripts
RUN mkdir -p /app/scripts

# Create a directory for Locust reports
RUN mkdir -p /app/reports

# Copy the traffic generation script into the container
# IMPORTANT: Corrected the destination path from /var/ossec/etc/ossec.conf
# to a more appropriate location like /app/scripts/
COPY scripts/generate_traffic.py /app/scripts/generate_traffic.py

# Set the working directory to where your script is located
WORKDIR /app/scripts

# Command to run the Locust traffic generation script in headless mode and save reports.
# -f generate_traffic.py: Specifies your Locust test file.
# --host=http://coraza-proxy-waf:8080: Sets the target host for Locust.
# --users 100: Total number of simulated users.
# --spawn-rate 10: Rate at which users are spawned per second.
# --run-time 10m: Run the test for a specific duration (e.g., 10 minutes).
#                 Remove this if you want it to run indefinitely until manually stopped.
# --headless: Runs Locust without the web UI.
# --html /app/reports/traffic_report.html: Generates a full HTML report.
# --csv /app/reports/traffic_stats: Generates CSV stats (will create traffic_stats_stats.csv and traffic_stats_failures.csv).
CMD ["locust", "-f", "generate_traffic.py", \
     "--host=http://coraza-proxy-waf:8080", \
     "--users", "100", "--spawn-rate", "10", \
     "--run-time", "10m", \
     "--headless", \
     "--html", "/app/reports/traffic_report.html", \
     "--csv", "/app/reports/traffic_stats"]


# You might want to expose the Locust web UI port (default 8089) if you want to access it from your host
# Expose the Locust web UI port (default 8089) - only needed if not using --headless
EXPOSE 8089