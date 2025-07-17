FROM python:3.9


WORKDIR /app


COPY ai-microservice/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy application's main files (Flask)---
COPY ai-microservice/*.py ./
# --- Copy the exported trained models  ---
COPY ai-microservice/trained-models/*.joblib ./trained-models/
COPY ai-microservice/trained-models/*.txt ./trained-models/
# --- Copy custom training code ---
# This is much cleaner than copying individual files.
# Make sure the source path '. /ml_pipeline_src' is correct relative to your build context (usually where docker-compose.yml is).
# The destination path '/app/ml_pipeline_src' will put all these files into a new folder named 'ml_pipeline_src' inside the container.
RUN mkdir -p /app/ml_pipeline_src
COPY training/*.py ./ml_pipeline_src

RUN mkdir -p /app/training/model-logs
# Create the empty all_models.txt file for logging
RUN touch /app/training/model-logs/all_models.txt

EXPOSE 5000


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        net-tools \
        procps \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*




# Start Wazuh agent first, then the Python app
CMD ["/bin/bash", "-c", "python app.py"]