FROM python:3.9


WORKDIR /app


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY *.py ./
COPY trained-models/*.joblib ./trained-models/

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