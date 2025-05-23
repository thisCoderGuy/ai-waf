#!/bin/bash


echo "Waiting for Wazuh manager API to be available..."
until curl -sS -k -u wazuh:wazuh -X GET https://wazuh-manager:55000 > /dev/null; do
  echo "Wazuh manager API not ready yet. Retrying in 5 seconds..."
  sleep 5
done
echo "Wazuh manager API is ready."


# Coraza Proxy Agent Enrollment
echo "Enrolling coraza-proxy agent..."
docker exec -it wazuh-manager bash -c "WAZUH_MANAGER_IP=\$(hostname -I | awk '{print \$1}') && \
  /var/ossec/bin/agent-auth -m \$WAZUH_MANAGER_IP -A coraza-proxy"


# AI Microservice Agent Enrollment
echo "Enrolling ai-microservice agent..."
docker exec -it wazuh-manager bash -c "WAZUH_MANAGER_IP=\$(hostname -I | awk '{print \$1}') && \
  /var/ossec/bin/agent-auth -m \$WAZUH_MANAGER_IP -A ai-microservice"


echo "Wazuh agent enrollment script finished."
