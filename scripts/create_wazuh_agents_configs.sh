#!/bin/bash


echo "Creating dummy Wazuh agent config placeholders..."
mkdir -p wazuh-agent-coraza-proxy
touch wazuh-agent-coraza-proxy/ossec.conf
mkdir -p wazuh-agent-ai-microservice
touch wazuh-agent-ai-microservice/ossec.conf


echo "Done."
