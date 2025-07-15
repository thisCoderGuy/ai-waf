# **Live Evaluation Mode using the Testbed**

This section guides you through setting up and running your testbed in **Live Evaluation Mode**. In this mode, the primary objective is to demonstrate and evaluate the real-time performance of your trained AI model when integrated into the security pipeline. This involves live traffic, AI-based classification by the ai-microservice, and monitoring through the Wazuh security stack.  

*Note: In this mode, all core application components and the entire Wazuh stack are started to simulate a complete security pipeline.*

## **Table of Contents**

* [Configuration Steps](#configuration-steps)  
* [Running the Testbed for Live Evaluation](#running-the-testbed-for-live-evaluation)  
* [Accessing Components for Evaluation](#accessing-components-for-evaluation)  
* [Stopping the Live Evaluation](#stopping-the-live-evaluation)

## **Configuration Steps**

For Live Evaluation Mode, ensure your AI model is trained and accessible by the ai-microservice. The ai-microservice is configured to load the model from the mounted volume.

* **Trained Model Path:** The ai-microservice expects the trained model to be at `/app/model/ai_waf_model.pkl` inside its container, which maps to `./training/trained_models` on your host. Ensure your training process has placed a valid model file there.  
* **AI Microservice URL:** The coraza-proxy is configured to send traffic to the ai-microservice for classification. This is set via the `AI_MICROSERVICE_URL` environment variable in `docker-compose.yml`.  
* Coraza Proxy Logging (for observation, not dataset generation):  
  While this mode focuses on evaluation, the coraza-proxy still logs traffic. You can adjust its logging behavior in `./coraza-proxy/config.go`. The `DefaultAIVerdictLabel` and `DefaultAIVulnerabilityTypeLabel` are set during dataset generation, but in live evaluation, the AI microservice's classification will determine the actual verdict/type. These labels in `config.go` would primarily be for demonstration purposes if you want to hardcode a default for unclassified traffic or for initial setup.  

  ```go
  // Example snippet from ./coraza-proxy/config.go  
  // These values are primarily for dataset generation.  
  // In Live Evaluation, the AI Microservice's verdict will override the default.  
  const (  
      DefaultAIVerdictLabel         = "benign" // benign or malicious  
      DefaultAIVulnerabilityTypeLabel = "none"  // none, sqli, xss, etc.  
  )
  ```

## **Running the Testbed for Live Evaluation**

To start all necessary services for Live Evaluation Mode, use the following Docker Compose command:  

`docker compose --profile app-core --profile live-eval up --build -d`

This command will:

* Build (if necessary) and start the coraza-proxy, ai-microservice, kali, and juice-shop containers (part of app-core).  
* Build (if necessary) and start the wazuh.manager, wazuh.indexer, and wazuh.dashboard containers (part of live-eval).  
* The kali container will execute its startup.sh script, which can be configured to initiate various traffic patterns, including attacks.  
* The coraza-proxy will intercept traffic to juice-shop, send it to the ai-microservice for real-time classification, and then forward it.  
* Wazuh agents (if configured in coraza-proxy, ai-microservice, juice-shop) will send security telemetry to the wazuh.manager, which then forwards it to the wazuh.indexer for storage and wazuh.dashboard for visualization.

## **Accessing Components for Evaluation**

Once the services are running, you can access various components to observe and interact with the testbed:

* Wazuh Dashboard:  
  Access the Wazuh Dashboard in your web browser to monitor security events, alerts, and the AI model's verdicts in real-time.  
  * **URL:** https://localhost:443  
  * **Credentials:** Refer to your docker-compose.yml for DASHBOARD_USERNAME and DASHBOARD_PASSWORD (default admin/SecretPassword).  
* Coraza Proxy (Web Application Firewall):  
  All traffic to the juice-shop application passes through the Coraza Proxy. You can direct your browser or attack tools to this address.  
  * **URL:** http://localhost:8080  
* Kali Linux Terminal:  
  Access the Kali container's terminal to manually launch attacks, run penetration testing tools, or execute automated traffic generation scripts (e.g., Locust).  

  `docker exec -it kali /bin/bash`

## **Stopping the Live Evaluation**

To stop all services running in Live Evaluation Mode:  

`docker compose --profile app-core --profile live-eval down`

This command will gracefully stop and remove all containers, networks, and volumes associated with both the app-core and live-eval profiles.