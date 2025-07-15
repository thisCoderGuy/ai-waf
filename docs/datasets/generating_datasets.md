# **Generating Datasets using the Testbed**

This section guides you through configuring and running your testbed in **Dataset Generation Mode**. In this mode, the focus is on creating high-quality, labeled datasets by simulating various web traffic patterns and meticulously logging the interactions.

*Note: In this mode, the Wazuh stack (Manager, Indexer, Dashboard) is **not** started, as it is primarily for live evaluation and monitoring.*

## **Table of Contents**

* [Configuration Steps](#configuration-steps)
  * [Traffic Generation Timing](#traffic-generation-timing)
  * [Type of Traffic Generated](#type-of-traffic-generated)
  * [Log Format and Labels](#log-format-and-labels)
* [Running the Testbed for Dataset Generation](#running-the-testbed-for-dataset-generation)
* [Stopping the Dataset Generation](#stopping-the-dataset-generation)


## **Configuration Steps**

Before running the testbed for dataset generation, you can customize the traffic and logging behavior:

### **Traffic Generation Timing**

Adjust the timing parameters for traffic generation within the kali container.

* **File:** `./kali/startup.sh`  
* **Parameters to change:**  
  * `DELAY_SECONDS`: Controls the delay before traffic generation begins.  
  * `--run-time`: Determines the duration for which traffic will be generated (e.g., 10m for 10 minutes).

```bash
# Example snippet from ./kali/startup.sh  
DELAY_SECONDS=60 # Delay before traffic generation starts  
# ...  
# Command to start traffic generation (e.g., Locust)  
locust --run-time 10m # Duration of traffic generation
```

### **Type of Traffic Generated**

Define the types of user behavior and attacks you want to simulate.

* **File:** `./kali/locust_tests/config.py`  
* **Parameters to change:**  
  * `USER_TASK_WEIGHTS`: A dictionary where keys are user/attacker types and values are their relative weights (integers only). Higher weights mean more frequent simulation of that behavior.

```python
# Example snippet from ./kali/locust_tests/config.py  
USER_TASK_WEIGHTS = { # ints only  
    "LegitimateUser": 12,  
    "SQLiAttacker": 0,  
    "XSSAttacker": 0,  
    "DirectoryTraversalAttacker": 0,  
    "EnumerationAttacker": 0,  
    "CSRFAttacker": 0,  
}
```

### **Log Format and Labels**

Configure how the coraza-proxy service logs the traffic, including the format and the labels used for your dataset.

* **File:** `./coraza-proxy/config.go`  
* **Parameters to change:**  
  * `loggerFormat`: Set to "csv" or "json" for the desired log file format.  
  * `loggerPath`: Specify the full path and filename for the output log file within the container (e.g., "/var/log/coraza/coraza-audit-benign.csv"). Remember this path is mapped to `./training/training-data/raw` on your host.  
  * `DefaultAIVerdictLabel`: A label to categorize the overall verdict (e.g., "benign", "malicious").  
  * `DefaultAIVulnerabilityTypeLabel`: A label to specify the type of vulnerability or attack (e.g., "none", "sqli", "xss").

```go
// Example snippet from ./coraza-proxy/config.go  
const loggerFormat \= "csv"  
const loggerPath \="/var/log/coraza/coraza-audit-benign.csv"  
// Default values for AI verdict and vulnerability type labels.  
// These can be modified here without touching the logger logic.  
const (  
    DefaultAIVerdictLabel         \= "benign" // benign or malicious  
    DefaultAIVulnerabilityTypeLabel \= "none"  // none, sqli, xss, etc.  
)
```

## **Running the Testbed for Dataset Generation**

Once you have configured the above parameters, use the following Docker Compose command to start the necessary services for dataset generation:  

`docker compose --profile app-core up --build -d`

This command will:

* Build (if necessary) and start the coraza-proxy, ai-microservice, kali, and juice-shop containers.  
* The kali container will execute its `startup.sh script,` which initiates the traffic generation based on your `locust_tests/config.py` settings.  
* The coraza-proxy will log the traffic to the specified loggerPath, which is mounted to `./training/training-data/raw` on your host machine. This is where your generated dataset will be stored.

You can access the Kali Linux container's terminal to initiate traffic generation or perform manual attacks:  

`docker exec -it kali /bin/bash`

## **Stopping the Dataset Generation**

After your desired data generation duration, you can stop the running services to collect your datasets:  

`docker compose --profile app-core down`

This command will stop and remove the app-core services, leaving your generated data in the `./training/training-data/raw` directory on your host, ready for the **Model Training Mode**.