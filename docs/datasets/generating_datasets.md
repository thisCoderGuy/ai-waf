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
  * `LOCUST_RUN_TIME`: Determines the maximum duration for which traffic will be generated (e.g., 10m for 10 minutes).

```bash
# Example snippet from ./kali/startup.sh  
DELAY_SECONDS="${DELAY_SECONDS:-10}"
# ...  
LOCUST_RUN_TIME="${LOCUST_RUN_TIME:-5h}" # Maximum duration of traffic generation
```

### **Type of Traffic Generated**

Define the types and durations of user behavior and attacks you want to simulate.

* **File:** `./kali/locust-tests/config.py`  
* **Parameters to change:**  
  * `PHASE_LENGTHS_SECONDS`: A dictionary where keys are user/attacker types and values are their relative durations (in seconds). 

```python
# Example snippet from ./kali/locust_tests/config.py  
PHASE_LENGTHS_SECONDS = { # in seconds
    "LegitimateUser":  1200,  
    "SQLiAttacker": 300,  
    "XSSAttacker": 200,  
    "DirectoryTraversalAttacker": 100,  
    "EnumerationAttacker": 100,  
    "CSRFAttacker": 100,  
}
```

### **Log Format and Labels**

Configure how the coraza-proxy service logs the traffic, including the format and the labels used for your dataset.

* **File:** `./coraza-proxy/config.go`  
* **Parameters to change:**  
  * `loggerFormat`: Set to "csv" or "json" for the desired log file format.  
  * `logFileName`: Specify the **filename for the output log file** within the container (e.g., "coraza-dataset.csv"). Remember this path is mapped to `./training/training-data/raw` on your host.  In addition, a timestamp will be added to the provided filename to ensure uniqueness and prevent overwriting of previous log files.
  * `DefaultAIVerdictLabel`: A default label to categorize the overall verdict (e.g., "UNKNOWN").  
  * `DefaultAIVulnerabilityTypeLabel`: A default label to specify the type of vulnerability or attack (e.g., "UNKNOWN_VULNERABILITY").

```go
// Example snippet from ./coraza-proxy/config.go  
const loggerFormat = "csv"  
const (
	logBaseDir = "/var/log/coraza/"

	// A timestamp will be added to the provided filename
	logFileName = "coraza-dataset.csv"

	loggerPath = logBaseDir + logFileName // e.g., "/var/log/coraza/coraza-audit-enum.csv"
)  
const (  
    DefaultAIVerdictLabel         = "UNKNOWN" // benign or malicious  
    DefaultAIVulnerabilityTypeLabel = "UNKNOWN_VULNERABILITY"  // none, sqli, xss, etc.  
)
```

## **Running the Testbed for Dataset Generation**

Once you have configured the above parameters, use the following Docker Compose command to start the necessary services for dataset generation:  

`docker compose --profile app-core --profile dataset-gen up --build -d`

This command will:

* Build (if necessary) and start the coraza-proxy, ai-microservice, kali, and juice-shop containers.  
* The kali container will execute its `startup.sh script,` which initiates the traffic generation based on your `locust_tests/config.py` settings.  
* The coraza-proxy will log the traffic to the specified loggerPath, which is mounted to `./training/training-data/raw` on your host machine. This is where your generated dataset will be stored.

You can access the Kali Linux container's terminal to initiate traffic generation or perform manual attacks:  

`docker exec -it kali-locust /bin/bash`

You can review logs from the containers, e.g.:  

`docker compose logs kali`

`docker compose logs coraza-proxy`

## **Stopping the Dataset Generation**

After your desired data generation duration, you can stop the running services to collect your datasets:  

`docker compose --profile app-core --profile dataset-gen down`

This command will stop and remove the app-core services, leaving your generated data in the `./training/training-data/raw` directory on your host, ready for the **Model Training Mode**.