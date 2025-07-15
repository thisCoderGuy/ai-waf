# **Testbed Architecture Overview**

The AI WAF project utilizes a robust and reproducible **Containerized Web Application Security Analytics Testbed**, built entirely with **Docker Compose**. This multi-component infrastructure is designed to simulate real-world web application interactions, including both legitimate user traffic and various attack vectors, while meticulously collecting security telemetry across its distinct operational modes.

## **Purpose of the Testbed**

The testbed serves as a controlled environment for:

* **Dataset Generation:** Creating high-quality, labeled datasets of web traffic (both benign and malicious) for training machine learning models.  
* **Model Training:** Providing a dedicated environment for developing, training, and optimizing AI models using the generated datasets.  
* **Live Evaluation:** Demonstrating and evaluating the real-time performance of the trained AI model when integrated into a full security pipeline, including a WAF and a Security Information and Event Management (SIEM) system.

## **Core Components**

The testbed is composed of several Docker services, each serving a specific role in the security pipeline. We extend our gratitude to the open-source communities behind these powerful tools:

* **Wazuh Stack** ([Wazuh](https://wazuh.com/) \- Manager, Indexer, Dashboard):  
  * A comprehensive open-source security platform.  
  * **Manager:** Collects and analyzes security data from agents, performing threat detection and alerting.  
  * **Indexer:** Stores and indexes the security events, making them searchable.  
  * **Dashboard:** Provides a web-based interface for visualizing alerts, managing agents, and exploring security data.  
  * **Role:** Primarily used in **Live Evaluation Mode** for real-time monitoring and analysis of WAF and AI microservice outputs.  
* **Coraza Proxy** ([Coraza WAF](https://coraza.io/) \- ModSecurity-compatible rules):  
  * An open-source Web Application Firewall (WAF) component.  
  * **Role:** Intercepts incoming web traffic, applies ModSecurity-compatible rules, and forwards requests to the target application. It can also send traffic to the ai-microservice for AI-based classification and logs all processed requests, which are crucial for dataset generation.  
* **AI Microservice** (Custom Flask-based application):  
  * A custom microservice built using the [Flask](https://flask.palletsprojects.com/) web framework.  
  * **Role:** Receives web request data from the coraza-proxy, performs real-time classification (e.g., benign/malicious verdict), and returns the result to the WAF. It loads the pre-trained model from a mounted volume.  
* **Juice Shop** ([OWASP Juice Shop](https://owasp.org/www-project-juice-shop/)):  
  * A deliberately insecure web application, part of the OWASP project.  
  * **Role:** Serves as the target application for both legitimate user interactions and simulated attacks, providing a realistic environment for the WAF to protect.  
* **Kali Linux** ([Kali Linux](https://www.kali.org/)):  
  * A popular open-source penetration testing distribution.  
  * **Role:** Used to generate various types of web traffic, including automated attacks (e.g., using [Locust](https://locust.io/)) and manual penetration testing, to simulate real-world scenarios for dataset generation and live evaluation.  
* **ML Trainer** (Leveraging various open-source ML libraries):  
  * A dedicated service for training machine learning models.  
  * **Role:** Consumes raw logs generated during dataset generation, preprocesses them, trains AI models, and saves the trained models for use by the ai-microservice.  
  * **Key Libraries/Frameworks:**  
    * [PyTorch](https://pytorch.org/)  
    * [TensorFlow](https://www.tensorflow.org/)  
    * [scikit-learn](https://scikit-learn.org/stable/)  
    * [Pandas](https://pandas.pydata.org/)  
    * [NumPy](https://numpy.org/)  
    * [Matplotlib](https://matplotlib.org/)  
    * [Seaborn](https://seaborn.pydata.org/)  
    * [Hugging Face Transformers](https://huggingface.co/docs/transformers/)  
    * [Hugging Face Datasets](https://huggingface.co/docs/datasets/)  
    * [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/)  
    * [Joblib](https://joblib.readthedocs.io/en/latest/)  
* **Docker** ([Docker Engine](https://www.google.com/search?q=https://www.docker.com/products/docker-engine/), [Docker Compose](https://docs.docker.com/compose/)):  
  * The foundational open-source platform for containerization and orchestrating multi-container applications.

## **Networking**

The testbed employs two distinct Docker networks to manage communication between services:

* **app-net:**  
  * Connects the core application components: coraza-proxy, ai-microservice, juice-shop, and kali.  
  * Facilitates internal communication (e.g., coraza-proxy talking to ai-microservice, kali attacking coraza-proxy).  
* **wazuh-net:**  
  * Connects the Wazuh stack components (wazuh.manager, wazuh.indexer, wazuh.dashboard) and enables Wazuh agents (running within coraza-proxy, ai-microservice, juice-shop) to communicate with the Wazuh Manager.  
  * Ensures security telemetry can be collected and sent to the SIEM.

## **Persistent Volumes**

Docker volumes are used to ensure data persistence and facilitate data sharing between services and the host machine:

* **Wazuh Volumes:** Multiple volumes (wazuh\_api\_configuration, wazuh\_etc, wazuh\_logs, etc.) are used to persist Wazuh's configuration, logs, and data, allowing the Wazuh stack to retain state across restarts.  
* **training/training-data/raw:** This host-mounted volume (/data/input in ml\_trainer, /var/log/coraza in coraza-proxy) is crucial for storing the raw web traffic logs generated by coraza-proxy during dataset generation.  
* **training/trained\_models:** This host-mounted volume (/models/output in ml\_trainer, /app/model in ai-microservice) serves as the central location for saving trained AI models from the ml\_trainer and loading them into the ai-microservice.  
* **kali/locust-reports:** Persists reports generated by Locust (traffic generation tool) from the kali container.

## **Operational Modes Integration**

The testbed's flexibility is achieved through Docker Compose profiles, which allow selective activation of services for different operational modes:

* **app-core profile:** Includes coraza-proxy, ai-microservice, kali, and juice-shop. These services are active during both **Dataset Generation Mode** (to produce traffic and logs) and **Live Evaluation Mode** (as the core application and WAF).  
* **live-eval profile:** Includes the entire Wazuh stack (wazuh.manager, wazuh.indexer, wazuh.dashboard). This profile is activated *in conjunction with app-core* for **Live Evaluation Mode** to enable real-time monitoring.  
* **train profile:** Includes only the ml\_trainer service. This profile is activated independently for **Model Training Mode**, ensuring that resource-intensive training tasks run in isolation.

## **Benefits of this Containerized Approach**

* **Reproducibility:** Docker ensures consistent environments across different machines, making research and development highly reproducible.  
* **Isolation:** Each service runs in its own isolated container, preventing conflicts and simplifying dependency management.  
* **Scalability:** While not explicitly configured for horizontal scaling, the modular design allows for future expansion.  
* **Ease of Setup:** Docker Compose simplifies the deployment and management of the multi-component architecture with single commands.  
* **Resource Management:** Profiles enable efficient resource utilization by only starting the necessary services for a given operational mode.

This robust testbed architecture provides a powerful and flexible platform for developing, evaluating, and demonstrating your Hybrid Traditional-AI WAF solution.