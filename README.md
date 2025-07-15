# **Hybrid Traditional-AI WAF**

## **Project Overview**

This research project focuses on developing and evaluating a **Hybrid Web Application Firewall (WAF)** that combines traditional rule-based detection with advanced **Artificial Intelligence (AI)** capabilities. The core objective is to enhance web application security by leveraging machine learning models to detect sophisticated and novel attack patterns that might bypass conventional WAF rules. 

This initiative is driven by the **critical lack of modern, high-quality, and labeled datasets** specifically tailored for training AI models in the domain of web application security, especially concerning the subtle evasion techniques (e.g., obfuscation, polymorphism) and context-dependent nature of diverse attack vectors that challenge traditional WAFs, often leading to false positives or missed detections. 

Our approach encompasses a full lifecycle: from generating meticulously labeled security datasets to training deep learning models and integrating them into a real-time defense system. This entire framework is built around a versatile **containerized testbed** that supports three distinct operational **phases**: **Dataset Generation**, **Model Training**, and **Live Evaluation**.

## **Operational Phases**

### **1\. Dataset Generation Phase**

This phase focuses on creating high-quality, labeled datasets for training machine learning models. It involves simulating various web traffic patterns and meticulously logging the interactions. To configure the generation of datasets, visit the [Generating Datasets](./docs/datasets/generating_datasets.md) page.

### **2\. Model Training Phase**

This phase focuses on the process of building and extending various machine learning models, ranging from **traditional machine learning algorithms** to **advanced deep learning architectures** like multi-input RNN or CNN classifiers, all utilizing the meticulously generated datasets. 

The project provides a **comprehensive framework** for training classification models. This framework supports experimentation and optimization across a spectrum of model complexities to achieve optimal threat detection. To learn more about the training framework, visit the [Training Framework](./docs/training/training_framework.md) page.

### **3\. Live Evaluation Phase**

This phase is dedicated to the **real-time evaluation and demonstration** of the trained AI model's performance when integrated into the security pipeline.

To learn more about the live evaluation mode, visit the [Live Evaluation Mode](./docs/live_evaluation/live_evaluation.md) page.

## **Testbed Architecture**

The project utilizes a robust and reproducible **Containerized Web Application Security Analytics Testbed**, built entirely with **Docker Compose**. This multi-component infrastructure is designed to simulate real-world web application interactions, including both legitimate user traffic and various attack vectors, while meticulously collecting security telemetry across three distinct operational modes. To learn more about the comprehensive test bed, visit the [Test Bed](./docs/testbed/test_bed.md) page.

## **Getting Started**

To set up and run this project, follow these steps:

### **Prerequisites**

* [Docker](https://docs.docker.com/get-docker/) (Docker Engine and Docker Compose)  
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### **Installation**

1. **Clone the repository:**  
   `git clone https://github.com/thisCoderGuy/ai-waf.git`

   `cd ai-waf`

2. **Build and start the Docker containers:**  
   `docker compose up \--build \-d`

   This command will build all necessary images and start the services in detached mode.

### **Accessing Components**

* **OWASP Juice Shop:** http://localhost:3000 (or your configured port)  
* **Wazuh Dashboard (Kibana/OpenSearch Dashboards):** http://localhost:5601 (default port, check docker-compose.yml for exact port)  
  * Default credentials: wazuh/wazuh (change immediately in production\!)  
* **Kali Linux Container:** You can exec into the Kali container to run tools manually or verify scripts.  
  `docker ps \# Find the container ID/name for Kali  `

  `docker exec \-it \<kali\_container\_id\_or\_name\> bash`

## **Usage**

This project supports three distinct operational phases, each with its own setup and execution steps:

### **1\. Running Dataset Generation**

Follow the instructions on the [Generating Datasets](./docs/datasets/generating_datasets.md) page to configure and execute the traffic generation and logging processes.

### **2\. Running Model Training**

Refer to the [Training Framework](./docs/training/training_framework.md) page for detailed guidance on preparing data, configuring models (traditional ML or deep learning), and initiating the training process.

### **3\. Running Live Evaluation**

Consult the [Live Evaluation Mode](./docs/live_evaluation/live_evaluation.md) page to learn how to deploy the AI Microservice, integrate it with Coraza, and observe real-time detection results through Wazuh.

## **Technologies Used**

* **Orchestration:** Docker, Docker Compose  
* **Web Application:** OWASP Juice Shop (Node.js/Express/Angular)  
* **Attack Simulation:** Kali Linux, Locust (Python)  
* **Web Application Firewall (WAF):** Coraza (Go, ModSecurity CRS compatible)  
* **SIEM/EDR:** Wazuh (Agents, Manager, Indexer, Dashboard)  
* **Machine Learning:** Python (PyTorch, scikit-learn, pandas, numpy), Custom Deep Learning Models (RNN, CNN)  
* **Logging & Data Storage:** File-based logs, Wazuh Indexer (OpenSearch/Elasticsearch)

## **Contributing**

We welcome contributions to this project\! If you'd like to contribute, please refer to our [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## **License**

This project is licensed under the [MIT License](./LICENSE.md).

## **Contact**

For questions, suggestions, or collaborations, please open an issue on this GitHub repository or reach out to [nelly.delessy@gmail.com](mailto:nelly.delessy@gmail.com).