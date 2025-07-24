# **Hybrid Traditional-AI WAF**


## **Table of Contents**

* [Project Overview](#project-overview)
* [Operational Modes](#operational-modes)
  * [1\. Dataset Generation Mode](#1-dataset-generation-mode)
  * [2\. Model Training Mode](#2-model-training-mode)
  * [3\. Live Evaluation Mode](#3-live-evaluation-mode)
* [Testbed Architecture](#testbed-architecture)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Accessing Components](#accessing-components)
* [Usage](#usage)
  * [1\. Running Dataset Generation](#1-running-dataset-generation)
  * [2\. Running Model Training](#2-running-model-training)
  * [3\. Running Live Evaluation](#3-running-live-evaluation)
* [Technologies Used](#technologies-used)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)




## **Project Overview**

This research project focuses on developing and evaluating a **Hybrid Web Application Firewall (WAF)** that combines traditional rule-based detection with advanced **Artificial Intelligence (AI)** capabilities. The core objective is to enhance web application security by leveraging machine learning models to detect sophisticated and novel attack patterns that might bypass conventional WAF rules. 

This initiative is driven by the **critical lack of modern, high-quality, and labeled datasets** specifically tailored for training AI models in the domain of web application security, especially concerning the subtle evasion techniques (e.g., obfuscation, polymorphism) and context-dependent nature of diverse attack vectors that challenge traditional WAFs, often leading to false positives or missed detections. 

Our approach encompasses a full lifecycle: from generating meticulously labeled security datasets to training deep learning models and integrating them into a real-time defense system. This entire framework is built around a versatile **containerized testbed** that supports three distinct operational **modes**: **Dataset Generation**, **Model Training**, and **Live Evaluation**.

## **Operational Modes**

### **1\. Dataset Generation Mode**

This mode focuses on creating high-quality, labeled datasets for training machine learning models. It involves simulating various web traffic patterns and meticulously logging the interactions. To configure the generation of datasets, visit the [Generating Datasets](./docs/datasets/generating_datasets.md) page.

### **2\. Model Training Mode**

This mode focuses on the process of building and extending various machine learning models, ranging from **traditional machine learning algorithms** to **advanced deep learning architectures** like multi-input RNN or CNN classifiers, all utilizing the meticulously generated datasets. 

The project provides a **comprehensive framework** for training classification models. To learn more about the model training mode, visit the [Model Training Mode](./docs/training/model_training.md) page.

This framework supports experimentation and optimization across a spectrum of model complexities to achieve optimal threat detection. To learn more about the training framework, visit the [Training Framework](./docs/training/training_framework.md) page.

### **3\. Live Evaluation Mode**

This mode is dedicated to the **real-time evaluation and demonstration** of the trained AI model's performance when integrated into the security pipeline.

To learn more about the live evaluation mode, visit the [Live Evaluation Mode](./docs/live_evaluation/live_evaluation.md) page.

## **Testbed Architecture**

The project utilizes a robust and reproducible **Containerized Web Application Security Analytics Testbed**, built entirely with **Docker Compose**. This multi-component infrastructure is designed to simulate real-world web application interactions, including both legitimate user traffic and various attack vectors, while meticulously collecting security telemetry across three distinct operational modes. To learn more about the comprehensive test bed, visit the [Test Bed](./docs/testbed/test_bed.md) page.

## **Getting Started**

To set up and run this project, follow these steps:

### **Prerequisites**

* [Docker](https://docs.docker.com/get-docker/) (Docker Engine and Docker Compose)  
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* **For GPU-accelerated training (Linux hosts with NVIDIA GPUs):**  
  * **NVIDIA Drivers:** Ensure the latest NVIDIA GPU drivers are installed on your host system. Verify with `nvidia-smi`.  
  * **NVIDIA Container Toolkit:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your host. This configures your Docker daemon to enable GPU access for containers. After installation, restart your Docker daemon (`sudo systemctl restart docker`).  
  * **Docker Daemon Configuration:** Confirm that your Docker daemon recognizes the nvidia runtime by running `docker info | grep Runtimes`. It should list nvidia. If not, re-run `sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker.  
* **For GPU-accelerated training (Windows hosts with NVIDIA GPUs):**  
  * **Windows Subsystem for Linux 2 (WSL2):** Ensure WSL2 is enabled and a Linux distribution (e.g., Ubuntu) is installed. Docker Desktop leverages WSL2 for GPU support.  
  * **NVIDIA Drivers:** Install the latest NVIDIA GPU drivers for Windows on your host system.  
  * **NVIDIA CUDA Toolkit for WSL:** Install the [NVIDIA CUDA Toolkit for WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) within your WSL2 Linux distribution. This provides the necessary CUDA libraries inside WSL2.  
  * **Docker Desktop Configuration:** Ensure Docker Desktop is configured to use the WSL2 backend (Settings \> General \> Use WSL 2 based engine). Docker Desktop should automatically expose the GPU to containers running in WSL2 if the above prerequisites are met. There is typically no explicit "GPU acceleration" toggle in Docker Desktop for Windows; it's handled via WSL2 integration.

### **Installation**

1. **Clone the repository:**  

   `git clone https://github.com/thisCoderGuy/ai-waf.git`

   `cd ai-waf`

2. **Build and start the Docker containers for your desired mode:**  

     Your *docker-compose.yml* uses profiles to manage different operational modes.  

   * To run the **Dataset Generation Mode**:  

     `docker compose --profile app-core --profile dataset-gen up --build -d`

   * To run the **Model Training Mode**: 

     The training service is configured to attempt to use your NVIDIA GPU if available and correctly set up on your host.  

     * **If you have an NVIDIA GPU:** Ensure all GPU prerequisites (drivers, toolkit, Docker config) are met.  

        `docker compose --profile train up --build` 

     * **If you do NOT have an NVIDIA GPU:** The training service is configured to request GPU resources. If no GPU is available or configured, Docker might fail to start this service with a "could not select device driver" error. You will need to either:  
       * **Remove the deploy section** from the training service in *docker-compose.yml* if you want to run CPU-only training.  
       * Or, consider using a CPU-only base image in training/Dockerfile (e.g., FROM python:3.9-slim-buster) and remove GPU-specific libraries from requirements.txt if you intend to train without a GPU.  
         
      *Note: The training service is designed to run its task and then exit. You will see its logs in your terminal.*  

   * To run the **Live Evaluation Mode**:  

     `docker compose --profile app-core --profile live-eval up --build -d`

3. **Stopping Services:**  
   
   * **To stop services for a specific profile (e.g., app-core and live-eval):**  

     `docker compose --profile app-core --profile live-eval down`

     *Note: When stopping services by profile, only the services explicitly activated by those profiles (and their dependencies) will be stopped. If you started services from multiple profiles, you may need to specify all active profiles to stop them.*


### **Accessing Components**

* **OWASP Juice Shop:** http://localhost:8080 (or your configured port)  
* **Wazuh Dashboard (Kibana/OpenSearch Dashboards):** https://localhost  
  * Default credentials: admin/SecretPassword (change immediately in production\!)  
* **Kali Linux Container:** You can exec into the Kali container to run tools manually or verify scripts.  
  `docker ps     # Find the container ID/name for Kali  `

  `docker exec -it <kali_container_id_or_name> bash`

## **Usage**

This project supports three distinct operational modes, each with its own setup and execution steps:

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