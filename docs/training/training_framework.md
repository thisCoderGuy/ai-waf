# **Training Framework Overview**

The training framework within this Hybrid Traditional-AI WAF project is a comprehensive system designed to facilitate the development, experimentation, and optimization of machine learning models for web application security. It acts as the backbone for transforming raw, labeled datasets into robust AI models capable of detecting sophisticated attack patterns.

## **Purpose and Objectives**

The primary objectives of this training framework are:

* **Model Development:** Provide a structured environment for building various types of machine learning models, ranging from traditional algorithms (e e.g., Support Vector Machines, Random Forests) to advanced deep learning architectures (e.g., Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), Transformers).  
* **Experimentation:** Enable easy experimentation with different model architectures, hyperparameters, and feature engineering techniques to identify the most effective approaches for threat detection.  
* **Optimization:** Support the fine-tuning of models to achieve optimal performance metrics, such as high accuracy, precision, recall, and F1-score, while minimizing false positives and false negatives.  
* **Reproducibility:** Ensure that training experiments can be easily reproduced, allowing for consistent results and reliable comparison of different models and configurations.  
* **Integration:** Produce trained models in a format that can be seamlessly integrated into the ai-microservice for real-time inference within the Live Evaluation Mode.

## **Key Components and Stages**

The training framework typically involves several interconnected stages, orchestrated by your training script (e.g., main.py or train\_model.py) running within the ml\_trainer container:

### **1\. Data Loading and Ingestion**

This initial stage focuses on reading the meticulously generated datasets from the designated input path.

* **Source:** Data is loaded from the ./training/training-data/raw directory on the host, which is mounted as /data/input/ inside the ml\_trainer container.  
* **Format Handling:** The framework expects to ingest data in formats like CSV or JSON, as configured in the coraza-proxy during Dataset Generation Mode.  
* **Libraries:** Common Python libraries like pandas are typically used for efficient data loading and initial inspection.

### **2\. Data Preprocessing and Feature Engineering**

Raw web traffic logs are not directly usable by machine learning models. This stage transforms the raw data into a numerical format suitable for model training.

* **Cleaning:** Handling missing values, removing irrelevant information.  
* **Normalization/Scaling:** Adjusting numerical features to a common scale.  
* **Encoding:** Converting categorical data (e.g., HTTP methods, attack types) into numerical representations (e.g., one-hot encoding).  
* **Text Processing:** For request bodies or URLs, this involves tokenization, vectorization (e.g., TF-IDF, Word Embeddings), and sequence padding for deep learning models.  
* **Labeling:** Ensuring that each data sample has an accurate label indicating whether it's benign or malicious, and potentially the type of attack. This relies on the labels set during dataset generation.  
* **Libraries:** numpy, pandas, and scikit-learn (for transformers, scalers) are commonly used. For text processing, libraries like transformers (Hugging Face) or tensorflow.keras.preprocessing.text are essential.

### **3\. Model Definition and Architecture**

This is where the core machine learning model is designed and instantiated.

* **Traditional ML Models:** For simpler, interpretable models, scikit-learn provides a wide range of algorithms (e.g., Logistic Regression, Support Vector Machines, Gradient Boosting, Random Forests).  
* **Deep Learning Models:** For complex pattern recognition, frameworks like **PyTorch** or **TensorFlow** are used to define neural network architectures. This can include:  
  * **CNNs:** Effective for extracting local features from sequences (e.g., character sequences in URLs or payloads).  
  * **RNNs/LSTMs/GRUs:** Suitable for processing sequential data and capturing temporal dependencies in traffic patterns.  
  * **Transformer Models:** Advanced architectures for complex natural language understanding tasks, highly effective for analyzing HTTP request content.  
* **Flexibility:** The framework is designed to allow easy swapping of different model architectures to compare their performance.

### **4\. Model Training and Optimization**

In this stage, the defined model learns from the preprocessed data.

* **Training Loop:** Iteratively feeding data to the model, computing loss, and updating model weights using optimizers (e.g., Adam, SGD).  
* **Hyperparameter Tuning:** Experimenting with learning rates, batch sizes, number of epochs, regularization strengths, and other model-specific parameters. Tools like accelerate (Hugging Face) can assist with distributed training and mixed-precision training for faster optimization on GPUs.  
* **Validation:** Using a separate validation dataset to monitor model performance during training and prevent overfitting.  
* **GPU Acceleration:** The ml\_trainer container is configured to leverage NVIDIA GPUs, significantly speeding up the training process for deep learning models.

### **5\. Model Evaluation**

After training, the model's performance is rigorously assessed on an unseen test dataset.

* **Metrics:** Evaluation typically involves metrics such as:  
  * **Accuracy:** Overall correctness.  
  * **Precision:** Proportion of true positives among all positive predictions (minimizing false positives).  
  * **Recall (Sensitivity):** Proportion of true positives among all actual positives (minimizing false negatives).  
  * **F1-Score:** Harmonic mean of precision and recall.  
  * **Confusion Matrix:** Detailed breakdown of true/false positives/negatives.  
  

### **6\. Model Saving and Versioning**

Once a satisfactory model is trained and evaluated, it is saved for later use in the ai-microservice.

* **Format:** Models are typically saved in formats native to their framework (e.g., .pt or .pth for PyTorch, .h5 or SavedModel format for TensorFlow/Keras, .pkl for scikit-learn models using joblib).  
* **Output Path:** Saved models are written to the /models/output/ directory inside the container, which is mounted to `./ai-microservice/trained-models` on your host.  
//TODO Use [MLflow as a Model Repository](https://docs.databricks.com/aws/en/mlflow/)   ([github](https://github.com/mlflow/mlflow))

By following these stages and leveraging the provided Dockerized environment, the training framework ensures a robust and efficient process for developing high-performing AI models for your WAF project.