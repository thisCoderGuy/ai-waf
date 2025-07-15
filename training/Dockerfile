# training/Dockerfile
#FROM nvcr.io/nvidia/pytorch:24.08-py3

# Use a generic NVIDIA CUDA base image for GPU support
# This image provides CUDA, cuDNN, and a base Ubuntu environment.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app

# Install Python and pip
# Set DEBIAN_FRONTEND to noninteractive to prevent prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies including PyTorch and TensorFlow
COPY requirements.txt .
# Use python3 -m pip to ensure the correct pip for python3 is used
RUN python3 -m pip install --no-cache-dir -r requirements.txt


# Copy your training script and any other necessary files
COPY . .

# Command to run your training script
#CMD ["python3", "main.py"] # Explicitly use python3
CMD python3 main.py 