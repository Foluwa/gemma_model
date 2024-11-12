# Use a Python image compatible with both Intel and Apple Silicon
FROM --platform=$BUILDPLATFORM python:3.9-slim

# Set up the working directory
WORKDIR /app

# Copy the project files to the working directory in the container, including model directory
COPY . /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean

# Install dependencies, specifying numpy<2.0 to ensure compatibility with TensorFlow
RUN pip install --no-cache-dir numpy==1.23.* && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir tensorflow-cpu-aws

# Set environment variables for CPU-only TensorFlow processing
ENV JAX_PLATFORMS="cpu"
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS="0"

# Expose port 8000
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]