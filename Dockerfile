# Use the specific Python version you requested
FROM python:3.10.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# wget: to download the model
# libgl1 & libglib2.0-0: required for OpenCV (used by SAHI/Ultralytics)
RUN apt-get update && apt-get install -y \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We use --extra-index-url to find the CPU-specific versions of PyTorch
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Download the YOLOv8 model using wget
# This ensures a clean, uncorrupted file inside the image
RUN wget -O yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

# Copy the rest of the application code
COPY . .

# Expose the Flask port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]