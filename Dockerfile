# Dockerfile for the FastAPI-based housing price prediction API
#
# This Dockerfile sets up a lightweight container for running a FastAPI-based
# housing price prediction API. The container is based on the official Python 3.12-slim 
# image, installs required dependencies, and runs the FastAPI application using Uvicorn.
#
# Key steps:
# - Uses Python 3.12-slim as the base image for a minimal environment.
# - Installs necessary Python packages listed in requirements.txt.
# - Copies API source code and pre-trained machine learning models into the container.
# - Exposes port 8000 for API access.
# - Runs the API using Uvicorn when the container starts.
#
# Usage:
# - Build the Docker image: `docker build -t housing-api .`
# - Run the container: `docker run -p 8000:8000 housing-api`
#
# Ensure that the models (`*.joblib`, `*.pkl`) and version information are
# available in the working directory during the build process.

# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code, model, and scaler files into the container
# COPY all py files including api.py and model_training files
COPY *.py .  
# COPY best_model.joblib and robust_scaler.joblib .
COPY *.joblib .
COPY *.pkl .
COPY version_info.txt .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the API when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]