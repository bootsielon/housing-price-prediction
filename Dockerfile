# Dockerfile

# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code, model, and scaler files into the container
COPY api.py .
COPY best_model.joblib .
COPY robust_scaler.joblib .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the API when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]