# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any, none explicitly needed for this Python app)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Create directories for data and models if they don't exist
RUN mkdir -p data/raw data/processed models

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
# Use gunicorn with uvicorn workers for production deployment
# --bind 0.0.0.0:8000 binds to all network interfaces on port 8000
# src.api.main:app refers to the 'app' object in 'main.py' inside the 'src/api' module
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "src.api.main:app"]

