# Base image
FROM python:3.9-slim

# Set environment variable to suppress pip warnings about root user
ENV PIP_ROOT_USER_ACTION=ignore

# Set working directory
WORKDIR /app

# Install Git
RUN apt-get update && apt-get install -y git

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install virtualenv to create a virtual environment
RUN pip install virtualenv

# Create a virtual environment inside the container
RUN virtualenv /venv  # Creates a virtual environment at /venv

# Use the virtual environment by default
# Update PATH to use the virtualenv's binaries
ENV PATH="/venv/bin:$PATH"

# Copy requirements file and install dependencies inside the virtual environment
COPY requirements.txt . 

# Install dependencies in the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1
