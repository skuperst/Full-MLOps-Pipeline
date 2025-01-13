# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

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
RUN pip install --no-cache-dir -r requirements.txt  # Install dependencies in the virtual environment

# Copy the rest of your application code
COPY . .

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Run DVC pipeline and then the Flask app (in that order)
CMD dvc repro -f && python src/flask_api.py
