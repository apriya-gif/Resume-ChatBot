# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy app files
COPY app/ ./app
COPY models/ ./models

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r app/requirements.txt

# Default command
CMD ["python", "app/llama_query.py"]
