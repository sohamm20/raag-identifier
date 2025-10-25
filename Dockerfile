# Multi-stage build for Raag Identifier
# Production-ready Docker container for inference

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM base as production

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY inference.py .
COPY evaluate.py .
COPY preprocess.py .

# Create directories for models and data
RUN mkdir -p /model /data /output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command (can be overridden)
CMD ["python", "inference.py", "--help"]

# Example usage:
# docker build -t raag-identifier .
# docker run -v $(pwd)/model:/model -v $(pwd)/data:/data raag-identifier \
#     python inference.py --model /model/best_model.pth --input /data/test.wav --output /data/predictions.jsonl
