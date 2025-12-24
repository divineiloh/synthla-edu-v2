# SYNTHLA-EDU V2 Docker Image
# Provides reproducible environment for educational synthetic data experiments

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt requirements-locked.txt ./

# Install Python dependencies (use locked versions for reproducibility)
RUN pip install --no-cache-dir -r requirements-locked.txt

# Copy application code
COPY synthla_edu_v2.py ./
COPY DATA_SCHEMA.md LICENSE VERSION README.md ./

# Create directories for data and outputs
RUN mkdir -p data/raw runs

# Set Python to run in unbuffered mode (see output immediately)
ENV PYTHONUNBUFFERED=1

# Default command shows help
CMD ["python", "synthla_edu_v2.py", "--help"]
