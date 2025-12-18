# Dockerfile for reproducible runs of SYNTHLA-EDU V2
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
COPY configs/ ./configs/

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose run command
ENTRYPOINT ["python", "-m", "synthla_edu_v2.run", "--config"]
CMD ["configs/quick.yaml"]
