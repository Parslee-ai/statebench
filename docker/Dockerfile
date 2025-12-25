# StateBench v1.0 Docker Image
#
# Build:
#   docker build -t statebench:latest -f docker/Dockerfile .
#
# Run:
#   docker run -v $(pwd)/data:/data -e OPENAI_API_KEY statebench:latest generate --help

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package
RUN pip install -e ".[dev]"

# Create data directory
RUN mkdir -p /data

# Set the entrypoint
ENTRYPOINT ["statebench"]

# Default command shows help
CMD ["--help"]
