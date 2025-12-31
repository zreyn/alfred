# Use Python 3.12 as base image
FROM python:3.12-slim

# Install system dependencies for audio processing and ML
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY f5-tts-service/pyproject.toml f5-tts-service/uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy pre-downloaded models, audio files, and source code
COPY f5-tts-service/ .

# Set Python path to include src
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose the service port
EXPOSE 11800

# Run the FastAPI service
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11800"]
