# Use Python 3.12 as base image
FROM python:3.12-slim

# Install system dependencies for audio, CUDA, and livekit-rtc
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libasound2-dev \
    portaudio19-dev \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY agent/pyproject.toml agent/uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the rest of the agent code
COPY agent/ .

# Set Python path to include src
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Run the livekit agent using the build-time virtualenv (skip uv bootstrapping)
CMD ["/app/.venv/bin/python", "src/main.py", "dev"]
