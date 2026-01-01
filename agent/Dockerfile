# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set frontend to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libasound2-dev \
    portaudio19-dev \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

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
