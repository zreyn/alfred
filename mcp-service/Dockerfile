# Use Python 3.12 as base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY mcp-service/pyproject.toml ./

# Install dependencies using uv
RUN uv sync --no-dev

# Copy source code
COPY mcp-service/src ./src

# Set Python path to include src
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose the MCP service port
EXPOSE 8000

# Run the FastMCP service
CMD ["/app/.venv/bin/python", "-m", "src.main"]
