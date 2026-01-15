"""
Configuration for the memory module.
"""

import os

# Ollama configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://local-ollama:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")

# Embedding dimensions for qwen3-embedding:0.6b
EMBEDDING_DIM = 1024

# Storage paths
MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", "/app/data/memory")

# Retrieval settings
SEMANTIC_TOP_K = 25
KEYWORD_TOP_K = 5

# Compression settings
BATCH_WINDOW_SIZE = 20  # Dialogues per compression batch

# LLM settings
LLM_TEMPERATURE = 0.1  # Low temperature for consistent compression
