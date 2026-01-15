"""
SimpleMem memory module for mcp-service.

Provides semantic memory capabilities with LanceDB vector storage,
Ollama embeddings, and LLM-based compression.
"""

from memory.models import Dialogue, MemoryEntry, MemorySearchResult, MemoryStats
from memory.repository import MemoryRepository

__all__ = [
    "Dialogue",
    "MemoryEntry",
    "MemorySearchResult",
    "MemoryStats",
    "MemoryRepository",
]
