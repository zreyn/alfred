"""
Data models for the memory module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class Dialogue:
    """A single dialogue entry from a conversation."""

    speaker: str
    content: str
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        elif isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class AtomicFact:
    """A compressed atomic fact extracted from dialogues."""

    content: str  # The lossless restatement with resolved references
    keywords: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    topic: Optional[str] = None


@dataclass
class MemoryEntry:
    """A stored memory with embedding and metadata."""

    id: str
    content: str
    vector: list[float]
    persons: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_atomic_fact(
        cls, fact: AtomicFact, vector: list[float]
    ) -> "MemoryEntry":
        """Create a MemoryEntry from an AtomicFact and its embedding."""
        return cls(
            id=str(uuid.uuid4()),
            content=fact.content,
            vector=vector,
            persons=fact.persons,
            locations=fact.locations,
            entities=fact.entities,
            keywords=fact.keywords,
            timestamp=fact.timestamp,
        )


@dataclass
class MemorySearchResult:
    """A search result with relevance score."""

    memory: MemoryEntry
    score: float


@dataclass
class MemoryStats:
    """Statistics about memory storage."""

    total_memories: int
    total_persons: int
    total_locations: int
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None
