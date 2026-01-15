from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Note:
    id: int
    content: str
    created_at: datetime
    updated_at: datetime
    tags: list[str] = field(default_factory=list)


@dataclass
class NoteCreate:
    content: str
    tags: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    note: Note
    rank: float
