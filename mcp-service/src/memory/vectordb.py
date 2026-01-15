"""
LanceDB vector database wrapper for memory storage.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

from memory.config import MEMORY_DB_PATH, EMBEDDING_DIM, SEMANTIC_TOP_K
from memory.models import MemoryEntry, MemorySearchResult, MemoryStats

logger = logging.getLogger(__name__)

# Table name for memories
TABLE_NAME = "memories"


class MemoryVectorDB:
    """Async wrapper for LanceDB vector database operations."""

    def __init__(self, db_path: str = MEMORY_DB_PATH):
        self.db_path = db_path
        self._db: Optional[lancedb.DBConnection] = None
        self._table: Optional[lancedb.table.Table] = None

    async def connect(self) -> None:
        """Initialize LanceDB connection and ensure table exists."""
        # Ensure directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        # Connect to LanceDB (sync API, but operations are fast)
        self._db = lancedb.connect(self.db_path)

        # Create or open table
        await self._ensure_table()

    async def _ensure_table(self) -> None:
        """Ensure the memories table exists with correct schema."""
        if self._db is None:
            raise RuntimeError("Database not connected")

        # Check if table exists
        existing_tables = self._db.table_names()

        if TABLE_NAME in existing_tables:
            self._table = self._db.open_table(TABLE_NAME)
            logger.info(f"Opened existing table: {TABLE_NAME}")
        else:
            # Create table with schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
                pa.field("persons", pa.list_(pa.string())),
                pa.field("locations", pa.list_(pa.string())),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("keywords", pa.list_(pa.string())),
                pa.field("timestamp", pa.string()),  # ISO format string
                pa.field("created_at", pa.string()),  # ISO format string
            ])

            # Create empty table with schema
            self._table = self._db.create_table(TABLE_NAME, schema=schema)
            logger.info(f"Created new table: {TABLE_NAME}")

    async def add_memories(self, entries: list[MemoryEntry]) -> int:
        """Add memory entries to the database.

        Args:
            entries: List of MemoryEntry objects to store.

        Returns:
            Number of entries added.
        """
        if self._table is None:
            raise RuntimeError("Table not initialized")

        if not entries:
            return 0

        # Convert to records for LanceDB
        records = []
        for entry in entries:
            records.append({
                "id": entry.id,
                "content": entry.content,
                "vector": entry.vector,
                "persons": entry.persons,
                "locations": entry.locations,
                "entities": entry.entities,
                "keywords": entry.keywords,
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else "",
                "created_at": entry.created_at.isoformat(),
            })

        # Add to table
        self._table.add(records)
        logger.info(f"Added {len(records)} memories to database")

        return len(records)

    async def search_semantic(
        self,
        query_vector: list[float],
        limit: int = SEMANTIC_TOP_K,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity.

        Args:
            query_vector: The query embedding vector.
            limit: Maximum number of results.

        Returns:
            List of MemorySearchResult sorted by relevance.
        """
        if self._table is None:
            raise RuntimeError("Table not initialized")

        # Perform vector search
        results = (
            self._table.search(query_vector)
            .limit(limit)
            .to_pandas()
        )

        # Convert to MemorySearchResult objects
        search_results = []
        for _, row in results.iterrows():
            entry = MemoryEntry(
                id=row["id"],
                content=row["content"],
                vector=row["vector"].tolist() if hasattr(row["vector"], "tolist") else row["vector"],
                persons=row["persons"] if row["persons"] is not None else [],
                locations=row["locations"] if row["locations"] is not None else [],
                entities=row["entities"] if row["entities"] is not None else [],
                keywords=row["keywords"] if row["keywords"] is not None else [],
                timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            # LanceDB returns _distance, convert to similarity score
            distance = row.get("_distance", 0.0)
            score = 1.0 / (1.0 + distance)  # Convert distance to similarity

            search_results.append(MemorySearchResult(memory=entry, score=score))

        return search_results

    async def search_keyword(
        self,
        keywords: list[str],
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """Search memories by keyword matching.

        Args:
            keywords: List of keywords to search for.
            limit: Maximum number of results.

        Returns:
            List of MemorySearchResult sorted by match count.
        """
        if self._table is None:
            raise RuntimeError("Table not initialized")

        # Get all entries and filter by keywords
        df = self._table.to_pandas()

        if df.empty:
            return []

        # Score each entry by keyword overlap
        results = []
        keywords_lower = [k.lower() for k in keywords]

        for _, row in df.iterrows():
            entry_keywords = row["keywords"] if row["keywords"] is not None else []
            entry_keywords_lower = [k.lower() for k in entry_keywords]

            # Count matching keywords
            matches = sum(1 for k in keywords_lower if k in entry_keywords_lower)

            # Also check content for keyword presence
            content_lower = row["content"].lower()
            content_matches = sum(1 for k in keywords_lower if k in content_lower)

            total_score = matches + (content_matches * 0.5)

            if total_score > 0:
                entry = MemoryEntry(
                    id=row["id"],
                    content=row["content"],
                    vector=row["vector"].tolist() if hasattr(row["vector"], "tolist") else row["vector"],
                    persons=row["persons"] if row["persons"] is not None else [],
                    locations=row["locations"] if row["locations"] is not None else [],
                    entities=row["entities"] if row["entities"] is not None else [],
                    keywords=entry_keywords,
                    timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                results.append(MemorySearchResult(memory=entry, score=total_score))

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def get_all_memories(self) -> list[MemoryEntry]:
        """Get all stored memories.

        Returns:
            List of all MemoryEntry objects.
        """
        if self._table is None:
            raise RuntimeError("Table not initialized")

        df = self._table.to_pandas()

        entries = []
        for _, row in df.iterrows():
            entry = MemoryEntry(
                id=row["id"],
                content=row["content"],
                vector=row["vector"].tolist() if hasattr(row["vector"], "tolist") else row["vector"],
                persons=row["persons"] if row["persons"] is not None else [],
                locations=row["locations"] if row["locations"] is not None else [],
                entities=row["entities"] if row["entities"] is not None else [],
                keywords=row["keywords"] if row["keywords"] is not None else [],
                timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            entries.append(entry)

        return entries

    async def get_stats(self) -> MemoryStats:
        """Get statistics about stored memories.

        Returns:
            MemoryStats object with storage statistics.
        """
        if self._table is None:
            raise RuntimeError("Table not initialized")

        df = self._table.to_pandas()

        if df.empty:
            return MemoryStats(
                total_memories=0,
                total_persons=0,
                total_locations=0,
            )

        # Collect unique persons and locations
        all_persons = set()
        all_locations = set()
        timestamps = []

        for _, row in df.iterrows():
            if row["persons"]:
                all_persons.update(row["persons"])
            if row["locations"]:
                all_locations.update(row["locations"])
            if row["created_at"]:
                timestamps.append(datetime.fromisoformat(row["created_at"]))

        oldest = min(timestamps) if timestamps else None
        newest = max(timestamps) if timestamps else None

        return MemoryStats(
            total_memories=len(df),
            total_persons=len(all_persons),
            total_locations=len(all_locations),
            oldest_memory=oldest,
            newest_memory=newest,
        )

    async def clear(self) -> None:
        """Delete all memories."""
        if self._db is None:
            raise RuntimeError("Database not connected")

        # Drop and recreate table
        if TABLE_NAME in self._db.table_names():
            self._db.drop_table(TABLE_NAME)
            logger.info(f"Dropped table: {TABLE_NAME}")

        # Recreate empty table
        await self._ensure_table()
        logger.info("Cleared all memories")

    async def close(self) -> None:
        """Close the database connection."""
        self._table = None
        self._db = None
        logger.info("Closed database connection")
