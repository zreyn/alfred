"""
High-level memory repository managing the full memory pipeline.
"""

import logging
from datetime import datetime
from typing import Optional

from memory.config import BATCH_WINDOW_SIZE, OLLAMA_HOST, LLM_MODEL, LLM_TEMPERATURE
from memory.models import Dialogue, MemoryEntry, MemorySearchResult, MemoryStats
from memory.embeddings import OllamaEmbeddings
from memory.compression import SemanticCompressor
from memory.vectordb import MemoryVectorDB
from memory.retrieval import HybridRetriever

import httpx

logger = logging.getLogger(__name__)


class MemoryRepository:
    """High-level interface for memory operations.

    Manages dialogue buffering, compression, storage, and retrieval.
    """

    def __init__(
        self,
        vector_db: Optional[MemoryVectorDB] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
        compressor: Optional[SemanticCompressor] = None,
    ):
        self.vector_db = vector_db or MemoryVectorDB()
        self.embeddings = embeddings or OllamaEmbeddings()
        self.compressor = compressor or SemanticCompressor()
        self.retriever: Optional[HybridRetriever] = None

        # Dialogue buffer for batch processing
        self._dialogue_buffer: list[Dialogue] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the repository and its components."""
        if self._initialized:
            return

        await self.vector_db.connect()
        self.retriever = HybridRetriever(self.vector_db, self.embeddings)
        self._initialized = True
        logger.info("Memory repository initialized")

    async def close(self) -> None:
        """Close all resources."""
        await self.embeddings.close()
        await self.compressor.close()
        await self.vector_db.close()
        self._initialized = False
        logger.info("Memory repository closed")

    def _ensure_initialized(self) -> None:
        """Ensure the repository is initialized."""
        if not self._initialized:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

    async def add_dialogue(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[str] = None,
    ) -> dict:
        """Add a single dialogue to the buffer.

        Args:
            speaker: Who said the dialogue.
            content: What was said.
            timestamp: Optional ISO timestamp.

        Returns:
            Dict with buffer status.
        """
        self._ensure_initialized()

        dialogue = Dialogue(
            speaker=speaker,
            content=content,
            timestamp=datetime.fromisoformat(timestamp) if timestamp else None,
        )
        self._dialogue_buffer.append(dialogue)

        logger.debug(f"Added dialogue to buffer: {speaker}: {content[:50]}...")

        return {
            "buffered": True,
            "buffer_size": len(self._dialogue_buffer),
        }

    async def add_dialogues(self, dialogues: list[dict]) -> dict:
        """Add multiple dialogues to the buffer.

        Args:
            dialogues: List of dialogue dicts with speaker, content, timestamp.

        Returns:
            Dict with buffer status.
        """
        self._ensure_initialized()

        for d in dialogues:
            dialogue = Dialogue(
                speaker=d["speaker"],
                content=d["content"],
                timestamp=datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else None,
            )
            self._dialogue_buffer.append(dialogue)

        logger.info(f"Added {len(dialogues)} dialogues to buffer")

        return {
            "buffered": True,
            "added_count": len(dialogues),
            "buffer_size": len(self._dialogue_buffer),
        }

    async def finalize(self) -> dict:
        """Process buffered dialogues into permanent memory.

        Compresses dialogues into atomic facts, generates embeddings,
        and stores in the vector database.

        Returns:
            Dict with processing results.
        """
        self._ensure_initialized()

        if not self._dialogue_buffer:
            return {
                "processed": False,
                "message": "No dialogues in buffer",
                "memories_created": 0,
            }

        dialogues = self._dialogue_buffer.copy()
        self._dialogue_buffer.clear()

        logger.info(f"Processing {len(dialogues)} buffered dialogues")

        # Compress dialogues into atomic facts
        facts = await self.compressor.compress_dialogues(dialogues)

        if not facts:
            return {
                "processed": True,
                "dialogues_processed": len(dialogues),
                "memories_created": 0,
                "message": "No memorable facts extracted",
            }

        # Generate embeddings for all facts
        fact_texts = [f.content for f in facts]
        embeddings = await self.embeddings.embed_batch(fact_texts)

        # Create memory entries
        entries = []
        for fact, embedding in zip(facts, embeddings):
            entry = MemoryEntry.from_atomic_fact(fact, embedding)
            entries.append(entry)

        # Store in vector database
        stored_count = await self.vector_db.add_memories(entries)

        logger.info(f"Created {stored_count} memories from {len(dialogues)} dialogues")

        return {
            "processed": True,
            "dialogues_processed": len(dialogues),
            "memories_created": stored_count,
        }

    async def query(self, query: str, limit: int = 10) -> dict:
        """Query memories and generate an answer.

        Args:
            query: The question to answer.
            limit: Maximum memories to retrieve.

        Returns:
            Dict with retrieved memories and synthesized answer.
        """
        self._ensure_initialized()

        if self.retriever is None:
            raise RuntimeError("Retriever not initialized")

        # Retrieve relevant memories
        results = await self.retriever.retrieve(query, limit=limit)

        if not results:
            return {
                "query": query,
                "memories_found": 0,
                "memories": [],
                "answer": "I don't have any memories related to that query.",
            }

        # Format memories for response
        memories = [
            {
                "content": r.memory.content,
                "score": r.score,
                "persons": r.memory.persons,
                "locations": r.memory.locations,
                "timestamp": r.memory.timestamp.isoformat() if r.memory.timestamp else None,
            }
            for r in results
        ]

        # Generate synthesized answer using LLM
        answer = await self._generate_answer(query, results)

        return {
            "query": query,
            "memories_found": len(results),
            "memories": memories,
            "answer": answer,
        }

    async def _generate_answer(
        self,
        query: str,
        results: list[MemorySearchResult],
    ) -> str:
        """Generate an answer from retrieved memories using LLM."""
        # Format context from memories
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. {result.memory.content}")

        context = "\n".join(context_parts)

        prompt = f"""Based ONLY on the following memories, answer the question. If the memories don't contain enough information, say so.

MEMORIES:
{context}

QUESTION: {query}

ANSWER:"""

        # Call LLM for answer
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "temperature": LLM_TEMPERATURE,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Unable to generate answer.")

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_persons: Optional[list[str]] = None,
    ) -> list[dict]:
        """Retrieve raw memories without answer generation.

        Args:
            query: The search query.
            limit: Maximum results.
            filter_persons: Optional filter by person names.

        Returns:
            List of memory dicts.
        """
        self._ensure_initialized()

        if self.retriever is None:
            raise RuntimeError("Retriever not initialized")

        results = await self.retriever.retrieve(query, limit=limit)

        # Apply person filter if specified
        if filter_persons:
            filter_set = {p.lower() for p in filter_persons}
            results = [
                r for r in results
                if any(p.lower() in filter_set for p in r.memory.persons)
            ]

        return [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "score": r.score,
                "persons": r.memory.persons,
                "locations": r.memory.locations,
                "keywords": r.memory.keywords,
                "timestamp": r.memory.timestamp.isoformat() if r.memory.timestamp else None,
                "created_at": r.memory.created_at.isoformat(),
            }
            for r in results
        ]

    async def get_stats(self) -> dict:
        """Get memory storage statistics.

        Returns:
            Dict with statistics.
        """
        self._ensure_initialized()

        stats = await self.vector_db.get_stats()

        return {
            "total_memories": stats.total_memories,
            "total_persons": stats.total_persons,
            "total_locations": stats.total_locations,
            "oldest_memory": stats.oldest_memory.isoformat() if stats.oldest_memory else None,
            "newest_memory": stats.newest_memory.isoformat() if stats.newest_memory else None,
            "buffer_size": len(self._dialogue_buffer),
        }

    async def clear(self) -> dict:
        """Clear all memories.

        Returns:
            Dict confirming deletion.
        """
        self._ensure_initialized()

        await self.vector_db.clear()
        self._dialogue_buffer.clear()

        return {
            "cleared": True,
            "message": "All memories have been deleted",
        }


# Global repository instance
_repository: Optional[MemoryRepository] = None


async def get_memory_repository() -> MemoryRepository:
    """Get or create the global memory repository instance."""
    global _repository

    if _repository is None:
        _repository = MemoryRepository()
        await _repository.initialize()

    return _repository
