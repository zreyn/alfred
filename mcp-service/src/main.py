from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from db.models import NoteCreate
from db.repository import search_notes as db_search_notes
from db.repository import store_note as db_store_note

from memory.repository import get_memory_repository

mcp = FastMCP("mcp-service")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


@mcp.tool()
async def store_note(content: str, tags: list[str] | None = None) -> dict:
    """Store a note with optional tags for later retrieval.

    Args:
        content: The content of the note to store.
        tags: Optional list of tags to categorize the note.

    Returns:
        The stored note with its assigned ID and metadata.
    """
    note_create = NoteCreate(content=content, tags=tags or [])
    note = await db_store_note(note_create)

    return {
        "id": note.id,
        "content": note.content,
        "tags": note.tags,
        "created_at": note.created_at.isoformat(),
    }


@mcp.tool()
async def search_notes(
    query: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search stored notes by content or tags.

    Args:
        query: Full-text search query to match against note content.
               Supports SQLite FTS5 syntax (AND, OR, NOT, phrases in quotes).
        tags: Filter results to notes that have ALL specified tags.
        limit: Maximum number of results to return (default: 10).

    Returns:
        List of matching notes with relevance ranking.
    """
    results = await db_search_notes(query=query, tags=tags, limit=limit)

    return [
        {
            "id": r.note.id,
            "content": r.note.content,
            "tags": r.note.tags,
            "created_at": r.note.created_at.isoformat(),
            "relevance": abs(r.rank),
        }
        for r in results
    ]


# =============================================================================
# Memory Tools (SimpleMem integration)
# =============================================================================


@mcp.tool()
async def memory_add(
    speaker: str,
    content: str,
    timestamp: str | None = None,
) -> dict:
    """Add a single dialogue to memory buffer.

    Dialogues are buffered until memory_finalize is called to process them
    into permanent semantic memories.

    Args:
        speaker: Who said the dialogue (e.g., "user", "assistant", person name).
        content: What was said.
        timestamp: Optional ISO 8601 timestamp (e.g., "2025-01-15T10:30:00").

    Returns:
        Buffer status including current buffer size.
    """
    repo = await get_memory_repository()
    return await repo.add_dialogue(speaker, content, timestamp)


@mcp.tool()
async def memory_add_batch(dialogues: list[dict]) -> dict:
    """Add multiple dialogues to memory buffer.

    Args:
        dialogues: List of dialogue objects, each with:
            - speaker (str): Who said the dialogue
            - content (str): What was said
            - timestamp (str, optional): ISO 8601 timestamp

    Returns:
        Buffer status including count added and total buffer size.
    """
    repo = await get_memory_repository()
    return await repo.add_dialogues(dialogues)


@mcp.tool()
async def memory_query(query: str, limit: int = 10) -> dict:
    """Query memories and get an AI-generated answer.

    Searches stored memories using hybrid semantic and keyword search,
    then synthesizes an answer based on the relevant memories.

    Args:
        query: The question to answer.
        limit: Maximum number of memories to retrieve (default: 10).

    Returns:
        Query results including:
        - memories_found: Number of relevant memories
        - memories: List of matching memories with scores
        - answer: AI-synthesized answer based on memories
    """
    repo = await get_memory_repository()
    return await repo.query(query, limit)


@mcp.tool()
async def memory_retrieve(
    query: str,
    limit: int = 10,
    filter_persons: list[str] | None = None,
) -> list[dict]:
    """Retrieve raw memories without answer generation.

    Use this when you need the raw memory data for custom processing
    rather than an AI-generated answer.

    Args:
        query: The search query.
        limit: Maximum number of results (default: 10).
        filter_persons: Optional filter to only return memories mentioning
                       these persons.

    Returns:
        List of memory objects with content, score, metadata.
    """
    repo = await get_memory_repository()
    return await repo.retrieve(query, limit, filter_persons)


@mcp.tool()
async def memory_stats() -> dict:
    """Get statistics about memory storage.

    Returns:
        Statistics including:
        - total_memories: Number of stored memories
        - total_persons: Number of unique persons mentioned
        - total_locations: Number of unique locations mentioned
        - oldest_memory: Timestamp of oldest memory
        - newest_memory: Timestamp of newest memory
        - buffer_size: Number of dialogues waiting to be processed
    """
    repo = await get_memory_repository()
    return await repo.get_stats()


@mcp.tool()
async def memory_clear() -> dict:
    """Clear all stored memories.

    WARNING: This permanently deletes all memories and cannot be undone.

    Returns:
        Confirmation of deletion.
    """
    repo = await get_memory_repository()
    return await repo.clear()


@mcp.tool()
async def memory_finalize() -> dict:
    """Process buffered dialogues into permanent memories.

    This compresses the buffered dialogues into atomic facts using an LLM,
    generates embeddings, and stores them in the vector database.

    Call this when a conversation session ends or when you want to
    persist the current dialogue buffer.

    Returns:
        Processing results including:
        - dialogues_processed: Number of dialogues processed
        - memories_created: Number of memories created
    """
    repo = await get_memory_repository()
    return await repo.finalize()


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
