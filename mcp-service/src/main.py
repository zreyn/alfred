from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from db.models import NoteCreate
from db.repository import search_notes as db_search_notes
from db.repository import store_note as db_store_note

mcp = FastMCP("mcp-service")


@mcp.tool()
def hello(name: str = "World") -> str:
    """Say hello to someone.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting message.
    """
    return f"Hello, {name}!"


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a greeting resource for a specific name."""
    return f"Hello, {name}! Welcome to the MCP service."


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


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


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
