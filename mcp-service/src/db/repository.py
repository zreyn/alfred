from datetime import datetime

import aiosqlite

from .database import get_db
from .models import Note, NoteCreate, SearchResult


async def store_note(note: NoteCreate) -> Note:
    """Store a new note with optional tags."""
    db = await get_db()

    cursor = await db.execute(
        "INSERT INTO notes (content) VALUES (?)",
        (note.content,),
    )
    note_id = cursor.lastrowid

    for tag_name in note.tags:
        tag_name_normalized = tag_name.lower().strip()
        await db.execute(
            "INSERT OR IGNORE INTO tags (name) VALUES (?)",
            (tag_name_normalized,),
        )
        cursor = await db.execute(
            "SELECT id FROM tags WHERE name = ?",
            (tag_name_normalized,),
        )
        tag_row = await cursor.fetchone()
        await db.execute(
            "INSERT INTO note_tags (note_id, tag_id) VALUES (?, ?)",
            (note_id, tag_row["id"]),
        )

    await db.commit()
    return await get_note_by_id(note_id)


async def search_notes(
    query: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> list[SearchResult]:
    """Search notes by content (FTS) and/or tags."""
    db = await get_db()
    results = []

    if query and tags:
        placeholders = ",".join("?" * len(tags))
        sql = f"""
            SELECT DISTINCT n.*, bm25(notes_fts) as rank
            FROM notes n
            JOIN notes_fts ON n.id = notes_fts.rowid
            JOIN note_tags nt ON n.id = nt.note_id
            JOIN tags t ON nt.tag_id = t.id
            WHERE notes_fts MATCH ? AND t.name IN ({placeholders})
            ORDER BY rank
            LIMIT ?
        """
        params = [query] + [t.lower().strip() for t in tags] + [limit]
    elif query:
        sql = """
            SELECT n.*, bm25(notes_fts) as rank
            FROM notes n
            JOIN notes_fts ON n.id = notes_fts.rowid
            WHERE notes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        params = [query, limit]
    elif tags:
        placeholders = ",".join("?" * len(tags))
        sql = f"""
            SELECT n.*, 0 as rank
            FROM notes n
            JOIN note_tags nt ON n.id = nt.note_id
            JOIN tags t ON nt.tag_id = t.id
            WHERE t.name IN ({placeholders})
            GROUP BY n.id
            HAVING COUNT(DISTINCT t.id) = ?
            ORDER BY n.created_at DESC
            LIMIT ?
        """
        params = [t.lower().strip() for t in tags] + [len(tags), limit]
    else:
        sql = """
            SELECT n.*, 0 as rank
            FROM notes n
            ORDER BY n.created_at DESC
            LIMIT ?
        """
        params = [limit]

    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()

    for row in rows:
        note = await _row_to_note(row)
        results.append(SearchResult(note=note, rank=row["rank"]))

    return results


async def get_note_by_id(note_id: int) -> Note | None:
    """Get a single note by ID with its tags."""
    db = await get_db()
    cursor = await db.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
    row = await cursor.fetchone()
    if row:
        return await _row_to_note(row)
    return None


async def _row_to_note(row: aiosqlite.Row) -> Note:
    """Convert database row to Note model with tags."""
    db = await get_db()
    cursor = await db.execute(
        """
        SELECT t.name FROM tags t
        JOIN note_tags nt ON t.id = nt.tag_id
        WHERE nt.note_id = ?
        """,
        (row["id"],),
    )
    tag_rows = await cursor.fetchall()
    tags = [r["name"] for r in tag_rows]

    return Note(
        id=row["id"],
        content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        tags=tags,
    )
