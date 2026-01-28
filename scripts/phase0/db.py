"""
SQLite FTS5 database operations for Phase 0 memory retrieval.

This module provides a SQLite-based full-text search system for engineering
memories extracted from code reviews. It uses SQLite's FTS5 (Full-Text Search 5)
extension to enable fast keyword-based retrieval using BM25 ranking.

Key Features:
    - FTS5 full-text indexing on situation variants
    - BM25 ranking for relevance scoring (lower scores = better matches)
    - Automatic sync between base table and FTS index via triggers
    - Support for FTS5 query syntax (AND, OR, NOT, "phrases", prefix*)

Database Structure:
    - memories: Base table storing all memory fields
    - memories_fts: FTS5 virtual table for full-text search
    - Triggers: Keep FTS index synchronized with base table changes

Usage:
    # Rebuild database from JSONL files
    uv run python scripts/phase0/db.py --rebuild

    # Programmatic usage
    from phase0.db import search_memories
    results = search_memories("data/phase0/memories/memories.db", "async function", limit=5)
"""

import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase0 import (
    load_memories,
    DEFAULT_MEMORIES_DIR,
    FIELD_ID,
    FIELD_SITUATION,
    FIELD_VARIANTS,
    FIELD_LESSON,
    FIELD_METADATA,
    FIELD_SOURCE,
    FIELD_RANK,
)

# Default database path
DEFAULT_DB_PATH = "data/phase0/memories/memories.db"


@contextmanager
def get_db_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    """
    Context manager for database connections ensuring proper resource cleanup.

    This ensures connections are always closed, even if exceptions occur.

    Args:
        db_path: Path to SQLite database file.

    Yields:
        SQLite connection object.

    Example:
        >>> with get_db_connection("memories.db") as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT COUNT(*) FROM memories")
    """
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _serialize_json_field(data: Optional[Dict[str, Any]]) -> str:
    """Serialize dictionary to JSON string for database storage."""
    return json.dumps(data or {}, ensure_ascii=False)


def _deserialize_json_field(json_str: Optional[str]) -> Dict[str, Any]:
    """Deserialize JSON string from database to dictionary."""
    if not json_str:
        return {}
    return json.loads(json_str)


def _row_to_memory_dict(row: sqlite3.Row, include_rank: bool = False) -> Dict[str, Any]:
    """Convert database row to memory dictionary."""
    try:
        variants = row[FIELD_VARIANTS]
    except IndexError:
        variants = None

    memory = {
        FIELD_ID: row[FIELD_ID],
        FIELD_SITUATION: row[FIELD_SITUATION],
        FIELD_VARIANTS: _deserialize_json_field(variants),
        FIELD_LESSON: row[FIELD_LESSON],
        FIELD_METADATA: _deserialize_json_field(row[FIELD_METADATA]),
        FIELD_SOURCE: _deserialize_json_field(row[FIELD_SOURCE]),
    }

    if include_rank:
        memory[FIELD_RANK] = row[FIELD_RANK]

    return memory


def create_database(db_path: str) -> None:
    """
    Create SQLite database with memories table and FTS5 full-text search index.

    This function sets up a complete search-enabled database structure with:
    1. Base table (memories) for storing structured memory data
    2. FTS5 virtual table (memories_fts) for fast full-text search
    3. Triggers to automatically keep the FTS index synchronized

    Args:
        db_path: Path where the SQLite database file should be created.
                 Any existing database at this path will be replaced.

    Note:
        The function drops any existing tables, so this is destructive!
        Always rebuild from JSONL source files rather than modifying in place.
    """
    with get_db_connection(db_path) as conn:
        cur = conn.cursor()

        # Drop existing tables if they exist (ensures clean rebuild)
        cur.execute("DROP TABLE IF EXISTS memories_fts")
        cur.execute("DROP TABLE IF EXISTS memories")

        # Create base table for storing all memory fields
        cur.execute(f"""
            CREATE TABLE memories (
                {FIELD_ID} TEXT PRIMARY KEY,
                {FIELD_SITUATION} TEXT NOT NULL,
                {FIELD_VARIANTS} TEXT,
                {FIELD_LESSON} TEXT NOT NULL,
                {FIELD_METADATA} TEXT,
                {FIELD_SOURCE} TEXT
            )
        """)

        # Create FTS5 virtual table for full-text search on situation variants
        cur.execute(f"""
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                situation_variant,
                memory_id UNINDEXED
            )
        """)

        # Trigger: Automatically index new memories when inserted
        cur.execute(f"""
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(situation_variant, memory_id)
                SELECT value, new.{FIELD_ID}
                FROM json_each(new.{FIELD_VARIANTS});
            END
        """)

        # Trigger: Remove all FTS index entries when memory is deleted
        cur.execute(f"""
            CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE memory_id = old.{FIELD_ID};
            END
        """)

        # Trigger: Update FTS index when memory is modified
        cur.execute(f"""
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                DELETE FROM memories_fts WHERE memory_id = old.{FIELD_ID};
                INSERT INTO memories_fts(situation_variant, memory_id)
                SELECT value, new.{FIELD_ID}
                FROM json_each(new.{FIELD_VARIANTS});
            END
        """)


def insert_memories(db_path: str, memories: List[Dict[str, Any]]) -> int:
    """
    Insert memories into the database, triggering automatic FTS indexing.

    Args:
        db_path: Path to SQLite database file.
        memories: List of memory dictionaries to insert.

    Returns:
        Count of successfully inserted records.

    Note:
        Uses INSERT OR REPLACE, so duplicate IDs will update existing records.
    """
    with get_db_connection(db_path) as conn:
        cur = conn.cursor()
        inserted = 0

        for mem in memories:
            try:
                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO memories
                    ({FIELD_ID}, {FIELD_SITUATION}, {FIELD_VARIANTS}, {FIELD_LESSON}, {FIELD_METADATA}, {FIELD_SOURCE})
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mem[FIELD_ID],
                        mem[FIELD_SITUATION],
                        _serialize_json_field(mem.get(FIELD_VARIANTS)),
                        mem[FIELD_LESSON],
                        _serialize_json_field(mem.get(FIELD_METADATA)),
                        _serialize_json_field(mem.get(FIELD_SOURCE)),
                    ),
                )
                inserted += 1
            except (KeyError, sqlite3.Error) as e:
                print(f"Warning: Skipping memory {mem.get(FIELD_ID, '?')}: {e}")

    return inserted


def get_memory_count(db_path: str) -> int:
    """
    Get the total number of memories stored in the database.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        Total count of memory records in the database.
    """
    with get_db_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        return cur.fetchone()[0]


def get_memory_by_id(db_path: str, memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific memory by its unique identifier.

    Args:
        db_path: Path to SQLite database file.
        memory_id: Unique memory identifier (e.g., "mem_abc123def456").

    Returns:
        Memory dictionary, or None if not found.
    """
    with get_db_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            f"""
            SELECT {FIELD_ID}, {FIELD_SITUATION}, {FIELD_VARIANTS}, {FIELD_LESSON},
                   {FIELD_METADATA}, {FIELD_SOURCE}
            FROM memories
            WHERE {FIELD_ID} = ?
            """,
            (memory_id,),
        )

        row = cur.fetchone()
        if row is None:
            return None

        return _row_to_memory_dict(row, include_rank=False)


def search_memories(
    db_path: str = DEFAULT_DB_PATH,
    query: str = "",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search memories using FTS5 full-text search on multiple situation variants.

    This function performs keyword-based search across all situation variants
    (3 per memory) and ranks results using the BM25 algorithm. When multiple
    variants of the same memory match, the memory is returned once with the
    best (lowest) rank score.

    Args:
        db_path: Path to SQLite database file.
                 Defaults to "data/phase0/memories/memories.db".
        query: Search query string. Supports FTS5 query syntax:
               - Simple terms: "async function"
               - Boolean: "async AND function", "react OR vue"
               - Negation: "function NOT deprecated"
               - Phrases: '"error handling"'
               - Prefix: "handl*" (matches handle, handler, handling)
        limit: Maximum number of results to return (default: 10).

    Returns:
        List of memory dictionaries ordered by relevance (best matches first).
        Each dict contains:
            - id: Unique memory identifier
            - situation_description: Primary variant for display
            - situation_variants: All 3 variants (array)
            - lesson: Actionable guidance
            - metadata: Dict with repo, language, severity, confidence
            - source: Dict with original code review context
            - rank: BM25 relevance score (LOWER is better - typically negative)

    Example:
        >>> results = search_memories("data/phase0/memories/memories.db", "async error", limit=5)
        >>> for r in results:
        ...     print(f"{r['id']}: {r['rank']:.2f}")
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute(
            f"""
            SELECT m.{FIELD_ID}, m.{FIELD_SITUATION}, m.situation_variants, m.{FIELD_LESSON},
                   m.{FIELD_METADATA}, m.{FIELD_SOURCE},
                   bm25(memories_fts) as {FIELD_RANK}
            FROM memories_fts fts
            JOIN memories m ON m.{FIELD_ID} = fts.memory_id
            WHERE memories_fts MATCH ?
            ORDER BY {FIELD_RANK}
            """,
            (query,),
        )

        # Deduplicate: keep first occurrence of each memory (best rank)
        seen_ids = set()
        results = []
        for row in cur.fetchall():
            mem_id = row[FIELD_ID]
            if mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)

            memory = {
                FIELD_ID: mem_id,
                FIELD_SITUATION: row[FIELD_SITUATION],
                "situation_variants": json.loads(row["situation_variants"]) if row["situation_variants"] else [],
                FIELD_LESSON: row[FIELD_LESSON],
                FIELD_METADATA: json.loads(row[FIELD_METADATA]) if row[FIELD_METADATA] else {},
                FIELD_SOURCE: json.loads(row[FIELD_SOURCE]) if row[FIELD_SOURCE] else {},
                FIELD_RANK: row[FIELD_RANK],
            }
            results.append(memory)

            if len(results) >= limit:
                break

        return results

    finally:
        conn.close()


def rebuild_database(
    db_path: str = DEFAULT_DB_PATH,
    memories_dir: str = DEFAULT_MEMORIES_DIR,
) -> None:
    """
    Rebuild the entire database from scratch using JSONL source files.

    Args:
        db_path: Path where SQLite database should be created.
                 WARNING: Existing database will be replaced!
        memories_dir: Directory containing memories_*.jsonl files.

    Side Effects:
        - Deletes existing database if present
        - Creates new database file
        - Prints progress information to stdout
    """
    print(f"Creating database at {db_path}...")
    create_database(db_path)

    print(f"Loading memories from {memories_dir}...")
    memories = load_memories(memories_dir)
    print(f"Found {len(memories)} memories")

    print("Inserting memories...")
    count = insert_memories(db_path, memories)
    print(f"Inserted {count} memories into database")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "--rebuild":
        print("SQLite FTS5 Memory Search Database Builder")
        print()
        print("Usage:")
        print("  uv run python scripts/phase0/db.py --rebuild")
        print()
        print("Description:")
        print("  Rebuilds the FTS5 search database from memories/*.jsonl files.")
        print("  This is a destructive operation that replaces the existing database.")
        print()
        print(f"  Database location: {DEFAULT_DB_PATH}")
        print(f"  Source JSONL files: {DEFAULT_MEMORIES_DIR}/memories_*.jsonl")
        sys.exit(1)

    rebuild_database()
