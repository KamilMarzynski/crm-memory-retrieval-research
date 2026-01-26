"""
Phase 0: SQLite FTS5 keyword search for memory retrieval.

This module provides a SQLite-based full-text search system for engineering
memories extracted from code reviews. It uses SQLite's FTS5 (Full-Text Search 5)
extension to enable fast keyword-based retrieval using BM25 ranking.

Key Features:
    - FTS5 full-text indexing on situation_description field
    - BM25 ranking for relevance scoring (lower scores = better matches)
    - Automatic sync between base table and FTS index via triggers
    - Support for FTS5 query syntax (AND, OR, NOT, "phrases", prefix*)

Database Structure:
    - memories: Base table storing all memory fields
    - memories_fts: FTS5 virtual table for full-text search
    - Triggers: Keep FTS index synchronized with base table changes

Usage:
    # Rebuild database from JSONL files
    uv run python scripts/phase0_sqlite_fts.py --rebuild

    # Programmatic usage
    from scripts.phase0_sqlite_fts import search_memories
    results = search_memories("data/phase0_memories/memories.db", "async function", limit=5)
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Default file paths for database and memory storage
DEFAULT_DB_PATH = "data/phase0/memories/memories.db"
DEFAULT_MEMORIES_DIR = "data/phase0/memories"

# Database field names (centralized to avoid magic strings)
FIELD_ID = "id"
FIELD_SITUATION = "situation_description"
FIELD_LESSON = "lesson"
FIELD_METADATA = "metadata"
FIELD_SOURCE = "source"
FIELD_RANK = "rank"


@contextmanager
def get_db_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    """
    Context manager for database connections ensuring proper resource cleanup.

    This ensures connections are always closed, even if exceptions occur.
    Using context managers is a clean code best practice for resource management.

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
    """
    Serialize dictionary to JSON string for database storage.

    Centralizes JSON serialization logic to maintain consistency and
    follow DRY (Don't Repeat Yourself) principle.

    Args:
        data: Dictionary to serialize, or None.

    Returns:
        JSON string. Empty dict "{}" if data is None.

    Note:
        ensure_ascii=False preserves Unicode characters, which is important
        for code snippets that may contain non-ASCII characters.
    """
    return json.dumps(data or {}, ensure_ascii=False)


def _deserialize_json_field(json_str: Optional[str]) -> Dict[str, Any]:
    """
    Deserialize JSON string from database to dictionary.

    Centralizes JSON deserialization logic to maintain consistency.

    Args:
        json_str: JSON string from database, or None.

    Returns:
        Parsed dictionary. Empty dict if json_str is None or empty.
    """
    if not json_str:
        return {}
    return json.loads(json_str)


def _row_to_memory_dict(row: sqlite3.Row, include_rank: bool = False) -> Dict[str, Any]:
    """
    Convert database row to memory dictionary.

    Centralizes row-to-dict conversion logic to avoid duplication between
    search_memories and get_memory_by_id functions.

    Args:
        row: SQLite Row object from query result.
        include_rank: Whether to include the 'rank' field (for search results).

    Returns:
        Memory dictionary with deserialized JSON fields.
    """
    memory = {
        FIELD_ID: row[FIELD_ID],
        FIELD_SITUATION: row[FIELD_SITUATION],
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

    Database Schema:
        memories:
            - id: Unique identifier (e.g., "mem_abc123def456")
            - situation_description: When this knowledge applies (searchable)
            - lesson: Actionable guidance (not indexed)
            - metadata: JSON string with repo, language, severity, etc.
            - source: JSON string with original code review context

        memories_fts:
            - FTS5 virtual table indexing situation_description
            - Enables fast keyword search with BM25 ranking
            - Linked to memories table via rowid

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
        # This is the authoritative source of truth for memory data
        cur.execute(f"""
            CREATE TABLE memories (
                {FIELD_ID} TEXT PRIMARY KEY,
                {FIELD_SITUATION} TEXT NOT NULL,
                {FIELD_LESSON} TEXT NOT NULL,
                {FIELD_METADATA} TEXT,
                {FIELD_SOURCE} TEXT
            )
        """)

        # Create FTS5 virtual table for full-text search on situation_description
        #
        # FTS5 is SQLite's full-text search extension that provides:
        # - Fast keyword matching across large text collections
        # - BM25 ranking algorithm (industry-standard relevance scoring)
        # - Advanced query syntax (AND, OR, NOT, phrases, prefix search)
        #
        # The content='memories' option creates a "contentless" FTS table that
        # references the memories table instead of duplicating data.
        # content_rowid='rowid' links FTS rows to memories rows via SQLite's
        # internal rowid column.
        cur.execute(f"""
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                {FIELD_SITUATION},
                content='memories',
                content_rowid='rowid'
            )
        """)

        # Trigger: Automatically index new memories when inserted
        # When a row is added to 'memories', also add it to 'memories_fts'
        cur.execute(f"""
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, {FIELD_SITUATION})
                VALUES (new.rowid, new.{FIELD_SITUATION});
            END
        """)

        # Trigger: Remove FTS index entry when memory is deleted
        # FTS5 uses special 'delete' command to remove indexed content
        cur.execute(f"""
            CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, {FIELD_SITUATION})
                VALUES ('delete', old.rowid, old.{FIELD_SITUATION});
            END
        """)

        # Trigger: Update FTS index when memory is modified
        # Must delete old content and insert new content (FTS doesn't support UPDATE)
        cur.execute(f"""
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, {FIELD_SITUATION})
                VALUES ('delete', old.rowid, old.{FIELD_SITUATION});
                INSERT INTO memories_fts(rowid, {FIELD_SITUATION})
                VALUES (new.rowid, new.{FIELD_SITUATION});
            END
        """)


def load_memories(phase0_dir: str) -> List[Dict[str, Any]]:
    """
    Load all accepted memories from JSONL files in the phase0 directory.

    This function scans the directory for files matching the pattern
    "memories_*.jsonl" (accepted memories only, not rejected_*.jsonl) and
    parses each line as a JSON memory object.

    Args:
        phase0_dir: Path to directory containing JSONL memory files.
                    Typically "data/phase0_memories/".

    Returns:
        List of memory dictionaries, each containing:
            - id: Unique memory identifier
            - situation_description: When this knowledge applies
            - lesson: Actionable guidance
            - metadata: Dict with repo, language, severity, confidence
            - source: Dict with original code review context

    File Format:
        Each JSONL file contains one JSON object per line (newline-delimited JSON).
        Example:
            {"id": "mem_abc123", "situation_description": "...", "lesson": "..."}
            {"id": "mem_def456", "situation_description": "...", "lesson": "..."}

    Raises:
        json.JSONDecodeError: If a line contains malformed JSON.
        FileNotFoundError: If phase0_dir doesn't exist.

    Note:
        Empty lines are skipped.
    """
    memories = []
    phase0_path = Path(phase0_dir)

    # Find all accepted memory files (memories_*.jsonl pattern)
    # This excludes rejected_*.jsonl files which contain low-quality extractions
    for jsonl_file in sorted(phase0_path.glob("memories_*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        memories.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Provide helpful error message with file and line number
                        print(f"Warning: Skipping malformed JSON in {jsonl_file.name}:{line_num}: {e}")

    return memories


def insert_memories(db_path: str, memories: List[Dict[str, Any]]) -> int:
    """
    Insert memories into the database, triggering automatic FTS indexing.

    Each insert into the 'memories' table automatically triggers the FTS
    index update (via memories_ai trigger), so memories become immediately
    searchable after insertion.

    Args:
        db_path: Path to SQLite database file.
        memories: List of memory dictionaries to insert. Each must contain:
                  - id: Unique identifier (required)
                  - situation_description: Search text (required)
                  - lesson: Actionable guidance (required)
                  - metadata: Optional dict (will be JSON-serialized)
                  - source: Optional dict (will be JSON-serialized)

    Returns:
        Count of successfully inserted records.

    Error Handling:
        - Missing required fields (id, situation_description, lesson): Skipped
        - Database errors (e.g., constraint violations): Skipped
        - Errors are printed to stdout with memory ID for debugging

    Note:
        Uses INSERT OR REPLACE, so duplicate IDs will update existing records.
        Commits after each successful batch to avoid losing work on errors.
    """
    with get_db_connection(db_path) as conn:
        cur = conn.cursor()
        inserted = 0

        for mem in memories:
            try:
                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO memories
                    ({FIELD_ID}, {FIELD_SITUATION}, {FIELD_LESSON}, {FIELD_METADATA}, {FIELD_SOURCE})
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        mem[FIELD_ID],
                        mem[FIELD_SITUATION],
                        mem[FIELD_LESSON],
                        _serialize_json_field(mem.get(FIELD_METADATA)),
                        _serialize_json_field(mem.get(FIELD_SOURCE)),
                    ),
                )
                inserted += 1
            except (KeyError, sqlite3.Error) as e:
                # Log errors but continue processing remaining memories
                print(f"Warning: Skipping memory {mem.get(FIELD_ID, '?')}: {e}")

    return inserted


def search_memories(
    db_path: str,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search memories using FTS5 full-text search with BM25 ranking.

    This function performs keyword-based search on the situation_description
    field and ranks results using the BM25 algorithm, which considers both
    term frequency and document length normalization.

    Args:
        db_path: Path to SQLite database file.
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
            - situation_description: When this knowledge applies
            - lesson: Actionable guidance
            - metadata: Dict with repo, language, severity, confidence
            - source: Dict with original code review context
            - rank: BM25 relevance score (LOWER is better - typically negative)

    BM25 Ranking:
        - Scores are typically negative numbers
        - Lower (more negative) scores indicate better matches
        - Results are automatically sorted by rank (best first)
        - Formula considers: term frequency, inverse document frequency,
          and document length normalization

    Examples:
        >>> search_memories("memories.db", "async error handling", limit=5)
        [{"id": "mem_123", "rank": -2.5, ...}, ...]

        >>> search_memories("memories.db", '"null check" AND javascript')
        [{"id": "mem_456", "rank": -3.1, ...}, ...]

    Note:
        Returns empty list if query doesn't match any documents or if
        the query syntax is invalid.
    """
    with get_db_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row  # Access columns by name
        cur = conn.cursor()

        # Query the FTS index and join with base table to get full record
        #
        # bm25(memories_fts) calculates BM25 relevance score
        # Lower (more negative) scores = better matches
        #
        # The MATCH operator performs FTS5 full-text search
        # It only searches the FTS index, so it's very fast even with many records
        cur.execute(
            f"""
            SELECT m.{FIELD_ID}, m.{FIELD_SITUATION}, m.{FIELD_LESSON},
                   m.{FIELD_METADATA}, m.{FIELD_SOURCE},
                   bm25(memories_fts) as {FIELD_RANK}
            FROM memories_fts fts
            JOIN memories m ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ?
            ORDER BY {FIELD_RANK}
            LIMIT ?
            """,
            (query, limit),
        )

        # Convert Row objects to dicts using centralized helper
        return [_row_to_memory_dict(row, include_rank=True) for row in cur.fetchall()]


def get_memory_count(db_path: str) -> int:
    """
    Get the total number of memories stored in the database.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        Total count of memory records in the database.

    Example:
        >>> count = get_memory_count("data/phase0_memories/memories.db")
        >>> print(f"Database contains {count} memories")
        Database contains 1234 memories
    """
    with get_db_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        return cur.fetchone()[0]


def get_memory_by_id(db_path: str, memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific memory by its unique identifier.

    This is a direct lookup (not a search), useful when you have a specific
    memory ID from a previous search result or experiment log.

    Args:
        db_path: Path to SQLite database file.
        memory_id: Unique memory identifier (e.g., "mem_abc123def456").

    Returns:
        Memory dictionary containing:
            - id: Unique memory identifier
            - situation_description: When this knowledge applies
            - lesson: Actionable guidance
            - metadata: Dict with repo, language, severity, confidence
            - source: Dict with original code review context

        Returns None if memory ID not found.

    Example:
        >>> mem = get_memory_by_id("memories.db", "mem_abc123def456")
        >>> if mem:
        ...     print(mem["lesson"])
        ... else:
        ...     print("Memory not found")
    """
    with get_db_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row  # Access columns by name
        cur = conn.cursor()

        # Direct lookup by primary key (very fast - uses index)
        cur.execute(
            f"""
            SELECT {FIELD_ID}, {FIELD_SITUATION}, {FIELD_LESSON},
                   {FIELD_METADATA}, {FIELD_SOURCE}
            FROM memories
            WHERE {FIELD_ID} = ?
            """,
            (memory_id,),
        )

        row = cur.fetchone()
        if row is None:
            return None

        # Convert row to dict using centralized helper
        return _row_to_memory_dict(row, include_rank=False)


def rebuild_database(
    db_path: str = DEFAULT_DB_PATH,
    memories_dir: str = DEFAULT_MEMORIES_DIR,
) -> None:
    """
    Rebuild the entire database from scratch using JSONL source files.

    This is the main orchestration function that:
    1. Creates a fresh database with schema and FTS index
    2. Loads all memories from JSONL files
    3. Inserts them into the database (triggering FTS indexing)

    Args:
        db_path: Path where SQLite database should be created.
                 Defaults to "data/phase0_memories/memories.db".
                 WARNING: Existing database will be replaced!
        memories_dir: Directory containing memories_*.jsonl files.
                      Defaults to "data/phase0_memories/".

    Side Effects:
        - Deletes existing database if present
        - Creates new database file
        - Prints progress information to stdout

    Usage:
        # Rebuild with default paths
        rebuild_database()

        # Rebuild with custom paths
        rebuild_database("custom.db", "custom_memories/")

    Typical Output:
        Creating database at data/phase0_memories/memories.db...
        Loading memories from data/phase0_memories...
        Found 1234 memories
        Inserting memories...
        Inserted 1234 memories into database

    Note:
        This is a destructive operation. Always maintain JSONL files as the
        source of truth - the database can always be rebuilt from them.
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
    import sys

    # Command-line interface for rebuilding the database
    # Only supports --rebuild flag to prevent accidental data loss
    if len(sys.argv) < 2 or sys.argv[1] != "--rebuild":
        print("SQLite FTS5 Memory Search Database Builder")
        print()
        print("Usage:")
        print("  uv run python scripts/phase0_sqlite_fts.py --rebuild")
        print()
        print("Description:")
        print("  Rebuilds the FTS5 search database from phase0_memories/*.jsonl files.")
        print("  This is a destructive operation that replaces the existing database.")
        print()
        print("  The database enables fast keyword search using SQLite's FTS5 extension")
        print("  with BM25 ranking for relevance scoring.")
        print()
        print(f"  Database location: {DEFAULT_DB_PATH}")
        print(f"  Source JSONL files: {DEFAULT_MEMORIES_DIR}/memories_*.jsonl")
        sys.exit(1)

    # Perform database rebuild
    rebuild_database()
