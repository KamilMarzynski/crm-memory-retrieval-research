"""
Phase 0: SQLite FTS5 keyword search for memory retrieval.

Loads memories from phase0 JSONL files into SQLite database with
FTS5 full-text search on situation_description field.

Usage:
    uv run python scripts/phase0_sqlite_fts.py --rebuild
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def create_database(db_path: str) -> None:
    """Create SQLite database with memories table and FTS5 index."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS memories_fts")
    cur.execute("DROP TABLE IF EXISTS memories")

    cur.execute("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            situation_description TEXT NOT NULL,
            lesson TEXT NOT NULL,
            metadata TEXT,
            source TEXT
        )
    """)

    cur.execute("""
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            situation_description,
            content='memories',
            content_rowid='rowid'
        )
    """)

    cur.execute("""
        CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, situation_description)
            VALUES (new.rowid, new.situation_description);
        END
    """)

    cur.execute("""
        CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, situation_description)
            VALUES ('delete', old.rowid, old.situation_description);
        END
    """)

    cur.execute("""
        CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, situation_description)
            VALUES ('delete', old.rowid, old.situation_description);
            INSERT INTO memories_fts(rowid, situation_description)
            VALUES (new.rowid, new.situation_description);
        END
    """)

    conn.commit()
    conn.close()


def load_memories(phase0_dir: str) -> List[Dict[str, Any]]:
    """Load all memories from JSONL files in phase0 directory."""
    memories = []
    phase0_path = Path(phase0_dir)

    for jsonl_file in phase0_path.glob("memories_*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    memories.append(json.loads(line))

    return memories


def insert_memories(db_path: str, memories: List[Dict[str, Any]]) -> int:
    """Insert memories into database. Returns count of inserted records."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    inserted = 0
    for mem in memories:
        try:
            cur.execute(
                """
                INSERT OR REPLACE INTO memories (id, situation_description, lesson, metadata, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    mem["id"],
                    mem["situation_description"],
                    mem["lesson"],
                    json.dumps(mem.get("metadata", {}), ensure_ascii=False),
                    json.dumps(mem.get("source", {}), ensure_ascii=False),
                ),
            )
            inserted += 1
        except (KeyError, sqlite3.Error) as e:
            print(f"Skipping memory {mem.get('id', '?')}: {e}")

    conn.commit()
    conn.close()
    return inserted


def search_memories(
    db_path: str,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search memories using FTS5 full-text search.

    Args:
        db_path: Path to SQLite database
        query: Search query (supports FTS5 syntax: AND, OR, NOT, "phrase", prefix*)
        limit: Maximum number of results

    Returns:
        List of memory dicts with added 'rank' field (lower is better match)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT m.id, m.situation_description, m.lesson, m.metadata, m.source,
               bm25(memories_fts) as rank
        FROM memories_fts fts
        JOIN memories m ON m.rowid = fts.rowid
        WHERE memories_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    )

    results = []
    for row in cur.fetchall():
        results.append({
            "id": row["id"],
            "situation_description": row["situation_description"],
            "lesson": row["lesson"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "source": json.loads(row["source"]) if row["source"] else {},
            "rank": row["rank"],
        })

    conn.close()
    return results


def get_memory_count(db_path: str) -> int:
    """Return total number of memories in database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM memories")
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_memory_by_id(db_path: str, memory_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single memory by its ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT id, situation_description, lesson, metadata, source FROM memories WHERE id = ?",
        (memory_id,),
    )

    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "situation_description": row["situation_description"],
        "lesson": row["lesson"],
        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        "source": json.loads(row["source"]) if row["source"] else {},
    }


# Default paths
DEFAULT_DB_PATH = "data/phase0_memories/memories.db"
DEFAULT_MEMORIES_DIR = "data/phase0_memories"


def rebuild_database(
    db_path: str = DEFAULT_DB_PATH,
    memories_dir: str = DEFAULT_MEMORIES_DIR,
) -> None:
    """Rebuild the database from all JSONL files in memories directory."""
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

    if len(sys.argv) < 2 or sys.argv[1] != "--rebuild":
        print("Usage: uv run python scripts/phase0_sqlite_fts.py --rebuild")
        print("Rebuilds the FTS5 search database from phase0_memories/*.jsonl files")
        sys.exit(1)

    rebuild_database()
