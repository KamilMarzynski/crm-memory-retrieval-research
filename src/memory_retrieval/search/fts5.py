import json
import sqlite3
from contextlib import AbstractContextManager
from typing import Any

from memory_retrieval.memories.loader import load_memories
from memory_retrieval.memories.schema import (
    FIELD_ID,
    FIELD_LESSON,
    FIELD_METADATA,
    FIELD_RANK,
    FIELD_SOURCE,
    FIELD_VARIANTS,
)
from memory_retrieval.search.base import SearchBackendBase, SearchResult
from memory_retrieval.search.db_utils import (
    deserialize_json_field,
    get_db_connection,
    serialize_json_field,
)


class FTS5Backend(SearchBackendBase):
    """Full-text search backend using SQLite FTS5 for keyword-based retrieval."""

    def _get_db_connection(self, db_path: str) -> AbstractContextManager[sqlite3.Connection]:
        """Return a context manager for a plain SQLite connection."""
        return get_db_connection(db_path)

    def create_database(self, db_path: str) -> None:
        """Create database schema with memories table and FTS5 index.

        Sets up automatic triggers to sync FTS5 index with the memories table,
        handling situation variants (multiple text representations per memory).

        Args:
            db_path: Path to the SQLite database file.
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("DROP TABLE IF EXISTS memories_fts")
            cursor.execute("DROP TABLE IF EXISTS memories")

            cursor.execute(f"""
                CREATE TABLE memories (
                    {FIELD_ID} TEXT PRIMARY KEY,
                    {FIELD_VARIANTS} TEXT NOT NULL,
                    {FIELD_LESSON} TEXT NOT NULL,
                    {FIELD_METADATA} TEXT,
                    {FIELD_SOURCE} TEXT
                )
            """)

            cursor.execute("""
                CREATE VIRTUAL TABLE memories_fts USING fts5(
                    situation_variant,
                    memory_id UNINDEXED
                )
            """)

            cursor.execute(f"""
                CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(situation_variant, memory_id)
                    SELECT value, new.{FIELD_ID}
                    FROM json_each(new.{FIELD_VARIANTS});
                END
            """)

            cursor.execute(f"""
                CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE memory_id = old.{FIELD_ID};
                END
            """)

            cursor.execute(f"""
                CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                    DELETE FROM memories_fts WHERE memory_id = old.{FIELD_ID};
                    INSERT INTO memories_fts(situation_variant, memory_id)
                    SELECT value, new.{FIELD_ID}
                    FROM json_each(new.{FIELD_VARIANTS});
                END
            """)

    def insert_memories(self, db_path: str, memories: list[dict[str, Any]]) -> int:
        """Insert memories into the database (FTS5 index updated via triggers).

        Args:
            db_path: Path to the SQLite database file.
            memories: List of memory dictionaries with id, variants, lesson, metadata, and source fields.

        Returns:
            Number of memories successfully inserted.
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            inserted = 0

            for memory in memories:
                try:
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO memories
                        ({FIELD_ID}, {FIELD_VARIANTS}, {FIELD_LESSON}, {FIELD_METADATA}, {FIELD_SOURCE})
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            memory[FIELD_ID],
                            serialize_json_field(memory.get(FIELD_VARIANTS)),
                            memory[FIELD_LESSON],
                            serialize_json_field(memory.get(FIELD_METADATA)),
                            serialize_json_field(memory.get(FIELD_SOURCE)),
                        ),
                    )
                    inserted += 1
                except (KeyError, sqlite3.Error) as e:
                    print(f"Warning: Skipping memory {memory.get(FIELD_ID, '?')}: {e}")

        return inserted

    def search(self, db_path: str, query: str, limit: int = 10) -> list[SearchResult]:
        """Search for memories using FTS5 full-text search with BM25 ranking.

        Deduplicates results since each memory can have multiple situation variants
        that may all match the query.

        Args:
            db_path: Path to the SQLite database file.
            query: FTS5 search query (supports operators like AND, OR, NOT, NEAR).
            limit: Maximum number of unique memories to return.

        Returns:
            List of SearchResult objects sorted by BM25 rank (best matches first).
        """
        with get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT m.{FIELD_ID}, m.{FIELD_VARIANTS}, m.{FIELD_LESSON},
                       m.{FIELD_METADATA}, m.{FIELD_SOURCE},
                       bm25(memories_fts) as {FIELD_RANK}
                FROM memories_fts fts
                JOIN memories m ON m.{FIELD_ID} = fts.memory_id
                WHERE memories_fts MATCH ?
                ORDER BY {FIELD_RANK}
                """,
                (query,),
            )

            seen_ids: set[str] = set()
            results: list[SearchResult] = []
            for row in cursor.fetchall():
                memory_id = row[FIELD_ID]
                if memory_id in seen_ids:
                    continue
                seen_ids.add(memory_id)

                variants = json.loads(row[FIELD_VARIANTS]) if row[FIELD_VARIANTS] else []
                rank = row[FIELD_RANK]

                results.append(
                    SearchResult(
                        id=memory_id,
                        situation=variants[0] if variants else "",
                        lesson=row[FIELD_LESSON],
                        metadata=deserialize_json_field(row[FIELD_METADATA]),
                        source=deserialize_json_field(row[FIELD_SOURCE]),
                        score=-rank,  # BM25 rank is negative; higher = better
                        raw_score=rank,
                        score_type="bm25_rank",
                    )
                )

                if len(results) >= limit:
                    break

            return results

    def rebuild_database(self, db_path: str, memories_dir: str) -> None:
        """Rebuild the entire database from scratch by loading memories.

        Args:
            db_path: Path to the SQLite database file to create/overwrite.
            memories_dir: Directory containing JSONL memory files.
        """
        print(f"Creating database at {db_path}...")
        self.create_database(db_path)

        print(f"Loading memories from {memories_dir}...")
        memories = load_memories(memories_dir)
        print(f"Found {len(memories)} memories")

        print("Inserting memories...")
        count = self.insert_memories(db_path, memories)
        print(f"Inserted {count} memories into database")

    def get_memory_count(self, db_path: str) -> int:
        """Get the total number of memories in the database.

        Args:
            db_path: Path to the SQLite database file.

        Returns:
            Total count of memories.
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]

    def get_memory_by_id(self, db_path: str, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by its ID.

        Args:
            db_path: Path to the SQLite database file.
            memory_id: The unique memory identifier.

        Returns:
            Memory dictionary with all fields including variants, or None if not found.
        """
        with get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {FIELD_ID}, {FIELD_VARIANTS}, {FIELD_LESSON},
                       {FIELD_METADATA}, {FIELD_SOURCE}
                FROM memories WHERE {FIELD_ID} = ?
                """,
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            variants = json.loads(row[FIELD_VARIANTS]) if row[FIELD_VARIANTS] else []
            return {
                FIELD_ID: row[FIELD_ID],
                FIELD_VARIANTS: variants,
                FIELD_LESSON: row[FIELD_LESSON],
                FIELD_METADATA: deserialize_json_field(row[FIELD_METADATA]),
                FIELD_SOURCE: deserialize_json_field(row[FIELD_SOURCE]),
            }
