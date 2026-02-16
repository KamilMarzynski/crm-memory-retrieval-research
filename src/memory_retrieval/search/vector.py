import sqlite3
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import ollama
import sqlite_vec

from memory_retrieval.memories.loader import load_memories
from memory_retrieval.memories.schema import (
    FIELD_DISTANCE,
    FIELD_ID,
    FIELD_LESSON,
    FIELD_METADATA,
    FIELD_SITUATION,
    FIELD_SOURCE,
)
from memory_retrieval.search.base import SearchResult
from memory_retrieval.search.db_utils import deserialize_json_field, serialize_json_field

DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_VECTOR_DIMENSIONS = 1024

# Known model → dimension mapping for automatic configuration
EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    "mxbai-embed-large": 1024,
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-large": 1024,
    "bge-m3": 1024,
}


def _serialize_f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


@contextmanager
def get_db_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    _load_sqlite_vec(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_confidence_from_distance(distance: float) -> str:
    if distance < 0.5:
        return "high"
    elif distance < 0.8:
        return "medium"
    elif distance < 1.2:
        return "low"
    else:
        return "very_low"


class VectorBackend:
    """Vector search backend using sqlite-vec for embedding-based retrieval.

    Args:
        embedding_model: Ollama model name for generating embeddings.
            Defaults to DEFAULT_EMBEDDING_MODEL ("mxbai-embed-large").
        vector_dimensions: Embedding dimension size. If None, auto-detected from
            EMBEDDING_MODEL_DIMENSIONS or falls back to DEFAULT_VECTOR_DIMENSIONS.
        ollama_host: Custom Ollama host URL. If None, uses the default client.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        vector_dimensions: int | None = None,
        ollama_host: str | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_dimensions = vector_dimensions or EMBEDDING_MODEL_DIMENSIONS.get(
            embedding_model, DEFAULT_VECTOR_DIMENSIONS
        )
        self._ollama_client: ollama.Client | None = (
            ollama.Client(host=ollama_host) if ollama_host else None
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using this backend's configured model."""
        client = self._ollama_client or ollama
        response = client.embed(model=self.embedding_model, input=text)
        return response["embeddings"][0]

    def create_database(self, db_path: str) -> None:
        """Create database schema with memories table and vector index.

        Args:
            db_path: Path to the SQLite database file.
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("DROP TABLE IF EXISTS vec_memories")
            cursor.execute("DROP TABLE IF EXISTS memories")

            cursor.execute(f"""
                CREATE TABLE memories (
                    {FIELD_ID} TEXT PRIMARY KEY,
                    {FIELD_SITUATION} TEXT NOT NULL,
                    {FIELD_LESSON} TEXT NOT NULL,
                    {FIELD_METADATA} TEXT,
                    {FIELD_SOURCE} TEXT
                )
            """)

            cursor.execute(f"""
                CREATE VIRTUAL TABLE vec_memories USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    situation_embedding float[{self.vector_dimensions}]
                )
            """)

    def insert_memories(self, db_path: str, memories: list[dict[str, Any]]) -> int:
        """Insert memories into the database and generate embeddings.

        Args:
            db_path: Path to the SQLite database file.
            memories: List of memory dictionaries with id, situation, lesson, metadata, and source fields.

        Returns:
            Number of memories successfully inserted.
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            inserted = 0

            for i, memory in enumerate(memories):
                memory_id = memory[FIELD_ID]
                situation = memory[FIELD_SITUATION]

                try:
                    print(f"  [{i + 1}/{len(memories)}] Embedding {memory_id}...")
                    embedding = self._get_embedding(situation)

                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO memories
                        ({FIELD_ID}, {FIELD_SITUATION}, {FIELD_LESSON}, {FIELD_METADATA}, {FIELD_SOURCE})
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            memory_id,
                            situation,
                            memory[FIELD_LESSON],
                            serialize_json_field(memory.get(FIELD_METADATA)),
                            serialize_json_field(memory.get(FIELD_SOURCE)),
                        ),
                    )

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO vec_memories (memory_id, situation_embedding)
                        VALUES (?, ?)
                        """,
                        (memory_id, _serialize_f32(embedding)),
                    )

                    inserted += 1
                except (KeyError, sqlite3.Error, ollama.ResponseError) as e:
                    print(f"  Warning: Skipping memory {memory_id}: {e}")

        return inserted

    def search(self, db_path: str, query: str, limit: int = 10) -> list[SearchResult]:
        """Search for memories using vector similarity.

        Args:
            db_path: Path to the SQLite database file.
            query: Search query text to embed and match against.
            limit: Maximum number of results to return (uses k parameter in sqlite-vec).

        Returns:
            List of SearchResult objects sorted by cosine distance (ascending).
        """
        query_embedding = self._get_embedding(query)

        with get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT m.{FIELD_ID}, m.{FIELD_SITUATION}, m.{FIELD_LESSON},
                       m.{FIELD_METADATA}, m.{FIELD_SOURCE},
                       v.distance as {FIELD_DISTANCE}
                FROM vec_memories v
                JOIN memories m ON m.{FIELD_ID} = v.memory_id
                WHERE v.situation_embedding MATCH ?
                    AND k = ?
                ORDER BY v.distance
                """,
                (_serialize_f32(query_embedding), limit),
            )

            results: list[SearchResult] = []
            for row in cursor.fetchall():
                distance = row[FIELD_DISTANCE]
                results.append(
                    SearchResult(
                        id=row[FIELD_ID],
                        situation=row[FIELD_SITUATION],
                        lesson=row[FIELD_LESSON],
                        metadata=deserialize_json_field(row[FIELD_METADATA]),
                        source=deserialize_json_field(row[FIELD_SOURCE]),
                        score=1.0 - distance,  # Invert so higher = better
                        raw_score=distance,
                        score_type="cosine_distance",
                    )
                )

            return results

    def rebuild_database(self, db_path: str, memories_dir: str) -> None:
        """Rebuild the entire database from scratch by loading memories and generating embeddings.

        Args:
            db_path: Path to the SQLite database file to create/overwrite.
            memories_dir: Directory containing JSONL memory files.
        """
        print(f"Creating database at {db_path}...")
        self.create_database(db_path)

        print(f"Loading memories from {memories_dir}...")
        memories = load_memories(memories_dir)
        print(f"Found {len(memories)} memories")

        print(
            f"Inserting memories (embedding via Ollama/{self.embedding_model}, dim={self.vector_dimensions})..."
        )
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
            Memory dictionary with all fields, or None if not found.
        """
        with get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {FIELD_ID}, {FIELD_SITUATION}, {FIELD_LESSON},
                       {FIELD_METADATA}, {FIELD_SOURCE}
                FROM memories WHERE {FIELD_ID} = ?
                """,
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                FIELD_ID: row[FIELD_ID],
                FIELD_SITUATION: row[FIELD_SITUATION],
                FIELD_LESSON: row[FIELD_LESSON],
                FIELD_METADATA: deserialize_json_field(row[FIELD_METADATA]),
                FIELD_SOURCE: deserialize_json_field(row[FIELD_SOURCE]),
            }

    def get_sample_memories(self, db_path: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get the first N memories from the database (not randomized).

        Args:
            db_path: Path to the SQLite database file.
            limit: Maximum number of memories to return.

        Returns:
            List of memory dictionaries.
        """
        with get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {FIELD_ID}, {FIELD_SITUATION}, {FIELD_LESSON},
                       {FIELD_METADATA}, {FIELD_SOURCE}
                FROM memories LIMIT ?
                """,
                (limit,),
            )
            return [
                {
                    FIELD_ID: row[FIELD_ID],
                    FIELD_SITUATION: row[FIELD_SITUATION],
                    FIELD_LESSON: row[FIELD_LESSON],
                    FIELD_METADATA: deserialize_json_field(row[FIELD_METADATA]),
                    FIELD_SOURCE: deserialize_json_field(row[FIELD_SOURCE]),
                }
                for row in cursor.fetchall()
            ]

    def get_random_sample_memories(self, db_path: str, n: int = 5) -> list[dict[str, Any]]:
        """Get N random memories from the database for use in prompt examples.

        Args:
            db_path: Path to the SQLite database file.
            n: Number of random memories to return.

        Returns:
            List of memory dictionaries (id, situation, lesson only).
        """
        with get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {FIELD_ID}, {FIELD_SITUATION}, {FIELD_LESSON}
                FROM memories ORDER BY RANDOM() LIMIT ?
                """,
                (n,),
            )
            return [
                {
                    FIELD_ID: row[FIELD_ID],
                    FIELD_SITUATION: row[FIELD_SITUATION],
                    FIELD_LESSON: row[FIELD_LESSON],
                }
                for row in cursor.fetchall()
            ]
