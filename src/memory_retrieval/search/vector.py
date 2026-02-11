import sqlite3
import struct
from contextlib import contextmanager
from typing import Any, Iterator

import ollama
import sqlite_vec

from memory_retrieval.memories.loader import load_memories
from memory_retrieval.memories.schema import (
    FIELD_ID,
    FIELD_SITUATION,
    FIELD_LESSON,
    FIELD_METADATA,
    FIELD_SOURCE,
    FIELD_DISTANCE,
)
from memory_retrieval.search.base import SearchResult
from memory_retrieval.search.db_utils import serialize_json_field, deserialize_json_field


# Ollama configuration
OLLAMA_HOST: str | None = None
OLLAMA_MODEL = "mxbai-embed-large"
VECTOR_DIMENSIONS = 1024

_ollama_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else None


def _serialize_f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def get_embedding(text: str) -> list[float]:
    client = _ollama_client or ollama
    response = client.embed(model=OLLAMA_MODEL, input=text)
    return response["embeddings"][0]


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
    def create_database(self, db_path: str) -> None:
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
                    situation_embedding float[{VECTOR_DIMENSIONS}]
                )
            """)

    def insert_memories(self, db_path: str, memories: list[dict[str, Any]]) -> int:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            inserted = 0

            for i, memory in enumerate(memories):
                memory_id = memory[FIELD_ID]
                situation = memory[FIELD_SITUATION]

                try:
                    print(f"  [{i + 1}/{len(memories)}] Embedding {memory_id}...")
                    embedding = get_embedding(situation)

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
        query_embedding = get_embedding(query)

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
                results.append(SearchResult(
                    id=row[FIELD_ID],
                    situation=row[FIELD_SITUATION],
                    lesson=row[FIELD_LESSON],
                    metadata=deserialize_json_field(row[FIELD_METADATA]),
                    source=deserialize_json_field(row[FIELD_SOURCE]),
                    score=1.0 - distance,  # Invert so higher = better
                    raw_score=distance,
                    score_type="cosine_distance",
                ))

            return results

    def rebuild_database(self, db_path: str, memories_dir: str) -> None:
        print(f"Creating database at {db_path}...")
        self.create_database(db_path)

        print(f"Loading memories from {memories_dir}...")
        memories = load_memories(memories_dir)
        print(f"Found {len(memories)} memories")

        print("Inserting memories (generating embeddings via Ollama)...")
        count = self.insert_memories(db_path, memories)
        print(f"Inserted {count} memories into database")

    def get_memory_count(self, db_path: str) -> int:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]

    def get_memory_by_id(self, db_path: str, memory_id: str) -> dict[str, Any] | None:
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

    def get_random_sample_memories(
        self, db_path: str, n: int = 5
    ) -> list[dict[str, Any]]:
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
