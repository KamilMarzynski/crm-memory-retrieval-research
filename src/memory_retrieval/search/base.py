import sqlite3
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from memory_retrieval.types import ScoreType


@dataclass
class SearchResult:
    id: str
    situation: str
    lesson: str
    metadata: dict[str, Any]
    source: dict[str, Any]
    score: float  # Higher = better (normalized)
    raw_score: float  # Original (BM25 rank or cosine distance)
    score_type: ScoreType  # Tightened from str


class SearchBackend(Protocol):
    def create_database(self, db_path: str) -> None: ...
    def insert_memories(self, db_path: str, memories: list[dict[str, Any]]) -> int: ...
    def search(self, db_path: str, query: str, limit: int = 10) -> list[SearchResult]: ...
    def rebuild_database(self, db_path: str, memories_dir: str) -> None: ...


class SearchBackendBase(ABC):
    """Abstract base class providing shared implementation for search backends.

    Subclasses must implement `_get_db_connection` to provide their backend-specific
    SQLite connection context manager (e.g. plain SQLite vs. sqlite-vec).
    """

    @abstractmethod
    def _get_db_connection(self, db_path: str) -> AbstractContextManager[sqlite3.Connection]:
        """Return a context manager that yields an open SQLite connection."""
        ...

    def get_memory_by_id(self, db_path: str, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by its ID.

        Returns the memory dict with id, situation/variants, lesson, metadata, source,
        or None if not found.
        """
        from memory_retrieval.memories.schema import (
            FIELD_ID,
            FIELD_LESSON,
            FIELD_METADATA,
            FIELD_SITUATION,
            FIELD_SOURCE,
        )
        from memory_retrieval.search.db_utils import deserialize_json_field

        with self._get_db_connection(db_path) as conn:
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
        """Get the first N memories from the database (deterministic order)."""
        from memory_retrieval.memories.schema import (
            FIELD_ID,
            FIELD_LESSON,
            FIELD_METADATA,
            FIELD_SITUATION,
            FIELD_SOURCE,
        )
        from memory_retrieval.search.db_utils import deserialize_json_field

        with self._get_db_connection(db_path) as conn:
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
