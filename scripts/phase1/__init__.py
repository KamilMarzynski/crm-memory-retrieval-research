"""
Phase 1: Vector search experiments for memory retrieval.

This module contains all code specific to Phase 1 experiments, which use
sqlite-vec vector similarity search with embeddings from a local Ollama model.

Modules:
    load_memories: Memory loading and field constants
    db: SQLite + sqlite-vec database operations

Typical Workflow:
    1. Extract memories: uv run python scripts/phase1/build_memories.py <file>.json
    2. Build database: uv run python scripts/phase1/db.py --rebuild
"""

from phase1.load_memories import (
    load_memories,
    DEFAULT_MEMORIES_DIR,
    FIELD_ID,
    FIELD_SITUATION,
    FIELD_LESSON,
    FIELD_METADATA,
    FIELD_SOURCE,
    FIELD_DISTANCE,
)
from phase1.db import (
    search_memories,
    get_memory_count,
    get_memory_by_id,
    DEFAULT_DB_PATH,
)

__all__ = [
    # memories
    "load_memories",
    "DEFAULT_MEMORIES_DIR",
    "FIELD_ID",
    "FIELD_SITUATION",
    "FIELD_LESSON",
    "FIELD_METADATA",
    "FIELD_SOURCE",
    "FIELD_DISTANCE",
    # db
    "search_memories",
    "get_memory_count",
    "get_memory_by_id",
    "DEFAULT_DB_PATH",
]
