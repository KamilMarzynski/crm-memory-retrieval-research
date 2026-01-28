"""
Phase 0: Keyword search experiments for memory retrieval.

This module contains all code specific to Phase 0 experiments, which validate
memory retrieval using SQLite FTS5 full-text search with BM25 ranking.

Modules:
    memories: Memory loading and field constants
    db: SQLite FTS5 database operations
    test_cases: Test case generation from raw PR data
    experiment: Retrieval experiment runner

Typical Workflow:
    1. Extract memories: uv run python scripts/phase0/build_memories.py <file>.json
    2. Build database: uv run python scripts/phase0/db.py --rebuild
    3. Generate test cases: uv run python scripts/phase0/test_cases.py
    4. Run experiments: uv run python scripts/phase0/experiment.py --all
"""

from phase0.memories import (
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
from phase0.db import (
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
    "FIELD_VARIANTS",
    "FIELD_LESSON",
    "FIELD_METADATA",
    "FIELD_SOURCE",
    "FIELD_RANK",
    # db
    "search_memories",
    "get_memory_count",
    "get_memory_by_id",
    "DEFAULT_DB_PATH",
]
