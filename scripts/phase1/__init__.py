"""
Phase 1: Vector search experiments for memory retrieval.

This module contains all code specific to Phase 1 experiments, which use
sqlite-vec vector similarity search with embeddings from a local Ollama model.

Modules:
    load_memories: Memory loading and field constants
    db: SQLite + sqlite-vec database operations

Typical Workflow:
    1. Extract memories: uv run python scripts/phase1/build_memories.py --all data/review_data
    2. Build database: uv run python scripts/phase1/db.py --rebuild
    3. Generate test cases: uv run python scripts/phase1/test_cases.py
    4. Run experiments: uv run python scripts/phase1/experiment.py --all

Note:
    Path discovery is now handled by the runs module. Use:
        from common.runs import get_latest_run
        run_dir = get_latest_run("phase1")
"""

from phase1.load_memories import (
    load_memories,
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
