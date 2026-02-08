"""
Memory loading utilities for Phase 2 experiments.

Re-exports all Phase 1 fields and adds Phase 2-specific constants
for reranker scores.

Constants:
    FIELD_RERANK_SCORE: Reranker relevance score field (Phase 2-specific)
    FIELD_SITUATION: Situation description field (from Phase 1)
    FIELD_DISTANCE: Vector distance field (from Phase 1)
    FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE: Common fields

Functions:
    load_memories: Load all accepted memories from JSONL files

Note:
    Phase 2 reuses Phase 1's memories and database.
    Use Phase 1 run paths for memory/DB access:
        from common.runs import get_latest_run, PHASE1
        phase1_run = get_latest_run(PHASE1)
        db_path = phase1_run / "memories" / "memories.db"
"""

from phase1.load_memories import (
    FIELD_ID,
    FIELD_SITUATION,
    FIELD_LESSON,
    FIELD_METADATA,
    FIELD_SOURCE,
    FIELD_DISTANCE,
    load_memories,
)

# Phase 2-specific field names
FIELD_RERANK_SCORE = "rerank_score"

__all__ = [
    "load_memories",
    "FIELD_ID",
    "FIELD_SITUATION",
    "FIELD_LESSON",
    "FIELD_METADATA",
    "FIELD_SOURCE",
    "FIELD_DISTANCE",
    "FIELD_RERANK_SCORE",
]
