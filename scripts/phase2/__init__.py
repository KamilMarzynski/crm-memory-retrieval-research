"""
Phase 2: Vector search + cross-encoder reranking experiments.

This module adds a reranking step after Phase 1's vector search to improve
precision. Uses bge-reranker-v2-m3 (CrossEncoder) to rescore candidates.

Phase 2 reuses Phase 1's memories, database, and test cases.
Only the experiment runner and reranker are Phase 2-specific.

Modules:
    load_memories: Memory field constants (re-exports Phase 1 + FIELD_RERANK_SCORE)
    reranker: Cross-encoder reranker (bge-reranker-v2-m3)
    experiment: Experiment runner (vector search + reranking)
    test_cases: Test case path resolution (from Phase 1 runs)

Typical Workflow:
    1. Ensure Phase 1 run exists with database and test cases
    2. Run experiments: uv run python scripts/phase2/experiment.py --all
    3. Analyze results in notebooks/phase2/phase2.ipynb
"""

from phase2.load_memories import (
    load_memories,
    FIELD_ID,
    FIELD_SITUATION,
    FIELD_LESSON,
    FIELD_METADATA,
    FIELD_SOURCE,
    FIELD_DISTANCE,
    FIELD_RERANK_SCORE,
)
from phase2.reranker import Reranker

__all__ = [
    # memories
    "load_memories",
    "FIELD_ID",
    "FIELD_SITUATION",
    "FIELD_LESSON",
    "FIELD_METADATA",
    "FIELD_SOURCE",
    "FIELD_DISTANCE",
    "FIELD_RERANK_SCORE",
    # reranker
    "Reranker",
]
