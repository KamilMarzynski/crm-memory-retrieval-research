"""
Memory loading utilities for Phase 0 experiments.

Re-exports shared constants and load_memories from common.load_memories,
and defines Phase 0-specific field constants.

Constants:
    DEFAULT_MEMORIES_DIR: Default directory for Phase 0 memory files
    FIELD_VARIANTS: Situation variants field (Phase 0-specific)
    FIELD_RANK: BM25 rank field (Phase 0-specific)
    FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE: Common fields

Functions:
    load_memories: Load all accepted memories from JSONL files
"""

from common.load_memories import FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE, load_memories

# Default paths
DEFAULT_MEMORIES_DIR = "data/phase0/memories"

# Phase 0-specific field names
FIELD_VARIANTS = "situation_variants"
FIELD_RANK = "rank"

__all__ = [
    "load_memories",
    "DEFAULT_MEMORIES_DIR",
    "FIELD_ID",
    "FIELD_VARIANTS",
    "FIELD_LESSON",
    "FIELD_METADATA",
    "FIELD_SOURCE",
    "FIELD_RANK",
]
