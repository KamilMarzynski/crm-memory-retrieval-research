"""
Memory loading utilities for Phase 1 experiments.

Re-exports shared constants and load_memories from common.load_memories,
and defines Phase 1-specific field constants.

Constants:
    DEFAULT_MEMORIES_DIR: Default directory for Phase 1 memory files
    FIELD_SITUATION: Situation description field (Phase 1-specific)
    FIELD_DISTANCE: Vector distance field (Phase 1-specific)
    FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE: Common fields

Functions:
    load_memories: Load all accepted memories from JSONL files
"""

from common.load_memories import FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE, load_memories

# Default paths
DEFAULT_MEMORIES_DIR = "data/phase1/memories"

# Phase 1-specific field names
FIELD_SITUATION = "situation_description"
FIELD_DISTANCE = "distance"

__all__ = [
    "load_memories",
    "DEFAULT_MEMORIES_DIR",
    "FIELD_ID",
    "FIELD_SITUATION",
    "FIELD_LESSON",
    "FIELD_METADATA",
    "FIELD_SOURCE",
    "FIELD_DISTANCE",
]
