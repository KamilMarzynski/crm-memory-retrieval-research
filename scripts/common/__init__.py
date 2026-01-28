"""
Common utilities shared across all phases and scripts.

This module provides generic functionality that is not specific to any
particular experiment phase, including file I/O, LLM API access, and
memory loading.

Modules:
    io: File I/O operations (load_json, save_json, ensure_dir)
    load_memories: Memory loading from JSONL files and shared field constants
    openrouter: OpenRouter LLM API client

Usage:
    from common.io import load_json, save_json
    from common.load_memories import load_memories, FIELD_ID
    from common.openrouter import call_openrouter
"""

from common.io import load_json, save_json, ensure_dir
from common.load_memories import load_memories, FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE
from common.openrouter import call_openrouter, OPENROUTER_URL, OPENROUTER_API_KEY_ENV

__all__ = [
    "load_json",
    "save_json",
    "ensure_dir",
    "load_memories",
    "FIELD_ID",
    "FIELD_LESSON",
    "FIELD_METADATA",
    "FIELD_SOURCE",
    "call_openrouter",
    "OPENROUTER_URL",
    "OPENROUTER_API_KEY_ENV",
]
