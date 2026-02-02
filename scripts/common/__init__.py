"""
Common utilities shared across all phases and scripts.

This module provides generic functionality that is not specific to any
particular experiment phase, including file I/O, LLM API access, memory
loading, and test case generation.

Modules:
    io: File I/O operations (load_json, save_json, ensure_dir)
    load_memories: Memory loading from JSONL files and shared field constants
    openrouter: OpenRouter LLM API client
    test_cases: Test case generation from raw PR data and memories

Usage:
    from common.io import load_json, save_json
    from common.load_memories import load_memories, FIELD_ID
    from common.openrouter import call_openrouter
    from common.test_cases import build_test_cases
"""

from common.io import load_json, save_json, ensure_dir
from common.load_memories import load_memories, FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE
from common.openrouter import call_openrouter, OPENROUTER_URL, OPENROUTER_API_KEY_ENV
from common.prompts import load_prompt, Prompt
from common.test_cases import (
    build_test_cases,
    build_test_case,
    filter_diff,
    get_ground_truth_memory_ids,
    FIELD_TEST_CASE_ID,
    FIELD_SOURCE_FILE,
    FIELD_PR_CONTEXT,
    FIELD_FILTERED_DIFF,
    FIELD_DIFF_STATS,
    FIELD_GROUND_TRUTH_IDS,
    FIELD_GROUND_TRUTH_COUNT,
    TEST_CASE_ID_PREFIX,
    EXCLUDED_FILE_PATTERNS,
)

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
    "load_prompt",
    "Prompt",
    "build_test_cases",
    "build_test_case",
    "filter_diff",
    "get_ground_truth_memory_ids",
    "FIELD_TEST_CASE_ID",
    "FIELD_SOURCE_FILE",
    "FIELD_PR_CONTEXT",
    "FIELD_FILTERED_DIFF",
    "FIELD_DIFF_STATS",
    "FIELD_GROUND_TRUTH_IDS",
    "FIELD_GROUND_TRUTH_COUNT",
    "TEST_CASE_ID_PREFIX",
    "EXCLUDED_FILE_PATTERNS",
]
