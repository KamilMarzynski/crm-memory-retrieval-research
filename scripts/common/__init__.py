"""
Common utilities shared across all phases and scripts.

This module provides generic functionality that is not specific to any
particular experiment phase, including file I/O and LLM API access.

Modules:
    io: File I/O operations (load_json, save_json, ensure_dir)
    openrouter: OpenRouter LLM API client

Usage:
    from common.io import load_json, save_json
    from common.openrouter import call_openrouter
"""

from common.io import load_json, save_json, ensure_dir
from common.openrouter import call_openrouter, OPENROUTER_URL, OPENROUTER_API_KEY_ENV

__all__ = [
    "load_json",
    "save_json",
    "ensure_dir",
    "call_openrouter",
    "OPENROUTER_URL",
    "OPENROUTER_API_KEY_ENV",
]
