"""
Shared query generation utilities for retrieval experiments.

Provides robust parsing of LLM-generated queries and shared configuration
constants used by both Phase 1 and Phase 2 experiment runners.

Functions:
    parse_queries_robust: Parse query list from LLM response with fallback strategies.

Constants:
    Query generation model defaults, length limits, and search configuration.
"""

import json
import re
from typing import List

# Model configuration
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1500

# Query generation limits
MAX_CONTEXT_LENGTH = 3000
MAX_DIFF_LENGTH = 12000
MAX_QUERIES_PER_EXPERIMENT = 20

# Query length targets (aligned with memory length for vector similarity)
TARGET_QUERY_WORDS_MIN = 20
TARGET_QUERY_WORDS_MAX = 50

# Search configuration
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_DISTANCE_THRESHOLD = 1.1

# Batch execution configuration
DEFAULT_SLEEP_BETWEEN_EXPERIMENTS = 1.0


def parse_queries_robust(response: str) -> List[str]:
    """
    Robustly parse query list from LLM response.

    Handles various output formats and malformed JSON with multiple
    fallback strategies:
        1. Try JSON array extraction
        2. Fallback to quoted string extraction (20-200 chars)
        3. Last resort: line-by-line parsing

    Args:
        response: Raw LLM response string.

    Returns:
        List of query strings, capped at MAX_QUERIES_PER_EXPERIMENT.
    """
    # Try direct JSON parse first
    try:
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:MAX_QUERIES_PER_EXPERIMENT]
    except json.JSONDecodeError:
        pass

    # Fallback: extract quoted strings (20-200 chars for reasonable queries)
    quoted = re.findall(r'"([^"]{20,200})"', response)
    if quoted:
        return quoted[:MAX_QUERIES_PER_EXPERIMENT]

    # Last resort: split by newlines and clean
    lines = []
    for line in response.split('\n'):
        line = line.strip()
        line = re.sub(r'^[\d\.\-\*]+\s*', '', line)  # Remove list markers
        line = line.strip('"\'')
        if 20 <= len(line) <= 200:
            lines.append(line)

    return lines[:MAX_QUERIES_PER_EXPERIMENT]


__all__ = [
    "parse_queries_robust",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "MAX_CONTEXT_LENGTH",
    "MAX_DIFF_LENGTH",
    "MAX_QUERIES_PER_EXPERIMENT",
    "TARGET_QUERY_WORDS_MIN",
    "TARGET_QUERY_WORDS_MAX",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_DISTANCE_THRESHOLD",
    "DEFAULT_SLEEP_BETWEEN_EXPERIMENTS",
]
