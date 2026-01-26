"""
Phase 0: Common utilities shared across phase0 scripts.

This module provides shared functionality for Phase 0 experiments including
file I/O operations, memory loading, and database search operations.

Functions are designed to be reusable across all phase0 scripts without
creating circular dependencies.

Usage:
    from phase0_common import load_json, save_json, load_memories, search_memories
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

# Default paths
DEFAULT_MEMORIES_DIR = "data/phase0/memories"
DEFAULT_DB_PATH = "data/phase0/memories/memories.db"

# Database field names
FIELD_ID = "id"
FIELD_SITUATION = "situation_description"
FIELD_LESSON = "lesson"
FIELD_METADATA = "metadata"
FIELD_SOURCE = "source"
FIELD_RANK = "rank"


def load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON file from disk.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        json.JSONDecodeError: If file contains malformed JSON.
        FileNotFoundError: If file doesn't exist.

    Example:
        >>> data = load_json("data/test_cases/pr_123.json")
        >>> print(data["test_case_id"])
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save dictionary as JSON file.

    Centralizes JSON serialization with consistent formatting.

    Args:
        data: Dictionary to save.
        path: Path where file should be written.

    Note:
        Uses ensure_ascii=False to preserve Unicode characters.
        Uses indent=2 for human-readable formatting.

    Example:
        >>> data = {"test_case_id": "tc_123", "recall": 0.85}
        >>> save_json(data, Path("results/experiment.json"))
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_memories(phase0_dir: str = DEFAULT_MEMORIES_DIR) -> List[Dict[str, Any]]:
    """
    Load all accepted memories from JSONL files in the phase0 directory.

    This function scans the directory for files matching the pattern
    "memories_*.jsonl" (accepted memories only, not rejected_*.jsonl) and
    parses each line as a JSON memory object.

    Args:
        phase0_dir: Path to directory containing JSONL memory files.
                    Defaults to "data/phase0/memories".

    Returns:
        List of memory dictionaries, each containing:
            - id: Unique memory identifier
            - situation_description: When this knowledge applies
            - lesson: Actionable guidance
            - metadata: Dict with repo, language, severity, confidence
            - source: Dict with original code review context

    Raises:
        FileNotFoundError: If phase0_dir doesn't exist.

    Example:
        >>> memories = load_memories("data/phase0/memories")
        >>> print(f"Loaded {len(memories)} memories")
        Loaded 13 memories

    Note:
        Empty lines are skipped. Malformed JSON lines generate warnings.
    """
    memories = []
    phase0_path = Path(phase0_dir)

    # Find all accepted memory files (memories_*.jsonl pattern)
    # This excludes rejected_*.jsonl files which contain low-quality extractions
    for jsonl_file in sorted(phase0_path.glob("memories_*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        memories.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Provide helpful error message with file and line number
                        print(f"Warning: Skipping malformed JSON in {jsonl_file.name}:{line_num}: {e}")

    return memories


def search_memories(
    db_path: str = DEFAULT_DB_PATH,
    query: str = "",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search memories using FTS5 full-text search with BM25 ranking.

    This function performs keyword-based search on the situation_description
    field and ranks results using the BM25 algorithm.

    Args:
        db_path: Path to SQLite database file.
                 Defaults to "data/phase0/memories/memories.db".
        query: Search query string. Supports FTS5 query syntax:
               - Simple terms: "async function"
               - Boolean: "async AND function", "react OR vue"
               - Negation: "function NOT deprecated"
               - Phrases: '"error handling"'
               - Prefix: "handl*" (matches handle, handler, handling)
        limit: Maximum number of results to return (default: 10).

    Returns:
        List of memory dictionaries ordered by relevance (best matches first).
        Each dict contains:
            - id: Unique memory identifier
            - situation_description: When this knowledge applies
            - lesson: Actionable guidance
            - metadata: Dict with repo, language, severity, confidence
            - source: Dict with original code review context
            - rank: BM25 relevance score (LOWER is better - typically negative)

    Example:
        >>> results = search_memories("data/phase0/memories/memories.db", "async error", limit=5)
        >>> for r in results:
        ...     print(f"{r['id']}: {r['rank']:.2f}")
        mem_abc123: -2.54
        mem_def456: -1.87

    Note:
        Returns empty list if query doesn't match any documents or if
        the query syntax is invalid.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        # Query the FTS index and join with base table to get full record
        # bm25(memories_fts) calculates BM25 relevance score
        # Lower (more negative) scores = better matches
        cur.execute(
            f"""
            SELECT m.{FIELD_ID}, m.{FIELD_SITUATION}, m.{FIELD_LESSON},
                   m.{FIELD_METADATA}, m.{FIELD_SOURCE},
                   bm25(memories_fts) as {FIELD_RANK}
            FROM memories_fts fts
            JOIN memories m ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ?
            ORDER BY {FIELD_RANK}
            LIMIT ?
            """,
            (query, limit),
        )

        # Convert Row objects to dicts
        results = []
        for row in cur.fetchall():
            memory = {
                FIELD_ID: row[FIELD_ID],
                FIELD_SITUATION: row[FIELD_SITUATION],
                FIELD_LESSON: row[FIELD_LESSON],
                FIELD_METADATA: json.loads(row[FIELD_METADATA]) if row[FIELD_METADATA] else {},
                FIELD_SOURCE: json.loads(row[FIELD_SOURCE]) if row[FIELD_SOURCE] else {},
                FIELD_RANK: row[FIELD_RANK],
            }
            results.append(memory)

        return results

    finally:
        conn.close()
