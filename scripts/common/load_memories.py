"""
Memory loading utilities shared across all phases.

This module provides the core function for loading memories from JSONL files
and the field name constants that are common to all phases.

Constants:
    FIELD_ID: Memory identifier field
    FIELD_LESSON: Lesson text field
    FIELD_METADATA: Metadata dict field
    FIELD_SOURCE: Source context field

Functions:
    load_memories: Load all accepted memories from JSONL files
"""

import json
from pathlib import Path
from typing import Any, Dict, List


# Database/memory field names common to all phases
FIELD_ID = "id"
FIELD_LESSON = "lesson"
FIELD_METADATA = "metadata"
FIELD_SOURCE = "source"


def load_memories(memories_dir: str) -> List[Dict[str, Any]]:
    """
    Load all accepted memories from JSONL files in a memories directory.

    Scans the directory for files matching "memories_*.jsonl" (excluding
    rejected_*.jsonl) and parses each line as a JSON memory object.

    Args:
        memories_dir: Path to directory containing JSONL memory files.

    Returns:
        List of memory dictionaries.

    Raises:
        FileNotFoundError: If memories_dir doesn't exist.
    """
    memories = []
    memories_path = Path(memories_dir)

    for jsonl_file in sorted(memories_path.glob("memories_*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        memories.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON in {jsonl_file.name}:{line_num}: {e}")

    return memories
