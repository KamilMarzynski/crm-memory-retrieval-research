"""
Memory loading utilities for Phase 0 experiments.

This module handles loading extracted memories from JSONL files and provides
centralized field name constants used across the phase0 codebase.

Constants:
    DEFAULT_MEMORIES_DIR: Default directory for memory files
    FIELD_*: Database/memory field names

Functions:
    load_memories: Load all accepted memories from JSONL files
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default paths
DEFAULT_MEMORIES_DIR = "data/phase0/memories"

# Database/memory field names (centralized to avoid magic strings)
FIELD_ID = "id"
FIELD_SITUATION = "situation_description"
FIELD_VARIANTS = "situation_variants"
FIELD_LESSON = "lesson"
FIELD_METADATA = "metadata"
FIELD_SOURCE = "source"
FIELD_RANK = "rank"


def load_memories(memories_dir: str = DEFAULT_MEMORIES_DIR) -> List[Dict[str, Any]]:
    """
    Load all accepted memories from JSONL files in the memories directory.

    This function scans the directory for files matching the pattern
    "memories_*.jsonl" (accepted memories only, not rejected_*.jsonl) and
    parses each line as a JSON memory object.

    Args:
        memories_dir: Path to directory containing JSONL memory files.
                      Defaults to "data/phase0/memories".

    Returns:
        List of memory dictionaries, each containing:
            - id: Unique memory identifier
            - situation_description: When this knowledge applies
            - situation_variants: List of 3 situation variants
            - lesson: Actionable guidance
            - metadata: Dict with repo, language, severity, confidence
            - source: Dict with original code review context

    Raises:
        FileNotFoundError: If memories_dir doesn't exist.

    Example:
        >>> memories = load_memories("data/phase0/memories")
        >>> print(f"Loaded {len(memories)} memories")
        Loaded 13 memories

    Note:
        Empty lines are skipped. Malformed JSON lines generate warnings.
    """
    memories = []
    memories_path = Path(memories_dir)

    # Find all accepted memory files (memories_*.jsonl pattern)
    # This excludes rejected_*.jsonl files which contain low-quality extractions
    for jsonl_file in sorted(memories_path.glob("memories_*.jsonl")):
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
