"""
File I/O utilities for loading and saving data.

This module provides centralized file operations with consistent error handling
and formatting options. All functions use UTF-8 encoding by default.

Functions:
    load_json: Load JSON file from disk
    save_json: Save dictionary as JSON file
    ensure_dir: Create directory if it doesn't exist
"""

import json
from pathlib import Path
from typing import Any, Dict


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


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Creates parent directories as needed (like mkdir -p).

    Args:
        path: Directory path to create.

    Example:
        >>> ensure_dir("data/phase0/memories")
    """
    Path(path).mkdir(parents=True, exist_ok=True)
