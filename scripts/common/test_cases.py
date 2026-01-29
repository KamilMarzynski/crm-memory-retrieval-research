"""
Test case generation utilities shared across all phases.

This module provides the core logic for generating test case files from raw
PR data and extracted memories. Each test case is a self-contained JSON file
that includes everything needed to run retrieval experiments.

Key Features:
    - Pre-filtered diffs (excludes lock files, generated code)
    - Pre-computed ground truth memory IDs (matches comment IDs)
    - One test case per PR (skips PRs with zero ground truth)
    - Reproducible snapshots for consistent experimentation

Functions:
    filter_diff: Remove excluded file patterns from git diff output
    get_ground_truth_memory_ids: Match comment IDs to memory IDs
    build_test_case: Build a single test case from raw PR data
    build_test_cases: Build all test cases from a directory
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from common.io import load_json, save_json
from common.load_memories import load_memories

# Test case field names
FIELD_TEST_CASE_ID = "test_case_id"
FIELD_SOURCE_FILE = "source_file"
FIELD_PR_CONTEXT = "pr_context"
FIELD_FILTERED_DIFF = "filtered_diff"
FIELD_METADATA = "metadata"
FIELD_DIFF_STATS = "diff_stats"
FIELD_GROUND_TRUTH_IDS = "ground_truth_memory_ids"
FIELD_GROUND_TRUTH_COUNT = "ground_truth_count"

# Test case ID prefix
TEST_CASE_ID_PREFIX = "tc_"

# Files to exclude from diff (meaningless for code review patterns)
EXCLUDED_FILE_PATTERNS = [
    r"package-lock\.json$",
    r"yarn\.lock$",
    r"pnpm-lock\.yaml$",
    r"\.snap$",
    r"__snapshots__/",
    r"\.min\.js$",
    r"\.min\.css$",
    r"\.map$",
    r"\.d\.ts$",
    r"dist/",
    r"build/",
    r"node_modules/",
    r"\.generated\.",
    r"migrations/\d+",
]


def filter_diff(full_diff: str) -> str:
    """
    Remove diff sections for files matching excluded patterns.

    Filters out lock files, build artifacts, and generated code that don't
    contain meaningful code review patterns.

    Args:
        full_diff: Complete git diff output.

    Returns:
        Filtered diff with excluded files removed.
    """
    if not full_diff:
        return ""

    lines = full_diff.split("\n")
    filtered_lines = []
    skip_until_next_file = False
    current_file = None

    for line in lines:
        if line.startswith("diff --git"):
            match = re.search(r"[ab]/(.+?)(?:\s|$)", line)
            if match:
                current_file = match.group(1)
                skip_until_next_file = any(
                    re.search(pattern, current_file) for pattern in EXCLUDED_FILE_PATTERNS
                )

            if not skip_until_next_file:
                filtered_lines.append(line)
        elif not skip_until_next_file:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def get_ground_truth_memory_ids(raw_path: str, all_memories: List[Dict[str, Any]]) -> Set[str]:
    """
    Get memory IDs that were extracted from this raw PR file.

    Matches comment IDs between raw PR data and extracted memories.

    Args:
        raw_path: Path to raw PR JSON file.
        all_memories: List of all extracted memories.

    Returns:
        Set of memory IDs that originated from this PR's code review comments.
    """
    raw_data = load_json(raw_path)
    comment_ids = {c.get("id") for c in raw_data.get("code_review_comments", [])}

    ground_truth_ids = set()
    for mem in all_memories:
        source_comment_id = mem.get("metadata", {}).get("source_comment_id")
        if source_comment_id in comment_ids:
            ground_truth_ids.add(mem["id"])

    return ground_truth_ids


def build_test_case(
    raw_file_path: str,
    all_memories: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Build a single test case from raw PR data and memories.

    Args:
        raw_file_path: Path to raw PR JSON file.
        all_memories: List of all extracted memories (from JSONL files).

    Returns:
        Test case dictionary, or None if no ground truth memories found.
    """
    raw_path = Path(raw_file_path)

    try:
        raw_data = load_json(str(raw_path))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Error loading raw file: {e}")
        return None

    ground_truth_ids = get_ground_truth_memory_ids(str(raw_path), all_memories)

    if not ground_truth_ids:
        return None

    full_diff = raw_data.get("full_diff", "")
    filtered_diff = filter_diff(full_diff)

    test_case = {
        FIELD_TEST_CASE_ID: f"{TEST_CASE_ID_PREFIX}{raw_path.stem}",
        FIELD_SOURCE_FILE: raw_path.name,
        FIELD_PR_CONTEXT: raw_data.get("context", ""),
        FIELD_FILTERED_DIFF: filtered_diff,
        FIELD_METADATA: raw_data.get("meta", {}),
        FIELD_DIFF_STATS: {
            "original_length": len(full_diff),
            "filtered_length": len(filtered_diff),
        },
        FIELD_GROUND_TRUTH_IDS: sorted(list(ground_truth_ids)),
        FIELD_GROUND_TRUTH_COUNT: len(ground_truth_ids),
    }

    return test_case


def build_test_cases(
    raw_dir: str,
    memories_dir: str,
    output_dir: str,
) -> None:
    """
    Build test cases from all raw PR files in directory.

    Args:
        raw_dir: Directory containing raw PR JSON files.
        memories_dir: Directory containing extracted memory JSONL files.
        output_dir: Directory where test case files should be written.
    """
    print("Loading all memories...")
    all_memories = load_memories(memories_dir)
    print(f"Loaded {len(all_memories)} memories")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(Path(raw_dir).glob("*.json"))
    created = 0
    skipped = 0

    for i, raw_file in enumerate(raw_files):
        print(f"[{i+1}/{len(raw_files)}] Processing {raw_file.name}...")

        try:
            test_case = build_test_case(str(raw_file), all_memories)

            if test_case is None:
                print(f"  Skipped (no ground truth memories)")
                skipped += 1
                continue

            tc_path = output_path / f"{raw_file.stem}.json"
            save_json(test_case, tc_path)

            print(f"  Created test case with {test_case[FIELD_GROUND_TRUTH_COUNT]} ground truth memories")
            created += 1

        except Exception as e:
            print(f"  Error: {e}")
            skipped += 1

    print("\n" + "=" * 60)
    print("TEST CASE GENERATION SUMMARY")
    print("=" * 60)
    print(f"Test cases created: {created}")
    print(f"PRs skipped (no ground truth): {skipped}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
