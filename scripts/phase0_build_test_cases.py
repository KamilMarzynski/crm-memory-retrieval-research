"""
Phase 0: Build self-contained test cases for retrieval experiments.

This module generates test case files from raw PR data and extracted memories.
Each test case is a self-contained JSON file that includes everything needed to
run retrieval experiments without re-processing raw data.

Key Features:
    - Pre-filtered diffs (excludes lock files, generated code)
    - Pre-computed ground truth memory IDs (matches comment IDs)
    - One test case per PR (skips PRs with zero ground truth)
    - Reproducible snapshots for consistent experimentation

Test Case Structure:
    Each test case includes:
    - Filtered diff (ready for LLM processing)
    - PR context and metadata
    - Ground truth memory IDs (for recall calculation)
    - Diff statistics

Usage:
    # Generate test cases (run after extracting memories and building database)
    uv run python scripts/phase0_build_test_cases.py

    # The script reads from:
    #   - data/review_data/*.json (raw PR data)
    #   - data/phase0_memories/*.jsonl (extracted memories)
    #
    # And writes to:
    #   - data/phase0_test_cases/*.json (test case files)

Example:
    >>> from phase0_build_test_cases import build_test_case
    >>> test_case = build_test_case("data/review_data/pr_12345.json", all_memories)
    >>> print(test_case["test_case_id"])
    tc_pr_12345
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from phase0_common import load_json, save_json, load_memories

# Default directory paths
DEFAULT_RAW_DIR = "data/review_data"
DEFAULT_PHASE0_DIR = "data/phase0/memories"
DEFAULT_OUTPUT_DIR = "data/phase0/test_cases"

# Test case field names (centralized to avoid magic strings)
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
# These patterns match lock files, build artifacts, and generated code
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


def _filter_diff(full_diff: str) -> str:
    """
    Remove diff sections for files matching excluded patterns.

    Filters out lock files, build artifacts, and generated code that don't
    contain meaningful code review patterns. This reduces noise in test cases.

    Args:
        full_diff: Complete git diff output.

    Returns:
        Filtered diff with excluded files removed.

    Note:
        Matches patterns defined in EXCLUDED_FILE_PATTERNS constant.
    """
    if not full_diff:
        return ""

    lines = full_diff.split("\n")
    filtered_lines = []
    skip_until_next_file = False
    current_file = None

    for line in lines:
        # Detect file header in diff
        if line.startswith("diff --git"):
            # Extract filename from diff header (e.g., "diff --git a/path b/path")
            match = re.search(r"[ab]/(.+?)(?:\s|$)", line)
            if match:
                current_file = match.group(1)
                # Check if this file should be excluded
                skip_until_next_file = any(
                    re.search(pattern, current_file) for pattern in EXCLUDED_FILE_PATTERNS
                )

            if not skip_until_next_file:
                filtered_lines.append(line)
        elif not skip_until_next_file:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _get_ground_truth_memory_ids(raw_path: str, all_memories: List[Dict[str, Any]]) -> Set[str]:
    """
    Get memory IDs that were extracted from this raw PR file.

    Matches comment IDs between raw PR data and extracted memories to determine
    which memories should be considered "ground truth" for this test case.

    Args:
        raw_path: Path to raw PR JSON file.
        all_memories: List of all extracted memories.

    Returns:
        Set of memory IDs that originated from this PR's code review comments.

    Example:
        >>> raw_data = _load_json("data/review_data/pr_123.json")
        >>> memories = load_memories("data/phase0/memories")
        >>> gt_ids = _get_ground_truth_memory_ids("data/review_data/pr_123.json", memories)
        >>> print(f"Found {len(gt_ids)} ground truth memories")
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

    This function processes raw PR data to create a self-contained test case
    that includes pre-filtered diffs and pre-computed ground truth memory IDs.

    Args:
        raw_file_path: Path to raw PR JSON file.
        all_memories: List of all extracted memories (from JSONL files).

    Returns:
        Test case dictionary containing:
            - test_case_id: Unique identifier (e.g., "tc_feature-JIRA-123")
            - source_file: Original raw filename
            - pr_context: Full PR description and requirements
            - filtered_diff: Pre-filtered code diff
            - metadata: PR metadata (branches, commit hash, repo info)
            - diff_stats: Original and filtered diff lengths
            - ground_truth_memory_ids: IDs of memories from this PR
            - ground_truth_count: Number of ground truth memories

        Returns None if:
            - Raw file cannot be loaded
            - No ground truth memories found (PR has no relevant memories)

    Example:
        >>> memories = load_memories("data/phase0_memories")
        >>> test_case = build_test_case("data/review_data/pr_123.json", memories)
        >>> if test_case:
        ...     print(f"Created {test_case['test_case_id']}")
    """
    raw_path = Path(raw_file_path)

    # Load raw PR data
    try:
        raw_data = load_json(str(raw_path))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ✗ Error loading raw file: {e}")
        return None

    # Find ground truth memory IDs by matching comment IDs
    ground_truth_ids = _get_ground_truth_memory_ids(str(raw_path), all_memories)

    # Skip if no ground truth (PR has no extracted memories)
    if not ground_truth_ids:
        return None

    # Filter diff to remove noise (lock files, generated code, etc.)
    full_diff = raw_data.get("full_diff", "")
    filtered_diff = _filter_diff(full_diff)

    # Build test case structure
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
    raw_dir: str = DEFAULT_RAW_DIR,
    phase0_dir: str = DEFAULT_PHASE0_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Build test cases from all raw PR files in directory.

    This is the main orchestration function that:
    1. Loads all memories from JSONL files (once)
    2. Processes each raw PR file
    3. Creates test case files (skips PRs with no ground truth)
    4. Prints summary statistics

    Args:
        raw_dir: Directory containing raw PR JSON files.
                 Defaults to "data/review_data".
        phase0_dir: Directory containing extracted memory JSONL files.
                    Defaults to "data/phase0_memories".
        output_dir: Directory where test case files should be written.
                    Defaults to "data/phase0_test_cases".
                    Directory will be created if it doesn't exist.

    Side Effects:
        - Creates output_dir if it doesn't exist
        - Writes JSON files to output_dir
        - Prints progress information to stdout

    Example:
        >>> build_test_cases()
        Loading all memories...
        Loaded 13 memories
        [1/5] Processing feature-JIRA-724.json...
          ✓ Created test case with 2 ground truth memories
        ...

    Note:
        Test cases are frozen snapshots. If you re-extract memories or modify
        filtering logic, you should regenerate test cases.
    """
    # Load all memories once (reused for all test cases)
    print("Loading all memories...")
    all_memories = load_memories(phase0_dir)
    print(f"Loaded {len(all_memories)} memories")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each raw PR file
    raw_files = sorted(Path(raw_dir).glob("*.json"))
    created = 0
    skipped = 0

    for i, raw_file in enumerate(raw_files):
        print(f"[{i+1}/{len(raw_files)}] Processing {raw_file.name}...")

        try:
            # Build test case
            test_case = build_test_case(str(raw_file), all_memories)

            if test_case is None:
                print(f"  ⊘ Skipped (no ground truth memories)")
                skipped += 1
                continue

            # Save test case to file
            tc_path = output_path / f"{raw_file.stem}.json"
            save_json(test_case, tc_path)

            print(f"  ✓ Created test case with {test_case[FIELD_GROUND_TRUTH_COUNT]} ground truth memories")
            created += 1

        except Exception as e:
            # Log error but continue processing remaining files
            print(f"  ✗ Error: {e}")
            skipped += 1

    # Print summary
    print("\n" + "=" * 60)
    print("TEST CASE GENERATION SUMMARY")
    print("=" * 60)
    print(f"Test cases created: {created}")
    print(f"PRs skipped (no ground truth): {skipped}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Command-line interface for generating test cases
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Phase 0 Test Case Generator")
        print()
        print("Usage:")
        print("  uv run python scripts/phase0_build_test_cases.py")
        print()
        print("Description:")
        print("  Generates self-contained test cases from raw PR data and extracted memories.")
        print("  Each test case includes pre-filtered diffs and pre-computed ground truth IDs.")
        print()
        print("  PRs with zero ground truth memories are skipped.")
        print()
        print("Input:")
        print(f"  Raw PR files: {DEFAULT_RAW_DIR}/*.json")
        print(f"  Extracted memories: {DEFAULT_PHASE0_DIR}/memories_*.jsonl")
        print()
        print("Output:")
        print(f"  Test cases: {DEFAULT_OUTPUT_DIR}/*.json")
        print()
        print("Workflow:")
        print("  1. Extract memories: uv run python scripts/pre0_build_memories.py <file>.json")
        print("  2. Build database: uv run python scripts/phase0_sqlite_fts.py --rebuild")
        print("  3. Generate test cases: uv run python scripts/phase0_build_test_cases.py")
        print("  4. Run experiments: uv run python scripts/phase0_experiment.py --all")
        sys.exit(0)

    # Generate test cases
    build_test_cases()
