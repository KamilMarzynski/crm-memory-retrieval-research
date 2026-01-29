"""
Test case generation for Phase 1 retrieval experiments.

Thin wrapper around common.test_cases with Phase 1-specific paths.

Usage:
    uv run python scripts/phase1/test_cases.py

Input:
    - data/review_data/*.json (raw PR files)
    - data/phase1/memories/*.jsonl (extracted memories)

Output:
    - data/phase1/test_cases/*.json (test case files)
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.test_cases import build_test_cases

# Phase 1 directory paths
DEFAULT_RAW_DIR = "data/review_data"
DEFAULT_MEMORIES_DIR = "data/phase1/memories"
DEFAULT_OUTPUT_DIR = "data/phase1/test_cases"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Phase 1 Test Case Generator")
        print()
        print("Usage:")
        print("  uv run python scripts/phase1/test_cases.py")
        print()
        print("Description:")
        print("  Generates self-contained test cases from raw PR data and extracted memories.")
        print("  Each test case includes pre-filtered diffs and pre-computed ground truth IDs.")
        print()
        print("  PRs with zero ground truth memories are skipped.")
        print()
        print("Input:")
        print(f"  Raw PR files: {DEFAULT_RAW_DIR}/*.json")
        print(f"  Extracted memories: {DEFAULT_MEMORIES_DIR}/memories_*.jsonl")
        print()
        print("Output:")
        print(f"  Test cases: {DEFAULT_OUTPUT_DIR}/*.json")
        print()
        print("Workflow:")
        print("  1. Extract memories: uv run python scripts/phase1/build_memories.py <file>.json")
        print("  2. Build database: uv run python scripts/phase1/db.py --rebuild")
        print("  3. Generate test cases: uv run python scripts/phase1/test_cases.py")
        print("  4. Run experiments: uv run python scripts/phase1/experiment.py --all")
        sys.exit(0)

    build_test_cases(DEFAULT_RAW_DIR, DEFAULT_MEMORIES_DIR, DEFAULT_OUTPUT_DIR)
