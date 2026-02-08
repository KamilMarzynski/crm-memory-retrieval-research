"""
Test case generation for Phase 1 retrieval experiments.

Thin wrapper around common.test_cases with Phase 1-specific paths.

Usage:
    # Uses latest run
    uv run python scripts/phase1/test_cases.py

    # Use specific run
    uv run python scripts/phase1/test_cases.py --run-id run_20260208_143022

Input:
    - data/review_data/*.json (raw PR files)
    - <run_dir>/memories/*.jsonl (extracted memories)

Output:
    - <run_dir>/test_cases/*.json (test case files)
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.test_cases import build_test_cases
from common.runs import get_latest_run, get_run, update_run_status, PHASE1

# Phase 1 directory paths
DEFAULT_RAW_DIR = "data/review_data"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 Test Case Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Description:
  Generates self-contained test cases from raw PR data and extracted memories.
  Each test case includes pre-filtered diffs and pre-computed ground truth IDs.

  PRs with zero ground truth memories are skipped.

Workflow:
  1. Extract memories: uv run python scripts/phase1/build_memories.py --all data/review_data
  2. Build database: uv run python scripts/phase1/db.py --rebuild
  3. Generate test cases: uv run python scripts/phase1/test_cases.py
  4. Run experiments: uv run python scripts/phase1/experiment.py --all
        """,
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Use specific run (default: latest run)",
    )

    args = parser.parse_args()

    # Determine run directory
    if args.run_id:
        run_dir = get_run(PHASE1, args.run_id)
        print(f"Using run: {args.run_id}")
    else:
        run_dir = get_latest_run(PHASE1)
        print(f"Using latest run: {run_dir.name}")

    memories_dir = str(run_dir / "memories")
    output_dir = str(run_dir / "test_cases")

    print(f"Input: {DEFAULT_RAW_DIR}/*.json")
    print(f"Memories: {memories_dir}/memories_*.jsonl")
    print(f"Output: {output_dir}/*.json")
    print()

    build_test_cases(DEFAULT_RAW_DIR, memories_dir, output_dir)

    # Count test cases created
    test_case_count = len(list(Path(output_dir).glob("*.json")))
    update_run_status(run_dir, "test_cases", {"count": test_case_count})
    print(f"\nRun status updated: {run_dir.name}")
