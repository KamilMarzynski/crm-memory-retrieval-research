"""
Test case access for Phase 2 retrieval experiments.

Phase 2 reuses Phase 1's test cases (no separate generation needed).
This module resolves test case paths from Phase 1 runs.

Usage:
    # List test cases from latest Phase 1 run
    uv run python scripts/phase2/test_cases.py

    # List test cases from specific Phase 1 run
    uv run python scripts/phase2/test_cases.py --phase1-run-id run_20260208_143022
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.io import load_json
from common.runs import get_latest_run, get_run, PHASE1


def get_test_cases_dir(phase1_run_id: str = None) -> Path:
    """
    Get test cases directory from a Phase 1 run.

    Args:
        phase1_run_id: Specific Phase 1 run ID, or None for latest.

    Returns:
        Path to the test_cases directory.
    """
    if phase1_run_id:
        phase1_run = get_run(PHASE1, phase1_run_id)
    else:
        phase1_run = get_latest_run(PHASE1)

    return phase1_run / "test_cases"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 Test Case Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Description:
  Lists test cases from a Phase 1 run. Phase 2 reuses Phase 1's
  test cases (same memories, same ground truth).

Examples:
  uv run python scripts/phase2/test_cases.py
  uv run python scripts/phase2/test_cases.py --phase1-run-id run_20260208_143022
        """,
    )
    parser.add_argument(
        "--phase1-run-id",
        default=None,
        help="Phase 1 run ID (default: latest)",
    )

    args = parser.parse_args()

    if args.phase1_run_id:
        phase1_run = get_run(PHASE1, args.phase1_run_id)
        print(f"Phase 1 run: {args.phase1_run_id}")
    else:
        phase1_run = get_latest_run(PHASE1)
        print(f"Phase 1 run (latest): {phase1_run.name}")

    test_cases_dir = phase1_run / "test_cases"
    print(f"Test cases dir: {test_cases_dir}")
    print()

    test_case_files = sorted(test_cases_dir.glob("*.json"))

    if not test_case_files:
        print("No test cases found. Run Phase 1 test case generation first.")
        sys.exit(1)

    print(f"Found {len(test_case_files)} test cases:")
    for f in test_case_files:
        tc = load_json(str(f))
        gt_count = tc.get("ground_truth_count", len(tc.get("ground_truth_memory_ids", [])))
        print(f"  {f.name} — {gt_count} ground truth memories")
