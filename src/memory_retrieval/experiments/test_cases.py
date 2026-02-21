import json
import re
from pathlib import Path
from typing import Any

from memory_retrieval.constants import EXCLUDED_FILE_PATTERNS
from memory_retrieval.infra.io import load_json, save_json
from memory_retrieval.memories.loader import load_memories

# Test case field names
FIELD_TEST_CASE_ID = "test_case_id"
FIELD_SOURCE_FILE = "source_file"
FIELD_PR_CONTEXT = "pr_context"
FIELD_FILTERED_DIFF = "filtered_diff"
FIELD_METADATA = "metadata"
FIELD_DIFF_STATS = "diff_stats"
FIELD_GROUND_TRUTH_IDS = "ground_truth_memory_ids"
FIELD_GROUND_TRUTH_COUNT = "ground_truth_count"

TEST_CASE_ID_PREFIX = "tc_"


def filter_diff(full_diff: str) -> str:
    """Filter out generated/lock files and build artifacts from a git diff.

    Removes diff chunks for files matching EXCLUDED_FILE_PATTERNS (lock files,
    minified files, snapshots, build outputs, etc.) to focus on meaningful code changes.

    Args:
        full_diff: Complete git diff output.

    Returns:
        Filtered diff with excluded files removed.
    """
    if not full_diff:
        return ""

    lines = full_diff.split("\n")
    filtered_lines: list[str] = []
    skip_until_next_file = False

    for line in lines:
        if line.startswith("diff --git"):
            match = re.search(r"[ab]/(.+?)(?:\s|$)", line)
            if match:
                current_file = match.group(1)
                skip_until_next_file = any(
                    pattern.search(current_file) for pattern in EXCLUDED_FILE_PATTERNS
                )

            if not skip_until_next_file:
                filtered_lines.append(line)
        elif not skip_until_next_file:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def get_ground_truth_memory_ids(raw_path: str, all_memories: list[dict[str, Any]]) -> set[str]:
    """Determine which memories are ground truth for a given PR.

    Matches memories to the PR by checking if their source_comment_id appears
    in the PR's code_review_comments list.

    Args:
        raw_path: Path to raw PR JSON file.
        all_memories: Complete list of all extracted memories.

    Returns:
        Set of memory IDs that are ground truth for this PR.
    """
    raw_data = load_json(raw_path)
    comment_ids = {comment.get("id") for comment in raw_data.get("code_review_comments", [])}

    ground_truth_ids: set[str] = set()
    for memory in all_memories:
        source_comment_id = memory.get("metadata", {}).get("source_comment_id")
        if source_comment_id in comment_ids:
            ground_truth_ids.add(memory["id"])

    return ground_truth_ids


def build_test_case(
    raw_file_path: str,
    all_memories: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build a single test case from a raw PR file.

    Creates a self-contained test case with filtered diff, context, and ground truth
    memory IDs. Skips PRs with no ground truth memories.

    Args:
        raw_file_path: Path to raw PR JSON file.
        all_memories: Complete list of all extracted memories.

    Returns:
        Test case dictionary with all fields, or None if no ground truth memories found.
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
    filtered = filter_diff(full_diff)

    test_case = {
        FIELD_TEST_CASE_ID: f"{TEST_CASE_ID_PREFIX}{raw_path.stem}",
        FIELD_SOURCE_FILE: raw_path.name,
        FIELD_PR_CONTEXT: raw_data.get("context", ""),
        FIELD_FILTERED_DIFF: filtered,
        FIELD_METADATA: raw_data.get("meta", {}),
        FIELD_DIFF_STATS: {
            "original_length": len(full_diff),
            "filtered_length": len(filtered),
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
    """Build test cases for all PRs in the raw data directory.

    Processes each raw PR JSON file, filters diffs, computes ground truth, and
    creates self-contained test case files. Skips PRs with no ground truth memories.

    Args:
        raw_dir: Directory containing raw PR JSON files.
        memories_dir: Directory containing JSONL memory files.
        output_dir: Directory where test case JSON files will be written.
    """
    all_memories = load_memories(memories_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(Path(raw_dir).glob("*.json"))
    created = 0
    skipped = 0

    for i, raw_file in enumerate(raw_files):
        try:
            test_case = build_test_case(str(raw_file), all_memories)

            if test_case is None:
                skipped += 1
                continue

            tc_path = output_path / f"{raw_file.stem}.json"
            save_json(test_case, tc_path)
            created += 1

        except Exception as error:
            print(f"[{i + 1}/{len(raw_files)}] {raw_file.stem} — ERROR: {error}")
            skipped += 1

    print(f"Test cases: {created} created, {skipped} skipped (no ground truth)")
