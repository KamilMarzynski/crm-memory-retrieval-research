import json
import re
from pathlib import Path
from typing import Any

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
                    re.search(pattern, current_file) for pattern in EXCLUDED_FILE_PATTERNS
                )

            if not skip_until_next_file:
                filtered_lines.append(line)
        elif not skip_until_next_file:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def get_ground_truth_memory_ids(
    raw_path: str, all_memories: list[dict[str, Any]]
) -> set[str]:
    raw_data = load_json(raw_path)
    comment_ids = {c.get("id") for c in raw_data.get("code_review_comments", [])}

    ground_truth_ids: set[str] = set()
    for mem in all_memories:
        source_comment_id = mem.get("metadata", {}).get("source_comment_id")
        if source_comment_id in comment_ids:
            ground_truth_ids.add(mem["id"])

    return ground_truth_ids


def build_test_case(
    raw_file_path: str,
    all_memories: list[dict[str, Any]],
) -> dict[str, Any] | None:
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
