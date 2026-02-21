import json
from pathlib import Path

from memory_retrieval.experiments.test_cases import build_test_case, filter_diff


# ---------- filter_diff ----------


def test_filter_diff_empty_string_returns_empty_string() -> None:
    assert filter_diff("") == ""


def test_filter_diff_keeps_regular_typescript_file() -> None:
    diff = "diff --git a/src/component.ts b/src/component.ts\n+export const x = 1;"
    result = filter_diff(diff)
    assert "component.ts" in result


def test_filter_diff_removes_package_lock_json() -> None:
    diff = (
        "diff --git a/package-lock.json b/package-lock.json\n"
        "+some lock content\n"
        "diff --git a/src/app.ts b/src/app.ts\n"
        "+export const x = 1;"
    )
    result = filter_diff(diff)
    assert "package-lock.json" not in result
    assert "app.ts" in result


def test_filter_diff_removes_yarn_lock() -> None:
    diff = "diff --git a/yarn.lock b/yarn.lock\n+lockfile content"
    result = filter_diff(diff)
    assert "yarn.lock" not in result


def test_filter_diff_removes_pnpm_lockfile() -> None:
    diff = "diff --git a/pnpm-lock.yaml b/pnpm-lock.yaml\n+lock content"
    result = filter_diff(diff)
    assert "pnpm-lock.yaml" not in result


def test_filter_diff_removes_minified_js() -> None:
    diff = "diff --git a/dist/bundle.min.js b/dist/bundle.min.js\n+minified code"
    result = filter_diff(diff)
    assert "bundle.min.js" not in result


def test_filter_diff_removes_source_map_files() -> None:
    diff = "diff --git a/dist/app.js.map b/dist/app.js.map\n+source map content"
    result = filter_diff(diff)
    assert ".map" not in result


def test_filter_diff_removes_typescript_declaration_files() -> None:
    diff = "diff --git a/dist/types.d.ts b/dist/types.d.ts\n+type declarations"
    result = filter_diff(diff)
    assert ".d.ts" not in result


def test_filter_diff_removes_snapshot_files() -> None:
    diff = (
        "diff --git a/tests/__snapshots__/Button.snap b/tests/__snapshots__/Button.snap\n"
        "+snapshot content"
    )
    result = filter_diff(diff)
    assert "Button.snap" not in result


def test_filter_diff_removes_dist_directory_files() -> None:
    diff = "diff --git a/dist/output.js b/dist/output.js\n+built output"
    result = filter_diff(diff)
    assert "dist/output.js" not in result


def test_filter_diff_mixed_diff_keeps_meaningful_removes_generated() -> None:
    diff = (
        "diff --git a/package-lock.json b/package-lock.json\n"
        "+lock content\n"
        "diff --git a/src/utils.ts b/src/utils.ts\n"
        "+export function foo() {}\n"
        "diff --git a/yarn.lock b/yarn.lock\n"
        "+lock content\n"
        "diff --git a/src/helpers.py b/src/helpers.py\n"
        "+def bar(): pass\n"
    )
    result = filter_diff(diff)
    assert "utils.ts" in result
    assert "helpers.py" in result
    assert "package-lock.json" not in result
    assert "yarn.lock" not in result


# ---------- build_test_case ----------


def _write_raw_pr_file(
    directory: Path,
    filename: str,
    comment_ids: list[str],
    diff: str = "",
) -> Path:
    raw_path = directory / filename
    raw_path.write_text(
        json.dumps(
            {
                "context": "Feature: Add authentication",
                "meta": {"sourceBranch": "feature/auth", "targetBranch": "main"},
                "full_diff": diff,
                "code_review_comments": [{"id": comment_id} for comment_id in comment_ids],
            }
        ),
        encoding="utf-8",
    )
    return raw_path


def _make_memory(memory_id: str, source_comment_id: str) -> dict:
    return {
        "id": memory_id,
        "situation_description": "A developer wrote code without proper authentication checks.",
        "lesson": "Always verify authentication tokens before processing requests.",
        "metadata": {"source_comment_id": source_comment_id},
    }


def test_build_test_case_returns_none_when_no_ground_truth_matches(tmp_path: Path) -> None:
    raw_path = _write_raw_pr_file(tmp_path, "pr001.json", comment_ids=["comment_x"])
    memories = [_make_memory("mem_aaa", "comment_unrelated")]
    result = build_test_case(str(raw_path), memories)
    assert result is None


def test_build_test_case_returns_dict_when_ground_truth_matches(tmp_path: Path) -> None:
    raw_path = _write_raw_pr_file(tmp_path, "pr001.json", comment_ids=["comment_x"])
    memories = [_make_memory("mem_aaa", "comment_x")]
    result = build_test_case(str(raw_path), memories)
    assert result is not None


def test_build_test_case_test_case_id_uses_file_stem(tmp_path: Path) -> None:
    raw_path = _write_raw_pr_file(tmp_path, "pr_review_42.json", comment_ids=["c1"])
    memories = [_make_memory("mem_aaa", "c1")]
    result = build_test_case(str(raw_path), memories)
    assert result is not None
    assert result["test_case_id"] == "tc_pr_review_42"


def test_build_test_case_ground_truth_ids_contains_matched_memory(tmp_path: Path) -> None:
    raw_path = _write_raw_pr_file(tmp_path, "pr001.json", comment_ids=["comment_x", "comment_y"])
    memories = [
        _make_memory("mem_aaa", "comment_x"),
        _make_memory("mem_bbb", "comment_unrelated"),
    ]
    result = build_test_case(str(raw_path), memories)
    assert result is not None
    assert "mem_aaa" in result["ground_truth_memory_ids"]
    assert "mem_bbb" not in result["ground_truth_memory_ids"]
    assert result["ground_truth_count"] == 1


def test_build_test_case_diff_stats_are_included(tmp_path: Path) -> None:
    raw_path = _write_raw_pr_file(
        tmp_path, "pr001.json", comment_ids=["c1"], diff="some meaningful diff content here"
    )
    memories = [_make_memory("mem_aaa", "c1")]
    result = build_test_case(str(raw_path), memories)
    assert result is not None
    assert "original_length" in result["diff_stats"]
    assert "filtered_length" in result["diff_stats"]
    assert result["diff_stats"]["original_length"] > 0
