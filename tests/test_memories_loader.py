import json
from pathlib import Path
from typing import Any

from memory_retrieval.memories.loader import load_memories


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_load_memories_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    memories = load_memories(tmp_path)
    assert memories == []


def test_load_memories_single_valid_file_loads_all_records(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "memories_001.jsonl",
        [{"id": "mem_aaa", "lesson": "Lesson A"}, {"id": "mem_bbb", "lesson": "Lesson B"}],
    )
    memories = load_memories(tmp_path)
    assert len(memories) == 2
    assert {memory["id"] for memory in memories} == {"mem_aaa", "mem_bbb"}


def test_load_memories_multiple_files_loads_all(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "memories_001.jsonl", [{"id": "m1"}])
    _write_jsonl(tmp_path / "memories_002.jsonl", [{"id": "m2"}, {"id": "m3"}])
    memories = load_memories(tmp_path)
    assert len(memories) == 3


def test_load_memories_non_matching_filenames_are_ignored(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "other_file.jsonl", [{"id": "should_not_load"}])
    _write_jsonl(tmp_path / "memories_001.jsonl", [{"id": "should_load"}])
    memories = load_memories(tmp_path)
    assert len(memories) == 1
    assert memories[0]["id"] == "should_load"


def test_load_memories_skips_malformed_json_lines(tmp_path: Path) -> None:
    file_path = tmp_path / "memories_001.jsonl"
    file_path.write_text(
        '{"id": "good_1"}\nnot valid json\n{"id": "good_2"}\n',
        encoding="utf-8",
    )
    memories = load_memories(tmp_path)
    assert len(memories) == 2
    loaded_ids = [memory["id"] for memory in memories]
    assert "good_1" in loaded_ids
    assert "good_2" in loaded_ids


def test_load_memories_skips_empty_lines(tmp_path: Path) -> None:
    file_path = tmp_path / "memories_001.jsonl"
    file_path.write_text(
        '{"id": "m1"}\n\n\n{"id": "m2"}\n',
        encoding="utf-8",
    )
    memories = load_memories(tmp_path)
    assert len(memories) == 2
