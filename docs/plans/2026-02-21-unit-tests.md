# Unit Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add comprehensive unit tests covering 9 core modules, with pytest properly configured as a dev dependency.

**Architecture:** Flat test files per module under `tests/`, using `tmp_path` and `monkeypatch` for filesystem isolation. No mocking of business logic — only external services (LLM/Ollama) are excluded. Each task writes one test file and verifies it passes before committing.

**Tech Stack:** pytest, pytest's built-in `tmp_path` and `monkeypatch` fixtures, SQLite (FTS5 tests), standard library only

---

## Task 1: Add pytest to dev dependencies and configure pytest

**Files:**
- Modify: `pyproject.toml`

**Step 1: Edit pyproject.toml**

In `pyproject.toml`, make two changes:

Change the `[dependency-groups]` section from:
```toml
[dependency-groups]
dev = [
    "nbstripout>=0.9.0",
    "ruff>=0.15.0",
]
```

To:
```toml
[dependency-groups]
dev = [
    "nbstripout>=0.9.0",
    "pytest>=8.0.0",
    "ruff>=0.15.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Sync dev dependencies**

```bash
uv sync --group dev
```

Expected: resolves and installs pytest.

**Step 3: Verify pytest runs**

```bash
uv run pytest --collect-only
```

Expected: collects existing tests in `tests/`, no errors.

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pytest to dev dependencies"
```

---

## Task 2: Create shared conftest.py

**Files:**
- Create: `tests/conftest.py`

**Step 1: Create the file**

```python
from pathlib import Path
from typing import Any

import pytest

from memory_retrieval.search.fts5 import FTS5Backend


def sample_memory(
    memory_id: str = "mem_test001",
    variants: list[str] | None = None,
    lesson: str = "Always validate your inputs.",
) -> dict[str, Any]:
    """Return a canonical memory dict for use in tests."""
    if variants is None:
        variants = ["A developer added code without proper input validation."]
    return {
        "id": memory_id,
        "situation_variants": variants,
        "lesson": lesson,
        "metadata": {"repo": "test-repo", "language": "py"},
        "source": {"file": "main.py"},
    }


@pytest.fixture
def tmp_fts5_db(tmp_path: Path) -> str:
    """Create a temporary FTS5 database with two sample memories pre-inserted."""
    db_path = str(tmp_path / "memories.db")
    backend = FTS5Backend()
    backend.create_database(db_path)
    backend.insert_memories(
        db_path,
        [
            sample_memory(
                "mem_001",
                ["Python code has type errors when types are wrong."],
                "Type check everything.",
            ),
            sample_memory(
                "mem_002",
                ["Code review missing without proper test coverage."],
                "Write tests first.",
            ),
        ],
    )
    return db_path
```

**Step 2: Verify conftest loads without error**

```bash
uv run pytest --collect-only
```

Expected: no import errors.

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared fixtures in conftest.py"
```

---

## Task 3: Test memories/helpers.py

**Files:**
- Create: `tests/test_memories_helpers.py`

**Step 1: Create the test file**

```python
from memory_retrieval.memories.helpers import (
    confidence_map,
    file_pattern,
    lang_from_file,
    short_repo_name,
    stable_id,
)


def test_stable_id_is_deterministic() -> None:
    id1 = stable_id("comment_1", "situation text", "lesson text")
    id2 = stable_id("comment_1", "situation text", "lesson text")
    assert id1 == id2


def test_stable_id_starts_with_mem_prefix() -> None:
    memory_id = stable_id("comment_1", "situation", "lesson")
    assert memory_id.startswith("mem_")


def test_stable_id_hash_is_12_characters() -> None:
    memory_id = stable_id("comment_1", "situation", "lesson")
    hash_part = memory_id[len("mem_"):]
    assert len(hash_part) == 12


def test_stable_id_differs_for_different_comment_ids() -> None:
    id1 = stable_id("comment_1", "same situation", "same lesson")
    id2 = stable_id("comment_2", "same situation", "same lesson")
    assert id1 != id2


def test_stable_id_differs_for_different_content() -> None:
    id1 = stable_id("c1", "situation A", "lesson A")
    id2 = stable_id("c1", "situation B", "lesson B")
    assert id1 != id2


def test_lang_from_file_returns_lowercase_extension() -> None:
    assert lang_from_file("module.py") == "py"
    assert lang_from_file("component.ts") == "ts"
    assert lang_from_file("style.CSS") == "css"


def test_lang_from_file_no_extension_returns_unknown() -> None:
    assert lang_from_file("Makefile") == "unknown"
    assert lang_from_file("DOCKERFILE") == "unknown"


def test_file_pattern_nested_path_produces_extension_glob() -> None:
    pattern = file_pattern("src/components/Button.tsx")
    assert pattern == "src/components/*.tsx"


def test_file_pattern_top_level_file_returns_filename() -> None:
    pattern = file_pattern("main.py")
    assert pattern == "main.py"


def test_file_pattern_no_extension_produces_wildcard() -> None:
    pattern = file_pattern("src/Makefile")
    assert pattern == "src/*"


def test_short_repo_name_extracts_last_path_segment() -> None:
    assert short_repo_name("/home/user/projects/my-repo") == "my-repo"
    assert short_repo_name("https://github.com/org/my-project") == "my-project"


def test_short_repo_name_empty_string_returns_unknown() -> None:
    assert short_repo_name("") == "unknown"


def test_confidence_map_high() -> None:
    assert confidence_map("high") == 1.0


def test_confidence_map_medium() -> None:
    assert confidence_map("medium") == 0.7


def test_confidence_map_low() -> None:
    assert confidence_map("low") == 0.4


def test_confidence_map_unknown_string_returns_default() -> None:
    assert confidence_map("very_high") == 0.5
    assert confidence_map("unknown") == 0.5


def test_confidence_map_none_returns_default() -> None:
    assert confidence_map(None) == 0.5


def test_confidence_map_is_case_insensitive() -> None:
    assert confidence_map("HIGH") == 1.0
    assert confidence_map("Medium") == 0.7
    assert confidence_map("LOW") == 0.4
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_memories_helpers.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_memories_helpers.py
git commit -m "test: add tests for memories/helpers.py"
```

---

## Task 4: Test memories/validators.py

**Files:**
- Create: `tests/test_memories_validators.py`

**Step 1: Create the test file**

```python
from memory_retrieval.memories.validators import (
    get_situation_validator,
    validate_lesson,
    validate_situation_v1,
    validate_situation_v2,
)


# ---------- validate_situation_v1 (char-length based) ----------


def test_validate_situation_v1_too_short_fails() -> None:
    is_valid, reason = validate_situation_v1("Too short.")
    assert not is_valid
    assert reason == "situation_too_short"


def test_validate_situation_v1_too_long_fails() -> None:
    is_valid, reason = validate_situation_v1("x" * 451 + ".")
    assert not is_valid
    assert reason == "situation_too_long"


def test_validate_situation_v1_valid_with_terminal_punctuation() -> None:
    is_valid, reason = validate_situation_v1("a" * 50 + ".")
    assert is_valid
    assert reason == "ok"


def test_validate_situation_v1_valid_without_terminal_punctuation() -> None:
    is_valid, reason = validate_situation_v1("a" * 50)
    assert is_valid
    assert reason == "ok_no_punct"


def test_validate_situation_v1_empty_string_fails() -> None:
    is_valid, _ = validate_situation_v1("")
    assert not is_valid


# ---------- validate_situation_v2 (word-count based) ----------


def test_validate_situation_v2_too_few_words_fails() -> None:
    short_situation = "only three words."  # 3 words
    is_valid, reason = validate_situation_v2(short_situation)
    assert not is_valid
    assert "situation_too_short" in reason


def test_validate_situation_v2_too_many_words_fails() -> None:
    long_situation = " ".join(["word"] * 75) + "."
    is_valid, reason = validate_situation_v2(long_situation)
    assert not is_valid
    assert "situation_too_long" in reason


def test_validate_situation_v2_valid_range_with_punctuation() -> None:
    valid_situation = " ".join(["word"] * 30) + "."
    is_valid, reason = validate_situation_v2(valid_situation)
    assert is_valid
    assert reason == "ok"


def test_validate_situation_v2_valid_range_without_punctuation() -> None:
    valid_situation = " ".join(["word"] * 30)
    is_valid, reason = validate_situation_v2(valid_situation)
    assert is_valid
    assert reason == "ok_no_punct"


# ---------- validate_lesson ----------


def test_validate_lesson_too_short_fails() -> None:
    is_valid, reason = validate_lesson("Too short.")
    assert not is_valid
    assert reason == "lesson_too_short"


def test_validate_lesson_too_long_fails() -> None:
    is_valid, reason = validate_lesson("x" * 221 + ".")
    assert not is_valid
    assert reason == "lesson_too_long"


def test_validate_lesson_valid() -> None:
    lesson = "Always validate input before passing it to external APIs."
    is_valid, reason = validate_lesson(lesson)
    assert is_valid
    assert reason == "ok"


def test_validate_lesson_empty_string_fails() -> None:
    is_valid, _ = validate_lesson("")
    assert not is_valid


# ---------- get_situation_validator ----------


def test_get_situation_validator_returns_v1_for_version_1() -> None:
    validator = get_situation_validator("1.0.0")
    assert validator is validate_situation_v1


def test_get_situation_validator_returns_v2_for_version_2() -> None:
    validator = get_situation_validator("2.0.0")
    assert validator is validate_situation_v2


def test_get_situation_validator_unknown_version_returns_callable() -> None:
    validator = get_situation_validator("99.0.0")
    # Must be callable and produce the expected tuple shape
    is_valid, reason = validator("a" * 50 + ".")
    assert isinstance(is_valid, bool)
    assert isinstance(reason, str)
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_memories_validators.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_memories_validators.py
git commit -m "test: add tests for memories/validators.py"
```

---

## Task 5: Test memories/loader.py

**Files:**
- Create: `tests/test_memories_loader.py`

**Step 1: Create the test file**

```python
import json
from pathlib import Path

from memory_retrieval.memories.loader import load_memories


def _write_jsonl(path: Path, records: list[dict]) -> None:
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
    assert memories[0]["id"] == "mem_aaa"


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
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_memories_loader.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_memories_loader.py
git commit -m "test: add tests for memories/loader.py"
```

---

## Task 6: Test search/db_utils.py

**Files:**
- Create: `tests/test_search_db_utils.py`

**Step 1: Create the test file**

```python
from memory_retrieval.search.db_utils import deserialize_json_field, serialize_json_field


def test_serialize_json_field_none_returns_empty_json_object() -> None:
    result = serialize_json_field(None)
    assert result == "{}"


def test_serialize_json_field_dict_returns_valid_json_string() -> None:
    result = serialize_json_field({"key": "value"})
    assert '"key"' in result
    assert '"value"' in result


def test_serialize_json_field_preserves_unicode() -> None:
    result = serialize_json_field({"message": "héllo wörld"})
    assert "héllo wörld" in result


def test_deserialize_json_field_none_returns_empty_dict() -> None:
    result = deserialize_json_field(None)
    assert result == {}


def test_deserialize_json_field_empty_string_returns_empty_dict() -> None:
    result = deserialize_json_field("")
    assert result == {}


def test_round_trip_preserves_simple_dict() -> None:
    original = {"key": "value", "count": 42}
    serialized = serialize_json_field(original)
    deserialized = deserialize_json_field(serialized)
    assert deserialized == original


def test_round_trip_preserves_nested_dict() -> None:
    original = {"outer": {"inner": [1, 2, 3]}}
    serialized = serialize_json_field(original)
    deserialized = deserialize_json_field(serialized)
    assert deserialized == original


def test_round_trip_preserves_unicode() -> None:
    original = {"text": "héllo wörld — こんにちは"}
    serialized = serialize_json_field(original)
    deserialized = deserialize_json_field(serialized)
    assert deserialized == original
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_search_db_utils.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_search_db_utils.py
git commit -m "test: add tests for search/db_utils.py"
```

---

## Task 7: Test infra/prompts.py

**Files:**
- Create: `tests/test_infra_prompts.py`

**Step 1: Create the test file**

```python
from pathlib import Path

import pytest

from memory_retrieval.infra.prompts import Prompt, _parse_prompt_file, _parse_semver, load_prompt


# ---------- _parse_semver ----------


def test_parse_semver_valid_string_returns_tuple() -> None:
    assert _parse_semver("1.2.3") == (1, 2, 3)
    assert _parse_semver("0.0.1") == (0, 0, 1)
    assert _parse_semver("10.20.30") == (10, 20, 30)


def test_parse_semver_too_few_parts_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _parse_semver("1.2")


def test_parse_semver_non_numeric_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _parse_semver("not.a.version")


def test_parse_semver_with_prefix_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _parse_semver("v1.2.3")


# ---------- _parse_prompt_file ----------


def test_parse_prompt_file_extracts_system_and_user_sections() -> None:
    content = "---system---\nSystem instructions here.\n---user---\nUser template here.\n"
    system, user = _parse_prompt_file(content)
    assert system == "System instructions here."
    assert user == "User template here."


def test_parse_prompt_file_no_valid_sections_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _parse_prompt_file("just plain text with no sections at all")


def test_parse_prompt_file_multiline_sections() -> None:
    content = "---system---\nLine one.\nLine two.\n---user---\nQuery: {query}\n"
    system, user = _parse_prompt_file(content)
    assert "Line one." in system
    assert "Line two." in system
    assert "{query}" in user


# ---------- Prompt.render ----------


def test_prompt_render_substitutes_all_kwargs() -> None:
    prompt = Prompt(name="test", version="1.0.0", system="Hello {name}.", user="Query: {query}.")
    messages = prompt.render(name="world", query="find bugs")
    assert messages[0]["content"] == "Hello world."
    assert messages[1]["content"] == "Query: find bugs."


def test_prompt_render_missing_key_becomes_empty_string() -> None:
    prompt = Prompt(name="test", version="1.0.0", system="Value: {missing_key}.", user="")
    messages = prompt.render()
    assert messages[0]["content"] == "Value: ."


def test_prompt_render_returns_system_and_user_messages() -> None:
    prompt = Prompt(name="test", version="1.0.0", system="System.", user="User.")
    messages = prompt.render()
    roles = [message["role"] for message in messages]
    assert "system" in roles
    assert "user" in roles


def test_prompt_version_tag_property() -> None:
    prompt = Prompt(name="my-prompt", version="2.1.0", system="", user="")
    assert prompt.version_tag == "my-prompt/v2.1.0"


# ---------- load_prompt ----------


def _write_prompt_file(directory: Path, version: str, system: str, user: str) -> None:
    content = f"---system---\n{system}\n---user---\n{user}\n"
    (directory / f"v{version}.md").write_text(content, encoding="utf-8")


def test_load_prompt_loads_specific_version(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "my-prompt"
    prompt_dir.mkdir()
    _write_prompt_file(prompt_dir, "1.0.0", "System v1.", "User v1.")
    _write_prompt_file(prompt_dir, "2.0.0", "System v2.", "User v2.")

    prompt = load_prompt("my-prompt", version="1.0.0", prompts_dir=tmp_path)
    assert prompt.system == "System v1."
    assert prompt.version == "1.0.0"


def test_load_prompt_loads_latest_when_version_is_none(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "my-prompt"
    prompt_dir.mkdir()
    _write_prompt_file(prompt_dir, "1.0.0", "System v1.", "User v1.")
    _write_prompt_file(prompt_dir, "2.0.0", "System v2.", "User v2.")
    _write_prompt_file(prompt_dir, "1.5.0", "System v1.5.", "User v1.5.")

    prompt = load_prompt("my-prompt", prompts_dir=tmp_path)
    assert prompt.system == "System v2."
    assert prompt.version == "2.0.0"


def test_load_prompt_missing_directory_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent-prompt", prompts_dir=tmp_path)


def test_load_prompt_missing_version_raises_file_not_found_error(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "my-prompt"
    prompt_dir.mkdir()
    _write_prompt_file(prompt_dir, "1.0.0", "System.", "User.")

    with pytest.raises(FileNotFoundError):
        load_prompt("my-prompt", version="9.9.9", prompts_dir=tmp_path)


def test_load_prompt_name_is_set_correctly(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "query-builder"
    prompt_dir.mkdir()
    _write_prompt_file(prompt_dir, "1.0.0", "System.", "User.")

    prompt = load_prompt("query-builder", prompts_dir=tmp_path)
    assert prompt.name == "query-builder"
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_infra_prompts.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_infra_prompts.py
git commit -m "test: add tests for infra/prompts.py"
```

---

## Task 8: Test infra/runs.py

**Files:**
- Create: `tests/test_infra_runs.py`

**Note:** `DATA_DIR` is a module-level `Path("data")` constant. We redirect it to `tmp_path` using `monkeypatch.setattr` to avoid touching real data directories.

**Step 1: Create the test file**

```python
import json
from pathlib import Path

import pytest

import memory_retrieval.infra.runs as runs_module
from memory_retrieval.infra.runs import (
    NoRunsFoundError,
    create_run,
    create_subrun,
    get_latest_run,
    get_run,
    get_subrun_db_path,
    list_runs,
    list_subruns,
    update_run_status,
)


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect DATA_DIR to a temp directory so tests never touch real data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(runs_module, "DATA_DIR", data_dir)
    return data_dir


# ---------- create_run ----------


def test_create_run_creates_all_four_subdirectories(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    assert (run_dir / "memories").is_dir()
    assert (run_dir / "test_cases").is_dir()
    assert (run_dir / "queries").is_dir()
    assert (run_dir / "results").is_dir()


def test_create_run_writes_run_json_with_expected_fields(isolated_data_dir: Path) -> None:
    run_id, run_dir = create_run("phase1")
    run_metadata = json.loads((run_dir / "run.json").read_text())
    assert run_metadata["run_id"] == run_id
    assert run_metadata["phase"] == "phase1"
    assert run_metadata["pipeline_status"] == {}
    assert "created_at" in run_metadata


def test_create_run_id_starts_with_run_prefix(isolated_data_dir: Path) -> None:
    run_id, _ = create_run("phase1")
    assert run_id.startswith("run_")


def test_create_run_description_stored_when_provided(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1", description="My test run")
    run_metadata = json.loads((run_dir / "run.json").read_text())
    assert run_metadata["description"] == "My test run"


# ---------- get_latest_run ----------


def test_get_latest_run_returns_run_dir_path(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    latest = get_latest_run("phase1")
    assert latest == run_dir


def test_get_latest_run_returns_newest_when_multiple_exist(isolated_data_dir: Path) -> None:
    run_id_a, _ = create_run("phase1")
    run_id_b, run_dir_b = create_run("phase1")
    latest = get_latest_run("phase1")
    # The latest should be the one with the lexicographically largest name (newest timestamp)
    assert latest.name >= run_id_a


def test_get_latest_run_raises_when_no_runs_exist(isolated_data_dir: Path) -> None:
    with pytest.raises(NoRunsFoundError):
        get_latest_run("phase1")


def test_get_latest_run_raises_when_runs_dir_does_not_exist(isolated_data_dir: Path) -> None:
    with pytest.raises(NoRunsFoundError):
        get_latest_run("phase_that_never_existed")


# ---------- get_run ----------


def test_get_run_returns_correct_path(isolated_data_dir: Path) -> None:
    run_id, run_dir = create_run("phase1")
    result = get_run("phase1", run_id)
    assert result == run_dir


def test_get_run_raises_file_not_found_for_missing_run(isolated_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_run("phase1", "run_nonexistent_20990101_000000")


# ---------- list_runs ----------


def test_list_runs_returns_empty_list_when_no_runs(isolated_data_dir: Path) -> None:
    result = list_runs("phase1")
    assert result == []


def test_list_runs_returns_all_runs(isolated_data_dir: Path) -> None:
    create_run("phase1")
    create_run("phase1")
    runs = list_runs("phase1")
    assert len(runs) == 2


def test_list_runs_includes_run_id_and_phase(isolated_data_dir: Path) -> None:
    run_id, _ = create_run("phase1")
    runs = list_runs("phase1")
    run_ids = [run["run_id"] for run in runs]
    assert run_id in run_ids


# ---------- update_run_status ----------


def test_update_run_status_persists_stage_data(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    update_run_status(run_dir, "build_memories", {"count": 42})
    run_metadata = json.loads((run_dir / "run.json").read_text())
    stage_data = run_metadata["pipeline_status"]["build_memories"]
    assert stage_data["count"] == 42


def test_update_run_status_adds_completed_at_timestamp(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    update_run_status(run_dir, "build_memories")
    run_metadata = json.loads((run_dir / "run.json").read_text())
    assert "completed_at" in run_metadata["pipeline_status"]["build_memories"]


def test_update_run_status_multiple_stages_are_all_stored(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    update_run_status(run_dir, "build_memories", {"count": 10})
    update_run_status(run_dir, "db", {"memory_count": 10})
    run_metadata = json.loads((run_dir / "run.json").read_text())
    assert "build_memories" in run_metadata["pipeline_status"]
    assert "db" in run_metadata["pipeline_status"]


# ---------- create_subrun ----------


def test_create_subrun_creates_results_directory(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    subrun_dir = create_subrun(run_dir, "nomic_embed")
    assert (subrun_dir / "results").is_dir()


def test_create_subrun_writes_subrun_json_with_expected_fields(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    subrun_dir = create_subrun(run_dir, "nomic_embed", description="Embedder swap test")
    subrun_metadata = json.loads((subrun_dir / "subrun.json").read_text())
    assert subrun_metadata["subrun_id"] == "nomic_embed"
    assert subrun_metadata["description"] == "Embedder swap test"
    assert "created_at" in subrun_metadata
    assert "parent_run_id" in subrun_metadata


def test_create_subrun_rejects_forward_slash_in_id(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    with pytest.raises(ValueError):
        create_subrun(run_dir, "bad/subrun_id")


def test_create_subrun_rejects_backslash_in_id(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    with pytest.raises(ValueError):
        create_subrun(run_dir, "bad\\subrun_id")


# ---------- list_subruns ----------


def test_list_subruns_returns_empty_list_when_no_subruns(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    result = list_subruns(run_dir)
    assert result == []


def test_list_subruns_returns_all_created_subruns(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    create_subrun(run_dir, "sub_a")
    create_subrun(run_dir, "sub_b")
    subruns = list_subruns(run_dir)
    subrun_ids = [sub["subrun_id"] for sub in subruns]
    assert "sub_a" in subrun_ids
    assert "sub_b" in subrun_ids


# ---------- get_subrun_db_path ----------


def test_get_subrun_db_path_falls_back_to_parent_memories_db(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    subrun_dir = create_subrun(run_dir, "reranker_only")
    db_path = get_subrun_db_path(subrun_dir)
    assert db_path == str(run_dir / "memories" / "memories.db")


def test_get_subrun_db_path_prefers_own_db_when_it_exists(isolated_data_dir: Path) -> None:
    _, run_dir = create_run("phase1")
    subrun_dir = create_subrun(run_dir, "new_embedder")
    own_db = subrun_dir / "memories.db"
    own_db.touch()
    db_path = get_subrun_db_path(subrun_dir)
    assert db_path == str(own_db)
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_infra_runs.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_infra_runs.py
git commit -m "test: add tests for infra/runs.py"
```

---

## Task 9: Test experiments/test_cases.py

**Files:**
- Create: `tests/test_experiments_test_cases.py`

**Step 1: Create the test file**

```python
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
        json.dumps({
            "context": "Feature: Add authentication",
            "meta": {"sourceBranch": "feature/auth", "targetBranch": "main"},
            "full_diff": diff,
            "code_review_comments": [{"id": comment_id} for comment_id in comment_ids],
        }),
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
    raw_path = _write_raw_pr_file(
        tmp_path, "pr001.json", comment_ids=["comment_x", "comment_y"]
    )
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
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_experiments_test_cases.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_experiments_test_cases.py
git commit -m "test: add tests for experiments/test_cases.py"
```

---

## Task 10: Test experiments/metrics.py pool/dedup functions

**Files:**
- Create: `tests/test_experiments_metrics.py`

**Step 1: Create the test file**

```python
from memory_retrieval.experiments.metrics import (
    pool_and_deduplicate_by_distance,
    pool_and_deduplicate_by_rerank_score,
)


# ---------- pool_and_deduplicate_by_distance ----------


def test_pool_by_distance_same_memory_keeps_minimum_distance() -> None:
    query_results = [
        {"results": [{"id": "m1", "distance": 0.5}]},
        {"results": [{"id": "m1", "distance": 0.2}]},  # lower = better
    ]
    pooled = pool_and_deduplicate_by_distance(query_results)
    assert len(pooled) == 1
    assert pooled[0]["distance"] == 0.2


def test_pool_by_distance_unique_memories_all_included() -> None:
    query_results = [
        {"results": [{"id": "m1", "distance": 0.3}]},
        {"results": [{"id": "m2", "distance": 0.1}]},
        {"results": [{"id": "m3", "distance": 0.7}]},
    ]
    pooled = pool_and_deduplicate_by_distance(query_results)
    assert len(pooled) == 3


def test_pool_by_distance_result_sorted_ascending_by_distance() -> None:
    query_results = [
        {"results": [
            {"id": "m1", "distance": 0.8},
            {"id": "m2", "distance": 0.2},
            {"id": "m3", "distance": 0.5},
        ]},
    ]
    pooled = pool_and_deduplicate_by_distance(query_results)
    distances = [result["distance"] for result in pooled]
    assert distances == sorted(distances)


def test_pool_by_distance_preserves_all_result_fields() -> None:
    query_results = [
        {"results": [{"id": "m1", "distance": 0.3, "situation": "A code situation."}]},
    ]
    pooled = pool_and_deduplicate_by_distance(query_results)
    assert pooled[0]["situation"] == "A code situation."


def test_pool_by_distance_empty_results_returns_empty_list() -> None:
    query_results: list[dict] = []
    pooled = pool_and_deduplicate_by_distance(query_results)
    assert pooled == []


# ---------- pool_and_deduplicate_by_rerank_score ----------


def test_pool_by_rerank_same_memory_keeps_maximum_score() -> None:
    per_query_reranked = [
        {"reranked": [{"id": "m1", "rerank_score": 0.3}]},
        {"reranked": [{"id": "m1", "rerank_score": 0.9}]},  # higher = better
    ]
    pooled = pool_and_deduplicate_by_rerank_score(per_query_reranked)
    assert len(pooled) == 1
    assert pooled[0]["rerank_score"] == 0.9


def test_pool_by_rerank_unique_memories_all_included() -> None:
    per_query_reranked = [
        {"reranked": [{"id": "m1", "rerank_score": 0.9}]},
        {"reranked": [{"id": "m2", "rerank_score": 0.5}]},
        {"reranked": [{"id": "m3", "rerank_score": 0.1}]},
    ]
    pooled = pool_and_deduplicate_by_rerank_score(per_query_reranked)
    assert len(pooled) == 3


def test_pool_by_rerank_result_sorted_descending_by_score() -> None:
    per_query_reranked = [
        {"reranked": [
            {"id": "m1", "rerank_score": 0.3},
            {"id": "m2", "rerank_score": 0.9},
            {"id": "m3", "rerank_score": 0.6},
        ]},
    ]
    pooled = pool_and_deduplicate_by_rerank_score(per_query_reranked)
    scores = [result["rerank_score"] for result in pooled]
    assert scores == sorted(scores, reverse=True)


def test_pool_by_rerank_empty_results_returns_empty_list() -> None:
    per_query_reranked: list[dict] = []
    pooled = pool_and_deduplicate_by_rerank_score(per_query_reranked)
    assert pooled == []
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_experiments_metrics.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_experiments_metrics.py
git commit -m "test: add tests for pool/dedup functions in experiments/metrics.py"
```

---

## Task 11: Test search/fts5.py

**Files:**
- Create: `tests/test_search_fts5.py`

**Step 1: Create the test file**

```python
from pathlib import Path

import pytest

from memory_retrieval.search.fts5 import FTS5Backend


def _make_memory(memory_id: str, variants: list[str], lesson: str) -> dict:
    return {
        "id": memory_id,
        "situation_variants": variants,
        "lesson": lesson,
        "metadata": {"repo": "test-repo"},
        "source": {},
    }


@pytest.fixture
def empty_fts5_db(tmp_path: Path) -> str:
    """An FTS5 database with schema created but no memories inserted."""
    db_path = str(tmp_path / "test.db")
    FTS5Backend().create_database(db_path)
    return db_path


def test_search_returns_matching_memory(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["Python type errors occur when passing wrong arguments"], "Always type check.")],
    )
    results = backend.search(empty_fts5_db, "Python type", limit=5)
    assert len(results) == 1
    assert results[0].id == "mem_001"


def test_search_returns_correct_lesson(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["A situation involving authentication"], "Always validate tokens.")],
    )
    results = backend.search(empty_fts5_db, "authentication")
    assert results[0].lesson == "Always validate tokens."


def test_search_deduplicates_multiple_matching_variants(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["test variant one about validation", "test variant two about validation"], "Lesson A.")],
    )
    # Both variants match "test", but only one result should be returned
    results = backend.search(empty_fts5_db, "test")
    assert len(results) == 1
    assert results[0].id == "mem_001"


def test_search_returns_no_results_for_unmatched_query(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["Python code review scenario"], "Use type hints.")],
    )
    results = backend.search(empty_fts5_db, "totally_unrelated_zzz")
    assert results == []


def test_search_score_is_positive_and_score_type_is_bm25(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["keyword search example with relevant terms"], "Lesson.")],
    )
    results = backend.search(empty_fts5_db, "keyword")
    assert results[0].score > 0
    assert results[0].score_type == "bm25_rank"


def test_search_respects_limit(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    memories = [
        _make_memory(f"mem_{index:03}", [f"test situation {index} about code"], f"Lesson {index}.")
        for index in range(10)
    ]
    backend.insert_memories(empty_fts5_db, memories)
    results = backend.search(empty_fts5_db, "test situation", limit=3)
    assert len(results) <= 3


def test_get_memory_count_returns_correct_count(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    memories = [
        _make_memory("m1", ["first situation"], "First lesson."),
        _make_memory("m2", ["second situation"], "Second lesson."),
        _make_memory("m3", ["third situation"], "Third lesson."),
    ]
    backend.insert_memories(empty_fts5_db, memories)
    assert backend.get_memory_count(empty_fts5_db) == 3


def test_get_memory_by_id_returns_full_memory(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["situation description text"], "Use this lesson.")],
    )
    memory = backend.get_memory_by_id(empty_fts5_db, "mem_001")
    assert memory is not None
    assert memory["id"] == "mem_001"
    assert "situation description text" in memory["situation_variants"]
    assert memory["lesson"] == "Use this lesson."


def test_get_memory_by_id_returns_none_for_nonexistent_id(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    memory = backend.get_memory_by_id(empty_fts5_db, "mem_nonexistent")
    assert memory is None


def test_insert_memories_returns_inserted_count(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    memories = [
        _make_memory("m1", ["first"], "First."),
        _make_memory("m2", ["second"], "Second."),
    ]
    count = backend.insert_memories(empty_fts5_db, memories)
    assert count == 2
```

**Step 2: Run and verify all pass**

```bash
uv run pytest tests/test_search_fts5.py -v
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/test_search_fts5.py
git commit -m "test: add tests for search/fts5.py"
```

---

## Task 12: Final verification — run full test suite

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass, no errors. Review the output and check for any unexpected failures.

**Step 2: Check coverage summary (optional)**

```bash
uv run pytest tests/ --tb=short -q
```

Expected: summary showing all tests passed.

**Step 3: Final commit (if any fixes were needed)**

If any tests revealed real bugs in the code, fix the bug (not the test), then:

```bash
git add -p
git commit -m "fix: <description of actual bug found>"
```
