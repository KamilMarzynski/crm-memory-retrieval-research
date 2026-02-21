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
    create_run("phase1")
    _, newer_run_dir = create_run("phase1")
    latest = get_latest_run("phase1")
    assert latest == newer_run_dir


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
