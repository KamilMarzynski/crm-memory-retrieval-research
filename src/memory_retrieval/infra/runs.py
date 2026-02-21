import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict


class PipelineStageStatus(TypedDict, total=False):
    completed_at: str
    count: int
    failed: int
    memory_count: int
    prompt_version: str | None
    distance_threshold: float | None


class RunMetadata(TypedDict, total=False):
    run_id: str
    created_at: str
    phase: str
    description: str | None
    pipeline_status: dict[str, PipelineStageStatus]
    run_dir: Path


class NoRunsFoundError(Exception):
    def __init__(self, phase: str, runs_dir: Path):
        self.phase = phase
        self.runs_dir = runs_dir
        super().__init__(
            f"No runs found for {phase}. "
            f"Expected runs in: {runs_dir}\n"
            f"Create a new run with: create_run('{phase}')"
        )


DATA_DIR = Path("data")
RUN_PREFIX = "run_"
RUN_METADATA_FILE = "run.json"
SUBRUNS_DIR = "subruns"
SUBRUN_METADATA_FILE = "subrun.json"

PHASE1 = "phase1"
PHASE2 = "phase2"


def _get_runs_dir(phase: str) -> Path:
    return DATA_DIR / phase / "runs"


def _generate_run_id() -> str:
    """Generate a timestamp-based run ID with microsecond precision.

    Microseconds prevent collisions when two runs are created within the same
    second (e.g. in batch_runner.py running multiple pipeline configurations).
    """
    return f"{RUN_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / RUN_METADATA_FILE
    if not metadata_path.exists():
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def _save_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = run_dir / RUN_METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _build_run_metadata(
    run_id: str,
    phase: str,
    description: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "phase": phase,
        "description": description,
        "pipeline_status": {},
    }


def create_run(
    phase: str,
    description: str | None = None,
) -> tuple[str, Path]:
    run_id = _generate_run_id()
    runs_dir = _get_runs_dir(phase)
    run_dir = runs_dir / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "memories").mkdir(exist_ok=True)
    (run_dir / "test_cases").mkdir(exist_ok=True)
    (run_dir / "queries").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)

    metadata = _build_run_metadata(run_id, phase, description)
    _save_run_metadata(run_dir, metadata)

    return run_id, run_dir


def get_latest_run(phase: str) -> Path:
    runs_dir = _get_runs_dir(phase)

    if not runs_dir.exists():
        raise NoRunsFoundError(phase, runs_dir)

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(RUN_PREFIX)],
        key=lambda x: x.name,
        reverse=True,
    )

    if not run_dirs:
        raise NoRunsFoundError(phase, runs_dir)

    return run_dirs[0]


def get_run(phase: str, run_id: str) -> Path:
    runs_dir = _get_runs_dir(phase)
    run_dir = runs_dir / run_id

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run '{run_id}' not found for {phase}. Expected directory: {run_dir}"
        )

    return run_dir


def list_runs(phase: str) -> list[dict[str, Any]]:
    runs_dir = _get_runs_dir(phase)

    if not runs_dir.exists():
        return []

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(RUN_PREFIX)],
        key=lambda x: x.name,
        reverse=True,
    )

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        metadata = _load_run_metadata(run_dir)
        metadata["run_dir"] = run_dir
        if "run_id" not in metadata:
            metadata["run_id"] = run_dir.name
        runs.append(metadata)

    return runs


def update_run_status(
    run_dir: Path,
    stage: str,
    info: dict[str, Any] | None = None,
) -> None:
    metadata = _load_run_metadata(run_dir)

    if "pipeline_status" not in metadata:
        metadata["pipeline_status"] = {}

    stage_info: dict[str, Any] = dict(info) if info else {}
    stage_info["completed_at"] = datetime.now().isoformat()

    metadata["pipeline_status"][stage] = stage_info

    _save_run_metadata(run_dir, metadata)


def update_config_fingerprint(
    run_dir: Path,
    fingerprint: dict[str, Any],
) -> None:
    """Store a config fingerprint in run.json for cross-run comparison."""
    metadata = _load_run_metadata(run_dir)
    metadata["config_fingerprint"] = fingerprint
    _save_run_metadata(run_dir, metadata)


# ---------------------------------------------------------------------------
# Subrun system — nested experiments inside a parent run
# ---------------------------------------------------------------------------


def _build_subrun_metadata(
    subrun_id: str,
    parent_run_id: str,
    description: str | None,
) -> dict[str, Any]:
    return {
        "subrun_id": subrun_id,
        "parent_run_id": parent_run_id,
        "created_at": datetime.now().isoformat(),
        "description": description,
        "pipeline_status": {},
    }


def create_subrun(
    parent_run_dir: Path,
    subrun_id: str,
    description: str | None = None,
) -> Path:
    """Create a subrun directory inside a parent run.

    Subruns share the parent's memories JSONL, test_cases, and queries.
    They have their own results/ directory and optionally their own memories.db
    (when the embedding model changes).

    Args:
        parent_run_dir: Path to the parent run directory.
        subrun_id: Short identifier for the subrun (e.g., "nomic_embed").
        description: Optional human-readable description.

    Returns:
        Path to the created subrun directory.
    """
    if not (parent_run_dir / RUN_METADATA_FILE).exists():
        raise ValueError(f"{parent_run_dir} is not a valid run directory (no {RUN_METADATA_FILE})")

    if "/" in subrun_id or "\\" in subrun_id:
        raise ValueError(f"subrun_id cannot contain path separators: {subrun_id}")

    subrun_dir = parent_run_dir / SUBRUNS_DIR / subrun_id
    subrun_dir.mkdir(parents=True, exist_ok=True)
    (subrun_dir / "results").mkdir(exist_ok=True)

    parent_metadata = _load_run_metadata(parent_run_dir)
    parent_run_id = parent_metadata.get("run_id", parent_run_dir.name)

    subrun_metadata = _build_subrun_metadata(subrun_id, parent_run_id, description)

    metadata_path = subrun_dir / SUBRUN_METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(subrun_metadata, f, indent=2, ensure_ascii=False)

    return subrun_dir


def list_subruns(parent_run_dir: Path) -> list[dict[str, Any]]:
    """List all subruns of a given parent run.

    Returns:
        List of subrun metadata dicts, each including a "subrun_dir" key.
    """
    subruns_dir = parent_run_dir / SUBRUNS_DIR
    if not subruns_dir.exists():
        return []

    subruns: list[dict[str, Any]] = []
    for subrun_dir in sorted(subruns_dir.iterdir()):
        if not subrun_dir.is_dir():
            continue
        metadata_path = subrun_dir / SUBRUN_METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["subrun_dir"] = subrun_dir
            subruns.append(metadata)

    return subruns


def get_subrun(parent_run_dir: Path, subrun_id: str) -> Path:
    """Get a specific subrun directory.

    Args:
        parent_run_dir: Path to the parent run directory.
        subrun_id: The subrun identifier.

    Returns:
        Path to the subrun directory.

    Raises:
        FileNotFoundError: If the subrun does not exist.
    """
    subrun_dir = parent_run_dir / SUBRUNS_DIR / subrun_id
    if not subrun_dir.exists():
        raise FileNotFoundError(
            f"Subrun '{subrun_id}' not found in {parent_run_dir}. Expected directory: {subrun_dir}"
        )
    return subrun_dir


def get_subrun_db_path(subrun_dir: Path) -> str:
    """Return the subrun's own memories.db if it exists, otherwise the parent's.

    For embedder-change subruns, the caller rebuilds the DB in the subrun dir.
    For reranker-only subruns, no DB is created and the parent's DB is used.

    Args:
        subrun_dir: Path to the subrun directory.

    Returns:
        String path to the appropriate memories.db file.
    """
    subrun_db = subrun_dir / "memories.db"
    if subrun_db.exists():
        return str(subrun_db)

    # Fall back to parent's memories.db
    parent_run_dir = subrun_dir.parent.parent  # subruns/<id> → subruns → parent
    parent_db = parent_run_dir / "memories" / "memories.db"
    return str(parent_db)


def get_subrun_paths(subrun_dir: Path) -> dict[str, str]:
    """Return all paths needed for experiments, resolving to parent paths where appropriate.

    Subruns share memories JSONL, test_cases, and queries from the parent.
    They have their own db_path (if embedder changed) and results_dir.

    Args:
        subrun_dir: Path to the subrun directory.

    Returns:
        Dict with keys: memories_dir, test_cases_dir, queries_dir, db_path, results_dir.
    """
    parent_run_dir = subrun_dir.parent.parent  # subruns/<id> → subruns → parent
    return {
        "memories_dir": str(parent_run_dir / "memories"),
        "test_cases_dir": str(parent_run_dir / "test_cases"),
        "queries_dir": str(parent_run_dir / "queries"),
        "db_path": get_subrun_db_path(subrun_dir),
        "results_dir": str(subrun_dir / "results"),
    }


def update_subrun_status(
    subrun_dir: Path,
    stage: str,
    info: dict[str, Any] | None = None,
) -> None:
    """Update pipeline_status in subrun.json (mirrors update_run_status for subruns)."""
    metadata_path = subrun_dir / SUBRUN_METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"No {SUBRUN_METADATA_FILE} found in {subrun_dir}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    if "pipeline_status" not in metadata:
        metadata["pipeline_status"] = {}

    stage_info: dict[str, Any] = dict(info) if info else {}
    stage_info["completed_at"] = datetime.now().isoformat()
    metadata["pipeline_status"][stage] = stage_info

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
