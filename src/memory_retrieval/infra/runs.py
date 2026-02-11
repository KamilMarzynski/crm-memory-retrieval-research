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

PHASE1 = "phase1"
PHASE2 = "phase2"


def _get_runs_dir(phase: str) -> Path:
    return DATA_DIR / phase / "runs"


def _generate_run_id() -> str:
    return f"{RUN_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / RUN_METADATA_FILE
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = run_dir / RUN_METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


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

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "phase": phase,
        "description": description,
        "pipeline_status": {},
    }
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
            f"Run '{run_id}' not found for {phase}. "
            f"Expected directory: {run_dir}"
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
