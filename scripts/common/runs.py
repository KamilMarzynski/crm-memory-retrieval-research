"""
Run isolation system for experiment tracking.

This module provides a system for isolating each pipeline execution ("run")
into its own directory, preventing output mixing between runs while allowing
automatic discovery of the latest outputs.

Directory Structure:
    data/<phase>/
      runs/
        run_20260208_143022/              # Timestamp-based run ID
          run.json                         # Run metadata
          memories/
            memories_*.jsonl
            rejected_*.jsonl
            memories.db
          test_cases/
            *.json
          results/
            results_*.json

Functions:
    create_run: Create new run directory with metadata
    get_latest_run: Get most recent run directory
    get_run: Get specific run by ID
    list_runs: List all runs with metadata
    update_run_status: Update run.json pipeline status

Example:
    >>> from common.runs import create_run, get_latest_run
    >>> run_id, run_dir = create_run("phase1")
    >>> print(f"Created run: {run_id}")
    >>> latest = get_latest_run("phase1")
    >>> print(f"Latest run: {latest}")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict


class PipelineStageStatus(TypedDict, total=False):
    """Status information for a pipeline stage."""
    completed_at: str
    count: int
    failed: int
    memory_count: int
    prompt_version: Optional[str]
    distance_threshold: Optional[float]


class RunMetadata(TypedDict, total=False):
    """Metadata structure for a run."""
    run_id: str
    created_at: str
    phase: str
    description: Optional[str]
    pipeline_status: Dict[str, PipelineStageStatus]
    run_dir: Path  # Added when returned from list_runs


class NoRunsFoundError(Exception):
    """Raised when no runs exist for a phase."""

    def __init__(self, phase: str, runs_dir: Path):
        self.phase = phase
        self.runs_dir = runs_dir
        super().__init__(
            f"No runs found for {phase}. "
            f"Expected runs in: {runs_dir}\n"
            f"Create a new run with: create_run('{phase}')"
        )


# Base data directory
DATA_DIR = Path("data")

# Run directory prefix
RUN_PREFIX = "run_"

# Run metadata filename
RUN_METADATA_FILE = "run.json"

# Phase identifiers
PHASE1 = "phase1"


def _get_runs_dir(phase: str) -> Path:
    """Get the runs directory for a phase."""
    return DATA_DIR / phase / "runs"


def _generate_run_id() -> str:
    """Generate a timestamp-based run ID."""
    return f"{RUN_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _load_run_metadata(run_dir: Path) -> RunMetadata:
    """Load run.json metadata from a run directory."""
    metadata_path = run_dir / RUN_METADATA_FILE
    if not metadata_path.exists():
        return RunMetadata()
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_run_metadata(run_dir: Path, metadata: RunMetadata) -> None:
    """Save run.json metadata to a run directory."""
    metadata_path = run_dir / RUN_METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def create_run(
    phase: str,
    description: Optional[str] = None,
) -> Tuple[str, Path]:
    """
    Create a new run directory with initial metadata.

    Creates the run directory structure with subdirectories for memories,
    test_cases, and results. Initializes run.json with creation timestamp.

    Args:
        phase: Phase identifier (e.g., "phase1").
        description: Optional description of the run purpose.

    Returns:
        Tuple of (run_id, run_dir) where run_id is the generated ID
        and run_dir is the Path to the run directory.

    Example:
        >>> run_id, run_dir = create_run("phase1", description="Testing v2 prompts")
        >>> print(f"Created: {run_dir}")
    """
    run_id = _generate_run_id()
    runs_dir = _get_runs_dir(phase)
    run_dir = runs_dir / run_id

    # Create directory structure
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "memories").mkdir(exist_ok=True)
    (run_dir / "test_cases").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)

    # Create initial metadata
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "phase": phase,
        "description": description,
        "pipeline_status": {},
    }
    _save_run_metadata(run_dir, metadata)

    return run_id, run_dir


def get_latest_run(phase: str) -> Path:
    """
    Get the most recent run directory for a phase.

    Runs are sorted by their timestamp-based IDs (lexicographic sort
    works correctly for YYYYMMDD_HHMMSS format).

    Args:
        phase: Phase identifier (e.g., "phase1").

    Returns:
        Path to the most recent run directory.

    Raises:
        NoRunsFoundError: If no runs exist for the phase.

    Example:
        >>> latest = get_latest_run("phase1")
        >>> print(f"Using run: {latest.name}")
    """
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
    """
    Get a specific run directory by ID.

    Args:
        phase: Phase identifier (e.g., "phase1").
        run_id: Run identifier (e.g., "run_20260208_143022").

    Returns:
        Path to the run directory.

    Raises:
        FileNotFoundError: If the run directory doesn't exist.

    Example:
        >>> run_dir = get_run("phase1", "run_20260208_143022")
        >>> db_path = run_dir / "memories" / "memories.db"
    """
    runs_dir = _get_runs_dir(phase)
    run_dir = runs_dir / run_id

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run '{run_id}' not found for {phase}. "
            f"Expected directory: {run_dir}"
        )

    return run_dir


def list_runs(phase: str) -> List[RunMetadata]:
    """
    List all runs for a phase with their metadata.

    Returns runs sorted by creation time (most recent first).

    Args:
        phase: Phase identifier (e.g., "phase1").

    Returns:
        List of dictionaries containing run metadata. Each dict includes:
        - run_id: The run identifier
        - run_dir: Path to the run directory
        - created_at: ISO timestamp of creation
        - description: Optional description
        - pipeline_status: Dict of stage completion info

    Example:
        >>> runs = list_runs("phase1")
        >>> for run in runs:
        ...     print(f"{run['run_id']}: {run.get('description', 'no description')}")
    """
    runs_dir = _get_runs_dir(phase)

    if not runs_dir.exists():
        return []

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(RUN_PREFIX)],
        key=lambda x: x.name,
        reverse=True,
    )

    runs: List[RunMetadata] = []
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
    info: Optional[PipelineStageStatus] = None,
) -> None:
    """
    Update the pipeline status in run.json.

    Records completion information for a pipeline stage.

    Args:
        run_dir: Path to the run directory.
        stage: Pipeline stage name (e.g., "build_memories", "db", "test_cases", "experiment").
        info: Optional dictionary with stage-specific info (e.g., {"count": 41}).
              Automatically adds "completed_at" timestamp.

    Example:
        >>> update_run_status(run_dir, "build_memories", {"count": 41})
        >>> update_run_status(run_dir, "db", {"memory_count": 41})
    """
    metadata = _load_run_metadata(run_dir)

    if "pipeline_status" not in metadata:
        metadata["pipeline_status"] = {}

    stage_info = info.copy() if info else {}
    stage_info["completed_at"] = datetime.now().isoformat()

    metadata["pipeline_status"][stage] = stage_info

    _save_run_metadata(run_dir, metadata)


__all__ = [
    "NoRunsFoundError",
    "RunMetadata",
    "PipelineStageStatus",
    "PHASE1",
    "create_run",
    "get_latest_run",
    "get_run",
    "list_runs",
    "update_run_status",
]
