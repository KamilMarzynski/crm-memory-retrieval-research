"""Batch experiment runner for unattended multi-run pipeline execution.

Supports two batch modes:

1. Full pipeline batch (BatchFullPipelineConfig):
   Runs N independent full pipelines: extract memories → build DB → build test cases
   → generate queries → run experiments → generate run summary.
   Each run is independent, isolated, and crash-safe.

2. Subrun batch (BatchSubrunConfig):
   Runs experiments inside new subruns of existing parent runs.
   Useful for testing a new reranker or embedder against already-generated queries.

Usage:
    from memory_retrieval.experiments.batch_runner import (
        BatchFullPipelineConfig,
        BatchSubrunConfig,
        run_full_pipeline_batch,
        run_subrun_batch,
    )

Background execution:
    nohup uv run python run_batch.py > /dev/null 2>&1 &
    # Or watch live:
    uv run python run_batch.py 2>&1 | tee -a data/phase2/batches/current_batch.log &

Check progress:
    tail -f data/phase2/batches/batch_20260218_143022/batch.log
"""

import json
import logging
import sys
import traceback as traceback_module
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_retrieval.experiments.comparison import (
    build_config_fingerprint,
    generate_run_summary,
)
from memory_retrieval.experiments.query_generation import (
    QueryGenerationConfig,
    generate_all_queries,
)
from memory_retrieval.experiments.runner import ExperimentConfig, run_all_experiments
from memory_retrieval.experiments.test_cases import build_test_cases
from memory_retrieval.infra.io import load_json
from memory_retrieval.infra.runs import (
    _load_run_metadata,
    _save_run_metadata,
    create_run,
    create_subrun,
    get_run,
    get_subrun_paths,
    update_config_fingerprint,
    update_run_status,
    update_subrun_status,
)
from memory_retrieval.memories.extractor import ExtractionConfig, extract_memories
from memory_retrieval.search.vector import DEFAULT_EMBEDDING_MODEL, VectorBackend

DATA_DIR = Path("data")
BATCHES_SUBDIR = "batches"
BATCH_PREFIX = "batch_"
BATCH_MANIFEST_FILE = "batch_manifest.json"
BATCH_LOG_FILE = "batch.log"
UNKNOWN_VERSION = "unknown"  # Fallback for config fields not recorded in run metadata


@dataclass
class BatchFullPipelineConfig:
    """Configuration for running N independent full pipeline runs unattended.

    Each run executes all pipeline stages: extract memories → build DB →
    build test cases → generate queries → run experiments → generate summary.
    """

    phase: str
    num_runs: int
    extraction_config: ExtractionConfig
    query_config: QueryGenerationConfig
    experiment_config: ExperimentConfig
    raw_data_dir: str = "data/review_data"
    description: str | None = None


@dataclass
class BatchSubrunConfig:
    """Configuration for running experiments inside new subruns of existing parent runs.

    Creates one subrun per parent run ID. Useful for testing a new reranker or
    embedder configuration against already-generated test cases and queries.
    """

    phase: str
    parent_run_ids: list[str]
    experiment_config: ExperimentConfig
    rebuild_db: bool = False
    description: str | None = None


@dataclass
class SingleRunOutcome:
    """Result of a single run within a batch."""

    run_index: int
    status: str  # "completed" | "failed"
    run_id: str | None  # None if run creation itself failed
    run_dir: Path | None
    started_at: str
    completed_at: str
    optimal_f1: float | None
    error: str | None
    traceback: str | None


@dataclass
class BatchOutcome:
    """Result of a complete batch execution."""

    batch_id: str
    batch_dir: Path
    outcomes: list[SingleRunOutcome]
    num_completed: int
    num_failed: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_full_pipeline_batch(config: BatchFullPipelineConfig) -> BatchOutcome:
    """Run N independent full pipeline runs unattended.

    Each run is crash-isolated — a failure in one run does not stop the batch.
    Progress is written to batch_manifest.json after each run.

    Args:
        config: Batch configuration specifying number of runs and pipeline settings.

    Returns:
        BatchOutcome with results from all runs.
    """
    batch_id = f"{BATCH_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_started_at = datetime.now().isoformat()
    batch_dir = DATA_DIR / config.phase / BATCHES_SUBDIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_batch_logging(batch_dir, batch_id)
    logger.info("=" * 60)
    logger.info("BATCH START: %s", batch_id)
    logger.info("Phase: %s | Runs requested: %d", config.phase, config.num_runs)
    if config.description:
        logger.info("Description: %s", config.description)
    logger.info("=" * 60)

    outcomes: list[SingleRunOutcome] = []

    for run_index in range(config.num_runs):
        logger.info("")
        logger.info("--- Run %d/%d ---", run_index + 1, config.num_runs)
        outcome = _run_one_full_pipeline(run_index, config, batch_id, logger)
        outcomes.append(outcome)
        _save_batch_manifest(
            batch_dir, batch_id, "full_pipeline", config, batch_started_at, outcomes
        )

        if outcome.status == "completed":
            f1_display = f"{outcome.optimal_f1:.4f}" if outcome.optimal_f1 is not None else "n/a"
            logger.info(
                "Run %d/%d COMPLETED — run_id=%s  optimal_f1=%s",
                run_index + 1,
                config.num_runs,
                outcome.run_id,
                f1_display,
            )
        else:
            logger.error("Run %d/%d FAILED — %s", run_index + 1, config.num_runs, outcome.error)

    num_completed = sum(1 for outcome in outcomes if outcome.status == "completed")
    num_failed = sum(1 for outcome in outcomes if outcome.status == "failed")

    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH COMPLETE: %s", batch_id)
    logger.info("Completed: %d / %d  |  Failed: %d", num_completed, config.num_runs, num_failed)
    logger.info("Batch dir: %s", batch_dir)
    logger.info("=" * 60)
    _log_summary_table(logger, outcomes)

    return BatchOutcome(
        batch_id=batch_id,
        batch_dir=batch_dir,
        outcomes=outcomes,
        num_completed=num_completed,
        num_failed=num_failed,
    )


def run_subrun_batch(config: BatchSubrunConfig) -> BatchOutcome:
    """Run experiments inside new subruns of existing parent runs.

    Creates one subrun per parent run ID. Each subrun is crash-isolated.

    Args:
        config: Subrun batch configuration with parent run IDs and experiment settings.

    Returns:
        BatchOutcome with results from all subruns.
    """
    batch_id = f"{BATCH_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_started_at = datetime.now().isoformat()
    batch_dir = DATA_DIR / config.phase / BATCHES_SUBDIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_batch_logging(batch_dir, batch_id)
    logger.info("=" * 60)
    logger.info("SUBRUN BATCH START: %s", batch_id)
    logger.info("Phase: %s | Parent runs: %d", config.phase, len(config.parent_run_ids))
    logger.info("Rebuild DB: %s", config.rebuild_db)
    if config.description:
        logger.info("Description: %s", config.description)
    logger.info("=" * 60)

    outcomes: list[SingleRunOutcome] = []

    for run_index, parent_run_id in enumerate(config.parent_run_ids):
        logger.info("")
        logger.info(
            "--- Subrun %d/%d (parent: %s) ---",
            run_index + 1,
            len(config.parent_run_ids),
            parent_run_id,
        )
        outcome = _run_one_subrun(run_index, parent_run_id, config, batch_id, logger)
        outcomes.append(outcome)
        _save_batch_manifest(batch_dir, batch_id, "subrun", config, batch_started_at, outcomes)

        if outcome.status == "completed":
            f1_display = f"{outcome.optimal_f1:.4f}" if outcome.optimal_f1 is not None else "n/a"
            logger.info(
                "Subrun %d/%d COMPLETED — run_id=%s  optimal_f1=%s",
                run_index + 1,
                len(config.parent_run_ids),
                outcome.run_id,
                f1_display,
            )
        else:
            logger.error(
                "Subrun %d/%d FAILED — %s",
                run_index + 1,
                len(config.parent_run_ids),
                outcome.error,
            )

    num_completed = sum(1 for outcome in outcomes if outcome.status == "completed")
    num_failed = sum(1 for outcome in outcomes if outcome.status == "failed")

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUBRUN BATCH COMPLETE: %s", batch_id)
    logger.info(
        "Completed: %d / %d  |  Failed: %d", num_completed, len(config.parent_run_ids), num_failed
    )
    logger.info("Batch dir: %s", batch_dir)
    logger.info("=" * 60)
    _log_summary_table(logger, outcomes)

    return BatchOutcome(
        batch_id=batch_id,
        batch_dir=batch_dir,
        outcomes=outcomes,
        num_completed=num_completed,
        num_failed=num_failed,
    )


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------


def _run_one_full_pipeline(
    run_index: int,
    config: BatchFullPipelineConfig,
    batch_id: str,
    logger: logging.Logger,
) -> SingleRunOutcome:
    """Execute one full pipeline run. Returns outcome regardless of success or failure."""
    started_at = datetime.now().isoformat()
    run_id: str | None = None
    run_dir: Path | None = None

    try:
        # Stage 1: Create run
        logger.info("[%d] Creating run...", run_index)
        run_id, run_dir = create_run(config.phase, description=config.description)
        logger.info("[%d] Created run: %s", run_index, run_id)

        # Record batch membership in run.json
        run_metadata = _load_run_metadata(run_dir)
        run_metadata["batch_id"] = batch_id
        run_metadata["batch_index"] = run_index
        _save_run_metadata(run_dir, run_metadata)

        memories_dir = str(run_dir / "memories")
        test_cases_dir = str(run_dir / "test_cases")
        queries_dir = str(run_dir / "queries")
        results_dir = str(run_dir / "results")
        db_path = str(run_dir / "memories" / "memories.db")

        # Stage 2: Extract memories from all raw data files
        logger.info("[%d] Extracting memories from %s...", run_index, config.raw_data_dir)
        raw_data_files = sorted(Path(config.raw_data_dir).glob("*.json"))
        if not raw_data_files:
            raise FileNotFoundError(f"No JSON files found in {config.raw_data_dir}")

        for raw_file in raw_data_files:
            extract_memories(str(raw_file), memories_dir, config.extraction_config)

        # Count after extraction — extract_memories may filter or merge records,
        # so we read the final JSONL output rather than counting input files.
        total_memories_written = sum(
            1
            for memory_file in Path(memories_dir).glob("memories_*.jsonl")
            for line in memory_file.open(encoding="utf-8")
            if line.strip()
        )

        update_run_status(
            run_dir,
            "build_memories",
            {
                "count": total_memories_written,
                "prompt_version": getattr(config.extraction_config, "prompt_version", None),
            },
        )
        logger.info("[%d] Extracted %d memories", run_index, total_memories_written)

        # Stage 3: Build vector database
        logger.info("[%d] Building vector database...", run_index)
        search_backend = config.experiment_config.search_backend
        if isinstance(search_backend, VectorBackend):
            search_backend.rebuild_database(db_path, memories_dir)
        else:
            raise TypeError(
                f"Full pipeline batch requires a VectorBackend, got {type(search_backend).__name__}"
            )

        update_run_status(run_dir, "db", {"memory_count": total_memories_written})
        logger.info("[%d] Database built: %s", run_index, db_path)

        # Stage 4: Build test cases
        logger.info("[%d] Building test cases...", run_index)
        build_test_cases(config.raw_data_dir, memories_dir, test_cases_dir)

        test_case_count = len(list(Path(test_cases_dir).glob("*.json")))
        update_run_status(run_dir, "test_cases", {"count": test_case_count})
        logger.info("[%d] Built %d test cases", run_index, test_case_count)

        # Stage 5: Generate queries (LLM — costs money)
        logger.info("[%d] Generating queries (LLM)...", run_index)
        generate_all_queries(
            test_cases_dir=test_cases_dir,
            queries_dir=queries_dir,
            config=config.query_config,
            db_path=db_path,
            search_backend=search_backend,
        )

        query_file_count = len(list(Path(queries_dir).glob("*.json")))
        update_run_status(
            run_dir,
            "query_generation",
            {
                "count": query_file_count,
                "model": config.query_config.model,
            },
        )
        logger.info("[%d] Generated %d query files", run_index, query_file_count)

        # Stage 6: Run experiments
        logger.info("[%d] Running experiments...", run_index)
        run_all_experiments(
            test_cases_dir=test_cases_dir,
            queries_dir=queries_dir,
            db_path=db_path,
            results_dir=results_dir,
            config=config.experiment_config,
        )

        result_count = len(list(Path(results_dir).glob("*.json")))
        update_run_status(run_dir, "experiment", {"count": result_count})
        logger.info("[%d] Ran %d experiments", run_index, result_count)

        # Stage 7: Build config fingerprint
        logger.info("[%d] Building config fingerprint...", run_index)
        fingerprint = _build_fingerprint_for_full_pipeline(run_dir, config)
        update_config_fingerprint(run_dir, fingerprint)

        # Stage 8: Generate run summary
        logger.info("[%d] Generating run summary...", run_index)
        rerank_strategies = (
            list(config.experiment_config.rerank_text_strategies.keys())
            if config.experiment_config.rerank_text_strategies
            else None
        )
        generate_run_summary(run_dir, strategies=rerank_strategies)

        optimal_f1 = _extract_optimal_f1_from_summary(run_dir)
        logger.info(
            "[%d] Run summary generated  optimal_f1=%s",
            run_index,
            f"{optimal_f1:.4f}" if optimal_f1 is not None else "n/a",
        )

        return SingleRunOutcome(
            run_index=run_index,
            status="completed",
            run_id=run_id,
            run_dir=run_dir,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            optimal_f1=optimal_f1,
            error=None,
            traceback=None,
        )

    except KeyboardInterrupt:
        raise
    except Exception as error:
        error_traceback = traceback_module.format_exc()
        logger.error("[%d] FAILED: %s", run_index, error, exc_info=True)

        return SingleRunOutcome(
            run_index=run_index,
            status="failed",
            run_id=run_id,
            run_dir=run_dir,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            optimal_f1=None,
            error=str(error),
            traceback=error_traceback,
        )


def _run_one_subrun(
    run_index: int,
    parent_run_id: str,
    config: BatchSubrunConfig,
    batch_id: str,
    logger: logging.Logger,
) -> SingleRunOutcome:
    """Execute one subrun. Returns outcome regardless of success or failure."""
    started_at = datetime.now().isoformat()
    subrun_id: str | None = None
    subrun_dir: Path | None = None

    try:
        # Stage 1: Locate parent run and create subrun
        logger.info("[%d] Locating parent run: %s", run_index, parent_run_id)
        parent_run_dir = get_run(config.phase, parent_run_id)

        subrun_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info("[%d] Creating subrun: %s", run_index, subrun_id)
        subrun_dir = create_subrun(parent_run_dir, subrun_id, description=config.description)

        # Record batch membership and config fingerprint in a single subrun.json write.
        # Fingerprint is built from parent_run_dir metadata (available now, before experiments),
        # so we can write everything upfront and let update_subrun_status handle the rest.
        subrun_metadata_path = subrun_dir / "subrun.json"
        with open(subrun_metadata_path, encoding="utf-8") as subrun_file:
            subrun_metadata = json.load(subrun_file)
        subrun_metadata["batch_id"] = batch_id
        subrun_metadata["batch_index"] = run_index
        subrun_metadata["config_fingerprint"] = _build_fingerprint_for_subrun(
            parent_run_dir, config
        )
        with open(subrun_metadata_path, "w", encoding="utf-8") as subrun_file:
            json.dump(subrun_metadata, subrun_file, indent=2, ensure_ascii=False)

        paths = get_subrun_paths(subrun_dir)
        parent_memories_dir = paths["memories_dir"]

        # Stage 2: Optionally rebuild DB with new embedder
        if config.rebuild_db:
            logger.info("[%d] Rebuilding DB in subrun dir...", run_index)
            search_backend = config.experiment_config.search_backend
            if not isinstance(search_backend, VectorBackend):
                raise TypeError(
                    f"rebuild_db=True requires a VectorBackend, got {type(search_backend).__name__}"
                )
            subrun_db_path = str(subrun_dir / "memories.db")
            search_backend.rebuild_database(subrun_db_path, parent_memories_dir)
            update_subrun_status(subrun_dir, "db", {"rebuilt": True})
            logger.info("[%d] DB rebuilt: %s", run_index, subrun_db_path)
        else:
            logger.info("[%d] Using parent DB: %s", run_index, paths["db_path"])

        # Stage 3: Run experiments
        logger.info("[%d] Running experiments...", run_index)
        run_all_experiments(
            test_cases_dir=paths["test_cases_dir"],
            queries_dir=paths["queries_dir"],
            db_path=paths["db_path"],
            results_dir=paths["results_dir"],
            config=config.experiment_config,
        )

        result_count = len(list(Path(paths["results_dir"]).glob("*.json")))
        update_subrun_status(subrun_dir, "experiment", {"count": result_count})
        logger.info("[%d] Ran %d experiments", run_index, result_count)

        # Stage 4: Generate run summary
        logger.info("[%d] Generating run summary...", run_index)
        rerank_strategies = (
            list(config.experiment_config.rerank_text_strategies.keys())
            if config.experiment_config.rerank_text_strategies
            else None
        )
        generate_run_summary(
            subrun_dir,
            strategies=rerank_strategies,
            results_dir=paths["results_dir"],
        )

        optimal_f1 = _extract_optimal_f1_from_summary(subrun_dir)
        logger.info(
            "[%d] Subrun summary generated  optimal_f1=%s",
            run_index,
            f"{optimal_f1:.4f}" if optimal_f1 is not None else "n/a",
        )

        return SingleRunOutcome(
            run_index=run_index,
            status="completed",
            run_id=subrun_id,
            run_dir=subrun_dir,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            optimal_f1=optimal_f1,
            error=None,
            traceback=None,
        )

    except KeyboardInterrupt:
        raise
    except Exception as error:
        error_traceback = traceback_module.format_exc()
        logger.error("[%d] FAILED: %s", run_index, error, exc_info=True)

        return SingleRunOutcome(
            run_index=run_index,
            status="failed",
            run_id=subrun_id,
            run_dir=subrun_dir,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            optimal_f1=None,
            error=str(error),
            traceback=error_traceback,
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_batch_logging(batch_dir: Path, batch_id: str) -> logging.Logger:
    """Create a logger that writes to both stdout and a per-batch log file."""
    logger = logging.getLogger(f"batch.{batch_id}")
    logger.setLevel(logging.DEBUG)

    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    log_file_path = batch_dir / BATCH_LOG_FILE
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _log_summary_table(logger: logging.Logger, outcomes: list[SingleRunOutcome]) -> None:
    """Print a summary table of all run outcomes."""
    logger.info("")
    logger.info("%-6s  %-30s  %-10s  %-10s  %s", "Index", "Run ID", "Status", "Optimal F1", "Error")
    logger.info("-" * 90)
    for outcome in outcomes:
        run_id_display = outcome.run_id or "(none)"
        f1_display = f"{outcome.optimal_f1:.4f}" if outcome.optimal_f1 is not None else "n/a"
        error_display = (outcome.error or "")[:40]
        logger.info(
            "%-6d  %-30s  %-10s  %-10s  %s",
            outcome.run_index,
            run_id_display,
            outcome.status,
            f1_display,
            error_display,
        )


# ---------------------------------------------------------------------------
# Batch manifest
# ---------------------------------------------------------------------------


def _save_batch_manifest(
    batch_dir: Path,
    batch_id: str,
    batch_type: str,
    config: BatchFullPipelineConfig | BatchSubrunConfig,
    created_at: str,
    outcomes: list[SingleRunOutcome],
) -> None:
    """Write batch_manifest.json incrementally after each run."""
    num_completed = sum(1 for outcome in outcomes if outcome.status == "completed")
    num_failed = sum(1 for outcome in outcomes if outcome.status == "failed")

    runs_data: list[dict[str, Any]] = []
    for outcome in outcomes:
        run_entry: dict[str, Any] = {
            "run_index": outcome.run_index,
            "run_id": outcome.run_id,
            "status": outcome.status,
            "started_at": outcome.started_at,
            "completed_at": outcome.completed_at,
            "optimal_f1": outcome.optimal_f1,
        }
        if outcome.run_dir is not None:
            run_entry["run_dir"] = str(outcome.run_dir)
        if outcome.error is not None:
            run_entry["error"] = outcome.error
        if outcome.traceback is not None:
            run_entry["traceback"] = outcome.traceback
        runs_data.append(run_entry)

    manifest: dict[str, Any] = {
        "batch_id": batch_id,
        "batch_type": batch_type,
        "phase": config.phase,
        "description": config.description,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "num_completed": num_completed,
        "num_failed": num_failed,
        "runs": runs_data,
    }

    manifest_path = batch_dir / BATCH_MANIFEST_FILE
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------


def _extract_optimal_f1_from_summary(run_dir: Path) -> float | None:
    """Extract the best available optimal F1 from run_summary.json.

    Tries post-rerank strategies first, then pre-rerank, then standard metrics.
    Returns None if summary does not exist or has no F1 data.
    """
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None

    try:
        summary = load_json(summary_path)
        macro = summary.get("macro_averaged", {})

        # Post-rerank: try each strategy's at_optimal_threshold
        post_rerank = macro.get("post_rerank", {})
        for strategy_data in post_rerank.values():
            f1 = strategy_data.get("at_optimal_threshold", {}).get("f1")
            if f1 is not None:
                return float(f1)

        # Pre-rerank: at_optimal_distance_threshold
        pre_rerank = macro.get("pre_rerank", {})
        f1 = pre_rerank.get("at_optimal_distance_threshold", {}).get("f1")
        if f1 is not None:
            return float(f1)

        # Standard (no reranking)
        f1 = macro.get("metrics", {}).get("f1")
        if f1 is not None:
            return float(f1)

    except Exception as error:
        # Corrupt or structurally unexpected summary — caller treats None as "not available"
        logging.getLogger(__name__).debug("Could not extract F1 from %s: %s", summary_path, error)

    return None


# ---------------------------------------------------------------------------
# Config fingerprinting
# ---------------------------------------------------------------------------


def _extract_experiment_fingerprint_fields(
    experiment_config: ExperimentConfig,
) -> tuple[str, str | None, list[str] | None]:
    """Extract the experiment-config portion of a fingerprint.

    Returns:
        (embedding_model, reranker_model, rerank_strategies) — fields shared
        by both full-pipeline and subrun fingerprint builders.
    """
    search_backend = experiment_config.search_backend
    embedding_model = (
        getattr(search_backend, "embedding_model", DEFAULT_EMBEDDING_MODEL)
        if isinstance(search_backend, VectorBackend)
        else UNKNOWN_VERSION
    )
    reranker = experiment_config.reranker
    reranker_model = reranker.model_name if reranker is not None else None
    rerank_strategies = (
        list(experiment_config.rerank_text_strategies.keys())
        if experiment_config.rerank_text_strategies
        else None
    )
    return embedding_model, reranker_model, rerank_strategies


def _build_fingerprint_for_full_pipeline(
    run_dir: Path,
    config: BatchFullPipelineConfig,
) -> dict[str, Any]:
    """Build a config fingerprint for a full pipeline run.

    Reads extraction prompt version from run.json pipeline_status (written by
    update_run_status during the extraction stage). Falls back to "unknown".
    """
    run_metadata = _load_run_metadata(run_dir)
    pipeline_status = run_metadata.get("pipeline_status", {})
    extraction_prompt_version = pipeline_status.get("build_memories", {}).get(
        "prompt_version", UNKNOWN_VERSION
    )

    embedding_model, reranker_model, rerank_strategies = _extract_experiment_fingerprint_fields(
        config.experiment_config
    )

    return build_config_fingerprint(
        extraction_prompt_version=extraction_prompt_version,
        embedding_model=embedding_model,
        search_backend="vector",
        search_limit=config.experiment_config.search_limit,
        distance_threshold=config.experiment_config.distance_threshold,
        query_model=config.query_config.model,
        query_prompt_version=str(config.query_config.prompt_version or "latest"),
        reranker_model=reranker_model,
        rerank_text_strategies=rerank_strategies,
    )


def _build_fingerprint_for_subrun(
    parent_run_dir: Path,
    config: BatchSubrunConfig,
) -> dict[str, Any]:
    """Build a config fingerprint for a subrun.

    Inherits extraction prompt version and query model from the parent run.
    Overrides search/reranking settings with the subrun's ExperimentConfig.
    """
    parent_metadata = _load_run_metadata(parent_run_dir)
    parent_pipeline_status = parent_metadata.get("pipeline_status", {})
    extraction_prompt_version = parent_pipeline_status.get("build_memories", {}).get(
        "prompt_version", UNKNOWN_VERSION
    )
    query_model = parent_pipeline_status.get("query_generation", {}).get("model", UNKNOWN_VERSION)
    query_prompt_version = parent_pipeline_status.get("query_generation", {}).get(
        "prompt_version", UNKNOWN_VERSION
    )

    embedding_model, reranker_model, rerank_strategies = _extract_experiment_fingerprint_fields(
        config.experiment_config
    )

    return build_config_fingerprint(
        extraction_prompt_version=extraction_prompt_version,
        embedding_model=embedding_model,
        search_backend="vector",
        search_limit=config.experiment_config.search_limit,
        distance_threshold=config.experiment_config.distance_threshold,
        query_model=query_model,
        query_prompt_version=query_prompt_version,
        reranker_model=reranker_model,
        rerank_text_strategies=rerank_strategies,
    )
