"""Cross-run comparison: fingerprinting, summaries, variance analysis, and A/B testing."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from retrieval_metrics.compare import (
    compare_run_groups as retrieval_compare_run_groups,
)
from retrieval_metrics.compare import (
    compute_variance_report as retrieval_compute_variance_report,
)
from retrieval_metrics.fingerprint import (
    build_fingerprint as retrieval_build_fingerprint,
)
from retrieval_metrics.fingerprint import (
    fingerprint_diff as retrieval_fingerprint_diff,
)

from retrieval_metrics.compute import reciprocal_rank
from retrieval_metrics.sweeps import find_optimal_entry

from memory_retrieval.constants import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_SEARCH_LIMIT
from memory_retrieval.experiments.metrics import pool_and_deduplicate_by_distance
from memory_retrieval.experiments.metrics_adapter import (
    build_macro_run_metric_extractor,
    build_per_case_metric_extractor,
    extract_metric_from_nested,
    macro_average_from_metric_dicts,
    threshold_sweep_from_experiments,
    top_n_sweep_from_experiments,
)
from memory_retrieval.infra.io import load_json, save_json

RUN_SUMMARY_FILE = "run_summary.json"


# ---------------------------------------------------------------------------
# Part 1: Config Fingerprint
# ---------------------------------------------------------------------------


def build_config_fingerprint(
    extraction_prompt_version: str,
    embedding_model: str,
    search_backend: str,
    search_limit: int,
    distance_threshold: float,
    query_model: str,
    query_prompt_version: str,
    reranker_model: str | None = None,
    rerank_text_strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Build a config fingerprint dict with a deterministic hash.

    The fingerprint captures every pipeline parameter that affects results.
    Same hash = same config; any result difference is LLM variance.
    """
    fingerprint_payload: dict[str, Any] = {
        "extraction_prompt_version": extraction_prompt_version,
        "embedding_model": embedding_model,
        "search_backend": search_backend,
        "search_limit": search_limit,
        "distance_threshold": distance_threshold,
        "query_model": query_model,
        "query_prompt_version": query_prompt_version,
        "reranker_model": reranker_model,
        "rerank_text_strategies": sorted(rerank_text_strategies)
        if rerank_text_strategies
        else None,
    }
    return retrieval_build_fingerprint(fingerprint_payload, hash_len=8)


def _compute_fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    """Compute first 8 chars of SHA-256 of sorted JSON (excluding hash itself)."""
    return retrieval_build_fingerprint(fingerprint, hash_len=8)["fingerprint_hash"]


def fingerprint_diff(
    fingerprint_a: dict[str, Any],
    fingerprint_b: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return changed fields between two fingerprints.

    Returns dict of {field: {"a": value_a, "b": value_b}} for fields that differ.
    """
    return retrieval_fingerprint_diff(fingerprint_a, fingerprint_b)


def _load_run_metadata_safe(run_dir: Path) -> dict[str, Any]:
    """Load run.json for a run directory, returning {} if not found or invalid."""
    run_json_path = run_dir / "run.json"
    if not run_json_path.exists():
        return {}
    try:
        return load_json(run_json_path)
    except Exception:
        return {}


class FingerprintReconstructor:
    """Reconstructs a config fingerprint from a completed run directory.

    Handles legacy runs that may not have all fields in run.json,
    falling back to scanning result files for pipeline parameters.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    def reconstruct(self) -> dict[str, Any]:
        """Reconstruct and return the full config fingerprint for the run."""
        run_metadata = _load_run_metadata_safe(self._run_dir)
        pipeline_status = run_metadata.get("pipeline_status", {})

        extraction_prompt_version = pipeline_status.get("build_memories", {}).get(
            "prompt_version", "unknown"
        )

        query_generation_status = pipeline_status.get("query_generation", {})
        query_model = query_generation_status.get("model", "")
        query_prompt_version = query_generation_status.get("prompt_version", "")

        search_limit = DEFAULT_SEARCH_LIMIT
        distance_threshold = DEFAULT_DISTANCE_THRESHOLD
        reranker_model = None
        rerank_text_strategies = None

        results_dir = self._run_dir / "results"
        if results_dir.exists():
            result_files = sorted(results_dir.glob("*.json"))
            if result_files:
                first_result = load_json(result_files[0])

                if not query_model:
                    query_generation = first_result.get("query_generation", {})
                    query_model = query_generation.get(
                        "model", first_result.get("model", "unknown")
                    )
                if not query_prompt_version:
                    query_generation = first_result.get("query_generation", {})
                    query_prompt_version = query_generation.get(
                        "prompt_version", first_result.get("prompt_version", "unknown")
                    )

                search_limit = first_result.get("search_limit", DEFAULT_SEARCH_LIMIT)
                distance_threshold = first_result.get(
                    "distance_threshold", DEFAULT_DISTANCE_THRESHOLD
                )
                reranker_model = first_result.get("reranker_model")

                if "rerank_strategies" in first_result:
                    rerank_text_strategies = list(first_result["rerank_strategies"].keys())

        # Normalize query prompt version: strip prefix if present (e.g., "memory-query/v3.0.0" -> "3.0.0")
        if "/" in query_prompt_version:
            version_part = query_prompt_version.split("/")[-1]
            if version_part.startswith("v"):
                query_prompt_version = version_part[1:]

        embedding_model = run_metadata.get("embedding_model", "mxbai-embed-large")

        return build_config_fingerprint(
            extraction_prompt_version=extraction_prompt_version,
            embedding_model=embedding_model,
            search_backend="vector",
            search_limit=search_limit,
            distance_threshold=distance_threshold,
            query_model=query_model,
            query_prompt_version=query_prompt_version,
            reranker_model=reranker_model,
            rerank_text_strategies=rerank_text_strategies,
        )


def reconstruct_fingerprint_from_run(run_dir: Path) -> dict[str, Any]:
    """Reconstruct a config fingerprint from existing run data. See FingerprintReconstructor."""
    return FingerprintReconstructor(run_dir).reconstruct()


# ---------------------------------------------------------------------------
# Part 2: Run Summary
# ---------------------------------------------------------------------------


def _get_reranked_results_for_strategy(
    result_data: dict[str, Any],
    strategy_name: str,
) -> list[dict[str, Any]]:
    """Get reranked_results for a strategy, handling single vs multi-strategy format."""
    if "rerank_strategies" in result_data and strategy_name in result_data["rerank_strategies"]:
        return result_data["rerank_strategies"][strategy_name]["reranked_results"]
    return result_data.get("reranked_results", [])


class RunSummaryGenerator:
    """Generates and saves run_summary.json for a completed experiment run.

    Processes all result files in the run directory, computes macro-averaged
    metrics, sweeps across distance thresholds and top-N values, and writes
    the summary to run_summary.json.
    """

    def __init__(
        self,
        run_dir: Path,
        strategies: list[str] | None = None,
        threshold_step: float = 0.005,
        top_n_max: int = 20,
        results_dir: str | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._strategies = strategies
        self._threshold_step = threshold_step
        self._top_n_max = top_n_max
        self._n_values = list(range(1, top_n_max + 1))
        self._results_dir_override = results_dir

    def generate(self) -> dict[str, Any]:
        """Generate the run summary and save it to run_summary.json."""
        run_id, run_metadata, successful_results, is_reranking, strategies = (
            self._load_run_context()
        )

        per_test_case, all_pre_rerank_metrics, distance_experiments = (
            self._process_per_test_case_metrics(
                successful_results,
                ground_truth_ids_map=None,
                is_reranking=is_reranking,
                strategies=strategies,
            )
        )

        summary: dict[str, Any] = {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "num_test_cases": len(successful_results),
            "per_test_case": per_test_case,
        }

        macro: dict[str, Any] = {}

        if is_reranking:
            baseline = self._build_baseline_section(distance_experiments)
            if baseline:
                summary["baseline"] = baseline

            rerank_strategies_section = self._build_rerank_strategies_section(
                successful_results, strategies
            )
            summary["rerank_strategies"] = rerank_strategies_section

            macro = self._build_rerank_macro_section(
                all_pre_rerank_metrics, summary.get("baseline", {}), rerank_strategies_section
            )
        else:
            all_metrics = [
                per_test_case[test_case_id]["metrics"]
                for test_case_id in per_test_case
                if "metrics" in per_test_case[test_case_id]
            ]
            macro["metrics"] = macro_average_from_metric_dicts(all_metrics)

        summary["macro_averaged"] = macro

        save_json(summary, self._run_dir / RUN_SUMMARY_FILE)
        return summary

    def _load_run_context(
        self,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]], bool, list[str] | None]:
        """Load run metadata and result files, detect reranking mode and strategies."""
        run_json_path = self._run_dir / "run.json"
        subrun_json_path = self._run_dir / "subrun.json"

        if run_json_path.exists():
            run_metadata = load_json(run_json_path)
            run_id = run_metadata.get("run_id", self._run_dir.name)
        elif subrun_json_path.exists():
            run_metadata = load_json(subrun_json_path)
            run_id = run_metadata.get("subrun_id", self._run_dir.name)
        else:
            run_metadata = {}
            run_id = self._run_dir.name

        resolved_results_dir = (
            Path(self._results_dir_override)
            if self._results_dir_override
            else self._run_dir / "results"
        )
        result_files = sorted(resolved_results_dir.glob("*.json"))
        if not result_files:
            raise FileNotFoundError(f"No result files found in {resolved_results_dir}")

        all_results = [load_json(file_path) for file_path in result_files]
        successful_results = [
            result
            for result in all_results
            if "pre_rerank_metrics" in result or "metrics" in result
        ]

        is_reranking = (
            "pre_rerank_metrics" in successful_results[0] if successful_results else False
        )

        strategies = self._strategies
        if strategies is None and is_reranking:
            first_result = successful_results[0]
            if "rerank_strategies" in first_result:
                strategies = list(first_result["rerank_strategies"].keys())
            else:
                strategies = ["default"]

        return run_id, run_metadata, successful_results, is_reranking, strategies

    def _process_per_test_case_metrics(
        self,
        successful_results: list[dict[str, Any]],
        ground_truth_ids_map: None,  # unused, kept for signature clarity
        is_reranking: bool,
        strategies: list[str] | None,
    ) -> tuple[dict[str, Any], list[dict[str, float]], list[dict[str, Any]]]:
        """Compute per-test-case metrics for all results.

        Returns (per_test_case, all_pre_rerank_metrics, distance_experiments).
        """
        per_test_case: dict[str, dict[str, Any]] = {}
        all_pre_rerank_metrics: list[dict[str, float]] = []
        distance_experiments: list[dict[str, Any]] = []

        for result in successful_results:
            test_case_id = result["test_case_id"]
            ground_truth_ids = set(result.get("ground_truth", {}).get("memory_ids", []))
            ground_truth_count = len(ground_truth_ids)
            num_queries = len(result.get("queries", []))

            test_case_entry: dict[str, Any] = {
                "ground_truth_count": ground_truth_count,
                "num_queries": num_queries,
            }

            if is_reranking:
                test_case_entry, pre_metrics_clean, pooled_by_distance = (
                    self._process_reranking_test_case(
                        result, ground_truth_ids, test_case_entry, strategies or []
                    )
                )
                all_pre_rerank_metrics.append(pre_metrics_clean)
                distance_experiments.append(
                    {
                        "ground_truth_ids": ground_truth_ids,
                        "ranked_results": pooled_by_distance,
                    }
                )
            else:
                metrics = result.get("metrics", {})
                test_case_entry["metrics"] = {
                    key: metrics.get(key, 0.0) for key in ["precision", "recall", "f1"]
                }

            per_test_case[test_case_id] = test_case_entry

        return per_test_case, all_pre_rerank_metrics, distance_experiments

    def _process_reranking_test_case(
        self,
        result: dict[str, Any],
        ground_truth_ids: set[str],
        test_case_entry: dict[str, Any],
        strategies: list[str],
    ) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
        """Process a single reranking test case, returning updated entry, pre-metrics, and pooled results."""
        pre_metrics = result["pre_rerank_metrics"]
        pre_metrics_clean = {
            key: pre_metrics[key] for key in ["precision", "recall", "f1"] if key in pre_metrics
        }

        pooled_by_distance = pool_and_deduplicate_by_distance(result.get("queries", []))
        pre_mrr = reciprocal_rank([entry["id"] for entry in pooled_by_distance], ground_truth_ids)
        pre_metrics_clean["mrr"] = pre_mrr

        pre_rerank_entry: dict[str, Any] = {}

        if pooled_by_distance:
            all_distances = [entry.get("distance", 0) for entry in pooled_by_distance]
            max_distance = max(all_distances) if all_distances else 1.5
            distance_thresholds = _build_threshold_range(0.0, max_distance, self._threshold_step)
            tc_distance_sweep = threshold_sweep_from_experiments(
                [{"ground_truth_ids": ground_truth_ids, "ranked_results": pooled_by_distance}],
                distance_thresholds,
                score_field="distance",
                higher_is_better=False,
            )
            optimal_distance = find_optimal_entry(tc_distance_sweep, metric_key="f1")
            pre_rerank_entry["distance_threshold"] = {
                "optimal_threshold": round(optimal_distance.get("threshold", 0.0), 4),
                "at_optimal": {
                    "precision": optimal_distance["precision"],
                    "recall": optimal_distance["recall"],
                    "f1": optimal_distance["f1"],
                    "mrr": optimal_distance["mrr"],
                },
            }

            tc_top_n_sweep = top_n_sweep_from_experiments(
                [{"ground_truth_ids": ground_truth_ids, "ranked_results": pooled_by_distance}],
                self._n_values,
            )
            optimal_top_n = find_optimal_entry(tc_top_n_sweep, metric_key="f1")
            pre_rerank_entry["top_n"] = {
                "optimal_n": optimal_top_n.get("top_n", 1),
                "at_optimal": {
                    "precision": optimal_top_n["precision"],
                    "recall": optimal_top_n["recall"],
                    "f1": optimal_top_n["f1"],
                    "mrr": optimal_top_n["mrr"],
                },
            }

        test_case_entry["pre_rerank"] = pre_rerank_entry

        post_rerank_entry: dict[str, Any] = {}
        for strategy_name in strategies:
            reranked = _get_reranked_results_for_strategy(result, strategy_name)
            if not reranked:
                continue

            strategy_entry: dict[str, Any] = {}

            all_scores = [entry["rerank_score"] for entry in reranked]
            if all_scores:
                max_score = max(all_scores)
                rerank_thresholds = _build_threshold_range(0.0, max_score, self._threshold_step)
                tc_rerank_sweep = threshold_sweep_from_experiments(
                    [{"ground_truth_ids": ground_truth_ids, "ranked_results": reranked}],
                    rerank_thresholds,
                    score_field="rerank_score",
                    higher_is_better=True,
                )
                optimal_rerank = find_optimal_entry(tc_rerank_sweep, metric_key="f1")
                strategy_entry["rerank_threshold"] = {
                    "optimal_threshold": round(optimal_rerank.get("threshold", 0.0), 4),
                    "at_optimal": {
                        "precision": optimal_rerank["precision"],
                        "recall": optimal_rerank["recall"],
                        "f1": optimal_rerank["f1"],
                        "mrr": optimal_rerank["mrr"],
                    },
                }

            tc_rerank_top_n = top_n_sweep_from_experiments(
                [{"ground_truth_ids": ground_truth_ids, "ranked_results": reranked}],
                self._n_values,
            )
            optimal_rerank_top_n = find_optimal_entry(tc_rerank_top_n, metric_key="f1")
            strategy_entry["top_n"] = {
                "optimal_n": optimal_rerank_top_n.get("top_n", 1),
                "at_optimal": {
                    "precision": optimal_rerank_top_n["precision"],
                    "recall": optimal_rerank_top_n["recall"],
                    "f1": optimal_rerank_top_n["f1"],
                    "mrr": optimal_rerank_top_n["mrr"],
                },
            }

            post_rerank_entry[strategy_name] = strategy_entry

        test_case_entry["post_rerank"] = post_rerank_entry
        return test_case_entry, pre_metrics_clean, pooled_by_distance

    def _build_baseline_section(self, distance_experiments: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the baseline distance-threshold and top-N sweep section."""
        if not distance_experiments:
            return {}

        all_max_distances = [
            max(entry.get("distance", 0) for entry in experiment["ranked_results"])
            for experiment in distance_experiments
            if experiment["ranked_results"]
        ]
        distance_global_max = max(all_max_distances) if all_max_distances else 1.5
        distance_thresholds = _build_threshold_range(0.0, distance_global_max, self._threshold_step)

        baseline_threshold_sweep = threshold_sweep_from_experiments(
            distance_experiments,
            distance_thresholds,
            score_field="distance",
            higher_is_better=False,
        )
        optimal_distance_threshold = find_optimal_entry(baseline_threshold_sweep, metric_key="f1")

        baseline_top_n_sweep = top_n_sweep_from_experiments(distance_experiments, self._n_values)
        optimal_distance_top_n = find_optimal_entry(baseline_top_n_sweep, metric_key="f1")

        return {
            "note": "Pre-rerank metrics from distance-sorted candidates (overfetched for reranking)",
            "distance_threshold_sweep": {
                "optimal_threshold": round(optimal_distance_threshold.get("threshold", 0.0), 4),
                "optimal_f1": round(optimal_distance_threshold["f1"], 4),
                "full_sweep": [
                    {
                        "threshold": round(entry["threshold"], 4),
                        "precision": round(entry["precision"], 4),
                        "recall": round(entry["recall"], 4),
                        "f1": round(entry["f1"], 4),
                        "mrr": round(entry["mrr"], 4),
                    }
                    for entry in baseline_threshold_sweep
                ],
            },
            "top_n_sweep": {
                "optimal_n": optimal_distance_top_n.get("top_n", 1),
                "optimal_f1": round(optimal_distance_top_n["f1"], 4),
                "full_sweep": [
                    {
                        "top_n": entry["top_n"],
                        "precision": round(entry["precision"], 4),
                        "recall": round(entry["recall"], 4),
                        "f1": round(entry["f1"], 4),
                        "mrr": round(entry["mrr"], 4),
                    }
                    for entry in baseline_top_n_sweep
                ],
            },
        }

    def _build_rerank_strategies_section(
        self,
        successful_results: list[dict[str, Any]],
        strategies: list[str] | None,
    ) -> dict[str, Any]:
        """Build the per-strategy threshold and top-N sweep section."""
        rerank_strategies_section: dict[str, Any] = {}
        for strategy_name in strategies or []:
            experiments_for_sweep = []
            for result in successful_results:
                ground_truth_ids = set(result.get("ground_truth", {}).get("memory_ids", []))
                reranked = _get_reranked_results_for_strategy(result, strategy_name)
                if reranked:
                    experiments_for_sweep.append(
                        {
                            "ground_truth_ids": ground_truth_ids,
                            "ranked_results": reranked,
                        }
                    )

            if not experiments_for_sweep:
                continue

            all_max_scores = [
                max(entry["rerank_score"] for entry in experiment["ranked_results"])
                for experiment in experiments_for_sweep
                if experiment["ranked_results"]
            ]
            global_max = max(all_max_scores) if all_max_scores else 1.0
            rerank_thresholds = _build_threshold_range(0.0, global_max, self._threshold_step)

            threshold_sweep_results = threshold_sweep_from_experiments(
                experiments_for_sweep,
                rerank_thresholds,
                score_field="rerank_score",
                higher_is_better=True,
            )
            optimal_threshold = find_optimal_entry(threshold_sweep_results, metric_key="f1")

            top_n_sweep_results = top_n_sweep_from_experiments(
                experiments_for_sweep, self._n_values
            )
            optimal_n_entry = find_optimal_entry(top_n_sweep_results, metric_key="f1")

            rerank_strategies_section[strategy_name] = {
                "threshold_sweep": {
                    "optimal_threshold": round(optimal_threshold.get("threshold", 0.0), 4),
                    "optimal_f1": round(optimal_threshold["f1"], 4),
                    "full_sweep": [
                        {
                            "threshold": round(entry["threshold"], 4),
                            "precision": round(entry["precision"], 4),
                            "recall": round(entry["recall"], 4),
                            "f1": round(entry["f1"], 4),
                            "mrr": round(entry["mrr"], 4),
                        }
                        for entry in threshold_sweep_results
                    ],
                },
                "top_n_sweep": {
                    "optimal_n": optimal_n_entry.get("top_n", 1),
                    "optimal_f1": round(optimal_n_entry["f1"], 4),
                    "full_sweep": [
                        {
                            "top_n": entry["top_n"],
                            "precision": round(entry["precision"], 4),
                            "recall": round(entry["recall"], 4),
                            "f1": round(entry["f1"], 4),
                            "mrr": round(entry["mrr"], 4),
                        }
                        for entry in top_n_sweep_results
                    ],
                },
            }

        return rerank_strategies_section

    def _build_rerank_macro_section(
        self,
        all_pre_rerank_metrics: list[dict[str, float]],
        baseline: dict[str, Any],
        rerank_strategies_section: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute macro-averaged metrics for reranking runs."""
        macro: dict[str, Any] = {}

        overfetched_macro = macro_average_from_metric_dicts(all_pre_rerank_metrics)
        pre_rerank_macro: dict[str, Any] = {"overfetched": overfetched_macro}

        distance_sweep_data = baseline.get("distance_threshold_sweep", {})
        if distance_sweep_data:
            pre_rerank_macro["at_optimal_distance_threshold"] = {
                "optimal_threshold": distance_sweep_data["optimal_threshold"],
                "precision": round(
                    _find_sweep_entry_by_value(
                        distance_sweep_data["full_sweep"],
                        "threshold",
                        distance_sweep_data["optimal_threshold"],
                    ).get("precision", 0.0),
                    4,
                ),
                "recall": round(
                    _find_sweep_entry_by_value(
                        distance_sweep_data["full_sweep"],
                        "threshold",
                        distance_sweep_data["optimal_threshold"],
                    ).get("recall", 0.0),
                    4,
                ),
                "f1": distance_sweep_data["optimal_f1"],
                "mrr": round(
                    _find_sweep_entry_by_value(
                        distance_sweep_data["full_sweep"],
                        "threshold",
                        distance_sweep_data["optimal_threshold"],
                    ).get("mrr", 0.0),
                    4,
                ),
            }

        top_n_sweep_data = baseline.get("top_n_sweep", {})
        if top_n_sweep_data:
            pre_rerank_macro["at_optimal_top_n"] = {
                "optimal_n": top_n_sweep_data["optimal_n"],
                "precision": round(
                    _find_sweep_entry_by_value(
                        top_n_sweep_data["full_sweep"],
                        "top_n",
                        top_n_sweep_data["optimal_n"],
                    ).get("precision", 0.0),
                    4,
                ),
                "recall": round(
                    _find_sweep_entry_by_value(
                        top_n_sweep_data["full_sweep"],
                        "top_n",
                        top_n_sweep_data["optimal_n"],
                    ).get("recall", 0.0),
                    4,
                ),
                "f1": top_n_sweep_data["optimal_f1"],
                "mrr": round(
                    _find_sweep_entry_by_value(
                        top_n_sweep_data["full_sweep"],
                        "top_n",
                        top_n_sweep_data["optimal_n"],
                    ).get("mrr", 0.0),
                    4,
                ),
            }

        macro["pre_rerank"] = pre_rerank_macro

        post_rerank_macro: dict[str, Any] = {}
        for strategy_name, strategy_data in rerank_strategies_section.items():
            threshold_data = strategy_data["threshold_sweep"]
            top_n_data = strategy_data["top_n_sweep"]

            optimal_threshold_entry = _find_sweep_entry_by_value(
                threshold_data["full_sweep"],
                "threshold",
                threshold_data["optimal_threshold"],
            )
            optimal_top_n_entry = _find_sweep_entry_by_value(
                top_n_data["full_sweep"],
                "top_n",
                top_n_data["optimal_n"],
            )

            post_rerank_macro[strategy_name] = {
                "at_optimal_threshold": {
                    "optimal_threshold": threshold_data["optimal_threshold"],
                    "precision": round(optimal_threshold_entry.get("precision", 0.0), 4),
                    "recall": round(optimal_threshold_entry.get("recall", 0.0), 4),
                    "f1": threshold_data["optimal_f1"],
                    "mrr": round(optimal_threshold_entry.get("mrr", 0.0), 4),
                },
                "at_optimal_top_n": {
                    "optimal_n": top_n_data["optimal_n"],
                    "precision": round(optimal_top_n_entry.get("precision", 0.0), 4),
                    "recall": round(optimal_top_n_entry.get("recall", 0.0), 4),
                    "f1": top_n_data["optimal_f1"],
                    "mrr": round(optimal_top_n_entry.get("mrr", 0.0), 4),
                },
            }

        macro["post_rerank"] = post_rerank_macro
        return macro


def generate_run_summary(
    run_dir: Path,
    strategies: list[str] | None = None,
    threshold_step: float = 0.005,
    top_n_max: int = 20,
    results_dir: str | None = None,
) -> dict[str, Any]:
    """Generate a run_summary.json from result files. Regenerable at any time.

    Produces a structured summary with fair baseline vs rerank comparison:
    - ``baseline``: distance threshold sweep and top-N sweep for pre-rerank candidates
    - ``rerank_strategies``: per-strategy threshold and top-N sweeps
    - ``macro_averaged``: pre_rerank (overfetched + at-optimal) and post_rerank per strategy

    See RunSummaryGenerator for implementation.
    """
    return RunSummaryGenerator(
        run_dir=run_dir,
        strategies=strategies,
        threshold_step=threshold_step,
        top_n_max=top_n_max,
        results_dir=results_dir,
    ).generate()


def _build_threshold_range(min_value: float, max_value: float, step: float) -> list[float]:
    """Build a list of threshold values from min to max with given step."""
    thresholds = []
    current = min_value
    while current <= max_value + step:
        thresholds.append(round(current, 4))
        current += step
    return thresholds


def _find_sweep_entry_by_value(
    full_sweep: list[dict[str, Any]],
    key: str,
    value: Any,
) -> dict[str, Any]:
    """Find the sweep entry whose key matches value (or closest match)."""
    if not full_sweep:
        return {}
    # Exact match first
    for entry in full_sweep:
        if entry.get(key) == value:
            return entry
    # Closest match for numeric keys
    try:
        return min(full_sweep, key=lambda entry: abs(entry.get(key, float("inf")) - value))
    except (TypeError, ValueError):
        return full_sweep[0]


# ---------------------------------------------------------------------------
# Part 3: Loading & Grouping
# ---------------------------------------------------------------------------


def load_run_summaries(
    phase: str,
    run_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load run summaries for all or selected runs in a phase.

    If a summary doesn't exist, generates it on the fly.
    """
    from memory_retrieval.infra.runs import list_runs

    runs = list_runs(phase)
    if run_ids is not None:
        runs = [run for run in runs if run["run_id"] in run_ids]

    summaries = []
    for run in runs:
        run_dir = run["run_dir"]
        summary_path = run_dir / RUN_SUMMARY_FILE

        if summary_path.exists():
            summary = load_json(summary_path)
        else:
            # Check if results exist before trying to generate
            results_dir = run_dir / "results"
            if not results_dir.exists() or not list(results_dir.glob("*.json")):
                continue
            summary = generate_run_summary(run_dir)

        # Attach fingerprint if available
        run_metadata = load_json(run_dir / "run.json")
        if "config_fingerprint" in run_metadata:
            summary["config_fingerprint"] = run_metadata["config_fingerprint"]

        summary["run_dir"] = str(run_dir)
        summaries.append(summary)

    return summaries


def load_subrun_summaries(
    phase: str,
    parent_run_ids: list[str] | None = None,
    subrun_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load run summaries for subruns of existing parent runs.

    Each returned summary has the same shape as load_run_summaries(), plus:
    - "parent_run_id": which parent run this subrun belongs to
    - "run_id": the subrun ID (e.g. "run_20260218_143022")

    Args:
        phase: "phase1" or "phase2".
        parent_run_ids: Restrict to subruns of these parent runs. If None, walks all runs.
        subrun_ids: Restrict to these subrun IDs only. If None, loads all subruns.

    Returns:
        List of summary dicts, one per subrun.
    """
    from memory_retrieval.infra.runs import get_subrun_paths, list_runs, list_subruns

    runs = list_runs(phase)
    if parent_run_ids is not None:
        runs = [run for run in runs if run["run_id"] in parent_run_ids]

    summaries = []
    for run in runs:
        parent_run_dir = run["run_dir"]
        for subrun_metadata in list_subruns(parent_run_dir):
            subrun_dir = subrun_metadata["subrun_dir"]
            current_subrun_id = subrun_metadata.get("subrun_id", subrun_dir.name)

            if subrun_ids is not None and current_subrun_id not in subrun_ids:
                continue

            summary_path = subrun_dir / RUN_SUMMARY_FILE
            if summary_path.exists():
                summary = load_json(summary_path)
            else:
                # Try generating on-the-fly if results exist
                paths = get_subrun_paths(subrun_dir)
                results_dir = Path(paths["results_dir"])
                if not results_dir.exists() or not list(results_dir.glob("*.json")):
                    continue
                summary = generate_run_summary(subrun_dir, results_dir=paths["results_dir"])

            # Attach fingerprint if stored in subrun.json
            if "config_fingerprint" in subrun_metadata:
                summary["config_fingerprint"] = subrun_metadata["config_fingerprint"]

            summary["run_id"] = current_subrun_id
            summary["parent_run_id"] = subrun_metadata.get("parent_run_id", parent_run_dir.name)
            summary["run_dir"] = str(subrun_dir)
            summaries.append(summary)

    return summaries


def group_runs_by_fingerprint(
    summaries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group run summaries by their config fingerprint hash.

    Returns dict of {fingerprint_hash: [summaries]}.
    Runs without a fingerprint are grouped under "unknown".
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        fingerprint = summary.get("config_fingerprint", {})
        hash_key = fingerprint.get("fingerprint_hash", "unknown")
        groups.setdefault(hash_key, []).append(summary)
    return groups


def average_per_case_metrics_across_runs(
    summaries: list[dict[str, Any]],
    metric_path: str = "post_rerank",
    strategy: str | None = None,
) -> dict[str, dict[str, float]]:
    """Average each test case's metrics across N runs to reduce LLM noise for A/B comparison.

    Args:
        summaries: List of run summaries (same config group).
        metric_path: "post_rerank" for reranking runs, "pre_rerank" for pre-rerank, "metrics" for standard.
        strategy: Strategy name (required for post_rerank).

    Returns:
        Dict of {test_case_id: {precision, recall, f1, mrr}} averaged across runs.
    """
    # Collect all test case IDs across all runs
    all_test_case_ids: set[str] = set()
    for summary in summaries:
        all_test_case_ids.update(summary.get("per_test_case", {}).keys())

    averaged: dict[str, dict[str, float]] = {}
    for test_case_id in sorted(all_test_case_ids):
        metric_values: list[dict[str, float]] = []
        for summary in summaries:
            per_tc = summary.get("per_test_case", {}).get(test_case_id, {})
            if metric_path == "post_rerank" and strategy:
                post_rerank = per_tc.get("post_rerank", {}).get(strategy, {})
                metrics = post_rerank.get("rerank_threshold", {}).get("at_optimal", {})
            elif metric_path == "pre_rerank":
                pre_rerank = per_tc.get("pre_rerank", {})
                metrics = pre_rerank.get("distance_threshold", {}).get("at_optimal", {})
            else:
                metrics = per_tc.get("metrics", {})

            if metrics and "f1" in metrics:
                metric_values.append(metrics)

        if metric_values:
            averaged[test_case_id] = {
                key: float(np.mean([m[key] for m in metric_values if key in m]))
                for key in ["precision", "recall", "f1", "mrr"]
                if any(key in m for m in metric_values)
            }

    return averaged


# ---------------------------------------------------------------------------
# Part 4: Statistical Analysis
# ---------------------------------------------------------------------------


def _extract_f1_from_macro(
    macro: dict[str, Any],
    metric_path: str,
    strategy: str | None,
) -> float | None:
    """Extract F1 from a macro_averaged or per_test_case metrics dict.

    Navigates the nested structure:
    - macro level: at_optimal_threshold / at_optimal_distance_threshold
    - per-test-case level: rerank_threshold.at_optimal / distance_threshold.at_optimal
    """
    return extract_metric_from_nested(macro, metric_path, strategy, metric_key="f1")


def compute_variance_report(
    summaries: list[dict[str, Any]],
    strategy: str | None = None,
    metric_path: str = "post_rerank",
) -> dict[str, Any]:
    """Compute variance statistics across same-config runs.

    Args:
        summaries: List of run summaries with the same config fingerprint.
        strategy: Strategy name (required for post_rerank).
        metric_path: "post_rerank", "pre_rerank", or "metrics".

    Returns:
        Dict with run-level and per-test-case variance statistics.
    """
    run_value_extractor = build_macro_run_metric_extractor(
        metric_path=metric_path, strategy=strategy
    )
    per_case_extractor = build_per_case_metric_extractor(metric_path=metric_path, strategy=strategy)
    report = retrieval_compute_variance_report(
        run_summaries=summaries,
        metric_key="f1",
        run_value_extractor=run_value_extractor,
        per_case_value_extractor=per_case_extractor,
    )

    # Preserve backward-compatible output keys.
    run_level = report.get("run_level", {})
    if "individual_values" in run_level:
        run_level["individual_f1_values"] = run_level.pop("individual_values")

    return {
        "num_runs": report.get("num_runs", len(summaries)),
        "strategy": strategy,
        "metric_path": metric_path,
        "run_level": run_level,
        "per_test_case": report.get("per_case", {}),
        **({"error": report["error"]} if "error" in report else {}),
    }


def compare_configs(
    summaries_a: list[dict[str, Any]],
    summaries_b: list[dict[str, Any]],
    strategy: str | None = None,
    metric_path: str = "post_rerank",
    bootstrap_iterations: int = 10000,
) -> dict[str, Any]:
    """Paired statistical comparison of two configs (A/B test).

    When multiple runs per config exist, averages each test case's F1 across runs first.

    Args:
        summaries_a: Run summaries for config A.
        summaries_b: Run summaries for config B.
        strategy: Strategy name (required for post_rerank).
        metric_path: "post_rerank", "pre_rerank", or "metrics".
        bootstrap_iterations: Number of bootstrap resamples.

    Returns:
        Dict with paired differences, bootstrap CI, Wilcoxon test, and significance assessment.
    """
    per_case_extractor = build_per_case_metric_extractor(metric_path=metric_path, strategy=strategy)
    comparison = retrieval_compare_run_groups(
        group_a=summaries_a,
        group_b=summaries_b,
        metric_key="f1",
        bootstrap_iterations=bootstrap_iterations,
        per_case_value_extractor=per_case_extractor,
    )

    # Preserve backward-compatible keys and shape.
    if "error" in comparison:
        common_case_ids = comparison.get("common_case_ids", [])
        return {
            "error": comparison["error"],
            "common_test_cases": common_case_ids,
        }

    paired_differences = []
    for entry in comparison.get("paired_differences", []):
        paired_differences.append(
            {
                "test_case_id": entry.get("test_case_id", entry["case_id"]),
                "f1_a": round(float(entry.get("f1_a", entry["metric_a"])), 4),
                "f1_b": round(float(entry.get("f1_b", entry["metric_b"])), 4),
                "diff": round(float(entry["diff"]), 4),
            }
        )

    return {
        "num_common_test_cases": comparison["num_common_cases"],
        "num_runs_a": comparison["num_runs_a"],
        "num_runs_b": comparison["num_runs_b"],
        "strategy": strategy,
        "metric_path": metric_path,
        "mean_f1_a": comparison["mean_a"],
        "mean_f1_b": comparison["mean_b"],
        "mean_diff": comparison["mean_diff"],
        "direction": comparison["direction"],
        "bootstrap_ci": comparison["bootstrap_ci"],
        "wilcoxon": comparison["wilcoxon"],
        "significance": comparison["significance"],
        "paired_differences": paired_differences,
    }
