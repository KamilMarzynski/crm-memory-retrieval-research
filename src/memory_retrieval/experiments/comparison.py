"""Cross-run comparison: fingerprinting, summaries, variance analysis, and A/B testing."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from memory_retrieval.experiments.metrics import (
    find_optimal_threshold,
    macro_average,
    pool_and_deduplicate_by_distance,
    reciprocal_rank,
    sweep_threshold,
    sweep_top_n,
)
from memory_retrieval.experiments.runner import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_SEARCH_LIMIT
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
    fingerprint: dict[str, Any] = {
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
    fingerprint["fingerprint_hash"] = _compute_fingerprint_hash(fingerprint)
    return fingerprint


def _compute_fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    """Compute first 8 chars of SHA-256 of sorted JSON (excluding hash itself)."""
    hashable = {key: value for key, value in fingerprint.items() if key != "fingerprint_hash"}
    canonical = json.dumps(hashable, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def fingerprint_diff(
    fingerprint_a: dict[str, Any],
    fingerprint_b: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return changed fields between two fingerprints.

    Returns dict of {field: {"a": value_a, "b": value_b}} for fields that differ.
    """
    diff = {}
    all_keys = set(fingerprint_a) | set(fingerprint_b)
    for key in sorted(all_keys):
        if key == "fingerprint_hash":
            continue
        value_a = fingerprint_a.get(key)
        value_b = fingerprint_b.get(key)
        if value_a != value_b:
            diff[key] = {"a": value_a, "b": value_b}
    return diff


def reconstruct_fingerprint_from_run(run_dir: Path) -> dict[str, Any]:
    """Reconstruct a config fingerprint from existing run data (backfill for old runs).

    Reads run.json and result files to extract pipeline parameters.
    """
    run_metadata = load_json(run_dir / "run.json")
    pipeline_status = run_metadata.get("pipeline_status", {})

    # Extraction prompt version from pipeline_status or default
    build_memories = pipeline_status.get("build_memories", {})
    extraction_prompt_version = build_memories.get("prompt_version", "unknown")

    # Query generation info: try pipeline_status first, then fall back to result files
    query_generation_status = pipeline_status.get("query_generation", {})
    query_model = query_generation_status.get("model", "")
    query_prompt_version = query_generation_status.get("prompt_version", "")

    # Search/reranking info from result files
    search_limit = DEFAULT_SEARCH_LIMIT
    distance_threshold = DEFAULT_DISTANCE_THRESHOLD
    reranker_model = None
    rerank_text_strategies = None

    results_dir = run_dir / "results"
    if results_dir.exists():
        result_files = sorted(results_dir.glob("*.json"))
        if result_files:
            first_result = load_json(result_files[0])

            # Try to get query info from result file if not in pipeline_status
            if not query_model:
                query_generation = first_result.get("query_generation", {})
                query_model = query_generation.get("model", first_result.get("model", "unknown"))
            if not query_prompt_version:
                query_generation = first_result.get("query_generation", {})
                query_prompt_version = query_generation.get(
                    "prompt_version", first_result.get("prompt_version", "unknown")
                )

            search_limit = first_result.get("search_limit", 20)
            distance_threshold = first_result.get("distance_threshold", 1.1)
            reranker_model = first_result.get("reranker_model")

            if "rerank_strategies" in first_result:
                rerank_text_strategies = list(first_result["rerank_strategies"].keys())

    # Normalize query prompt version: strip prefix if present (e.g., "memory-query/v3.0.0" -> "3.0.0")
    if "/" in query_prompt_version:
        version_part = query_prompt_version.split("/")[-1]
        if version_part.startswith("v"):
            query_prompt_version = version_part[1:]

    return build_config_fingerprint(
        extraction_prompt_version=extraction_prompt_version,
        embedding_model="mxbai-embed-large",
        search_backend="vector",
        search_limit=search_limit,
        distance_threshold=distance_threshold,
        query_model=query_model,
        query_prompt_version=query_prompt_version,
        reranker_model=reranker_model,
        rerank_text_strategies=rerank_text_strategies,
    )


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


def generate_run_summary(
    run_dir: Path,
    strategies: list[str] | None = None,
    threshold_step: float = 0.005,
    top_n_max: int = 20,
) -> dict[str, Any]:
    """Generate a run_summary.json from result files. Regenerable at any time.

    Args:
        run_dir: Path to the run directory.
        strategies: List of rerank strategy names to include. If None, auto-detected from results.
        threshold_step: Step size for threshold sweep.
        top_n_max: Maximum top-N value for sweep.
    """
    run_metadata = load_json(run_dir / "run.json")
    run_id = run_metadata.get("run_id", run_dir.name)

    results_dir = run_dir / "results"
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    all_results = [load_json(file_path) for file_path in result_files]
    successful_results = [
        result for result in all_results if "pre_rerank_metrics" in result or "metrics" in result
    ]

    is_reranking = "pre_rerank_metrics" in successful_results[0] if successful_results else False

    # Auto-detect strategies
    if strategies is None and is_reranking:
        first_result = successful_results[0]
        if "rerank_strategies" in first_result:
            strategies = list(first_result["rerank_strategies"].keys())
        else:
            strategies = ["default"]

    per_test_case: dict[str, dict[str, Any]] = {}
    all_pre_rerank_metrics: list[dict[str, float]] = []

    # Collect per-test-case data
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
            pre_metrics = result["pre_rerank_metrics"]
            pre_metrics_clean = {
                key: pre_metrics[key] for key in ["precision", "recall", "f1"] if key in pre_metrics
            }
            # Compute MRR for pre-rerank
            pooled_by_distance = pool_and_deduplicate_by_distance(result.get("queries", []))
            pre_mrr = reciprocal_rank(
                [entry["id"] for entry in pooled_by_distance], ground_truth_ids
            )
            pre_metrics_clean["mrr"] = pre_mrr
            test_case_entry["pre_rerank"] = pre_metrics_clean
            all_pre_rerank_metrics.append(pre_metrics_clean)

            # Per-strategy post-rerank at optimal threshold
            test_case_entry["post_rerank"] = {}
            for strategy_name in strategies:
                reranked = _get_reranked_results_for_strategy(result, strategy_name)
                if not reranked:
                    continue

                # Find optimal threshold for this test case
                all_scores = [entry["rerank_score"] for entry in reranked]
                if all_scores:
                    max_score = max(all_scores)
                    sweep_thresholds = _build_threshold_range(0.0, max_score, threshold_step)
                    tc_sweep = sweep_threshold(
                        [{"ground_truth_ids": ground_truth_ids, "ranked_results": reranked}],
                        sweep_thresholds,
                        score_field="rerank_score",
                        higher_is_better=True,
                    )
                    optimal = find_optimal_threshold(tc_sweep, metric="f1")
                    test_case_entry["post_rerank"][strategy_name] = {
                        "optimal_threshold": optimal.get("threshold", 0.0),
                        "at_optimal": {
                            "precision": optimal["precision"],
                            "recall": optimal["recall"],
                            "f1": optimal["f1"],
                            "mrr": optimal["mrr"],
                        },
                    }
        else:
            # Standard (no reranking) metrics
            metrics = result.get("metrics", {})
            test_case_entry["metrics"] = {
                key: metrics.get(key, 0.0) for key in ["precision", "recall", "f1"]
            }

        per_test_case[test_case_id] = test_case_entry

    # Macro-averaged metrics
    summary: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "num_test_cases": len(successful_results),
        "per_test_case": per_test_case,
    }

    macro: dict[str, Any] = {}

    if is_reranking:
        macro["pre_rerank"] = macro_average(all_pre_rerank_metrics)

        # Post-rerank: macro-average across strategies with sweep
        macro["post_rerank"] = {}
        for strategy_name in strategies:
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

            # Threshold sweep
            all_max_scores = [
                max(entry["rerank_score"] for entry in experiment["ranked_results"])
                for experiment in experiments_for_sweep
                if experiment["ranked_results"]
            ]
            global_max = max(all_max_scores) if all_max_scores else 1.0
            sweep_thresholds = _build_threshold_range(0.0, global_max, threshold_step)
            threshold_sweep_results = sweep_threshold(
                experiments_for_sweep,
                sweep_thresholds,
                score_field="rerank_score",
                higher_is_better=True,
            )
            optimal_threshold = find_optimal_threshold(threshold_sweep_results, metric="f1")

            macro["post_rerank"][strategy_name] = {
                "optimal_threshold": optimal_threshold.get("threshold", 0.0),
                "at_optimal": {
                    "precision": optimal_threshold["precision"],
                    "recall": optimal_threshold["recall"],
                    "f1": optimal_threshold["f1"],
                    "mrr": optimal_threshold["mrr"],
                },
            }

            # Store full threshold sweep for first/best strategy
            if "threshold_sweep" not in summary:
                summary["threshold_sweep"] = {
                    "strategy": strategy_name,
                    "optimal_threshold": optimal_threshold.get("threshold", 0.0),
                    "optimal_f1": optimal_threshold["f1"],
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
                }

        # Top-N sweep (using first strategy)
        primary_strategy = strategies[0]
        top_n_experiments = []
        for result in successful_results:
            ground_truth_ids = set(result.get("ground_truth", {}).get("memory_ids", []))
            reranked = _get_reranked_results_for_strategy(result, primary_strategy)
            if reranked:
                top_n_experiments.append(
                    {
                        "ground_truth_ids": ground_truth_ids,
                        "ranked_results": reranked,
                    }
                )

        if top_n_experiments:
            n_values = list(range(1, top_n_max + 1))
            top_n_sweep_results = sweep_top_n(top_n_experiments, n_values)
            optimal_n_entry = find_optimal_threshold(top_n_sweep_results, metric="f1")
            summary["top_n_sweep"] = {
                "strategy": primary_strategy,
                "optimal_n": optimal_n_entry.get("top_n", 1),
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
            }
    else:
        # Standard metrics macro-average
        all_metrics = [
            per_test_case[tc_id]["metrics"]
            for tc_id in per_test_case
            if "metrics" in per_test_case[tc_id]
        ]
        macro["metrics"] = macro_average(all_metrics)

    summary["macro_averaged"] = macro

    # Save
    summary_path = run_dir / RUN_SUMMARY_FILE
    save_json(summary, summary_path)
    return summary


def _build_threshold_range(min_value: float, max_value: float, step: float) -> list[float]:
    """Build a list of threshold values from min to max with given step."""
    thresholds = []
    current = min_value
    while current <= max_value + step:
        thresholds.append(round(current, 4))
        current += step
    return thresholds


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
                metrics = post_rerank.get("at_optimal", {})
            elif metric_path == "pre_rerank":
                metrics = per_tc.get("pre_rerank", {})
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
    """Extract F1 from a macro_averaged or per_test_case metrics dict."""
    if metric_path == "post_rerank" and strategy:
        return macro.get("post_rerank", {}).get(strategy, {}).get("at_optimal", {}).get("f1")
    elif metric_path == "pre_rerank":
        return macro.get("pre_rerank", {}).get("f1")
    else:
        return macro.get("metrics", {}).get("f1")


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
    num_runs = len(summaries)
    if num_runs < 2:
        return {
            "num_runs": num_runs,
            "error": "Need at least 2 runs for variance analysis",
        }

    # Run-level: macro-averaged F1 per run
    run_f1_values = []
    for summary in summaries:
        f1 = _extract_f1_from_macro(summary.get("macro_averaged", {}), metric_path, strategy)
        if f1 is not None:
            run_f1_values.append(f1)

    run_level = _compute_summary_stats(run_f1_values)
    run_level["individual_f1_values"] = run_f1_values

    # Per-test-case variance
    all_test_case_ids: set[str] = set()
    for summary in summaries:
        all_test_case_ids.update(summary.get("per_test_case", {}).keys())

    per_test_case_variance: dict[str, dict[str, Any]] = {}
    for test_case_id in sorted(all_test_case_ids):
        f1_values = []
        for summary in summaries:
            per_tc = summary.get("per_test_case", {}).get(test_case_id, {})
            f1 = _extract_f1_from_macro(per_tc, metric_path, strategy)
            if f1 is not None:
                f1_values.append(f1)

        if len(f1_values) >= 2:
            per_test_case_variance[test_case_id] = _compute_summary_stats(f1_values)

    return {
        "num_runs": num_runs,
        "strategy": strategy,
        "metric_path": metric_path,
        "run_level": run_level,
        "per_test_case": per_test_case_variance,
    }


def _compute_summary_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, 95% CI, and CV for a list of values."""
    num_values = len(values)
    if num_values < 2:
        return {
            "mean": values[0] if values else 0.0,
            "std": 0.0,
            "ci_95_lower": values[0] if values else 0.0,
            "ci_95_upper": values[0] if values else 0.0,
            "cv": 0.0,
            "n": num_values,
        }

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    # 95% CI using t-distribution
    t_value = scipy_stats.t.ppf(0.975, df=num_values - 1)
    margin = t_value * std / np.sqrt(num_values)

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_95_lower": round(mean - margin, 4),
        "ci_95_upper": round(mean + margin, 4),
        "cv": round(std / mean, 4) if mean > 0 else 0.0,
        "n": num_values,
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
    # Average per-case metrics across runs within each config
    averaged_a = average_per_case_metrics_across_runs(summaries_a, metric_path, strategy)
    averaged_b = average_per_case_metrics_across_runs(summaries_b, metric_path, strategy)

    # Find common test cases
    common_test_cases = sorted(set(averaged_a.keys()) & set(averaged_b.keys()))
    if len(common_test_cases) < 3:
        return {
            "error": f"Only {len(common_test_cases)} common test cases — need at least 3 for comparison",
            "common_test_cases": common_test_cases,
        }

    # Compute paired differences
    paired_differences: list[dict[str, Any]] = []
    f1_diffs: list[float] = []
    for test_case_id in common_test_cases:
        f1_a = averaged_a[test_case_id].get("f1", 0.0)
        f1_b = averaged_b[test_case_id].get("f1", 0.0)
        diff = f1_b - f1_a  # positive = B is better
        f1_diffs.append(diff)
        paired_differences.append(
            {
                "test_case_id": test_case_id,
                "f1_a": round(f1_a, 4),
                "f1_b": round(f1_b, 4),
                "diff": round(diff, 4),
            }
        )

    diffs_array = np.array(f1_diffs)
    mean_diff = float(np.mean(diffs_array))

    # 1. Bootstrap CI of mean delta
    rng = np.random.default_rng(seed=42)
    bootstrap_means = []
    for _ in range(bootstrap_iterations):
        resample = rng.choice(diffs_array, size=len(diffs_array), replace=True)
        bootstrap_means.append(float(np.mean(resample)))
    bootstrap_means_arr = np.array(bootstrap_means)
    bootstrap_ci_lower = float(np.percentile(bootstrap_means_arr, 2.5))
    bootstrap_ci_upper = float(np.percentile(bootstrap_means_arr, 97.5))

    # 2. Wilcoxon signed-rank test
    try:
        wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(diffs_array, alternative="two-sided")
        wilcoxon_result = {
            "statistic": float(wilcoxon_stat),
            "p_value": float(wilcoxon_p),
        }
    except ValueError:
        # All differences are zero
        wilcoxon_result = {"statistic": 0.0, "p_value": 1.0}

    # 3. Practical significance
    abs_mean_diff = abs(mean_diff)
    p_value = wilcoxon_result["p_value"]

    if abs_mean_diff >= 0.03 and p_value < 0.05:
        significance = "significant"
    elif abs_mean_diff >= 0.01 and p_value < 0.10:
        significance = "marginal"
    else:
        significance = "not_significant"

    # Direction
    if mean_diff > 0.001:
        direction = "b_better"
    elif mean_diff < -0.001:
        direction = "a_better"
    else:
        direction = "equivalent"

    return {
        "num_common_test_cases": len(common_test_cases),
        "num_runs_a": len(summaries_a),
        "num_runs_b": len(summaries_b),
        "strategy": strategy,
        "metric_path": metric_path,
        "mean_f1_a": round(float(np.mean([averaged_a[tc]["f1"] for tc in common_test_cases])), 4),
        "mean_f1_b": round(float(np.mean([averaged_b[tc]["f1"] for tc in common_test_cases])), 4),
        "mean_diff": round(mean_diff, 4),
        "direction": direction,
        "bootstrap_ci": {
            "lower": round(bootstrap_ci_lower, 4),
            "upper": round(bootstrap_ci_upper, 4),
            "excludes_zero": not (bootstrap_ci_lower <= 0 <= bootstrap_ci_upper),
        },
        "wilcoxon": wilcoxon_result,
        "significance": significance,
        "paired_differences": paired_differences,
    }
