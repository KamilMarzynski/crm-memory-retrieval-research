from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from retrieval_metrics.compute import compute_set_metrics
from retrieval_metrics.diagnostics import analyze_query_diagnostics

from memory_retrieval.constants import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_SEARCH_LIMIT
from memory_retrieval.experiments.metrics import pool_and_deduplicate_by_rerank_score
from memory_retrieval.experiments.metrics_adapter import metric_point_to_dict
from memory_retrieval.infra.io import load_json, save_json
from memory_retrieval.memories.helpers import get_confidence_from_distance
from memory_retrieval.memories.schema import FIELD_DISTANCE, FIELD_RERANK_SCORE, FIELD_SITUATION
from memory_retrieval.search.base import SearchBackend
from memory_retrieval.search.reranker import Reranker
from memory_retrieval.types import RerankerStrategies, TextStrategyFn


@dataclass
class ExperimentConfig:
    """Configuration for running retrieval experiments against pre-generated queries."""

    search_backend: SearchBackend
    search_limit: int = DEFAULT_SEARCH_LIMIT
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    reranker: Reranker | None = None
    rerank_text_strategies: RerankerStrategies | None = None

    def __post_init__(self) -> None:
        if self.search_limit <= 0:
            raise ValueError(f"search_limit must be > 0, got {self.search_limit}")
        if not (0 < self.distance_threshold <= 2.0):
            raise ValueError(f"distance_threshold must be in (0, 2], got {self.distance_threshold}")


def _compute_pre_rerank_metrics(
    query_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    distance_threshold: float,
) -> dict[str, Any]:
    """Compute pre-rerank precision/recall/F1 for all results within the distance threshold."""
    pre_rerank_threshold_ids: set[str] = {
        entry["id"]
        for query_result in query_results
        for entry in query_result["results"]
        if entry.get("distance", 0) <= distance_threshold
    }
    metrics = metric_point_to_dict(compute_set_metrics(pre_rerank_threshold_ids, ground_truth_ids))
    total_unique = len(
        {entry["id"] for query_result in query_results for entry in query_result["results"]}
    )
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "total_unique_retrieved": total_unique,
        "total_within_threshold": len(pre_rerank_threshold_ids),
        "ground_truth_retrieved": len(pre_rerank_threshold_ids & ground_truth_ids),
    }


def _execute_queries(
    queries: list[str],
    db_path: str,
    config: ExperimentConfig,
    ground_truth_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Run all queries against the search backend.

    Tags each result entry with is_ground_truth and the appropriate score field
    (distance for vector backends, rank for FTS5).

    Returns (query_results, all_retrieved_ids).
    """
    all_retrieved_ids: set[str] = set()
    query_results: list[dict[str, Any]] = []

    for query in queries:
        results = config.search_backend.search(db_path, query, limit=config.search_limit)
        retrieved_ids = {result.id for result in results}
        all_retrieved_ids.update(retrieved_ids)

        result_entries: list[dict[str, Any]] = []
        for result in results:
            entry: dict[str, Any] = {
                "id": result.id,
                "situation": result.situation,
                "lesson": result.lesson,
                "is_ground_truth": result.id in ground_truth_ids,
            }
            if result.score_type == "cosine_distance":
                entry["distance"] = result.raw_score
                entry["confidence"] = get_confidence_from_distance(result.raw_score)
            else:
                entry["rank"] = result.raw_score
            result_entries.append(entry)

        query_results.append(
            {
                "query": query,
                "word_count": len(query.split()),
                "result_count": len(results),
                "results": result_entries,
            }
        )

    return query_results, all_retrieved_ids


def _run_rerank_strategy(
    reranker: Reranker,
    query_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    text_fn: TextStrategyFn | None = None,
    text_field: str = "situation",
) -> dict[str, Any]:
    """Run reranking for a single text strategy and return all pooled results.

    Collects all (query, document) pairs across every query into a single
    model.predict() call, then redistributes scores back per query.
    All candidates are stored — no top-N filtering here.
    Notebooks handle top-N and threshold analysis downstream.
    """
    # Collect all (query, document) pairs in one flat list, carrying each candidate alongside
    all_pairs: list[tuple[str, str]] = []
    pair_metadata: list[tuple[int, dict[str, Any]]] = []  # (query_idx, candidate)

    for query_idx, query_result in enumerate(query_results):
        for candidate in query_result["results"]:
            doc_text = text_fn(candidate) if text_fn else candidate[text_field]
            all_pairs.append((query_result["query"], doc_text))
            pair_metadata.append((query_idx, candidate))

    # Single model.predict() call for all pairs
    all_scores = reranker.score_all_pairs(all_pairs)

    # Redistribute scores back to their query buckets
    scored_candidates_per_query: list[list[dict[str, Any]]] = [[] for _ in query_results]
    for (query_idx, candidate), score in zip(pair_metadata, all_scores, strict=True):
        enriched = dict(candidate)
        enriched[FIELD_RERANK_SCORE] = score
        scored_candidates_per_query[query_idx].append(enriched)

    # Sort each query's results by rerank score descending
    all_reranked_per_query = []
    for query_result, scored_candidates in zip(
        query_results, scored_candidates_per_query, strict=True
    ):
        reranked_for_query = sorted(
            scored_candidates,
            key=lambda candidate: candidate[FIELD_RERANK_SCORE],
            reverse=True,
        )
        all_reranked_per_query.append(
            {"query": query_result["query"], "reranked": reranked_for_query}
        )

    pooled_reranked = pool_and_deduplicate_by_rerank_score(all_reranked_per_query)

    return {
        "pooled_count": len(pooled_reranked),
        "reranked_results": [
            {
                "id": result["id"],
                "rerank_score": result[FIELD_RERANK_SCORE],
                "distance": result.get(FIELD_DISTANCE, result.get("distance", 0)),
                "situation": result.get(FIELD_SITUATION, result.get("situation", "")),
                "lesson": result.get("lesson", ""),
                "is_ground_truth": result["id"] in ground_truth_ids,
            }
            for result in pooled_reranked
        ],
        "per_query_reranked": [
            {
                "query": per_query["query"],
                "reranked": [
                    {
                        "id": result["id"],
                        "rerank_score": result[FIELD_RERANK_SCORE],
                        "distance": result.get(FIELD_DISTANCE, result.get("distance", 0)),
                        "is_ground_truth": result["id"] in ground_truth_ids,
                    }
                    for result in per_query["reranked"]
                ],
            }
            for per_query in all_reranked_per_query
        ],
    }


def _print_experiment_summary(
    all_results: list[dict[str, Any]],
    config: ExperimentConfig,
) -> None:
    """Print aggregate metrics after all experiments complete."""
    success_key = "pre_rerank_metrics" if config.reranker is not None else "metrics"
    successful = [
        experiment_result for experiment_result in all_results if success_key in experiment_result
    ]

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    if not successful:
        print("No successful experiments")
        if len(successful) < len(all_results):
            print(f"Failed experiments: {len(all_results) - len(successful)}")
        return

    if config.reranker is not None:
        print(f"Experiments run: {len(successful)}")

        avg_pre_n = sum(
            experiment_result["pre_rerank_metrics"]["total_within_threshold"]
            for experiment_result in successful
        ) / len(successful)

        print(
            f"\nPre-rerank metrics (all candidates within distance threshold, ~{avg_pre_n:.0f} avg):"
        )
        for metric in ["recall", "precision", "f1"]:
            avg_value = sum(
                experiment_result["pre_rerank_metrics"][metric] for experiment_result in successful
            ) / len(successful)
            print(f"  {metric:<12} {avg_value:.3f}")

        print(
            "\nReranked results stored — use notebook analysis cells for top-N and threshold sweeps."
        )
    else:
        avg_recall = sum(
            experiment_result["metrics"]["recall"] for experiment_result in successful
        ) / len(successful)
        avg_precision = sum(
            experiment_result["metrics"]["precision"] for experiment_result in successful
        ) / len(successful)
        avg_f1 = sum(experiment_result["metrics"]["f1"] for experiment_result in successful) / len(
            successful
        )
        total_gt = sum(
            experiment_result["ground_truth"]["count"] for experiment_result in successful
        )
        total_retrieved = sum(
            experiment_result["metrics"]["ground_truth_retrieved"]
            for experiment_result in successful
        )

        print(f"Experiments run: {len(successful)}")
        print(f"Total ground truth memories: {total_gt}")
        print(f"Total retrieved: {total_retrieved}")
        print("\nAGGREGATE METRICS:")
        print(f"  Average recall:    {avg_recall:.1%}")
        print(f"  Average precision: {avg_precision:.1%}")
        print(f"  Average F1:        {avg_f1:.3f}")

    if len(successful) < len(all_results):
        print(f"Failed experiments: {len(all_results) - len(successful)}")


def _build_reranking_fields(
    query_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    queries: list[str],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run reranking for all configured strategies and return result fields.

    Called when config.reranker is not None. Each strategy's results are pooled
    and deduplicated. The primary (first) strategy populates top-level keys;
    additional strategies are stored under rerank_strategies.
    """
    extra: dict[str, Any] = {
        "distance_threshold": config.distance_threshold,
        "reranker_model": config.reranker.model_name,  # type: ignore[union-attr]
        "pre_rerank_metrics": _compute_pre_rerank_metrics(
            query_results, ground_truth_ids, config.distance_threshold
        ),
        "rerank_queries": queries,
    }

    strategies = config.rerank_text_strategies or {"default": None}
    strategies_results: dict[str, dict[str, Any]] = {}

    for strategy_name, text_fn in strategies.items():
        strategies_results[strategy_name] = _run_rerank_strategy(
            config.reranker,  # type: ignore[arg-type]
            query_results,
            ground_truth_ids,
            text_fn=text_fn,
            text_field="situation",
        )

    # Primary strategy (first one) populates top-level keys
    primary_name = next(iter(strategies))
    primary = strategies_results[primary_name]
    extra["reranked_results"] = primary["reranked_results"]
    extra["per_query_reranked"] = primary["per_query_reranked"]

    if len(strategies) > 1:
        extra["rerank_strategies"] = {
            name: {key: value for key, value in strategy.items() if key != "pooled_count"}
            for name, strategy in strategies_results.items()
        }

    return extra


def _build_standard_fields(
    query_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    queries: list[str],
    all_retrieved_ids: set[str],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Compute metrics for the standard (no reranking) retrieval path.

    Uses distance-based filtering for vector results; falls back to all
    retrieved IDs when results contain no distance scores (FTS5).
    """
    # Vector results have a "distance" key; FTS5 results have a "rank" key.
    # Use distance-based filtering only when results contain distance scores.
    uses_distance_filtering = any(
        "distance" in entry for query_result in query_results for entry in query_result["results"]
    )

    if uses_distance_filtering:
        filtered_retrieved_ids: set[str] = {
            entry["id"]
            for query_result in query_results
            for entry in query_result["results"]
            if entry.get("distance", 0) <= config.distance_threshold
        }
    else:
        filtered_retrieved_ids = all_retrieved_ids

    metrics = metric_point_to_dict(compute_set_metrics(filtered_retrieved_ids, ground_truth_ids))

    query_analysis = analyze_query_diagnostics(
        query_results,
        relevance_key="is_ground_truth",
        score_key="distance",
    )
    for query_entry in query_analysis.get("best_queries", []):
        query_entry["avg_distance"] = query_entry.pop("avg_score", None)
        query_entry["min_distance"] = query_entry.pop("min_score", None)
    for query_entry in query_analysis.get("worst_queries", []):
        query_entry["avg_distance"] = query_entry.pop("avg_score", None)
        query_entry["min_distance"] = query_entry.pop("min_score", None)

    return {
        "distance_threshold": config.distance_threshold,
        "metrics": {
            "total_queries": len(queries),
            "total_unique_retrieved": len(all_retrieved_ids),
            "total_within_threshold": len(filtered_retrieved_ids),
            "ground_truth_retrieved": len(filtered_retrieved_ids & ground_truth_ids),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "query_analysis": query_analysis,
        "retrieved_ground_truth_ids": sorted(list(filtered_retrieved_ids & ground_truth_ids)),
        "missed_ground_truth_ids": sorted(list(ground_truth_ids - filtered_retrieved_ids)),
    }


def run_experiment(
    test_case_path: str,
    query_file_path: str,
    db_path: str,
    results_dir: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run a retrieval experiment using pre-generated queries.

    Loads queries from query_file_path (generated by generate_queries_for_test_case),
    runs search and optional reranking, computes metrics against ground truth.
    """
    test_case = load_json(test_case_path)
    query_data = load_json(query_file_path)
    meta = test_case.get("metadata", {})
    ground_truth_ids = set(test_case.get("ground_truth_memory_ids", []))

    queries = query_data.get("queries", [])
    test_case_id = test_case.get("test_case_id", "unknown")

    query_results, all_retrieved_ids = _execute_queries(queries, db_path, config, ground_truth_ids)

    result: dict[str, Any] = {
        "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "test_case_id": test_case_id,
        "source_file": test_case.get("source_file", "unknown"),
        "pr_context": f"{meta.get('sourceBranch', '?')} -> {meta.get('targetBranch', '?')}",
        "query_generation": {
            "model": query_data.get("model", "unknown"),
            "prompt_version": query_data.get("prompt_version", "unknown"),
            "generated_at": query_data.get("generated_at", "unknown"),
        },
        "search_limit": config.search_limit,
        "diff_stats": test_case.get("diff_stats", {}),
        "ground_truth": {
            "memory_ids": sorted(list(ground_truth_ids)),
            "count": len(ground_truth_ids),
        },
        "queries": query_results,
    }

    if query_data.get("sample_memories_used"):
        result["sample_memories_used"] = query_data["sample_memories_used"]

    if config.reranker is not None:
        result.update(_build_reranking_fields(query_results, ground_truth_ids, queries, config))
    else:
        result.update(
            _build_standard_fields(
                query_results, ground_truth_ids, queries, all_retrieved_ids, config
            )
        )

    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_dir) / f"results_{test_case_id}_{result['experiment_id']}.json"
    save_json(result, results_path)

    return result


def run_all_experiments(
    test_cases_dir: str,
    queries_dir: str,
    db_path: str,
    results_dir: str,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """Run experiments for all test cases using pre-generated queries.

    Each test case is matched to its query file by filename stem
    (e.g. test_cases/tc_review_1.json -> queries/tc_review_1.json).
    """
    test_case_files = sorted(Path(test_cases_dir).glob("*.json"))
    queries_path = Path(queries_dir)
    all_results: list[dict[str, Any]] = []

    for i, test_case_file in enumerate(test_case_files):
        query_file = queries_path / test_case_file.name
        if not query_file.exists():
            print(
                f"[{i + 1}/{len(test_case_files)}] {test_case_file.stem} — skipped (no query file)"
            )
            all_results.append({"test_case_file": test_case_file.name, "error": "no query file"})
            continue

        try:
            experiment_result = run_experiment(
                str(test_case_file),
                query_file_path=str(query_file),
                db_path=db_path,
                results_dir=results_dir,
                config=config,
            )
            all_results.append(experiment_result)
            print(f"[{i + 1}/{len(test_case_files)}] {test_case_file.stem} — done")
        except Exception as error:
            print(f"[{i + 1}/{len(test_case_files)}] {test_case_file.stem} — ERROR: {error}")
            all_results.append({"test_case_file": test_case_file.name, "error": str(error)})

    _print_experiment_summary(all_results, config)
    return all_results
