from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_retrieval.experiments.metrics import analyze_query_performance, compute_metrics
from memory_retrieval.infra.io import load_json, save_json
from memory_retrieval.memories.schema import FIELD_DISTANCE, FIELD_RERANK_SCORE, FIELD_SITUATION
from memory_retrieval.search.base import SearchBackend
from memory_retrieval.search.reranker import Reranker
from memory_retrieval.search.vector import VectorBackend, get_confidence_from_distance

DEFAULT_SEARCH_LIMIT = 20
DEFAULT_DISTANCE_THRESHOLD = 1.1


@dataclass
class ExperimentConfig:
    """Configuration for running retrieval experiments against pre-generated queries."""

    search_backend: SearchBackend
    search_limit: int = DEFAULT_SEARCH_LIMIT
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    reranker: Reranker | None = None
    rerank_text_strategies: dict[str, Callable[[dict[str, Any]], str]] | None = None


def _pool_and_deduplicate(
    query_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool results from all queries and deduplicate by memory ID."""
    best_by_memory_id: dict[str, dict[str, Any]] = {}

    for query_result in query_results:
        for result in query_result.get("results", []):
            memory_id = result["id"]
            current_distance = result.get("distance", float("inf"))
            best_distance = (
                best_by_memory_id[memory_id].get("distance", float("inf"))
                if memory_id in best_by_memory_id
                else float("inf")
            )
            if current_distance < best_distance:
                best_by_memory_id[memory_id] = result

    return sorted(best_by_memory_id.values(), key=lambda x: x.get("distance", 0))


def _pool_and_deduplicate_by_rerank_score(
    per_query_reranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool reranked results from all queries, keeping the highest rerank score per memory."""
    best_by_memory_id: dict[str, dict[str, Any]] = {}

    for query_result in per_query_reranked:
        for result in query_result["reranked"]:
            memory_id = result["id"]
            rerank_score = result.get(FIELD_RERANK_SCORE, float("-inf"))
            if memory_id not in best_by_memory_id or rerank_score > best_by_memory_id[
                memory_id
            ].get(FIELD_RERANK_SCORE, float("-inf")):
                best_by_memory_id[memory_id] = result

    return sorted(
        best_by_memory_id.values(), key=lambda x: x.get(FIELD_RERANK_SCORE, 0), reverse=True
    )


def _run_rerank_strategy(
    reranker: Reranker,
    query_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    text_fn: Callable[[dict[str, Any]], str] | None = None,
    text_field: str = "situation",
) -> dict[str, Any]:
    """Run reranking for a single text strategy and return all pooled results.

    All candidates are reranked and stored — no top-N filtering is applied here.
    Notebooks handle top-N and threshold analysis downstream.
    """
    all_reranked_per_query = []
    for query_result in query_results:
        reranked_for_query = reranker.rerank(
            query_result["query"],
            query_result["results"],
            top_n=None,
            text_field=text_field,
            text_fn=text_fn,
        )
        all_reranked_per_query.append(
            {
                "query": query_result["query"],
                "reranked": reranked_for_query,
            }
        )

    pooled_reranked = _pool_and_deduplicate_by_rerank_score(all_reranked_per_query)

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
    diff_stats = test_case.get("diff_stats", {})

    queries = query_data.get("queries", [])

    test_case_id = test_case.get("test_case_id", "unknown")

    # Execute search for each query
    all_retrieved_ids: set[str] = set()
    query_results: list[dict[str, Any]] = []
    is_vector = isinstance(config.search_backend, VectorBackend)

    for query in queries:
        results = config.search_backend.search(db_path, query, limit=config.search_limit)
        retrieved_ids = {result.id for result in results}
        all_retrieved_ids.update(retrieved_ids)

        query_result_entries = []
        for result in results:
            result_entry: dict[str, Any] = {
                "id": result.id,
                "situation": result.situation,
                "lesson": result.lesson,
                "is_ground_truth": result.id in ground_truth_ids,
            }
            if is_vector:
                result_entry["distance"] = result.raw_score
                result_entry["confidence"] = get_confidence_from_distance(result.raw_score)
            else:
                result_entry["rank"] = result.raw_score

            query_result_entries.append(result_entry)

        query_results.append(
            {
                "query": query,
                "word_count": len(query.split()),
                "result_count": len(results),
                "results": query_result_entries,
            }
        )

    # --- Build result structure ---
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
        "diff_stats": diff_stats,
        "ground_truth": {
            "memory_ids": sorted(list(ground_truth_ids)),
            "count": len(ground_truth_ids),
        },
        "queries": query_results,
    }

    if query_data.get("sample_memories_used"):
        result["sample_memories_used"] = query_data["sample_memories_used"]

    if config.reranker is not None:
        # --- Reranking path ---
        result["distance_threshold"] = config.distance_threshold
        result["reranker_model"] = config.reranker.model_name

        # Pre-rerank metrics (all candidates within distance threshold)
        pre_rerank_threshold_ids: set[str] = {
            entry["id"]
            for query_result in query_results
            for entry in query_result["results"]
            if entry.get("distance", 0) <= config.distance_threshold
        }
        pre_rerank_metrics = compute_metrics(pre_rerank_threshold_ids, ground_truth_ids)

        result["pre_rerank_metrics"] = {
            **pre_rerank_metrics,
            "total_unique_retrieved": len(all_retrieved_ids),
            "total_within_threshold": len(pre_rerank_threshold_ids),
            "ground_truth_retrieved": len(pre_rerank_threshold_ids & ground_truth_ids),
        }

        # Per-query reranking with strategy support
        strategies = config.rerank_text_strategies or {"default": None}

        result["rerank_queries"] = queries
        strategies_results: dict[str, dict[str, Any]] = {}

        for strategy_name, text_fn in strategies.items():
            strategy = _run_rerank_strategy(
                config.reranker,
                query_results,
                ground_truth_ids,
                text_fn=text_fn,
                text_field="situation",
            )
            strategies_results[strategy_name] = strategy

        # Primary strategy (first one) populates top-level keys
        primary_name = next(iter(strategies))
        primary = strategies_results[primary_name]
        result["reranked_results"] = primary["reranked_results"]
        result["per_query_reranked"] = primary["per_query_reranked"]

        if len(strategies) > 1:
            result["rerank_strategies"] = {
                name: {k: v for k, v in strategy.items() if k != "pooled_count"}
                for name, strategy in strategies_results.items()
            }

    else:
        # --- Standard (no reranking) path ---
        result["distance_threshold"] = config.distance_threshold

        if is_vector:
            filtered_retrieved_ids: set[str] = {
                entry["id"]
                for query_result in query_results
                for entry in query_result["results"]
                if entry.get("distance", 0) <= config.distance_threshold
            }
        else:
            filtered_retrieved_ids = all_retrieved_ids

        metrics = compute_metrics(filtered_retrieved_ids, ground_truth_ids)
        query_analysis = analyze_query_performance(query_results)

        result["metrics"] = {
            "total_queries": len(queries),
            "total_unique_retrieved": len(all_retrieved_ids),
            "total_within_threshold": len(filtered_retrieved_ids),
            "ground_truth_retrieved": len(filtered_retrieved_ids & ground_truth_ids),
            **metrics,
        }
        result["query_analysis"] = query_analysis
        result["retrieved_ground_truth_ids"] = sorted(
            list(filtered_retrieved_ids & ground_truth_ids)
        )
        result["missed_ground_truth_ids"] = sorted(list(ground_truth_ids - filtered_retrieved_ids))

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

    # Print summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    success_key = "pre_rerank_metrics" if config.reranker is not None else "metrics"
    successful = [
        experiment_result for experiment_result in all_results if success_key in experiment_result
    ]

    if config.reranker is not None:
        if successful:
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
                    experiment_result["pre_rerank_metrics"][metric]
                    for experiment_result in successful
                ) / len(successful)
                print(f"  {metric:<12} {avg_value:.3f}")

            print(
                "\nReranked results stored — use notebook analysis cells for top-N and threshold sweeps."
            )
        else:
            print("No successful experiments")
    else:
        if successful:
            avg_recall = sum(
                experiment_result["metrics"]["recall"] for experiment_result in successful
            ) / len(successful)
            avg_precision = sum(
                experiment_result["metrics"]["precision"] for experiment_result in successful
            ) / len(successful)
            avg_f1 = sum(
                experiment_result["metrics"]["f1"] for experiment_result in successful
            ) / len(successful)
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
        else:
            print("No successful experiments")

    if len(successful) < len(all_results):
        print(f"Failed experiments: {len(all_results) - len(successful)}")

    return all_results
