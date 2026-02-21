from __future__ import annotations

from typing import Any

from retrieval_metrics.compute import (
    compute_set_metrics,
    compute_threshold_metrics,
    compute_top_n_metrics,
)
from retrieval_metrics.diagnostics import analyze_query_diagnostics

from memory_retrieval.experiments.metrics_adapter import restriction_evaluation_to_dict
from memory_retrieval.memories.schema import FIELD_RERANK_SCORE


def compute_metrics(
    retrieved_ids: set[str],
    ground_truth_ids: set[str],
) -> dict[str, float]:
    """Compute precision, recall, and F1 score for retrieval results."""
    point = compute_set_metrics(retrieved_ids, ground_truth_ids)
    return {
        "recall": round(point.recall, 4),
        "precision": round(point.precision, 4),
        "f1": round(point.f1, 4),
    }


def analyze_query_performance(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze per-query performance metrics across all queries in an experiment."""
    diagnostics = analyze_query_diagnostics(
        query_results, relevance_key="is_ground_truth", score_key="distance"
    )

    # Keep backward-compatible key names for existing notebooks and scripts.
    for query_entry in diagnostics.get("best_queries", []):
        query_entry["avg_distance"] = query_entry.pop("avg_score", None)
        query_entry["min_distance"] = query_entry.pop("min_score", None)

    for query_entry in diagnostics.get("worst_queries", []):
        query_entry["avg_distance"] = query_entry.pop("avg_score", None)
        query_entry["min_distance"] = query_entry.pop("min_score", None)

    return diagnostics


def pool_and_deduplicate_by_distance(
    query_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool results from all queries and deduplicate by memory ID, keeping best (min) distance."""
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

    return sorted(best_by_memory_id.values(), key=lambda entry: entry.get("distance", 0))


def pool_and_deduplicate_by_rerank_score(
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
        best_by_memory_id.values(),
        key=lambda entry: entry.get(FIELD_RERANK_SCORE, 0),
        reverse=True,
    )


def compute_metrics_at_top_n(
    ranked_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    top_n: int,
    id_field: str = "id",
) -> dict[str, Any]:
    """Compute P/R/F1/MRR for the top-N results from a ranked list."""
    evaluation = compute_top_n_metrics(ranked_results, ground_truth_ids, top_n, id_key=id_field)
    return restriction_evaluation_to_dict(evaluation)


def compute_metrics_at_threshold(
    ranked_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    threshold: float,
    score_field: str,
    higher_is_better: bool,
    id_field: str = "id",
) -> dict[str, Any]:
    """Compute P/R/F1/MRR for results that pass a score threshold."""
    evaluation = compute_threshold_metrics(
        ranked_results,
        ground_truth_ids,
        threshold,
        score_key=score_field,
        higher_is_better=higher_is_better,
        id_key=id_field,
    )
    return restriction_evaluation_to_dict(evaluation, include_accepted_count=True)
