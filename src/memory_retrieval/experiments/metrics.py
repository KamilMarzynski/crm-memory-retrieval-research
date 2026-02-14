import statistics
from typing import Any

from memory_retrieval.memories.schema import FIELD_RERANK_SCORE


def compute_metrics(
    retrieved_ids: set[str],
    ground_truth_ids: set[str],
) -> dict[str, float]:
    """Compute precision, recall, and F1 score for retrieval results.

    Args:
        retrieved_ids: Set of memory IDs that were retrieved by the system.
        ground_truth_ids: Set of memory IDs that should have been retrieved (ground truth).

    Returns:
        Dictionary with 'recall', 'precision', and 'f1' scores (all rounded to 4 decimals).
        Returns zeros if ground_truth_ids is empty.
    """
    if not ground_truth_ids:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    hits = len(retrieved_ids & ground_truth_ids)
    recall = hits / len(ground_truth_ids)
    precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def analyze_query_performance(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze per-query performance metrics across all queries in an experiment.

    Args:
        query_results: List of query result dictionaries, each containing:
            - query: The query text
            - results: List of search results with is_ground_truth and distance fields

    Returns:
        Dictionary with aggregate query statistics:
            - queries_with_hits: Count of queries that retrieved at least one ground truth memory
            - total_queries: Total number of queries analyzed
            - query_hit_rate: Fraction of queries with hits
            - best_queries: Top 3 queries by number of ground truth hits
            - worst_queries: Bottom 3 queries by number of ground truth hits
            - avg_word_count: Average length of queries in words
    """
    query_stats = []

    for query_result in query_results:
        hits = sum(1 for result in query_result.get("results", []) if result.get("is_ground_truth"))
        total = len(query_result.get("results", []))
        distances = [
            result["distance"] for result in query_result.get("results", []) if "distance" in result
        ]

        query_stats.append(
            {
                "query": query_result["query"],
                "word_count": len(query_result["query"].split()),
                "hits": hits,
                "precision": hits / total if total > 0 else 0,
                "avg_distance": statistics.mean(distances) if distances else None,
                "min_distance": min(distances) if distances else None,
            }
        )

    queries_with_hits = sum(1 for query_stat in query_stats if query_stat["hits"] > 0)
    total_queries = len(query_stats)

    return {
        "queries_with_hits": queries_with_hits,
        "total_queries": total_queries,
        "query_hit_rate": queries_with_hits / total_queries if total_queries > 0 else 0,
        "best_queries": sorted(query_stats, key=lambda x: -x["hits"])[:3],
        "worst_queries": sorted(query_stats, key=lambda x: x["hits"])[:3],
        "avg_word_count": statistics.mean([query_stat["word_count"] for query_stat in query_stats])
        if query_stats
        else 0,
    }


# ---------------------------------------------------------------------------
# Pooling / deduplication
# ---------------------------------------------------------------------------


def pool_and_deduplicate_by_distance(
    query_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool results from all queries and deduplicate by memory ID, keeping best (min) distance.

    Args:
        query_results: List of per-query result dicts, each with a "results" list
            containing dicts with at least "id" and "distance" keys.

    Returns:
        List of result dicts sorted by distance ascending (best first).
    """
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


def pool_and_deduplicate_by_rerank_score(
    per_query_reranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool reranked results from all queries, keeping the highest rerank score per memory.

    Args:
        per_query_reranked: List of per-query dicts, each with a "reranked" list
            containing dicts with at least "id" and FIELD_RERANK_SCORE keys.

    Returns:
        List of result dicts sorted by rerank score descending (best first).
    """
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


# ---------------------------------------------------------------------------
# MRR building block
# ---------------------------------------------------------------------------


def reciprocal_rank(ranked_ids: list[str], ground_truth_ids: set[str]) -> float:
    """Return 1/rank of the first ground truth hit in a ranked list, or 0 if none found.

    Args:
        ranked_ids: Ordered list of memory IDs (best first).
        ground_truth_ids: Set of ground truth memory IDs.

    Returns:
        Reciprocal rank (1/rank) of first hit, or 0.0.
    """
    for rank, memory_id in enumerate(ranked_ids, 1):
        if memory_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Single test case metrics at cutoff
# ---------------------------------------------------------------------------


def compute_metrics_at_top_n(
    ranked_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    top_n: int,
    id_field: str = "id",
) -> dict[str, Any]:
    """Compute P/R/F1/MRR for the top-N results from a ranked list.

    Args:
        ranked_results: List of result dicts, pre-sorted by relevance (best first).
        ground_truth_ids: Set of ground truth memory IDs.
        top_n: Number of top results to consider.
        id_field: Key name for the memory ID in each result dict.

    Returns:
        Dict with precision, recall, f1, mrr, and retrieved_ids.
    """
    top_results = ranked_results[:top_n]
    retrieved_ids = {result[id_field] for result in top_results}
    ranked_id_list = [result[id_field] for result in top_results]

    ground_truth_count = len(ground_truth_ids)
    hits = len(retrieved_ids & ground_truth_ids)
    num_retrieved = len(retrieved_ids)

    precision = hits / num_retrieved if num_retrieved > 0 else 0.0
    recall = hits / ground_truth_count if ground_truth_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mrr = reciprocal_rank(ranked_id_list, ground_truth_ids)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "retrieved_ids": retrieved_ids,
    }


def compute_metrics_at_threshold(
    ranked_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    threshold: float,
    score_field: str,
    higher_is_better: bool,
    id_field: str = "id",
) -> dict[str, Any]:
    """Compute P/R/F1/MRR for results that pass a score threshold.

    Args:
        ranked_results: List of result dicts, pre-sorted by relevance (best first).
        ground_truth_ids: Set of ground truth memory IDs.
        threshold: Score threshold for filtering.
        score_field: Key name for the score value in each result dict.
        higher_is_better: If True, accept results with score >= threshold.
            If False, accept results with score <= threshold.
        id_field: Key name for the memory ID in each result dict.

    Returns:
        Dict with precision, recall, f1, mrr, retrieved_ids, and accepted_count.
    """
    if higher_is_better:
        accepted = [result for result in ranked_results if result[score_field] >= threshold]
    else:
        accepted = [result for result in ranked_results if result[score_field] <= threshold]

    retrieved_ids = {result[id_field] for result in accepted}
    ranked_id_list = [result[id_field] for result in accepted]

    ground_truth_count = len(ground_truth_ids)
    hits = len(retrieved_ids & ground_truth_ids)
    num_retrieved = len(retrieved_ids)

    precision = hits / num_retrieved if num_retrieved > 0 else 0.0
    recall = hits / ground_truth_count if ground_truth_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mrr = reciprocal_rank(ranked_id_list, ground_truth_ids)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "retrieved_ids": retrieved_ids,
        "accepted_count": num_retrieved,
    }


# ---------------------------------------------------------------------------
# Macro-averaged sweeps
# ---------------------------------------------------------------------------


def sweep_top_n(
    experiments: list[dict[str, Any]],
    n_values: list[int],
    id_field: str = "id",
) -> list[dict[str, Any]]:
    """Macro-averaged P/R/F1/MRR sweep across top-N values.

    Args:
        experiments: List of experiment dicts, each with:
            - ground_truth_ids: set[str] of ground truth memory IDs
            - ranked_results: list[dict] pre-sorted by relevance (best first)
        n_values: List of N values to sweep.
        id_field: Key name for the memory ID in each result dict.

    Returns:
        List of dicts, one per N value: {top_n, precision, recall, f1, mrr}.
    """
    sweep_results = []
    for top_n in n_values:
        per_case_metrics = []
        for experiment in experiments:
            metrics = compute_metrics_at_top_n(
                experiment["ranked_results"],
                experiment["ground_truth_ids"],
                top_n,
                id_field=id_field,
            )
            per_case_metrics.append(metrics)
        averaged = macro_average(per_case_metrics)
        averaged["top_n"] = top_n
        sweep_results.append(averaged)
    return sweep_results


def sweep_threshold(
    experiments: list[dict[str, Any]],
    thresholds: list[float],
    score_field: str,
    higher_is_better: bool,
    id_field: str = "id",
) -> list[dict[str, Any]]:
    """Macro-averaged P/R/F1/MRR sweep across score thresholds.

    Args:
        experiments: List of experiment dicts, each with:
            - ground_truth_ids: set[str] of ground truth memory IDs
            - ranked_results: list[dict] pre-sorted by relevance, each with score_field
        thresholds: List of threshold values to sweep.
        score_field: Key name for the score value in each result dict.
        higher_is_better: If True, accept results with score >= threshold.
            If False, accept results with score <= threshold.
        id_field: Key name for the memory ID in each result dict.

    Returns:
        List of dicts, one per threshold: {threshold, precision, recall, f1, mrr}.
    """
    sweep_results = []
    for threshold in thresholds:
        per_case_metrics = []
        for experiment in experiments:
            metrics = compute_metrics_at_threshold(
                experiment["ranked_results"],
                experiment["ground_truth_ids"],
                threshold,
                score_field,
                higher_is_better,
                id_field=id_field,
            )
            per_case_metrics.append(metrics)
        averaged = macro_average(per_case_metrics)
        averaged["threshold"] = threshold
        sweep_results.append(averaged)
    return sweep_results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def macro_average(per_case_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Average a list of metric dicts (precision, recall, f1, mrr).

    Args:
        per_case_metrics: List of dicts, each with at least precision, recall, f1, mrr keys.

    Returns:
        Dict with same metric keys, averaged values.
    """
    if not per_case_metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}

    metric_keys = ["precision", "recall", "f1", "mrr"]
    result = {}
    for key in metric_keys:
        values = [metrics[key] for metrics in per_case_metrics if key in metrics]
        result[key] = sum(values) / len(values) if values else 0.0
    return result


def find_optimal_threshold(
    sweep_results: list[dict[str, Any]],
    metric: str = "f1",
) -> dict[str, Any]:
    """Find the sweep entry maximizing the given metric.

    Args:
        sweep_results: Output from sweep_threshold() or sweep_top_n().
        metric: Metric key to optimize (default: "f1").

    Returns:
        Dict with threshold (or top_n), precision, recall, f1, mrr, and index.
    """
    best_index = 0
    best_value = -1.0
    for index, entry in enumerate(sweep_results):
        if entry.get(metric, 0.0) > best_value:
            best_value = entry[metric]
            best_index = index

    result = dict(sweep_results[best_index])
    result["index"] = best_index
    return result
