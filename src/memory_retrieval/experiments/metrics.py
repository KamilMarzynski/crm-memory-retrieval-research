import statistics
from typing import Any


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
        distances = [result["distance"] for result in query_result.get("results", []) if "distance" in result]

        query_stats.append({
            "query": query_result["query"],
            "word_count": len(query_result["query"].split()),
            "hits": hits,
            "precision": hits / total if total > 0 else 0,
            "avg_distance": statistics.mean(distances) if distances else None,
            "min_distance": min(distances) if distances else None,
        })

    queries_with_hits = sum(1 for query_stat in query_stats if query_stat["hits"] > 0)
    total_queries = len(query_stats)

    return {
        "queries_with_hits": queries_with_hits,
        "total_queries": total_queries,
        "query_hit_rate": queries_with_hits / total_queries if total_queries > 0 else 0,
        "best_queries": sorted(query_stats, key=lambda x: -x["hits"])[:3],
        "worst_queries": sorted(query_stats, key=lambda x: x["hits"])[:3],
        "avg_word_count": statistics.mean([query_stat["word_count"] for query_stat in query_stats]) if query_stats else 0,
    }
