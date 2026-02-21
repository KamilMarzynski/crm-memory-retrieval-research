from memory_retrieval.experiments.metrics import (
    pool_and_deduplicate_by_distance,
    pool_and_deduplicate_by_rerank_score,
)


# ---------- pool_and_deduplicate_by_distance ----------


def test_pool_by_distance_same_memory_keeps_minimum_distance() -> None:
    query_results = [
        {"results": [{"id": "m1", "distance": 0.5}]},
        {"results": [{"id": "m1", "distance": 0.2}]},  # lower = better
    ]
    pooled = pool_and_deduplicate_by_distance(query_results)
    assert len(pooled) == 1
    assert pooled[0]["distance"] == 0.2


def test_pool_by_distance_unique_memories_all_included() -> None:
    query_results = [
        {"results": [{"id": "m1", "distance": 0.3}]},
        {"results": [{"id": "m2", "distance": 0.1}]},
        {"results": [{"id": "m3", "distance": 0.7}]},
    ]
    pooled = pool_and_deduplicate_by_distance(query_results)
    assert len(pooled) == 3


def test_pool_by_distance_result_sorted_ascending_by_distance() -> None:
    query_results = [
        {
            "results": [
                {"id": "m1", "distance": 0.8},
                {"id": "m2", "distance": 0.2},
                {"id": "m3", "distance": 0.5},
            ]
        },
    ]
    pooled = pool_and_deduplicate_by_distance(query_results)
    distances = [result["distance"] for result in pooled]
    assert distances == sorted(distances)


def test_pool_by_distance_preserves_all_result_fields() -> None:
    query_results = [
        {"results": [{"id": "m1", "distance": 0.3, "situation": "A code situation."}]},
    ]
    pooled = pool_and_deduplicate_by_distance(query_results)
    assert pooled[0]["situation"] == "A code situation."


def test_pool_by_distance_empty_results_returns_empty_list() -> None:
    query_results: list[dict] = []
    pooled = pool_and_deduplicate_by_distance(query_results)
    assert pooled == []


# ---------- pool_and_deduplicate_by_rerank_score ----------


def test_pool_by_rerank_same_memory_keeps_maximum_score() -> None:
    per_query_reranked = [
        {"reranked": [{"id": "m1", "rerank_score": 0.3}]},
        {"reranked": [{"id": "m1", "rerank_score": 0.9}]},  # higher = better
    ]
    pooled = pool_and_deduplicate_by_rerank_score(per_query_reranked)
    assert len(pooled) == 1
    assert pooled[0]["rerank_score"] == 0.9


def test_pool_by_rerank_unique_memories_all_included() -> None:
    per_query_reranked = [
        {"reranked": [{"id": "m1", "rerank_score": 0.9}]},
        {"reranked": [{"id": "m2", "rerank_score": 0.5}]},
        {"reranked": [{"id": "m3", "rerank_score": 0.1}]},
    ]
    pooled = pool_and_deduplicate_by_rerank_score(per_query_reranked)
    assert len(pooled) == 3


def test_pool_by_rerank_result_sorted_descending_by_score() -> None:
    per_query_reranked = [
        {
            "reranked": [
                {"id": "m1", "rerank_score": 0.3},
                {"id": "m2", "rerank_score": 0.9},
                {"id": "m3", "rerank_score": 0.6},
            ]
        },
    ]
    pooled = pool_and_deduplicate_by_rerank_score(per_query_reranked)
    scores = [result["rerank_score"] for result in pooled]
    assert scores == sorted(scores, reverse=True)


def test_pool_by_rerank_empty_results_returns_empty_list() -> None:
    per_query_reranked: list[dict] = []
    pooled = pool_and_deduplicate_by_rerank_score(per_query_reranked)
    assert pooled == []
