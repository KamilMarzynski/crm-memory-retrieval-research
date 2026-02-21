import pytest

from memory_retrieval.experiments.runner import (
    ExperimentConfig,
    _compute_pre_rerank_metrics,
)
from memory_retrieval.search.fts5 import FTS5Backend


def test_experiment_config_validates_search_limit() -> None:
    with pytest.raises(ValueError, match="search_limit"):
        ExperimentConfig(search_backend=FTS5Backend(), search_limit=0)


def test_experiment_config_validates_distance_threshold_too_low() -> None:
    with pytest.raises(ValueError, match="distance_threshold"):
        ExperimentConfig(search_backend=FTS5Backend(), distance_threshold=0.0)


def test_experiment_config_validates_distance_threshold_too_high() -> None:
    with pytest.raises(ValueError, match="distance_threshold"):
        ExperimentConfig(search_backend=FTS5Backend(), distance_threshold=2.5)


def test_compute_pre_rerank_metrics_empty_retrieval() -> None:
    query_results = [{"query": "test", "results": [], "word_count": 1, "result_count": 0}]
    ground_truth_ids = {"mem_abc"}
    metrics = _compute_pre_rerank_metrics(
        query_results=query_results,
        ground_truth_ids=ground_truth_ids,
        distance_threshold=1.1,
    )
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0


def test_compute_pre_rerank_metrics_perfect_retrieval() -> None:
    query_results = [
        {
            "query": "test query",
            "results": [{"id": "mem_abc", "distance": 0.3}],
            "word_count": 2,
            "result_count": 1,
        }
    ]
    ground_truth_ids = {"mem_abc"}
    metrics = _compute_pre_rerank_metrics(
        query_results=query_results,
        ground_truth_ids=ground_truth_ids,
        distance_threshold=1.1,
    )
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_compute_pre_rerank_metrics_distance_threshold_filters() -> None:
    """Results beyond the threshold should be excluded from metric computation."""
    query_results = [
        {
            "query": "test query",
            "results": [
                {"id": "mem_abc", "distance": 0.3},
                {"id": "mem_def", "distance": 1.5},  # beyond threshold
            ],
            "word_count": 2,
            "result_count": 2,
        }
    ]
    ground_truth_ids = {"mem_abc", "mem_def"}
    metrics = _compute_pre_rerank_metrics(
        query_results=query_results,
        ground_truth_ids=ground_truth_ids,
        distance_threshold=1.1,
    )
    # Only mem_abc passes the threshold — recall is 0.5, precision is 1.0
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["total_within_threshold"] == 1
