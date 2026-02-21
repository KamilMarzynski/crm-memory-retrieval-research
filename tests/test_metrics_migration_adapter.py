import pytest
from retrieval_metrics.compute import compute_threshold_metrics, compute_top_n_metrics

from memory_retrieval.experiments.metrics import (
    compute_metrics,
    compute_metrics_at_threshold,
    compute_metrics_at_top_n,
)
from memory_retrieval.experiments.metrics_adapter import (
    macro_average_from_metric_dicts as macro_average,
    restriction_evaluation_to_dict,
    threshold_sweep_from_experiments,
    top_n_sweep_from_experiments,
)


def _sample_ranked_results() -> list[dict[str, float | str]]:
    return [
        {"id": "a", "distance": 0.10, "rerank_score": 0.95},
        {"id": "x", "distance": 0.20, "rerank_score": 0.70},
        {"id": "b", "distance": 0.35, "rerank_score": 0.30},
    ]


def test_compute_metrics_top_n_matches_package_evaluation() -> None:
    ranked_results = _sample_ranked_results()
    ground_truth_ids = {"a", "b"}

    legacy_result = compute_metrics_at_top_n(ranked_results, ground_truth_ids, top_n=2)
    package_result = restriction_evaluation_to_dict(
        compute_top_n_metrics(ranked_results, ground_truth_ids, top_n=2)
    )

    assert legacy_result["precision"] == package_result["precision"]
    assert legacy_result["recall"] == package_result["recall"]
    assert legacy_result["f1"] == package_result["f1"]
    assert legacy_result["mrr"] == package_result["mrr"]
    assert legacy_result["retrieved_ids"] == package_result["retrieved_ids"]


def test_compute_metrics_threshold_matches_package_evaluation() -> None:
    ranked_results = _sample_ranked_results()
    ground_truth_ids = {"a", "b"}

    legacy_result = compute_metrics_at_threshold(
        ranked_results,
        ground_truth_ids,
        threshold=0.65,
        score_field="rerank_score",
        higher_is_better=True,
    )
    package_result = restriction_evaluation_to_dict(
        compute_threshold_metrics(
            ranked_results,
            ground_truth_ids,
            threshold=0.65,
            score_key="rerank_score",
            higher_is_better=True,
        ),
        include_accepted_count=True,
    )

    assert legacy_result["precision"] == package_result["precision"]
    assert legacy_result["recall"] == package_result["recall"]
    assert legacy_result["f1"] == package_result["f1"]
    assert legacy_result["mrr"] == package_result["mrr"]
    assert legacy_result["retrieved_ids"] == package_result["retrieved_ids"]
    assert legacy_result["accepted_count"] == package_result["accepted_count"]


def test_compute_metrics_rounding_compatibility() -> None:
    result = compute_metrics(retrieved_ids={"a", "x"}, ground_truth_ids={"a", "b"})
    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_macro_average_keeps_legacy_metric_shape() -> None:
    result = macro_average(
        [
            {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "mrr": 1.0},
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0},
        ]
    )

    assert set(result.keys()) == {"precision", "recall", "f1", "mrr"}
    assert result["precision"] == 0.5
    assert result["recall"] == 0.25
    assert result["f1"] == pytest.approx(0.33335)
    assert result["mrr"] == 0.5


def test_adapter_sweeps_support_custom_id_field() -> None:
    experiments = [
        {
            "ground_truth_ids": {"m1"},
            "ranked_results": [
                {"memory_id": "m1", "distance": 0.1},
                {"memory_id": "m2", "distance": 0.3},
            ],
        }
    ]

    top_n_rows = top_n_sweep_from_experiments(experiments, n_values=[1], id_field="memory_id")
    threshold_rows = threshold_sweep_from_experiments(
        experiments,
        thresholds=[0.2],
        score_field="distance",
        higher_is_better=False,
        id_field="memory_id",
    )

    assert top_n_rows[0]["precision"] == 1.0
    assert top_n_rows[0]["recall"] == 1.0
    assert threshold_rows[0]["precision"] == 1.0
    assert threshold_rows[0]["recall"] == 1.0
