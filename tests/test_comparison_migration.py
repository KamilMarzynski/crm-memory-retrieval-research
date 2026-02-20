import json
from pathlib import Path

from memory_retrieval.experiments.comparison import (
    compare_configs,
    compute_variance_report,
    generate_run_summary,
)


def _summary(run_id: str, case_values: dict[str, float]) -> dict:
    mean_value = sum(case_values.values()) / len(case_values)
    return {
        "run_id": run_id,
        "macro_averaged": {
            "pre_rerank": {
                "at_optimal_distance_threshold": {"f1": round(mean_value, 4)},
            }
        },
        "per_test_case": {
            case_id: {
                "pre_rerank": {
                    "distance_threshold": {
                        "at_optimal": {"f1": value},
                    }
                }
            }
            for case_id, value in case_values.items()
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_compute_variance_report_preserves_legacy_shape() -> None:
    summaries = [
        _summary("run_a", {"tc1": 0.40, "tc2": 0.50, "tc3": 0.60}),
        _summary("run_b", {"tc1": 0.45, "tc2": 0.55, "tc3": 0.65}),
    ]

    report = compute_variance_report(summaries, metric_path="pre_rerank")

    assert report["num_runs"] == 2
    assert report["metric_path"] == "pre_rerank"
    assert "run_level" in report
    assert "individual_f1_values" in report["run_level"]
    assert set(report["per_test_case"].keys()) == {"tc1", "tc2", "tc3"}


def test_compare_configs_preserves_legacy_shape() -> None:
    summaries_a = [
        _summary("a1", {"tc1": 0.40, "tc2": 0.50, "tc3": 0.60}),
        _summary("a2", {"tc1": 0.45, "tc2": 0.52, "tc3": 0.62}),
    ]
    summaries_b = [
        _summary("b1", {"tc1": 0.48, "tc2": 0.56, "tc3": 0.66}),
        _summary("b2", {"tc1": 0.50, "tc2": 0.58, "tc3": 0.68}),
    ]

    comparison = compare_configs(
        summaries_a,
        summaries_b,
        metric_path="pre_rerank",
        bootstrap_iterations=500,
    )

    assert comparison["num_common_test_cases"] == 3
    assert comparison["metric_path"] == "pre_rerank"
    assert comparison["direction"] in {"a_better", "b_better", "no_change"}
    assert "bootstrap_ci" in comparison
    assert "wilcoxon" in comparison
    assert "significance" in comparison
    assert len(comparison["paired_differences"]) == 3
    assert {"test_case_id", "f1_a", "f1_b", "diff"} <= set(
        comparison["paired_differences"][0].keys()
    )


def test_generate_run_summary_smoke_non_rerank(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_non_rerank"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)

    _write_json(run_dir / "run.json", {"run_id": "run_non_rerank"})
    _write_json(
        results_dir / "tc1.json",
        {
            "test_case_id": "tc1",
            "ground_truth": {"memory_ids": ["m1"]},
            "queries": [
                {
                    "query": "q1",
                    "results": [
                        {"id": "m1", "distance": 0.1, "is_ground_truth": True},
                        {"id": "m2", "distance": 0.2, "is_ground_truth": False},
                    ],
                }
            ],
            "metrics": {"precision": 0.5, "recall": 1.0, "f1": 0.6667},
        },
    )

    summary = generate_run_summary(run_dir, threshold_step=0.5, top_n_max=3)

    assert summary["run_id"] == "run_non_rerank"
    assert "macro_averaged" in summary
    assert "metrics" in summary["macro_averaged"]
    assert (run_dir / "run_summary.json").exists()


def test_generate_run_summary_smoke_rerank(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_rerank"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)

    _write_json(run_dir / "run.json", {"run_id": "run_rerank"})
    _write_json(
        results_dir / "tc1.json",
        {
            "test_case_id": "tc1",
            "ground_truth": {"memory_ids": ["m1"]},
            "queries": [
                {
                    "query": "q1",
                    "results": [
                        {"id": "m1", "distance": 0.1, "is_ground_truth": True},
                        {"id": "m2", "distance": 0.2, "is_ground_truth": False},
                    ],
                }
            ],
            "pre_rerank_metrics": {"precision": 0.5, "recall": 1.0, "f1": 0.6667},
            "reranked_results": [
                {"id": "m1", "rerank_score": 0.9, "distance": 0.1, "is_ground_truth": True},
                {"id": "m2", "rerank_score": 0.1, "distance": 0.2, "is_ground_truth": False},
            ],
        },
    )

    summary = generate_run_summary(
        run_dir,
        strategies=["default"],
        threshold_step=0.5,
        top_n_max=3,
    )

    assert summary["run_id"] == "run_rerank"
    assert "baseline" in summary
    assert "rerank_strategies" in summary
    assert "macro_averaged" in summary
    assert "pre_rerank" in summary["macro_averaged"]
    assert "post_rerank" in summary["macro_averaged"]
    assert "default" in summary["macro_averaged"]["post_rerank"]


def test_compare_configs_requires_enough_common_cases() -> None:
    comparison = compare_configs(
        [_summary("a", {"tc1": 0.4, "tc2": 0.5})],
        [_summary("b", {"tc1": 0.6, "tc2": 0.7})],
        metric_path="pre_rerank",
        bootstrap_iterations=100,
    )
    assert "error" in comparison
    assert comparison["common_test_cases"] == ["tc1", "tc2"]
