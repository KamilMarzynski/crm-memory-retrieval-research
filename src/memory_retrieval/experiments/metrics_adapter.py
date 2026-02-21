"""Backward-compatibility adapter for the retrieval_metrics migration (Feb 2026).

DEPRECATED: This module maps retrieval_metrics dataclasses to legacy dict shapes
expected by existing notebook code. Once notebooks are migrated to import from
retrieval_metrics directly, this module can be retired.

Retirement path:
1. Update all notebooks to import MetricPoint, SweepResult from retrieval_metrics directly
2. Remove calls to metric_point_to_dict() and extract_metric_from_nested()
3. Delete this module
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from retrieval_metrics.aggregate import aggregate_cases
from retrieval_metrics.restrictions import ThresholdRestriction, TopNRestriction
from retrieval_metrics.sweeps import run_sweep
from retrieval_metrics.types import (
    AveragingConfig,
    CaseMetrics,
    MetricPoint,
    RestrictionEvaluation,
    SweepCase,
)

RunMetricExtractor = Callable[[Mapping[str, Any], str], float | None]
PerCaseMetricExtractor = Callable[[Mapping[str, Any], str], dict[str, float]]


def metric_point_to_dict(
    metric_point: MetricPoint, round_digits: int | None = None
) -> dict[str, float]:
    """Convert a MetricPoint into the project's dict shape."""
    values = {
        "precision": float(metric_point.precision),
        "recall": float(metric_point.recall),
        "f1": float(metric_point.f1),
        "mrr": float(metric_point.mrr),
    }
    if round_digits is not None:
        return {key: round(value, round_digits) for key, value in values.items()}
    return values


def restriction_evaluation_to_dict(
    evaluation: RestrictionEvaluation,
    include_accepted_count: bool = False,
) -> dict[str, Any]:
    """Convert a RestrictionEvaluation into legacy result fields."""
    payload: dict[str, Any] = {
        **metric_point_to_dict(evaluation.metrics),
        "retrieved_ids": set(evaluation.retrieved_ids),
    }
    if include_accepted_count:
        payload["accepted_count"] = evaluation.accepted_count
    return payload


def macro_average_from_metric_dicts(
    per_case_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Macro-average precision/recall/F1/MRR from metric dicts."""
    if not per_case_metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}

    case_metrics = [
        CaseMetrics(
            case_id=f"case_{index}",
            metrics=MetricPoint(
                precision=float(metrics.get("precision", 0.0)),
                recall=float(metrics.get("recall", 0.0)),
                f1=float(metrics.get("f1", 0.0)),
                mrr=float(metrics.get("mrr", 0.0)),
            ),
        )
        for index, metrics in enumerate(per_case_metrics)
    ]
    macro = aggregate_cases(case_metrics, mode="macro")
    return metric_point_to_dict(macro)


def top_n_sweep_from_experiments(
    experiments: Sequence[Mapping[str, Any]],
    n_values: Sequence[int],
    id_field: str = "id",
) -> list[dict[str, Any]]:
    """Run a macro-averaged top-N sweep over legacy experiment inputs."""
    sweep_cases = _legacy_experiments_to_sweep_cases(experiments, id_field=id_field)
    restrictions = [TopNRestriction(top_n=top_n) for top_n in n_values]
    sweep_rows = run_sweep(
        cases=sweep_cases,
        restriction_or_sweep_spec=restrictions,
        averaging_config=AveragingConfig(mode="macro"),
    )
    return [
        {
            "top_n": int(row["top_n"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "mrr": float(row["mrr"]),
        }
        for row in sweep_rows
    ]


def threshold_sweep_from_experiments(
    experiments: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    score_field: str,
    higher_is_better: bool,
    id_field: str = "id",
) -> list[dict[str, Any]]:
    """Run a macro-averaged threshold sweep over legacy experiment inputs."""
    sweep_cases = _legacy_experiments_to_sweep_cases(experiments, id_field=id_field)
    restrictions = [
        ThresholdRestriction(
            threshold=threshold,
            score_key=score_field,
            higher_is_better=higher_is_better,
        )
        for threshold in thresholds
    ]
    sweep_rows = run_sweep(
        cases=sweep_cases,
        restriction_or_sweep_spec=restrictions,
        averaging_config=AveragingConfig(mode="macro"),
    )
    return [
        {
            "threshold": float(row["threshold"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "mrr": float(row["mrr"]),
        }
        for row in sweep_rows
    ]


def build_macro_run_metric_extractor(
    metric_path: str = "post_rerank",
    strategy: str | None = None,
) -> RunMetricExtractor:
    """Create extractor for run-level metric value from legacy macro_averaged payloads."""

    def extractor(summary: Mapping[str, Any], metric_key: str) -> float | None:
        return extract_metric_from_nested(
            summary.get("macro_averaged", {}), metric_path, strategy, metric_key
        )

    return extractor


def build_per_case_metric_extractor(
    metric_path: str = "post_rerank",
    strategy: str | None = None,
) -> PerCaseMetricExtractor:
    """Create extractor for per-test-case metric maps from legacy run summaries."""

    def extractor(summary: Mapping[str, Any], metric_key: str) -> dict[str, float]:
        per_case = summary.get("per_test_case", {})
        extracted: dict[str, float] = {}
        for test_case_id, test_case_metrics in per_case.items():
            value = extract_metric_from_nested(test_case_metrics, metric_path, strategy, metric_key)
            if value is not None:
                extracted[str(test_case_id)] = float(value)
        return extracted

    return extractor


def extract_metric_from_nested(
    payload: Mapping[str, Any],
    metric_path: str,
    strategy: str | None,
    metric_key: str = "f1",
) -> float | None:
    """Extract metric value from legacy nested summary structures."""
    if metric_path == "post_rerank" and strategy:
        post_rerank = payload.get("post_rerank", {}).get(strategy, {})
        macro_value = post_rerank.get("at_optimal_threshold", {}).get(metric_key)
        if macro_value is not None:
            return float(macro_value)
        per_case_value = (
            post_rerank.get("rerank_threshold", {}).get("at_optimal", {}).get(metric_key)
        )
        if per_case_value is not None:
            return float(per_case_value)
        return None

    if metric_path == "pre_rerank":
        pre_rerank = payload.get("pre_rerank", {})
        macro_value = pre_rerank.get("at_optimal_distance_threshold", {}).get(metric_key)
        if macro_value is not None:
            return float(macro_value)
        per_case_value = (
            pre_rerank.get("distance_threshold", {}).get("at_optimal", {}).get(metric_key)
        )
        if per_case_value is not None:
            return float(per_case_value)
        return None

    standard_value = payload.get("metrics", {}).get(metric_key)
    if standard_value is not None:
        return float(standard_value)

    return None


def _legacy_experiments_to_sweep_cases(
    experiments: Sequence[Mapping[str, Any]],
    id_field: str,
) -> list[SweepCase]:
    sweep_cases: list[SweepCase] = []
    for index, experiment in enumerate(experiments):
        ranked_results = _normalize_ranked_results_for_id_field(
            experiment.get("ranked_results", []),
            id_field=id_field,
        )
        relevant_ids = set(experiment.get("ground_truth_ids", set()))
        sweep_cases.append(
            SweepCase(
                case_id=f"case_{index}",
                ranked_results=ranked_results,
                relevant_ids=relevant_ids,
            )
        )
    return sweep_cases


def _normalize_ranked_results_for_id_field(
    ranked_results: Sequence[Mapping[str, Any]],
    id_field: str,
) -> list[Mapping[str, Any]]:
    if id_field == "id":
        return list(ranked_results)

    normalized: list[Mapping[str, Any]] = []
    for result in ranked_results:
        if "id" in result:
            normalized.append(result)
            continue
        if id_field in result:
            normalized.append({"id": result[id_field], **result})
            continue
        normalized.append(result)
    return normalized
