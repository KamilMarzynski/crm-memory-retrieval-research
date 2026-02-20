# Retrieval Metrics Migration Notes

## Overview
This repository now consumes the standalone `retrieval-metrics` package via a local editable path dependency:

- External package path: `/Users/mayk/Projects/private/retrieval-metrics`
- Dependency wiring: `/Users/mayk/.codex/worktrees/bec6/crm-memory-retrieval-research/pyproject.toml`
- `uv` source override: `[tool.uv.sources] retrieval-metrics = { path = "...", editable = true }`

Core metric math, aggregation, sweeps, diagnostics, comparison stats, and fingerprinting are now delegated to the package.

## Adapter-First Compatibility
Project-specific schemas are preserved through:

- `/Users/mayk/.codex/worktrees/bec6/crm-memory-retrieval-research/src/memory_retrieval/experiments/metrics_adapter.py`

The adapter maps package dataclasses and generic outputs into existing legacy dict/JSON shapes used by run summaries and notebooks.

## Old to New Mapping

`memory_retrieval.experiments.metrics` compatibility facade:

- `compute_metrics(...)` -> `retrieval_metrics.compute.compute_set_metrics(...)` + rounding
- `compute_metrics_at_top_n(...)` -> `retrieval_metrics.compute.compute_top_n_metrics(...)`
- `compute_metrics_at_threshold(...)` -> `retrieval_metrics.compute.compute_threshold_metrics(...)`
- `sweep_top_n(...)` -> adapter `top_n_sweep_from_experiments(...)` (package sweep engine)
- `sweep_threshold(...)` -> adapter `threshold_sweep_from_experiments(...)` (package sweep engine)
- `find_optimal_threshold(...)` -> `retrieval_metrics.sweeps.find_optimal_entry(...)`
- `analyze_query_performance(...)` -> `retrieval_metrics.diagnostics.analyze_query_diagnostics(...)` + legacy key aliases

`memory_retrieval.experiments.comparison` delegations:

- `build_config_fingerprint(...)` -> `retrieval_metrics.fingerprint.build_fingerprint(...)`
- `fingerprint_diff(...)` -> `retrieval_metrics.fingerprint.fingerprint_diff(...)`
- `compute_variance_report(...)` -> `retrieval_metrics.compare.compute_variance_report(...)`
- `compare_configs(...)` -> `retrieval_metrics.compare.compare_run_groups(...)`

The `comparison.py` public API and return schema remain stable for existing notebook/batch workflows.

## Notebook Migration Notes

Target notebooks were updated to reduce duplicated metric extraction logic and rely on package-backed APIs:

- `notebooks/comparison/cross_run_comparison.ipynb`
- `notebooks/comparison/subrun_comparison.ipynb`
- `notebooks/comparison/rerank_strategy_comparison.ipynb`
- `notebooks/phase1/phase1.ipynb`
- `notebooks/phase1/phase1_threshold_analysis.ipynb`
- `notebooks/phase2/phase1_reranking_comparison.ipynb`
- `notebooks/phase2/phase2.ipynb`

Changes include:

- shared nested extraction via `metrics_adapter.extract_metric_from_nested`
- package compute functions in notebook-local wrappers where legacy dict shape is required
- adapter sweep functions for legacy experiment payloads
- adapter-backed nested F1 extraction in all comparison notebooks (cross-run, subrun, rerank-strategy)

## Validation

- External package checks:
  - `uv run --with ruff ruff check .`
  - `uv run --with pytest pytest`
- Main repo checks:
  - `uv run --with ruff ruff check src run_batch.py`
  - `uv run --with pytest pytest -q`

Added migration-focused tests:

- `/Users/mayk/.codex/worktrees/bec6/crm-memory-retrieval-research/tests/test_metrics_migration_adapter.py`
- `/Users/mayk/.codex/worktrees/bec6/crm-memory-retrieval-research/tests/test_comparison_migration.py`
