"""Template script for unattended batch experiment runs.

Edit the config sections below, then run:

    uv run python run_batch.py

For background execution (fire and forget):

    nohup uv run python run_batch.py > /dev/null 2>&1 &

To watch live output while running in the background:

    uv run python run_batch.py 2>&1 | tee -a data/phase2/batches/current_batch.log &

To check progress on a running batch:

    tail -f data/phase2/batches/batch_<timestamp>/batch.log

After completion, load results into cross_run_comparison.ipynb:

    summaries = load_run_summaries("phase2", run_ids=[o.run_id for o in outcome.outcomes if o.run_id])

---

THREE USAGE PATTERNS
====================

1. Config A vs Config B (A/B test)
   Run two separate batches of 10 runs each, then compare via compare_configs().

2. Variance measurement
   Run one batch of 20 runs with identical config, analyze via compute_variance_report().

3. Deterministic subrun swap
   Run 10 parent runs (config A), then one subrun per parent with new embedder/reranker.
   Compare as paired 10 vs 10 using run IDs from both batches.
"""

# ============================================================
# OPTION A: Full pipeline batch (extract → DB → test cases → queries → experiments)
# ============================================================

from memory_retrieval.experiments.batch_runner import (
    BatchFullPipelineConfig,
    BatchSubrunConfig,
    run_subrun_batch,
)
from memory_retrieval.experiments.query_generation import QueryGenerationConfig
from memory_retrieval.experiments.runner import ExperimentConfig
from memory_retrieval.infra.runs import PHASE2
from memory_retrieval.memories.extractor import ExtractionConfig, SituationFormat
from memory_retrieval.search.reranker import Reranker
from memory_retrieval.search.vector import SentenceTransformerVectorBackend, VectorBackend

RERANK_TEXT_STRATEGIES = {
    "situation_only": lambda c: c["situation"],
    "situation_and_lesson": lambda c: f"situation: {c['situation']}; lesson: {c.get('lesson', '')}",
}

# --- Full pipeline config ---
full_pipeline_config = BatchFullPipelineConfig(
    phase=PHASE2,
    num_runs=20,
    extraction_config=ExtractionConfig(
        situation_format=SituationFormat.SINGLE,
        prompts_dir="data/prompts/phase2",
        prompt_version="3.0.0",
        model="anthropic/claude-haiku-4.5",
    ),
    query_config=QueryGenerationConfig(
        prompts_dir="data/prompts/phase2",
        prompt_version="3.0.0",
        model="anthropic/claude-sonnet-4.5",
    ),
    experiment_config=ExperimentConfig(
        search_backend=VectorBackend(),
        reranker=Reranker(),
        rerank_text_strategies=RERANK_TEXT_STRATEGIES,
    ),
    raw_data_dir="data/review_data",
    description="Config - bge-reranker-v2-m3 + mxbai-embed-large + prompt v3 situation haiku 4.5 + prompt v3 query sonnet 4.5 + situation_only and situation_and_lesson",
)

# ============================================================
# OPTION B: Subrun batch (re-run experiments on existing parent runs)
# Useful for testing a new reranker or search config without re-extracting memories.
# ============================================================

# Load run IDs from a previous full pipeline batch (copy from batch_manifest.json)
parent_run_ids = [
    # batch_20260218_171450 (10 runs)
    "run_20260218_171450",
    "run_20260218_172142",
    "run_20260218_172742",
    "run_20260218_173357",
    "run_20260218_174001",
    "run_20260218_174543",
    "run_20260218_175141",
    "run_20260218_175735",
    "run_20260218_180331",
    "run_20260218_180931",
    # batch_20260218_202350 (20 runs)
    "run_20260218_202350",
    "run_20260218_202928",
    "run_20260218_203520",
    "run_20260218_204111",
    "run_20260218_204655",
    "run_20260218_205227",
    "run_20260218_205757",
    "run_20260218_210347",
    "run_20260218_210936",
    "run_20260218_211505",
    "run_20260218_212111",
    "run_20260218_212700",
    "run_20260218_213251",
    "run_20260218_213822",
    "run_20260218_214416",
    "run_20260218_214957",
    "run_20260218_215532",
    "run_20260218_220111",
    "run_20260218_220653",
    "run_20260218_221228",
]

subrun_config = BatchSubrunConfig(
    phase=PHASE2,
    parent_run_ids=parent_run_ids,
    experiment_config=ExperimentConfig(
        search_backend=SentenceTransformerVectorBackend(
            model_name="perplexity-ai/pplx-embed-context-v1-0.6B"
        ),
        reranker=Reranker(),
        rerank_text_strategies=RERANK_TEXT_STRATEGIES,
    ),
    rebuild_db=True,  # True when switching embedding model
    description="Subrun batch — pplx-embed-v1-0.6B",
)

# ============================================================
# RUN — uncomment whichever mode you want
# ============================================================

if __name__ == "__main__":
    # --- Full pipeline batch ---
    # outcome = run_full_pipeline_batch(full_pipeline_config)

    # --- Subrun batch ---
    outcome = run_subrun_batch(subrun_config)

    # Print summary
    print(f"\nBatch complete: {outcome.batch_dir}")
    print(f"Completed: {outcome.num_completed} / {len(outcome.outcomes)}")
    print(f"Failed: {outcome.num_failed}")
    print()
    print("Completed run IDs:")
    for single_run_outcome in outcome.outcomes:
        if single_run_outcome.run_id and single_run_outcome.status == "completed":
            f1_display = (
                f"{single_run_outcome.optimal_f1:.4f}"
                if single_run_outcome.optimal_f1 is not None
                else "n/a"
            )
            print(
                f"  [{single_run_outcome.run_index}] {single_run_outcome.run_id}  F1={f1_display}"
            )
    print()
    print("To compare results, load into cross_run_comparison.ipynb:")
    print(
        "  run_ids = "
        + repr(
            [
                single_run_outcome.run_id
                for single_run_outcome in outcome.outcomes
                if single_run_outcome.run_id and single_run_outcome.status == "completed"
            ]
        )
    )
