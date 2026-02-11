# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for experimenting with memory retrieval techniques for the [code-review-mentat](https://github.com/KamilMarzynski/code-review-mentat) project. Processes real-world code review data using AI to extract reusable engineering knowledge as structured memories.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Export code review data to CSV (for external tools)
uv run python export_review_data_to_csv.py
```

All experiment workflows are run from Jupyter notebooks in `notebooks/`. See `notebooks/phase1/phase1.ipynb`, `notebooks/phase1/phase1_threshold_analysis.ipynb`, and `notebooks/phase2/phase2.ipynb`.

## Package Structure

```
src/
  memory_retrieval/
    memories/                    # Domain: Memory extraction & schema
      schema.py                  # All FIELD_* constants
      loader.py                  # load_memories from JSONL
      helpers.py                 # stable_id, lang_from_file, file_pattern, etc.
      validators.py              # situation/lesson validators with version registry
      extractor.py               # Unified extraction (SituationFormat.SINGLE or VARIANTS)

    search/                      # Domain: Retrieval backends
      base.py                    # SearchResult dataclass + SearchBackend Protocol
      db_utils.py                # Shared: get_db_connection, serialize/deserialize JSON
      fts5.py                    # FTS5Backend class (keyword search)
      vector.py                  # VectorBackend class (embedding search via sqlite-vec)
      reranker.py                # Reranker class (cross-encoder reranking)

    experiments/                 # Domain: Running & evaluating experiments
      runner.py                  # run_experiment/run_all via ExperimentConfig (search + metrics only)
      metrics.py                 # compute_metrics, analyze_query_performance
      query_generation.py        # QueryGenerationConfig, generate_all_queries (LLM query generation)
      test_cases.py              # build_test_cases, filter_diff

    infra/                       # Infrastructure utilities
      io.py                      # load_json, save_json, ensure_dir
      llm.py                     # call_openrouter (OpenRouter API client)
      prompts.py                 # load_prompt, Prompt dataclass (versioned prompts)
      runs.py                    # Run isolation system

data/
  prompts/                       # Versioned prompt templates per phase
    phase0/                      # FTS-specific prompts
    phase1/                      # Vector search prompts
    phase2/                      # Reranking prompts
  review_data/                   # Input: Raw code review JSON files
  phase0/                        # Phase 0: Keyword search data
  phase1/runs/                   # Phase 1: Vector search runs (isolated)
    <run_id>/
      memories/                  # Extracted memory JSONL files + memories.db
      test_cases/                # Ground truth test case JSON files
      queries/                   # Pre-generated LLM query JSON files
      results/                   # Experiment result JSON files
      run.json                   # Run metadata and pipeline status
  phase2/runs/                   # Phase 2: Reranking runs (same structure)

notebooks/
  phase1/phase1.ipynb                              # Full pipeline: extract → DB → test cases → queries → experiments
  phase1/phase1_threshold_analysis.ipynb           # Distance threshold analysis
  phase2/phase2.ipynb                              # Full pipeline with reranking: extract → DB → test cases → queries → experiments
  phase2/phase1_reranking_comparison.ipynb         # Reranking on phase1 data (comparison tool)
```

## Architecture

### Key Design Patterns

**SearchBackend Protocol** (`search/base.py`): All search backends implement a common protocol with `create_database`, `insert_memories`, `search`, and `rebuild_database` methods. Results are returned as `SearchResult` dataclasses with normalized fields.

**ExperimentConfig** (`experiments/runner.py`): Experiment runner controlled by a config dataclass with search/reranking parameters only (no LLM config). When `config.reranker is not None`, the runner adds pool+dedup+rerank steps and computes pre/post metrics. Otherwise it computes standard metrics. Experiments consume pre-generated query files.

**QueryGenerationConfig** (`experiments/query_generation.py`): Controls LLM-based query generation (model, prompt, sample memories). Query generation is a separate step that saves queries as JSON files, decoupled from experiment execution.

**ExtractionConfig** (`memories/extractor.py`): Controls memory extraction format. `SituationFormat.SINGLE` produces single situation descriptions (for vector search), `SituationFormat.VARIANTS` produces 3 semicolon-separated variants (for FTS5).

### Notebook Usage

```python
from memory_retrieval.search.vector import VectorBackend
from memory_retrieval.search.reranker import Reranker
from memory_retrieval.experiments.query_generation import generate_all_queries, QueryGenerationConfig
from memory_retrieval.experiments.runner import run_all_experiments, ExperimentConfig
from memory_retrieval.infra.runs import get_latest_run

run_dir = get_latest_run("phase1")
vector_backend = VectorBackend()

# Step 1: Generate queries (costs money — LLM API call)
query_config = QueryGenerationConfig(
    prompts_dir="data/prompts/phase1",
    model="anthropic/claude-sonnet-4.5",
)
generate_all_queries(
    test_cases_dir=str(run_dir / "test_cases"),
    queries_dir=str(run_dir / "queries"),
    config=query_config,
    db_path=str(run_dir / "memories" / "memories.db"),
    search_backend=vector_backend,
)

# Step 2: Run experiments (free — local search + metrics)
config = ExperimentConfig(
    search_backend=vector_backend,
    reranker=Reranker(),          # None for standard (no reranking) path
    rerank_top_n=4,
)
results = run_all_experiments(
    test_cases_dir=str(run_dir / "test_cases"),
    queries_dir=str(run_dir / "queries"),
    db_path=str(run_dir / "memories" / "memories.db"),
    results_dir=str(run_dir / "results"),
    config=config,
)
```

### Run Isolation System (Phase 1 & Phase 2)

Phase 1 and Phase 2 use a run isolation system to prevent output mixing between pipeline executions:

- **create_run(phase)**: Creates a new run directory with timestamp-based ID
- **get_latest_run(phase)**: Gets the most recent run (for chained operations)
- **get_run(phase, run_id)**: Gets a specific run by ID (for re-running experiments)
- **list_runs(phase)**: Lists all runs with metadata
- **update_run_status(run_dir, stage, info)**: Updates run.json with pipeline status

Each run contains a `run.json` with metadata:
```json
{
  "run_id": "run_20260208_143022",
  "created_at": "2026-02-08T14:30:22",
  "phase": "phase1",
  "pipeline_status": {
    "build_memories": {"completed_at": "...", "count": 41},
    "db": {"completed_at": "...", "memory_count": 41},
    "test_cases": {"completed_at": "...", "count": 11},
    "query_generation": {"completed_at": "...", "count": 11, "total_queries": 55},
    "experiment": {"completed_at": "...", "count": 11}
  }
}
```

### Data Pipeline

1. **Input**: Raw code review JSON files in `data/review_data/` containing PR context, metadata, and code review comments with fields like severity, confidence, code snippets, and user notes

2. **Memory Extraction** (`memories/extractor.py`):
   - Two-stage AI processing via OpenRouter API
   - Stage 1: Extract concrete situation description (25-60 words)
   - Stage 2: Extract actionable lesson (imperative, max 160 chars)
   - Quality validation rejects generic or malformed outputs

3. **Search Database** (`search/vector.py`):
   - Loads JSONL memories into SQLite with sqlite-vec extension
   - Generates embeddings via Ollama (mxbai-embed-large)
   - Supports cosine distance ranking

4. **Test Case Generation** (`experiments/test_cases.py`):
   - Processes raw PR data and extracted memories
   - Filters diffs (removes lock files, generated code, etc.)
   - Computes ground truth memory IDs by matching comment IDs
   - Creates self-contained test case files (skips PRs with no memories)

5. **Query Generation** (`experiments/query_generation.py`):
   - Generates search queries via LLM from PR context and diff
   - Saves query files as JSON in `queries/` directory (one per test case)
   - Separate step from experiments — queries are generated once, experiments run many times

6. **Retrieval Experiment** (`experiments/runner.py`):
   - Loads test case and pre-generated queries
   - Searches using vector similarity, calculates recall/precision/F1

7. **Reranking Experiment** (`experiments/runner.py` with `reranker` config):
   - Can run independently (phase2.ipynb) or on Phase 1 data (phase1_reranking_comparison.ipynb)
   - Loads pre-generated queries, runs vector search
   - **Per-query reranking**: each query's results are reranked independently against that query using the cross-encoder (bge-reranker-v2-m3), then pooled and deduplicated by best rerank score
   - Takes top-N after pooling (default: 4)
   - Computes metrics before and after reranking for comparison
   - Stores all pooled reranked candidates in `reranked_results` (sorted by rerank score descending) for downstream sweep analysis

### Memory Schema

```json
{
  "id": "mem_<12-char-hash>",
  "situation_description": "Concrete description of the code situation (25-60 words)",
  "lesson": "Actionable imperative guidance",
  "metadata": { "repo", "file_pattern", "language", "severity", "confidence" },
  "source": { "file", "line", "code_snippet", "comment", "pr_context" }
}
```

## Code Style & Quality Standards

### Variable Naming (CRITICAL)

**NEVER use cryptic abbreviations or single-letter variable names.** This codebase prioritizes readability over brevity. Always use descriptive, self-documenting variable names.

**Forbidden patterns:**
```python
# ❌ NEVER DO THIS
r = get_result()           # What is 'r'? result? rank? response?
qr = query_results         # Ambiguous - query_result singular or plural?
mid = memory["id"]         # Unclear - could be anything with 'mid'
d = calculate_distance()   # Single letter tells nothing
p, r, f1 = metrics         # Unpacking to cryptic names
tc_id = test_case["id"]    # Unnecessary abbreviation
gt_ids = ground_truth      # 'gt' requires mental translation
```

**Required patterns:**
```python
# ✅ ALWAYS DO THIS
result = get_result()
query_result = query_results[0]  # Singular, not 'qr'
memory_id = memory["id"]
distance = calculate_distance()
precision, recall, f1_score = metrics
test_case_id = test_case["id"]
ground_truth_ids = ground_truth
```

**Specific naming conventions for this project:**
- `result` NOT `r` (for search results, experiment results, etc.)
- `query_result` NOT `qr` (for individual query result objects)
- `memory_id` NOT `mid` (for memory identifiers)
- `distance` NOT `d` (for vector distances)
- `threshold` NOT `t` (for distance/score thresholds)
- `precision`, `recall`, `f1_score` NOT `p`, `r`, `f1`
- `reciprocal_rank` NOT `rr` (for MRR calculations)
- `experiment`, `experiment_data`, `experiment_result` NOT `exp`, `ed`, `er`
- `ground_truth_ids` NOT `gt_ids` (spell out "ground_truth")
- `test_case_id` NOT `tc_id` (spell out "test_case")
- `num_*` NOT `n_*` or `n` (e.g., `num_queries`, `num_memories`)
- Collections: `precisions`, `recalls`, `f1_scores` NOT `ps`, `rs`, `f1s`

**Acceptable short names (Python conventions):**
- `i`, `j`, `k` in simple `for i in range(...)` loops ONLY
- `_` for intentionally unused values
- Database cursors can be `cursor` (NOT `cur`)

**Rationale:** Research code is read far more often than written. Code that takes 30 seconds longer to write but saves 5 minutes of comprehension time for every reader (including future you) is a massive win. Cryptic abbreviations create maintenance burden, slow development, and lead to bugs.

### Code Quality Enforcement

All code must be readable and maintainable:
- Descriptive variable names (see above)
- Clear function names that describe what they do
- Type hints where helpful for clarity
- Comments explaining *why*, not *what* (code should be self-documenting)
- Consistent formatting via project standards

## Research Context & Experimentation Guide

### What This Research Is About

The goal is to build a **memory retrieval system** for the code-review-mentat tool. When a developer opens a PR, the system should automatically retrieve relevant past lessons (memories) from a knowledge base to enrich the AI code review. The core research question is: **how do we retrieve the right memories given a PR's diff and context?**

### Production Flow (What We're Optimizing For)

In production, the app will:
1. Receive a PR with diff + Jira/Confluence context
2. Generate multiple search queries from that context via LLM
3. Run vector search for each query against the memories database
4. Rerank each query's results independently using a cross-encoder
5. Pool results across all queries, deduplicate by best rerank score
6. Return top-N memories to include in the code review prompt

Each experiment evaluates this flow against ground truth (known memory-to-PR mappings from real code reviews).

### Experiment Phases

- **Phase 0** (FTS5): Keyword-based search baseline. Historical, not actively iterated.
- **Phase 1** (Vector search): Embedding-based retrieval. The main baseline. Iterates on prompts, models, and distance thresholds.
- **Phase 2** (Reranking): Adds cross-encoder reranking on top of vector search. Two modes:
  - `phase2.ipynb`: Full independent pipeline (own memories, DB, test cases) — for iterating on extraction prompts + reranking together
  - `phase1_reranking_comparison.ipynb`: Reuses Phase 1 data — isolates the reranking impact without changing anything else

### Key Metrics

- **Recall**: fraction of ground truth memories that appear in the retrieved set
- **Precision**: fraction of retrieved memories that are ground truth
- **F1**: harmonic mean of precision and recall (primary optimization target)
- **MRR**: mean reciprocal rank of the first ground truth hit

All metrics are **macro-averaged** (computed per test case, then averaged across test cases).

### What to Vary in Experiments

| Lever | Where | Notes |
|---|---|---|
| Query generation prompt | `data/prompts/<phase>/memory-query/` | Controls what queries the LLM generates from the PR |
| Query generation model | `QueryGenerationConfig.model` | Smarter models generate better queries |
| Memory extraction prompt | `data/prompts/<phase>/situation/`, `lesson/` | Controls memory quality (situation + lesson text) |
| Number of queries | Prompt-driven | More queries = broader recall, more noise |
| Search limit per query | `ExperimentConfig.search_limit` | How many candidates per query (default: 20) |
| Distance threshold | `ExperimentConfig.distance_threshold` | Pre-rerank filter for vector distance |
| Rerank top-N | `ExperimentConfig.rerank_top_n` | How many results to keep after reranking |
| Reranker model | `Reranker(model_name=...)` | Cross-encoder model for reranking |
| Embedding model | Ollama model in `VectorBackend` | Currently `mxbai-embed-large` |

## Environment

- Python 3.13 (uv package manager)
- Requires `OPENROUTER_API_KEY` environment variable
- Phase 1 and Phase 2 require Ollama with `mxbai-embed-large` model for embeddings
- Phase 2 additionally requires `sentence-transformers` package (for bge-reranker-v2-m3 cross-encoder)
- Data files in `data/` are gitignored (contains sensitive real-world code reviews)
