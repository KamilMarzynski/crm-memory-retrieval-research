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
      runner.py                  # Unified run_experiment/run_all via ExperimentConfig
      metrics.py                 # compute_metrics, analyze_query_performance
      query_generation.py        # parse_queries_robust + constants
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
  phase2/runs/                   # Phase 2: Reranking runs (isolated)

notebooks/
  phase1/phase1.ipynb                    # Full pipeline: extract → DB → test cases → experiments
  phase1/phase1_threshold_analysis.ipynb # Distance threshold analysis
  phase2/phase2.ipynb                    # Reranking experiments
```

## Architecture

### Key Design Patterns

**SearchBackend Protocol** (`search/base.py`): All search backends implement a common protocol with `create_database`, `insert_memories`, `search`, and `rebuild_database` methods. Results are returned as `SearchResult` dataclasses with normalized fields.

**ExperimentConfig** (`experiments/runner.py`): Unified experiment runner controlled by a single config dataclass. When `config.reranker is not None`, the runner adds pool+dedup+rerank steps and computes pre/post metrics. Otherwise it computes standard metrics.

**ExtractionConfig** (`memories/extractor.py`): Controls memory extraction format. `SituationFormat.SINGLE` produces single situation descriptions (for vector search), `SituationFormat.VARIANTS` produces 3 semicolon-separated variants (for FTS5).

### Notebook Usage

```python
from memory_retrieval.search.vector import VectorBackend
from memory_retrieval.search.reranker import Reranker
from memory_retrieval.experiments.runner import run_all_experiments, ExperimentConfig
from memory_retrieval.infra.runs import get_latest_run

config = ExperimentConfig(
    search_backend=VectorBackend(),
    reranker=Reranker(),          # None for standard (no reranking) path
    rerank_top_n=4,
    prompts_dir="data/prompts/phase2",
)
run_dir = get_latest_run("phase1")
results = run_all_experiments(
    str(run_dir / "test_cases"),
    str(run_dir / "memories" / "memories.db"),
    str(run_dir / "results"),
    config,
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

5. **Retrieval Experiment** (`experiments/runner.py`):
   - Loads test case with pre-computed ground truth
   - Generates search queries via LLM from PR context and diff
   - Searches using vector similarity, calculates recall/precision/F1

6. **Reranking Experiment** (`experiments/runner.py` with `reranker` config):
   - Reuses Phase 1 database and test cases (no separate memory extraction)
   - Generates queries, runs vector search, pools and deduplicates results
   - Reranks candidates using cross-encoder (bge-reranker-v2-m3)
   - Takes top-N after reranking (default: 4)
   - Computes metrics before and after reranking for comparison

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

## Environment

- Python 3.13 (uv package manager)
- Requires `OPENROUTER_API_KEY` environment variable
- Phase 1 requires Ollama with `mxbai-embed-large` model for embeddings
- Phase 2 requires `sentence-transformers` package (for bge-reranker-v2-m3 cross-encoder)
- Data files in `data/` are gitignored (contains sensitive real-world code reviews)
