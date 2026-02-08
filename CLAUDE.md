# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for experimenting with memory retrieval techniques for the [code-review-mentat](https://github.com/KamilMarzynski/code-review-mentat) project. Processes real-world code review data using AI to extract reusable engineering knowledge as structured memories.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Export code review data to CSV (for external tools)
uv run python scripts/export_review_data_to_csv.py

# === Phase 0: Keyword Search ===
# Extract memories from raw code review data (requires OPENROUTER_API_KEY)
uv run python scripts/phase0/build_memories.py data/review_data/<file>.json

# Build/rebuild SQLite FTS5 search database
uv run python scripts/phase0/db.py --rebuild

# Build test cases from raw data and extracted memories
uv run python scripts/phase0/test_cases.py

# Run retrieval experiment on all test cases
uv run python scripts/phase0/experiment.py --all

# === Phase 1: Vector Search (with Run Isolation) ===
# Each pipeline run creates an isolated directory with all outputs

# Extract memories (creates new run)
uv run python scripts/phase1/build_memories.py --all data/review_data

# Build vector database (uses latest run)
uv run python scripts/phase1/db.py --rebuild

# Generate test cases (uses latest run)
uv run python scripts/phase1/test_cases.py

# Run experiments (uses latest run)
uv run python scripts/phase1/experiment.py --all

# Use specific run (for re-running experiments)
uv run python scripts/phase1/experiment.py --all --run-id run_20260208_143022

# Test memory search
uv run python scripts/fetch_memories.py "your search query"
```

## Data Structure

```
data/
├── review_data/          # Input: Raw code review JSON files
├── phase0/               # Phase 0: Keyword search experiments
│   ├── memories/         # Extracted memories (JSONL) + SQLite DB
│   │   ├── memories_*.jsonl  # Accepted memories
│   │   ├── rejected_*.jsonl  # Rejected memories
│   │   └── memories.db       # FTS5 search database
│   ├── test_cases/       # Self-contained test cases for experiments
│   │   └── *.json        # One test case per PR (filtered diff + ground truth IDs)
│   └── results/          # Experiment results (JSON)
└── phase1/               # Phase 1: Vector search experiments
    └── runs/             # Run isolation: each pipeline execution is isolated
        ├── run_20260208_143022/  # Timestamp-based run ID
        │   ├── run.json          # Run metadata and pipeline status
        │   ├── memories/         # Extracted memories for this run
        │   │   ├── memories_*.jsonl
        │   │   ├── rejected_*.jsonl
        │   │   └── memories.db   # Vector search database (sqlite-vec)
        │   ├── test_cases/       # Test cases for this run
        │   │   └── *.json
        │   └── results/          # Experiment results for this run
        │       └── results_*.json
        └── run_20260209_101530/  # Another run (outputs isolated)
            └── ...
```

## Scripts

```
scripts/
├── common/                  # Shared utilities across all phases
│   ├── io.py               # load_json, save_json, ensure_dir
│   ├── load_memories.py    # Memory loading from JSONL, shared field constants
│   ├── openrouter.py       # OpenRouter LLM API client
│   ├── runs.py             # Run isolation system for experiment tracking
│   └── test_cases.py       # Test case generation utilities
│
├── phase0/                  # Phase 0: Keyword search experiments
│   ├── build_memories.py   # Extract memories via LLM (CLI: <file> | --all)
│   ├── db.py               # SQLite FTS5 database (CLI: --rebuild)
│   ├── load_memories.py    # Phase 0 field constants, re-exports common
│   ├── test_cases.py       # Test case generation (CLI)
│   └── experiment.py       # Experiment runner (CLI: <file> | --all)
│
├── phase1/                  # Phase 1: Vector search experiments
│   ├── build_memories.py   # Extract memories (CLI: --all, --run-id)
│   ├── db.py               # SQLite + sqlite-vec database (CLI: --rebuild, --run-id)
│   ├── load_memories.py    # Phase 1 field constants
│   ├── test_cases.py       # Test case generation (CLI: --run-id)
│   └── experiment.py       # Experiment runner (CLI: --all, --run-id)
│
├── fetch_memories.py        # CLI: Test memory search
└── export_review_data_to_csv.py  # CLI: Export to CSV
```

## Architecture

### Run Isolation System (Phase 1)

Phase 1 uses a run isolation system to prevent output mixing between pipeline executions:

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

2. **Memory Extraction** (`scripts/phase1/build_memories.py`):
   - Two-stage AI processing via OpenRouter API
   - Stage 1: Extract concrete situation description (25-60 words)
   - Stage 2: Extract actionable lesson (imperative, max 160 chars)
   - Quality validation rejects generic or malformed outputs

3. **Search Database** (`scripts/phase1/db.py`):
   - Loads JSONL memories into SQLite with sqlite-vec extension
   - Generates embeddings via Ollama (mxbai-embed-large)
   - Supports cosine distance ranking

4. **Test Case Generation** (`scripts/phase1/test_cases.py`):
   - Processes raw PR data and extracted memories
   - Filters diffs (removes lock files, generated code, etc.)
   - Computes ground truth memory IDs by matching comment IDs
   - Creates self-contained test case files (skips PRs with no memories)

5. **Retrieval Experiment** (`scripts/phase1/experiment.py`):
   - Loads test case with pre-computed ground truth
   - Generates search queries via LLM from PR context and diff
   - Searches using vector similarity, calculates recall/precision/F1

### Memory Schema (Phase 1)

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
- Data files in `data/` are gitignored (contains sensitive real-world code reviews)
