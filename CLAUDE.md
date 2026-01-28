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

# Extract memories from raw code review data (requires OPENROUTER_API_KEY)
uv run python scripts/phase0/build_memories.py data/review_data/<file>.json

# Build/rebuild SQLite FTS5 search database
uv run python scripts/phase0/db.py --rebuild

# Build test cases from raw data and extracted memories
uv run python scripts/phase0/test_cases.py

# Test memory search
uv run python scripts/fetch_memories.py "your search query"

# Run retrieval experiment on single test case (requires OPENROUTER_API_KEY)
uv run python scripts/phase0/experiment.py data/phase0/test_cases/<file>.json

# Run retrieval experiment on all test cases
uv run python scripts/phase0/experiment.py --all
```

## Data Structure

```
data/
├── review_data/          # Input: Raw code review JSON files
└── phase0/               # Phase 0: Keyword search experiments
    ├── memories/         # Extracted memories (JSONL) + SQLite DB
    │   ├── memories_*.jsonl  # Accepted memories
    │   ├── rejected_*.jsonl  # Rejected memories
    │   └── memories.db       # FTS5 search database
    ├── test_cases/       # Self-contained test cases for experiments
    │   └── *.json        # One test case per PR (filtered diff + ground truth IDs)
    └── results/          # Experiment results (JSON)
```

## Scripts

```
scripts/
├── common/                  # Shared utilities across all phases
│   ├── io.py               # load_json, save_json, ensure_dir
│   └── openrouter.py       # OpenRouter LLM API client
│
├── phase0/                  # Phase 0: Keyword search experiments
│   ├── build_memories.py   # Extract memories via LLM (CLI: <file> | --all)
│   ├── db.py               # SQLite FTS5 database (CLI: --rebuild)
│   ├── memories.py         # Memory loading, field constants
│   ├── test_cases.py       # Test case generation (CLI)
│   └── experiment.py       # Experiment runner (CLI: <file> | --all)
│
├── fetch_memories.py        # CLI: Test memory search
└── export_review_data_to_csv.py  # CLI: Export to CSV
```

## Architecture

### Data Pipeline

1. **Input**: Raw code review JSON files in `data/review_data/` containing PR context, metadata, and code review comments with fields like severity, confidence, code snippets, and user notes

2. **Memory Extraction** (`scripts/pre0_build_memories.py`):
   - Two-stage AI processing via OpenRouter API
   - Stage 1: Extract concrete situation description (2 sentences max, 40-450 chars)
   - Stage 2: Extract actionable lesson (imperative, max 160 chars)
   - Quality validation rejects generic or malformed outputs
   - Confidence filtering rejects memories with fused confidence < 0.6

3. **Search Database** (`scripts/phase0_sqlite_fts.py`):
   - Loads JSONL memories into SQLite with FTS5 full-text search
   - Indexes all `situation_variants` for keyword search
   - Supports BM25 ranking

4. **Test Case Generation** (`scripts/phase0_build_test_cases.py`):
   - Processes raw PR data and extracted memories
   - Filters diffs (removes lock files, generated code, etc.)
   - Computes ground truth memory IDs by matching comment IDs
   - Creates self-contained test case files (skips PRs with no memories)
   - Each test case includes: filtered diff, PR context, metadata, ground truth IDs

5. **Retrieval Experiment** (`scripts/phase0_experiment.py`):
   - Loads test case with pre-computed ground truth
   - Generates search queries via LLM from PR context and diff
   - Searches memories.db, calculates recall against ground truth

### Memory Schema

```json
{
  "id": "mem_<12-char-hash>",
  "situation_variants": ["Variant 1 (use for display)", "Variant 2", "Variant 3"],
  "lesson": "Actionable imperative guidance",
  "metadata": { "repo", "file_pattern", "language", "severity", "confidence" },
  "source": { "file", "line", "code_snippet", "comment", "pr_context" }
}
```

## Environment

- Python 3.13 (uv package manager)
- Requires `OPENROUTER_API_KEY` environment variable
- Data files in `data/` are gitignored (contains sensitive real-world code reviews)
