# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for experimenting with memory retrieval techniques for the [code-review-mentat](https://github.com/KamilMarzynski/code-review-mentat) project. Processes real-world code review data using AI to extract reusable engineering knowledge as structured memories.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Extract memories from raw code review data (requires OPENROUTER_API_KEY)
uv run python scripts/pre0_build_memories.py data/review_data/<file>.json

# Build/rebuild SQLite FTS5 search database
uv run python scripts/phase0_sqlite_fts.py --rebuild

# Test memory search
uv run python scripts/fetch_memories.py "your search query"

# Run retrieval experiment on single file (requires OPENROUTER_API_KEY)
uv run python scripts/phase0_experiment.py data/review_data/<file>.json

# Run retrieval experiment on all files
uv run python scripts/phase0_experiment.py --all
```

## Data Structure

```
data/
├── review_data/          # Input: Raw code review JSON files
├── phase0_memories/      # Output: Extracted memories (JSONL) + SQLite DB
│   ├── memories_*.jsonl  # Accepted memories
│   ├── rejected_*.jsonl  # Rejected memories
│   └── memories.db       # FTS5 search database
└── phase0_results/       # Output: Experiment results (JSON)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `pre0_build_memories.py` | Extract memories from raw code review JSON via LLM |
| `phase0_sqlite_fts.py` | Build SQLite FTS5 database from JSONL memories |
| `phase0_experiment.py` | Run retrieval experiments, measure recall |
| `fetch_memories.py` | CLI tool to test memory search |

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
   - Indexes `situation_description` field for keyword search
   - Supports BM25 ranking

4. **Retrieval Experiment** (`scripts/phase0_experiment.py`):
   - Takes raw PR data (context + diff)
   - Generates search queries via LLM
   - Searches memories, calculates recall against ground truth

### Memory Schema

```json
{
  "id": "mem_<12-char-hash>",
  "situation_description": "When this knowledge applies",
  "lesson": "Actionable imperative guidance",
  "metadata": { "repo", "file_pattern", "language", "severity", "confidence" },
  "source": { "file", "line", "code_snippet", "comment", "pr_context" }
}
```

## Environment

- Python 3.13 (uv package manager)
- Requires `OPENROUTER_API_KEY` environment variable
- Data files in `data/` are gitignored (contains sensitive real-world code reviews)
