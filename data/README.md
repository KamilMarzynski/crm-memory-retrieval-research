# OUTDATED


# Data Directory

This directory contains both input data and output results for the memory retrieval research project.
All data files are gitignored (sensitive), but directory structure and README files are tracked.

## Directory Structure

```
data/
├── review_data/          # Input: Raw code review JSON files
├── phase0_memories/      # Output: Extracted memories (JSONL) + SQLite DB
└── phase0_results/       # Output: Experiment results (JSON)
```

## Data Flow

```
1. Collect code reviews    →  data/review_data/*.json
2. Extract memories        →  data/phase0_memories/*.jsonl
3. Build search database   →  data/phase0_memories/memories.db
4. Run retrieval experiments →  data/phase0_results/*.json
```

## Scripts

| Script | Input | Output |
|--------|-------|--------|
| `pre0_build_memories.py` | `review_data/*.json` | `phase0_memories/*.jsonl` |
| `phase0_sqlite_fts.py` | `phase0_memories/*.jsonl` | `phase0_memories/memories.db` |
| `phase0_experiment.py` | `review_data/*.json` + DB | `phase0_results/*.json` |
| `fetch_memories.py` | Query string | Console output |
