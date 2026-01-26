# Phase 0 Test Cases

Self-contained test cases for Phase 0 memory retrieval experiments.

## Overview

Test cases are generated from raw PR data (`data/review_data/`) and extracted memories (`data/phase0/memories/`) by the `phase0_build_test_cases.py` script. Each test case includes everything needed to run retrieval experiments without re-processing raw data.

## Generation

```bash
# Generate test cases (run after extracting memories and building database)
uv run python scripts/phase0_build_test_cases.py
```

**Source Data:**
- Raw PR files from `data/review_data/*.json`
- Extracted memories from `data/phase0/memories/*.jsonl`

**Filtering:**
- PRs with zero ground truth memories are skipped
- Diffs are pre-filtered (removes lock files, generated code, etc.)
- Ground truth is pre-computed by matching comment IDs

## File Structure

One JSON file per PR:

```
phase0_test_cases/
├── feature-VAPI-724-to-release-v1-33-0.json
├── feature-VAPI-732-be-multiple-auth-strategies-on-endpoint-8h-to-release-v1-33-0.json
└── ...
```

## Schema

```json
{
  "test_case_id": "tc_<pr-name>",
  "source_file": "<original-raw-file>.json",
  "pr_context": "Full PR description and requirements",
  "filtered_diff": "Pre-filtered code diff (lock files removed)",
  "metadata": {
    "sourceBranch": "feature/VAPI-xxx",
    "targetBranch": "release/v1.xx.x",
    "gatheredAt": "ISO timestamp",
    "gatheredFromCommit": "commit hash",
    "repoPath": "local path",
    "repoRemote": "git remote URL",
    "version": "1.0"
  },
  "diff_stats": {
    "original_length": 12345,
    "filtered_length": 10234
  },
  "ground_truth_memory_ids": [
    "mem_<hash1>",
    "mem_<hash2>"
  ],
  "ground_truth_count": 2
}
```

## Usage

Test cases are consumed by `phase0_experiment.py`:

```bash
# Run single test case
uv run python scripts/phase0_experiment.py data/phase0/test_cases/<file>.json

# Run all test cases
uv run python scripts/phase0_experiment.py --all
```

**Experiment Flow:**
1. Load test case (includes pre-filtered diff and ground truth IDs)
2. Generate search queries via LLM from `pr_context` and `filtered_diff`
3. Search `memories.db` with generated queries
4. Calculate recall against `ground_truth_memory_ids`
5. Save results to `data/phase0/results/`

## Ground Truth

**Ground truth memory IDs** are memories that were extracted from the same PR's code review comments. These represent the "correct" memories that should ideally be retrieved when processing this PR.

**Matching Logic:**
- Extract comment IDs from raw PR data
- Find memories where `metadata.source_comment_id` matches those comment IDs
- These become the ground truth for recall measurement

## Notes

- Test cases are frozen snapshots for reproducibility
- If you re-extract memories or modify filtering, regenerate test cases
- Test case IDs follow the format: `tc_<source-file-stem>`
- All test cases have at least 1 ground truth memory (zero-memory PRs are skipped)
