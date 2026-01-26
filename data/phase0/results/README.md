# Phase 0 Experiment Results

Contains results from memory retrieval experiments using FTS5 keyword search.

## How Data Gets Here

Run retrieval experiments:
```bash
# Single file
uv run python scripts/phase0_experiment.py data/review_data/<file>.json

# All files
uv run python scripts/phase0_experiment.py --all
```

## Files

- `results_<raw-file-name>_exp_<timestamp>.json` - Experiment results

## Result JSON Schema

```json
{
  "experiment_id": "exp_20260121_181026",
  "raw_file": "bugfix-JIRA-123-to-release-v1-32-1.json",
  "pr_context": "bugfix/JIRA-123 -> release/v1.32.1",
  "model": "anthropic/claude-sonnet-4.5",
  "diff_stats": {
    "original_length": 18569,
    "filtered_length": 18569
  },
  "ground_truth": {
    "memory_ids": ["mem_xxx", "mem_yyy"],
    "count": 2
  },
  "queries": [
    {
      "query": "test assertion mismatch",
      "result_count": 2,
      "results": [
        {
          "id": "mem_xxx",
          "rank": -3.77,
          "situation": "Test description contradicts...",
          "is_ground_truth": true
        }
      ]
    }
  ],
  "metrics": {
    "total_queries": 8,
    "total_unique_retrieved": 5,
    "ground_truth_retrieved": 2,
    "recall": 0.667
  },
  "retrieved_ground_truth_ids": ["mem_xxx", "mem_yyy"],
  "missed_ground_truth_ids": ["mem_zzz"]
}
```

## Metrics

- **recall**: Proportion of ground truth memories successfully retrieved (0.0 - 1.0)
- **ground_truth_retrieved**: Count of memories from this PR that were found
- **total_unique_retrieved**: Total unique memories retrieved across all queries
