"""
Phase 0 Experiment: Validate memory retrieval using FTS5 keyword search.

This module runs retrieval experiments to measure how well keyword-based search
(using SQLite FTS5 with BM25 ranking) can retrieve relevant engineering memories
when given a code review context.

Experiment Flow:
    1. Load test case (includes pre-filtered diff and ground truth memory IDs)
    2. Generate search queries via LLM from PR context and diff
    3. Execute queries against memories.db using FTS5 search
    4. Calculate recall: (retrieved ground truth) / (total ground truth)
    5. Save detailed results for analysis

Usage:
    # Run single experiment
    uv run python scripts/phase0/experiment.py data/phase0/test_cases/<file>.json

    # Run all experiments
    uv run python scripts/phase0/experiment.py --all

    # Show help
    uv run python scripts/phase0/experiment.py --help
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import load_json, save_json, call_openrouter, OPENROUTER_API_KEY_ENV
from phase0 import search_memories, DEFAULT_DB_PATH

# Model configuration
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1500

# Query generation limits
MAX_CONTEXT_LENGTH = 3000
MAX_DIFF_LENGTH = 12000
MAX_QUERIES_PER_EXPERIMENT = 20

# Search configuration
DEFAULT_SEARCH_LIMIT = 5

# Default paths
DEFAULT_TEST_CASES_DIR = "data/phase0/test_cases"
DEFAULT_RESULTS_DIR = "data/phase0/results"

# Batch execution configuration
DEFAULT_SLEEP_BETWEEN_EXPERIMENTS = 1.0


def _build_query_generation_prompt(context: str, filtered_diff: str) -> List[Dict[str, str]]:
    """
    Build prompt for LLM to generate search queries from PR context.

    Args:
        context: PR context including description, requirements, and notes.
        filtered_diff: Code diff with build artifacts removed.

    Returns:
        List of message dicts suitable for chat completion API.
    """
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "\n... (truncated)"
    if len(filtered_diff) > MAX_DIFF_LENGTH:
        filtered_diff = filtered_diff[:MAX_DIFF_LENGTH] + "\n... (truncated)"

    system = """Generate search queries to retrieve relevant code review patterns from a vector database.

QUERY GENERATION STRATEGY:

1. Identify the code structure in the diff:
   - Is it a test file, mapper, service, validator, helper, model?
   - What methods/functions are being changed?

2. Identify technical patterns (NOT business logic):
   - Type patterns: optional, nullable, union types, generics
   - Logic patterns: conditionals, loops, error handling
   - Test patterns: edge cases, coverage gaps, assertion issues
   - API patterns: field changes, breaking changes, versioning

3. Identify specific gaps or issues:
   - What's missing? Use phrases like "missing test", "no validation", "lacks error handling"
   - What's inconsistent? Use phrases like "handles X but not Y"
   - What might break? Use phrases like "breaking change", "external dependency"

4. Generate queries that combine structure + pattern + gap:
   - "test file optional parameter missing edge case"
   - "mapper method undefined vs null handling"
   - "service boolean logic missing test coverage"

QUERY RULES:
- 4-8 words per query
- Include code structure when relevant (test file, mapper, service)
- Use technical terms: optional, undefined, null, nullable, edge case, union type
- Describe gaps: "missing", "lacks", "no", "doesn't handle"
- Avoid business domain terms (replace with generic technical terms)
- Generate 5-8 queries from different angles

OUTPUT: JSON array of query strings only.

EXAMPLE OUTPUT:
["test file optional parameter edge case", "mapper method undefined parent object", "missing test completely undefined input", "optional object test coverage nested vs parent", "test suite nullable parameter all scenarios"]"""

    user = f"""PR CONTEXT:
{context}

CODE DIFF:
{filtered_diff}

Generate search queries to retrieve relevant engineering memories. Return ONLY a JSON array of query strings."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_queries_from_response(response: str) -> List[str]:
    """
    Extract query list from LLM response.

    Args:
        response: Raw LLM response string.

    Returns:
        List of query strings, capped at MAX_QUERIES_PER_EXPERIMENT.
    """
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            return json.loads(match.group())[:MAX_QUERIES_PER_EXPERIMENT]
    except json.JSONDecodeError:
        pass

    # Fallback: parse line-by-line
    queries = []
    for line in response.split("\n"):
        line = line.strip().strip("-").strip("*").strip('"').strip()
        if line and len(line) > 3:
            queries.append(line)

    return queries[:MAX_QUERIES_PER_EXPERIMENT]


def run_experiment(
    test_case_path: str,
    db_path: str = DEFAULT_DB_PATH,
    model: str = DEFAULT_MODEL,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Run retrieval experiment for a single test case.

    Args:
        test_case_path: Path to test case JSON file.
        db_path: Path to SQLite database with FTS5 index.
        model: OpenRouter model identifier.
        results_dir: Directory where result files should be saved.

    Returns:
        Dictionary containing experiment results and metrics.

    Raises:
        SystemExit: If OPENROUTER_API_KEY environment variable is not set.
    """
    api_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Missing {OPENROUTER_API_KEY_ENV} environment variable")

    test_case = load_json(test_case_path)
    context = test_case.get("pr_context", "")
    filtered_diff = test_case.get("filtered_diff", "")
    meta = test_case.get("metadata", {})
    ground_truth_ids = set(test_case.get("ground_truth_memory_ids", []))
    diff_stats = test_case.get("diff_stats", {})

    print(f"Test case: {test_case.get('test_case_id', 'unknown')}")
    print(f"Diff stats: {diff_stats.get('original_length', 0)} -> {diff_stats.get('filtered_length', 0)} chars")
    print(f"Ground truth memories: {len(ground_truth_ids)}")

    print(f"Generating queries via {model}...")
    prompt = _build_query_generation_prompt(context, filtered_diff)
    response = call_openrouter(api_key, model, prompt, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS)
    queries = _parse_queries_from_response(response)
    print(f"Generated {len(queries)} queries")

    all_retrieved_ids: Set[str] = set()
    query_results = []

    for query in queries:
        results = search_memories(db_path, query, limit=DEFAULT_SEARCH_LIMIT)
        retrieved_ids = {r["id"] for r in results}
        all_retrieved_ids.update(retrieved_ids)

        query_results.append({
            "query": query,
            "result_count": len(results),
            "results": [
                {
                    "id": r["id"],
                    "rank": r["rank"],
                    "situation": (r["situation_variants"][0] if r["situation_variants"] else "")[:100],
                    "is_ground_truth": r["id"] in ground_truth_ids,
                }
                for r in results
            ],
        })

    retrieved_ground_truth = all_retrieved_ids & ground_truth_ids
    recall = len(retrieved_ground_truth) / len(ground_truth_ids) if ground_truth_ids else 0.0

    results = {
        "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "test_case_id": test_case.get("test_case_id", "unknown"),
        "source_file": test_case.get("source_file", "unknown"),
        "pr_context": f"{meta.get('sourceBranch', '?')} -> {meta.get('targetBranch', '?')}",
        "model": model,
        "diff_stats": diff_stats,
        "ground_truth": {
            "memory_ids": sorted(list(ground_truth_ids)),
            "count": len(ground_truth_ids),
        },
        "queries": query_results,
        "metrics": {
            "total_queries": len(queries),
            "total_unique_retrieved": len(all_retrieved_ids),
            "ground_truth_retrieved": len(retrieved_ground_truth),
            "recall": round(recall, 4),
        },
        "retrieved_ground_truth_ids": sorted(list(retrieved_ground_truth)),
        "missed_ground_truth_ids": sorted(list(ground_truth_ids - all_retrieved_ids)),
    }

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_dir) / f"results_{test_case.get('test_case_id', 'unknown')}_{results['experiment_id']}.json"
    save_json(results, results_path)

    print("\n" + "=" * 60)
    print(f"EXPERIMENT RESULTS: {test_case.get('test_case_id', 'unknown')}")
    print("=" * 60)
    print(f"Source: {test_case.get('source_file', 'unknown')}")
    print(f"PR: {results['pr_context']}")
    print(f"Model: {model}")
    print(f"Queries generated: {len(queries)}")
    print(f"Ground truth memories: {len(ground_truth_ids)}")
    print(f"Retrieved (unique): {len(all_retrieved_ids)}")
    print(f"Ground truth retrieved: {len(retrieved_ground_truth)}")
    print(f"\nRECALL: {recall:.1%}")
    print("=" * 60)

    if ground_truth_ids - all_retrieved_ids:
        print(f"\nMissed {len(ground_truth_ids - all_retrieved_ids)} memories:")
        for mid in sorted(ground_truth_ids - all_retrieved_ids):
            print(f"  - {mid}")

    print(f"\nResults saved to: {results_path}")
    return results


def run_all_experiments(
    test_cases_dir: str = DEFAULT_TEST_CASES_DIR,
    db_path: str = DEFAULT_DB_PATH,
    model: str = DEFAULT_MODEL,
    results_dir: str = DEFAULT_RESULTS_DIR,
    sleep_between: float = DEFAULT_SLEEP_BETWEEN_EXPERIMENTS,
) -> List[Dict[str, Any]]:
    """
    Run experiments for all test cases in directory.

    Args:
        test_cases_dir: Directory containing test case JSON files.
        db_path: Path to SQLite database with FTS5 index.
        model: OpenRouter model identifier.
        results_dir: Directory where result files should be saved.
        sleep_between: Seconds to sleep between experiments.

    Returns:
        List of result dictionaries from each experiment.
    """
    test_case_files = sorted(Path(test_cases_dir).glob("*.json"))
    all_results = []

    for i, tc_file in enumerate(test_case_files):
        print(f"\n[{i+1}/{len(test_case_files)}] Processing {tc_file.name}")

        try:
            result = run_experiment(
                str(tc_file),
                db_path=db_path,
                model=model,
                results_dir=results_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {tc_file.name}: {e}")
            all_results.append({"test_case_file": tc_file.name, "error": str(e)})

        if i < len(test_case_files) - 1:
            time.sleep(sleep_between)

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    successful = [r for r in all_results if "metrics" in r]
    if successful:
        avg_recall = sum(r["metrics"]["recall"] for r in successful) / len(successful)
        total_gt = sum(r["ground_truth"]["count"] for r in successful)
        total_retrieved = sum(r["metrics"]["ground_truth_retrieved"] for r in successful)

        print(f"Experiments run: {len(successful)}")
        print(f"Total ground truth memories: {total_gt}")
        print(f"Total retrieved: {total_retrieved}")
        print(f"Average recall: {avg_recall:.1%}")
    else:
        print("No successful experiments")

    if len(successful) < len(all_results):
        print(f"Failed experiments: {len(all_results) - len(successful)}")

    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
        print("Phase 0 Retrieval Experiment Runner")
        print()
        print("Usage:")
        print("  uv run python scripts/phase0/experiment.py <test-case.json>")
        print("  uv run python scripts/phase0/experiment.py --all")
        print("  uv run python scripts/phase0/experiment.py --help")
        print()
        print("Description:")
        print("  Runs memory retrieval experiments using FTS5 keyword search with BM25 ranking.")
        print("  Measures recall against pre-computed ground truth memory IDs.")
        print()
        print("Requirements:")
        print(f"  - {OPENROUTER_API_KEY_ENV} environment variable (for LLM API access)")
        print(f"  - Test cases in {DEFAULT_TEST_CASES_DIR}/ (generated by test_cases.py)")
        print(f"  - Memory database at {DEFAULT_DB_PATH} (built by db.py)")
        print()
        print("Options:")
        print("  <test-case.json>  Run single experiment on specified test case")
        print("  --all             Run experiments on all test cases in directory")
        print("  --help, -h        Show this help message")
        print()
        print("Output:")
        print(f"  Result files saved to {DEFAULT_RESULTS_DIR}/results_*.json")
        sys.exit(0)

    if sys.argv[1] == "--all":
        run_all_experiments()
    else:
        run_experiment(sys.argv[1])
