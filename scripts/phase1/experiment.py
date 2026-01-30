"""
Phase 1 Experiment: Validate memory retrieval using vector similarity search.

This module runs retrieval experiments to measure how well vector-based search
(using sqlite-vec with cosine similarity) can retrieve relevant engineering memories
when given a code review context.

Experiment Flow:
    1. Load test case (includes pre-filtered diff and ground truth memory IDs)
    2. Generate search queries via LLM from PR context and diff
    3. Execute queries against memories.db using vector similarity search
    4. Calculate recall: (retrieved ground truth) / (total ground truth)
    5. Save detailed results for analysis

Usage:
    # Run single experiment
    uv run python scripts/phase1/experiment.py data/phase1/test_cases/<file>.json

    # Run all experiments
    uv run python scripts/phase1/experiment.py --all

    # Show help
    uv run python scripts/phase1/experiment.py --help

Requirements:
    - OPENROUTER_API_KEY environment variable (for LLM API access)
    - Ollama running locally with mxbai-embed-large model (for embeddings)
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import load_json, save_json, call_openrouter, OPENROUTER_API_KEY_ENV
from phase1 import search_memories, DEFAULT_DB_PATH, FIELD_SITUATION, FIELD_DISTANCE
from phase1.db import get_random_sample_memories, get_confidence_from_distance

# Prompt version constants for tracking
QUERY_PROMPT_VERSION_V1 = "query_v1.0"
QUERY_PROMPT_VERSION_V2 = "query_v2.0"

# Model configuration
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1500

# Query generation limits
MAX_CONTEXT_LENGTH = 3000
MAX_DIFF_LENGTH = 12000
MAX_QUERIES_PER_EXPERIMENT = 20

# Query length targets (aligned with memory length for vector similarity)
TARGET_QUERY_WORDS_MIN = 20
TARGET_QUERY_WORDS_MAX = 50

# Search configuration
DEFAULT_SEARCH_LIMIT = 5
MAX_DISTANCE_THRESHOLD = 2.0

# Default paths
DEFAULT_TEST_CASES_DIR = "data/phase1/test_cases"
DEFAULT_RESULTS_DIR = "data/phase1/results"

# Batch execution configuration
DEFAULT_SLEEP_BETWEEN_EXPERIMENTS = 1.0


def _build_query_generation_prompt_v1(
    context: str, filtered_diff: str
) -> List[Dict[str, str]]:
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

    system = """You generate vector-search queries to retrieve engineering "situations" from a database.

        The database entries ("situations") have this style:
        - 2–3 short sentences describing a reusable code review pattern
        - Concrete technical terms: optional, undefined, null, nullable, union type, edge case, optional chaining, mapper, validator, test coverage, breaking change, external client
        - Describes WHEN it applies and WHAT is missing/inconsistent
        - Avoids code identifiers, file names, and implementation details
        - No solutions, no advice, no "should fix" language

        Your job is to rewrite the PR context + diff into queries that LOOK LIKE those situation descriptions.
        This is dense retrieval: prioritize semantic match over exact keyword match."""

    user = f"""PR CONTEXT:
        PR CONTEXT:
        {context}

        CODE DIFF:
        {filtered_diff}

        TASK
        Generate queries to retrieve relevant engineering memories from a vector database.

        RULES
        - Output ONLY a JSON array of strings.
        - Generate 6–10 queries.
        - Each query MUST be 2–3 short sentences.
        - Each query MUST describe a situation/pattern (WHEN/WHERE it applies) and the gap (missing/contradiction/inconsistency).
        - Do NOT include advice or fixes (no "should", "need to", "add", "change", "refactor", "ensure", "avoid").
        - Do NOT include file names, function names, variable names, or code identifiers.
        - Prefer technical vocabulary over domain vocabulary; include domain terms only if they are essential and prominent in the diff.
        - Across the whole list, cover at least these variations when applicable:
        - Test-related phrasing: "Test file ..." / "test description ..." / "assertion logic ..."
        - Nullability phrasing: "optional input ..." / "undefined vs null ..." / "nullable ..."
        - Structural phrasing: "Mapper ..." / "Service method ..." / "Validator ..."
        - Synonyms for gaps: "missing" / "lacks" / "doesn't handle" / "contradicts" / "inconsistent"

        QUALITY BAR
        Before output, check each query: “Could this sentence plausibly be stored as a situation_description in my database?”
        Return only the JSON array, no extra text.
"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_query_generation_prompt_v2(
    context: str,
    filtered_diff: str,
    sample_memories: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Build improved prompt for pattern-based query generation.

    Version: 2.0
    Output length: 20-50 words per query (aligned with memory length)

    Key improvements over v1:
    - Includes actual memory examples for semantic alignment
    - Aligned query length to 20-50 words
    - Added self-verification step
    - Hybrid vocabulary guidance

    Args:
        context: PR context including description, requirements, and notes.
        filtered_diff: Code diff with build artifacts removed.
        sample_memories: Real memories from database to ground the LLM.

    Returns:
        List of message dicts suitable for chat completion API.
    """
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "\n... (truncated)"
    if len(filtered_diff) > MAX_DIFF_LENGTH:
        filtered_diff = filtered_diff[:MAX_DIFF_LENGTH] + "\n... (truncated)"

    # Format actual memory examples (critical for semantic alignment)
    memory_examples = "\n".join([
        f"- \"{m[FIELD_SITUATION]}\""
        for m in sample_memories[:5]
    ])

    system = f"""You generate search queries for a code review memory database.

The database contains patterns like these REAL EXAMPLES:
{memory_examples}

Your queries MUST sound like these entries to achieve semantic match.

QUERY RULES:
- Each query: 20-50 words (1-2 sentences)
- Describe a pattern/situation, NOT a solution
- Include: code structure + technical pattern + gap/issue
- NO file names, function names, or identifiers
- NO advice verbs: "should", "need to", "ensure", "avoid"

VOCABULARY GUIDANCE:
Prefer structural/technical terms over domain-specific ones.
Instead of naming WHAT the code does (authentication, payment, pagination),
describe HOW it does it (strategy pattern, numeric parameters, nested object access).

Domain terms ARE acceptable when they add clarity that technical terms cannot capture.
Example: "pagination boundary" may be clearer than "numeric range limit".

SELF-CHECK before including each query:
1. Does it describe a pattern (not a solution)?
2. Could it plausibly exist in the example database above?
3. Is it 20-50 words?
Remove any queries that fail these checks.

Generate 6-10 diverse queries covering different aspects of the diff."""

    user = f"""PR CONTEXT:
{context}

CODE DIFF:
{filtered_diff}

Generate queries that would retrieve relevant memories.
Output ONLY a JSON array of strings."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_queries_from_response(response: str) -> List[str]:
    """
    Extract query list from LLM response (v1 parser).

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


def parse_queries_robust(response: str) -> List[str]:
    """
    Robustly parse query list from LLM response.

    Handles various output formats and malformed JSON with multiple
    fallback strategies.

    Args:
        response: Raw LLM response string.

    Returns:
        List of query strings, capped at MAX_QUERIES_PER_EXPERIMENT.
    """
    # Try direct JSON parse first
    try:
        # Look for JSON array
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:MAX_QUERIES_PER_EXPERIMENT]
    except json.JSONDecodeError:
        pass

    # Fallback: extract quoted strings (20-200 chars for reasonable queries)
    quoted = re.findall(r'"([^"]{20,200})"', response)
    if quoted:
        return quoted[:MAX_QUERIES_PER_EXPERIMENT]

    # Last resort: split by newlines and clean
    lines = []
    for line in response.split('\n'):
        line = line.strip()
        line = re.sub(r'^[\d\.\-\*]+\s*', '', line)  # Remove list markers
        line = line.strip('"\'')
        if 20 <= len(line) <= 200:
            lines.append(line)

    return lines[:MAX_QUERIES_PER_EXPERIMENT]


def analyze_query_performance(query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze which queries perform well vs poorly.

    Args:
        query_results: List of query result dictionaries from experiment.

    Returns:
        Dictionary with query performance statistics.
    """
    import statistics

    query_stats = []

    for query_result in query_results:
        hits = sum(1 for r in query_result.get("results", []) if r.get("is_ground_truth"))
        total = len(query_result.get("results", []))
        distances = [r["distance"] for r in query_result.get("results", [])]

        query_stats.append({
            "query": query_result["query"],
            "word_count": len(query_result["query"].split()),
            "hits": hits,
            "precision": hits / total if total > 0 else 0,
            "avg_distance": statistics.mean(distances) if distances else None,
            "min_distance": min(distances) if distances else None,
        })

    queries_with_hits = sum(1 for q in query_stats if q["hits"] > 0)
    total_queries = len(query_stats)

    return {
        "queries_with_hits": queries_with_hits,
        "total_queries": total_queries,
        "query_hit_rate": queries_with_hits / total_queries if total_queries > 0 else 0,
        "best_queries": sorted(query_stats, key=lambda x: -x["hits"])[:3],
        "worst_queries": sorted(query_stats, key=lambda x: x["hits"])[:3],
        "avg_word_count": statistics.mean([q["word_count"] for q in query_stats]) if query_stats else 0,
    }


def run_experiment(
    test_case_path: str,
    db_path: str = DEFAULT_DB_PATH,
    model: str = DEFAULT_MODEL,
    results_dir: str = DEFAULT_RESULTS_DIR,
    prompt_version: str = "v2",
) -> Dict[str, Any]:
    """
    Run retrieval experiment for a single test case.

    Args:
        test_case_path: Path to test case JSON file.
        db_path: Path to SQLite database with sqlite-vec index.
        model: OpenRouter model identifier.
        results_dir: Directory where result files should be saved.
        prompt_version: Prompt version to use ("v1" or "v2").

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
    print(
        f"Diff stats: {diff_stats.get('original_length', 0)} -> {diff_stats.get('filtered_length', 0)} chars"
    )
    print(f"Ground truth memories: {len(ground_truth_ids)}")

    # Select prompt version and parser
    parser: Callable[[str], List[str]]
    sample_memories: List[Dict[str, Any]]

    if prompt_version == "v2":
        # Get sample memories for v2 prompt (grounds LLM in database vocabulary)
        sample_memories = get_random_sample_memories(db_path, n=5)
        prompt = _build_query_generation_prompt_v2(context, filtered_diff, sample_memories)
        parser = parse_queries_robust
        version_tag = QUERY_PROMPT_VERSION_V2
        print(f"Using prompt v2 with {len(sample_memories)} sample memories")
    else:
        prompt = _build_query_generation_prompt_v1(context, filtered_diff)
        parser = _parse_queries_from_response
        version_tag = QUERY_PROMPT_VERSION_V1
        sample_memories = []
        print("Using prompt v1")

    print(f"Generating queries via {model}...")
    response = call_openrouter(
        api_key,
        model,
        prompt,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    queries = parser(response)
    print(f"Generated {len(queries)} queries")

    all_retrieved_ids: Set[str] = set()
    query_results = []

    for query in queries:
        results = search_memories(db_path, query, limit=DEFAULT_SEARCH_LIMIT)
        results = [r for r in results if r[FIELD_DISTANCE] <= MAX_DISTANCE_THRESHOLD]
        retrieved_ids = {r["id"] for r in results}
        all_retrieved_ids.update(retrieved_ids)

        query_results.append(
            {
                "query": query,
                "word_count": len(query.split()),
                "result_count": len(results),
                "results": [
                    {
                        "id": r["id"],
                        "distance": r[FIELD_DISTANCE],
                        "confidence": get_confidence_from_distance(r[FIELD_DISTANCE]),
                        "situation": r[FIELD_SITUATION],
                        "is_ground_truth": r["id"] in ground_truth_ids,
                    }
                    for r in results
                ],
            }
        )

    retrieved_ground_truth = all_retrieved_ids & ground_truth_ids
    recall = (
        len(retrieved_ground_truth) / len(ground_truth_ids) if ground_truth_ids else 0.0
    )

    # Calculate precision and F1
    total_retrieved = sum(len(qr["results"]) for qr in query_results)
    total_ground_truth_hits = sum(
        sum(1 for r in qr["results"] if r["is_ground_truth"])
        for qr in query_results
    )
    precision = total_ground_truth_hits / total_retrieved if total_retrieved > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Analyze per-query performance
    query_analysis = analyze_query_performance(query_results)

    results = {
        "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "test_case_id": test_case.get("test_case_id", "unknown"),
        "source_file": test_case.get("source_file", "unknown"),
        "pr_context": f"{meta.get('sourceBranch', '?')} -> {meta.get('targetBranch', '?')}",
        "model": model,
        "prompt_version": version_tag,
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
            "total_results": total_retrieved,
            "total_ground_truth_hits": total_ground_truth_hits,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        },
        "query_analysis": query_analysis,
        "retrieved_ground_truth_ids": sorted(list(retrieved_ground_truth)),
        "missed_ground_truth_ids": sorted(list(ground_truth_ids - all_retrieved_ids)),
    }

    # Include sample memories used in v2 prompt for debugging
    if sample_memories:
        results["sample_memories_used"] = [m[FIELD_SITUATION] for m in sample_memories]

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = (
        Path(results_dir)
        / f"results_{test_case.get('test_case_id', 'unknown')}_{results['experiment_id']}.json"
    )
    save_json(results, results_path)

    print("\n" + "=" * 60)
    print(f"EXPERIMENT RESULTS: {test_case.get('test_case_id', 'unknown')}")
    print("=" * 60)
    print(f"Source: {test_case.get('source_file', 'unknown')}")
    print(f"PR: {results['pr_context']}")
    print(f"Model: {model}")
    print(f"Prompt version: {version_tag}")
    print(f"Queries generated: {len(queries)}")
    print(f"Avg query length: {query_analysis['avg_word_count']:.1f} words")
    print(f"Ground truth memories: {len(ground_truth_ids)}")
    print(f"Retrieved (unique): {len(all_retrieved_ids)}")
    print(f"Ground truth retrieved: {len(retrieved_ground_truth)}")
    print("\nMETRICS:")
    print(f"  Recall:    {recall:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Query hit rate: {query_analysis['query_hit_rate']:.1%}")
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
    prompt_version: str = "v2",
) -> List[Dict[str, Any]]:
    """
    Run experiments for all test cases in directory.

    Args:
        test_cases_dir: Directory containing test case JSON files.
        db_path: Path to SQLite database with sqlite-vec index.
        model: OpenRouter model identifier.
        results_dir: Directory where result files should be saved.
        sleep_between: Seconds to sleep between experiments.
        prompt_version: Prompt version to use ("v1" or "v2").

    Returns:
        List of result dictionaries from each experiment.
    """
    test_case_files = sorted(Path(test_cases_dir).glob("*.json"))
    all_results = []

    for i, tc_file in enumerate(test_case_files):
        print(f"\n[{i + 1}/{len(test_case_files)}] Processing {tc_file.name}")

        try:
            result = run_experiment(
                str(tc_file),
                db_path=db_path,
                model=model,
                results_dir=results_dir,
                prompt_version=prompt_version,
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
        avg_precision = sum(r["metrics"]["precision"] for r in successful) / len(successful)
        avg_f1 = sum(r["metrics"]["f1"] for r in successful) / len(successful)
        total_gt = sum(r["ground_truth"]["count"] for r in successful)
        total_retrieved = sum(
            r["metrics"]["ground_truth_retrieved"] for r in successful
        )

        print(f"Prompt version: {prompt_version}")
        print(f"Experiments run: {len(successful)}")
        print(f"Total ground truth memories: {total_gt}")
        print(f"Total retrieved: {total_retrieved}")
        print("\nAGGREGATE METRICS:")
        print(f"  Average recall:    {avg_recall:.1%}")
        print(f"  Average precision: {avg_precision:.1%}")
        print(f"  Average F1:        {avg_f1:.3f}")
    else:
        print("No successful experiments")

    if len(successful) < len(all_results):
        print(f"Failed experiments: {len(all_results) - len(successful)}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1 Retrieval Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Description:
  Runs memory retrieval experiments using vector similarity search
  with sqlite-vec and cosine distance ranking.
  Measures recall, precision, and F1 against pre-computed ground truth memory IDs.

Requirements:
  - {OPENROUTER_API_KEY_ENV} environment variable (for LLM API access)
  - Ollama running locally with mxbai-embed-large model (for embeddings)
  - Test cases in {DEFAULT_TEST_CASES_DIR}/ (generated by test_cases.py)
  - Memory database at {DEFAULT_DB_PATH} (built by db.py)

Output:
  Result files saved to {DEFAULT_RESULTS_DIR}/results_*.json

Examples:
  uv run python scripts/phase1/experiment.py data/phase1/test_cases/test_001.json
  uv run python scripts/phase1/experiment.py --all
  uv run python scripts/phase1/experiment.py --all --prompt-version v1
        """,
    )
    parser.add_argument(
        "test_case",
        nargs="?",
        help="Path to test case JSON file (or use --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run experiments on all test cases in directory",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default="v2",
        help="Prompt version to use (default: v2 with sample memories and aligned lengths)",
    )

    args = parser.parse_args()

    if not args.test_case and not args.all:
        parser.print_help()
        sys.exit(0)

    if args.all:
        run_all_experiments(prompt_version=args.prompt_version)
    else:
        run_experiment(args.test_case, prompt_version=args.prompt_version)
