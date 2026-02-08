"""
Phase 2 Experiment: Vector search + cross-encoder reranking.

This module runs retrieval experiments that combine Phase 1's vector search
with a cross-encoder reranker to improve precision. The pipeline:
    1. Load test case (from Phase 1 run)
    2. Generate search queries via LLM
    3. Execute queries against memories.db using vector similarity search
    4. Pool and deduplicate results across all queries
    5. Rerank pooled candidates using cross-encoder (bge-reranker-v2-m3)
    6. Take top-N reranked results
    7. Calculate metrics: recall, precision, F1 (both before and after reranking)
    8. Save detailed results for analysis

Usage:
    # Run all experiments (uses latest Phase 1 and Phase 2 runs)
    uv run python scripts/phase2/experiment.py --all

    # Use specific Phase 1 run
    uv run python scripts/phase2/experiment.py --all --phase1-run-id run_20260208_143022

    # Custom rerank settings
    uv run python scripts/phase2/experiment.py --all --rerank-top-n 5

Requirements:
    - OPENROUTER_API_KEY environment variable (for LLM query generation)
    - Ollama running locally with mxbai-embed-large model (for embeddings)
    - Phase 1 run with built database and test cases
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import load_json, save_json, call_openrouter, OPENROUTER_API_KEY_ENV
from common.prompts import load_prompt
from common.runs import (
    create_run,
    get_latest_run,
    get_run,
    update_run_status,
    PHASE1,
    PHASE2,
)
from common.query_generation import (
    parse_queries_robust,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    MAX_CONTEXT_LENGTH,
    MAX_DIFF_LENGTH,
    MAX_QUERIES_PER_EXPERIMENT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_SLEEP_BETWEEN_EXPERIMENTS,
)

from phase1.db import search_memories, get_random_sample_memories, get_confidence_from_distance
from phase2.load_memories import FIELD_SITUATION, FIELD_DISTANCE, FIELD_RERANK_SCORE
from phase2.reranker import Reranker

# Phase 1 prompts directory (reuse Phase 1 query generation prompts)
_PROMPTS_DIR = Path(__file__).parent.parent / "phase1" / "prompts"

# Reranking configuration
DEFAULT_RERANK_TOP_N = 4


def _pool_and_deduplicate(
    query_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Pool results from all queries and deduplicate by memory ID.

    Keeps the result with the best (lowest) vector distance for each memory.

    Args:
        query_results: List of query result dicts, each containing "results" list.

    Returns:
        Deduplicated list of result dicts, sorted by distance (ascending).
    """
    best: Dict[str, Dict[str, Any]] = {}

    for qr in query_results:
        for r in qr.get("results", []):
            mid = r["id"]
            if mid not in best or r[FIELD_DISTANCE] < best[mid][FIELD_DISTANCE]:
                best[mid] = r

    return sorted(best.values(), key=lambda x: x[FIELD_DISTANCE])


def _compute_metrics(
    retrieved_ids: Set[str],
    ground_truth_ids: Set[str],
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for a set of retrieved IDs.

    Args:
        retrieved_ids: Set of retrieved memory IDs.
        ground_truth_ids: Set of ground truth memory IDs.

    Returns:
        Dictionary with recall, precision, f1 values.
    """
    if not ground_truth_ids:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    hits = len(retrieved_ids & ground_truth_ids)
    recall = hits / len(ground_truth_ids)
    precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def run_experiment(
    test_case_path: str,
    db_path: str,
    reranker: Reranker,
    results_dir: str,
    model: str = DEFAULT_MODEL,
    prompt_version: Optional[str] = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
) -> Dict[str, Any]:
    """
    Run retrieval experiment with reranking for a single test case.

    Pipeline:
        1. Generate queries from PR context via LLM
        2. Vector search for each query (top search_limit results)
        3. Pool and deduplicate across all queries
        4. Rerank pooled candidates with cross-encoder
        5. Take top rerank_top_n results
        6. Compute metrics (before and after reranking)

    Args:
        test_case_path: Path to test case JSON file.
        db_path: Path to Phase 1 SQLite database with sqlite-vec index.
        reranker: Initialized Reranker instance.
        results_dir: Directory where result files should be saved.
        model: OpenRouter model identifier for query generation.
        prompt_version: Prompt semver string (e.g. "2.0.0"). None for latest.
        distance_threshold: Cosine distance threshold for pre-rerank metrics.
        search_limit: Maximum results per query from vector search.
        rerank_top_n: Number of results to keep after reranking.

    Returns:
        Dictionary containing experiment results and metrics.
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
    print(f"Ground truth memories: {len(ground_truth_ids)}")

    # Load prompt template (reuse Phase 1 prompts)
    query_prompt = load_prompt("memory-query", version=prompt_version, prompts_dir=_PROMPTS_DIR)

    # Truncate inputs
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "\n... (truncated)"
    if len(filtered_diff) > MAX_DIFF_LENGTH:
        filtered_diff = filtered_diff[:MAX_DIFF_LENGTH] + "\n... (truncated)"

    # Get sample memories for v2+ prompts
    sample_memories: List[Dict[str, Any]] = []
    if query_prompt.version >= "2.0.0":
        sample_memories = get_random_sample_memories(db_path, n=5)
        memory_examples = "\n".join([
            f'- "{m[FIELD_SITUATION]}"'
            for m in sample_memories[:5]
        ])
        print(f"Using prompt {query_prompt.version_tag} with {len(sample_memories)} sample memories")
    else:
        memory_examples = ""
        print(f"Using prompt {query_prompt.version_tag}")

    messages = query_prompt.render(
        context=context,
        filtered_diff=filtered_diff,
        memory_examples=memory_examples,
    )

    # Generate queries via LLM
    print(f"Generating queries via {model}...")
    response = call_openrouter(
        api_key,
        model,
        messages,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    queries = parse_queries_robust(response)
    print(f"Generated {len(queries)} queries")

    # Execute vector search for each query
    all_retrieved_ids: Set[str] = set()
    query_results = []

    for query in queries:
        results = search_memories(db_path, query, limit=search_limit)
        retrieved_ids = {r["id"] for r in results}
        all_retrieved_ids.update(retrieved_ids)

        query_results.append({
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
        })

    # --- Pre-rerank metrics (Phase 1 equivalent) ---
    pre_rerank_threshold_ids: Set[str] = {
        r["id"]
        for qr in query_results
        for r in qr["results"]
        if r["distance"] <= distance_threshold
    }
    pre_rerank_metrics = _compute_metrics(pre_rerank_threshold_ids, ground_truth_ids)

    # --- Pool, deduplicate, and rerank ---
    pooled_candidates = _pool_and_deduplicate(query_results)
    print(f"Pooled candidates (deduplicated): {len(pooled_candidates)}")

    # Build a representative query for reranking (concatenate top queries)
    # Use the query that retrieved the most ground truth as the reranking query
    best_query = queries[0] if queries else ""
    best_query_hits = 0
    for qr in query_results:
        hits = sum(1 for r in qr["results"] if r["is_ground_truth"])
        if hits > best_query_hits:
            best_query_hits = hits
            best_query = qr["query"]

    print(f"Reranking with best query ({best_query_hits} GT hits): {best_query[:80]}...")
    reranked = reranker.rerank(best_query, pooled_candidates, top_n=rerank_top_n)

    # --- Post-rerank metrics ---
    reranked_ids = {r["id"] for r in reranked}
    post_rerank_metrics = _compute_metrics(reranked_ids, ground_truth_ids)

    # Build result structure
    results = {
        "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "test_case_id": test_case.get("test_case_id", "unknown"),
        "source_file": test_case.get("source_file", "unknown"),
        "pr_context": f"{meta.get('sourceBranch', '?')} -> {meta.get('targetBranch', '?')}",
        "model": model,
        "prompt_version": query_prompt.version_tag,
        "reranker_model": reranker.model_name,
        "distance_threshold": distance_threshold,
        "search_limit": search_limit,
        "rerank_top_n": rerank_top_n,
        "rerank_query": best_query,
        "diff_stats": diff_stats,
        "ground_truth": {
            "memory_ids": sorted(list(ground_truth_ids)),
            "count": len(ground_truth_ids),
        },
        "queries": query_results,
        "pre_rerank_metrics": {
            **pre_rerank_metrics,
            "total_unique_retrieved": len(all_retrieved_ids),
            "total_within_threshold": len(pre_rerank_threshold_ids),
            "ground_truth_retrieved": len(pre_rerank_threshold_ids & ground_truth_ids),
        },
        "post_rerank_metrics": {
            **post_rerank_metrics,
            "reranked_count": len(reranked),
            "ground_truth_retrieved": len(reranked_ids & ground_truth_ids),
        },
        "reranked_results": [
            {
                "id": r["id"],
                "rerank_score": r[FIELD_RERANK_SCORE],
                "distance": r[FIELD_DISTANCE],
                "situation": r[FIELD_SITUATION],
                "lesson": r.get("lesson", ""),
                "is_ground_truth": r["id"] in ground_truth_ids,
            }
            for r in reranked
        ],
        "retrieved_ground_truth_ids": sorted(list(reranked_ids & ground_truth_ids)),
        "missed_ground_truth_ids": sorted(list(ground_truth_ids - reranked_ids)),
    }

    # Include sample memories for debugging
    if sample_memories:
        results["sample_memories_used"] = [m[FIELD_SITUATION] for m in sample_memories]

    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = (
        Path(results_dir)
        / f"results_{test_case.get('test_case_id', 'unknown')}_{results['experiment_id']}.json"
    )
    save_json(results, results_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT RESULTS: {test_case.get('test_case_id', 'unknown')}")
    print("=" * 60)
    print(f"Source: {test_case.get('source_file', 'unknown')}")
    print(f"Queries generated: {len(queries)}")
    print(f"Ground truth memories: {len(ground_truth_ids)}")
    print(f"Pooled candidates: {len(pooled_candidates)}")
    print(f"Reranked top-{rerank_top_n}: {len(reranked)}")
    print()
    print(f"PRE-RERANK METRICS (threshold={distance_threshold}):")
    print(f"  Recall:    {pre_rerank_metrics['recall']:.1%}")
    print(f"  Precision: {pre_rerank_metrics['precision']:.1%}")
    print(f"  F1 Score:  {pre_rerank_metrics['f1']:.3f}")
    print()
    print(f"POST-RERANK METRICS (top-{rerank_top_n}):")
    print(f"  Recall:    {post_rerank_metrics['recall']:.1%}")
    print(f"  Precision: {post_rerank_metrics['precision']:.1%}")
    print(f"  F1 Score:  {post_rerank_metrics['f1']:.3f}")
    print()

    f1_delta = post_rerank_metrics["f1"] - pre_rerank_metrics["f1"]
    print(f"F1 DELTA: {f1_delta:+.3f} ({'improvement' if f1_delta > 0 else 'regression'})")
    print("=" * 60)

    if ground_truth_ids - reranked_ids:
        print(f"\nMissed {len(ground_truth_ids - reranked_ids)} memories:")
        for mid in sorted(ground_truth_ids - reranked_ids):
            print(f"  - {mid}")

    print(f"\nResults saved to: {results_path}")
    return results


def run_all_experiments(
    test_cases_dir: str,
    db_path: str,
    reranker: Reranker,
    results_dir: str,
    model: str = DEFAULT_MODEL,
    sleep_between: float = DEFAULT_SLEEP_BETWEEN_EXPERIMENTS,
    prompt_version: Optional[str] = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
) -> List[Dict[str, Any]]:
    """
    Run experiments for all test cases in directory.

    Args:
        test_cases_dir: Directory containing test case JSON files.
        db_path: Path to Phase 1 SQLite database.
        reranker: Initialized Reranker instance.
        results_dir: Directory where result files should be saved.
        model: OpenRouter model identifier.
        sleep_between: Seconds to sleep between experiments.
        prompt_version: Prompt semver string. None for latest.
        distance_threshold: Cosine distance threshold for pre-rerank metrics.
        search_limit: Maximum results per query from vector search.
        rerank_top_n: Number of results after reranking.

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
                reranker=reranker,
                results_dir=results_dir,
                model=model,
                prompt_version=prompt_version,
                distance_threshold=distance_threshold,
                search_limit=search_limit,
                rerank_top_n=rerank_top_n,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {tc_file.name}: {e}")
            all_results.append({"test_case_file": tc_file.name, "error": str(e)})

        if i < len(test_case_files) - 1:
            time.sleep(sleep_between)

    # Print summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    successful = [r for r in all_results if "post_rerank_metrics" in r]
    if successful:
        avg_pre = {
            "recall": sum(r["pre_rerank_metrics"]["recall"] for r in successful) / len(successful),
            "precision": sum(r["pre_rerank_metrics"]["precision"] for r in successful) / len(successful),
            "f1": sum(r["pre_rerank_metrics"]["f1"] for r in successful) / len(successful),
        }
        avg_post = {
            "recall": sum(r["post_rerank_metrics"]["recall"] for r in successful) / len(successful),
            "precision": sum(r["post_rerank_metrics"]["precision"] for r in successful) / len(successful),
            "f1": sum(r["post_rerank_metrics"]["f1"] for r in successful) / len(successful),
        }

        print(f"Experiments run: {len(successful)}")
        print(f"Rerank top-n: {rerank_top_n}")
        print()
        print(f"{'Metric':<12} {'Pre-rerank':>12} {'Post-rerank':>12} {'Delta':>10}")
        print("-" * 48)
        for metric in ["recall", "precision", "f1"]:
            pre = avg_pre[metric]
            post = avg_post[metric]
            delta = post - pre
            print(f"{metric:<12} {pre:>12.3f} {post:>12.3f} {delta:>+10.3f}")
    else:
        print("No successful experiments")

    if len(successful) < len(all_results):
        print(f"Failed experiments: {len(all_results) - len(successful)}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2 Retrieval Experiment Runner (Vector Search + Reranking)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Description:
  Runs memory retrieval experiments combining vector similarity search
  (Phase 1) with cross-encoder reranking for improved precision.

  Pipeline: vector search (top-N) -> pool & dedup -> rerank -> top-K

Requirements:
  - {OPENROUTER_API_KEY_ENV} environment variable (for LLM API access)
  - Ollama running locally with mxbai-embed-large model (for embeddings)
  - Phase 1 run with database and test cases

Output:
  Result files saved to <phase2_run_dir>/results/results_*.json

Examples:
  uv run python scripts/phase2/experiment.py --all
  uv run python scripts/phase2/experiment.py --all --rerank-top-n 5
  uv run python scripts/phase2/experiment.py --all --phase1-run-id run_20260208_143022
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
        help="Run experiments on all test cases",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Phase 2 run ID for results (default: create new run)",
    )
    parser.add_argument(
        "--phase1-run-id",
        default=None,
        help="Phase 1 run ID for database/test cases (default: latest)",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Prompt semver to use (e.g. '2.0.0'). Defaults to latest.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help=f"Distance threshold for pre-rerank metrics (default: {DEFAULT_DISTANCE_THRESHOLD})",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Vector search candidates per query (default: {DEFAULT_SEARCH_LIMIT})",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=DEFAULT_RERANK_TOP_N,
        help=f"Results to keep after reranking (default: {DEFAULT_RERANK_TOP_N})",
    )

    args = parser.parse_args()

    if not args.test_case and not args.all:
        parser.print_help()
        sys.exit(0)

    # Resolve Phase 1 run (for DB and test cases)
    if args.phase1_run_id:
        phase1_run = get_run(PHASE1, args.phase1_run_id)
        print(f"Phase 1 run: {args.phase1_run_id}")
    else:
        phase1_run = get_latest_run(PHASE1)
        print(f"Phase 1 run (latest): {phase1_run.name}")

    db_path = str(phase1_run / "memories" / "memories.db")
    test_cases_dir = str(phase1_run / "test_cases")

    # Resolve or create Phase 2 run (for results)
    if args.run_id:
        phase2_run = get_run(PHASE2, args.run_id)
        print(f"Phase 2 run: {args.run_id}")
    else:
        run_id, phase2_run = create_run(
            PHASE2,
            description=f"Reranking experiment (phase1: {phase1_run.name})",
        )
        # Store Phase 1 reference in run metadata
        update_run_status(phase2_run, "config", {
            "phase1_run_id": phase1_run.name,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "rerank_top_n": args.rerank_top_n,
            "search_limit": args.search_limit,
        })
        print(f"Phase 2 run (new): {run_id}")

    results_dir = str(phase2_run / "results")

    print(f"Database: {db_path}")
    print(f"Test cases: {test_cases_dir}")
    print(f"Results: {results_dir}")
    print()

    # Initialize reranker (model loads lazily on first use)
    reranker = Reranker()

    if args.all:
        all_results = run_all_experiments(
            test_cases_dir=test_cases_dir,
            db_path=db_path,
            reranker=reranker,
            results_dir=results_dir,
            prompt_version=args.prompt_version,
            distance_threshold=args.distance_threshold,
            search_limit=args.search_limit,
            rerank_top_n=args.rerank_top_n,
        )
        # Update run status
        successful = [r for r in all_results if "post_rerank_metrics" in r]
        avg_f1 = sum(r["post_rerank_metrics"]["f1"] for r in successful) / len(successful) if successful else 0
        update_run_status(phase2_run, "experiment", {
            "count": len(successful),
            "failed": len(all_results) - len(successful),
            "avg_f1_post_rerank": round(avg_f1, 4),
            "rerank_top_n": args.rerank_top_n,
            "prompt_version": args.prompt_version,
        })
        print(f"\nRun status updated: {phase2_run.name}")
    else:
        run_experiment(
            args.test_case,
            db_path=db_path,
            reranker=reranker,
            results_dir=results_dir,
            prompt_version=args.prompt_version,
            distance_threshold=args.distance_threshold,
            search_limit=args.search_limit,
            rerank_top_n=args.rerank_top_n,
        )
