import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_retrieval.infra.io import load_json, save_json
from memory_retrieval.infra.llm import call_openrouter, OPENROUTER_API_KEY_ENV
from memory_retrieval.infra.prompts import load_prompt
from memory_retrieval.memories.schema import FIELD_SITUATION, FIELD_DISTANCE, FIELD_RERANK_SCORE
from memory_retrieval.search.base import SearchBackend
from memory_retrieval.search.reranker import Reranker
from memory_retrieval.search.vector import VectorBackend, get_confidence_from_distance
from memory_retrieval.experiments.metrics import compute_metrics, analyze_query_performance
from memory_retrieval.experiments.query_generation import (
    parse_queries_robust,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    MAX_CONTEXT_LENGTH,
    MAX_DIFF_LENGTH,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_SLEEP_BETWEEN_EXPERIMENTS,
)


@dataclass
class ExperimentConfig:
    search_backend: SearchBackend
    prompts_dir: str | Path = "data/prompts/phase1"
    prompt_version: str | None = None
    model: str = DEFAULT_MODEL
    search_limit: int = DEFAULT_SEARCH_LIMIT
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    reranker: Reranker | None = None
    rerank_top_n: int = 4
    rerank_text_strategies: dict[str, Callable[[dict[str, Any]], str]] | None = None
    use_sample_memories: bool = True
    sleep_between: float = DEFAULT_SLEEP_BETWEEN_EXPERIMENTS


def _get_random_sample_memories(
    backend: SearchBackend, db_path: str, n: int = 5
) -> list[dict[str, Any]]:
    """Get sample memories if the backend supports it."""
    if isinstance(backend, VectorBackend):
        return backend.get_random_sample_memories(db_path, n=n)
    return []


def _pool_and_deduplicate(
    query_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool results from all queries and deduplicate by memory ID."""
    best: dict[str, dict[str, Any]] = {}

    for qr in query_results:
        for r in qr.get("results", []):
            mid = r["id"]
            current_dist = r.get("distance", float("inf"))
            best_dist = best[mid].get("distance", float("inf")) if mid in best else float("inf")
            if current_dist < best_dist:
                best[mid] = r

    return sorted(best.values(), key=lambda x: x.get("distance", 0))


def _pool_and_deduplicate_by_rerank_score(
    per_query_reranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool reranked results from all queries, keeping the highest rerank score per memory."""
    best: dict[str, dict[str, Any]] = {}

    for qr in per_query_reranked:
        for r in qr["reranked"]:
            mid = r["id"]
            score = r.get(FIELD_RERANK_SCORE, float("-inf"))
            if mid not in best or score > best[mid].get(FIELD_RERANK_SCORE, float("-inf")):
                best[mid] = r

    return sorted(best.values(), key=lambda x: x.get(FIELD_RERANK_SCORE, 0), reverse=True)


def _run_rerank_strategy(
    reranker: Reranker,
    query_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    rerank_top_n: int,
    text_fn: Callable[[dict[str, Any]], str] | None = None,
    text_field: str = "situation",
) -> dict[str, Any]:
    """Run reranking for a single text strategy and return metrics + results."""
    all_reranked_per_query = []
    for qr in query_results:
        reranked_for_query = reranker.rerank(
            qr["query"], qr["results"], top_n=None,
            text_field=text_field, text_fn=text_fn,
        )
        all_reranked_per_query.append({
            "query": qr["query"],
            "reranked": reranked_for_query,
        })

    pooled_reranked = _pool_and_deduplicate_by_rerank_score(all_reranked_per_query)
    top_reranked = pooled_reranked[:rerank_top_n]
    reranked_ids = {r["id"] for r in top_reranked}
    post_metrics = compute_metrics(reranked_ids, ground_truth_ids)

    return {
        "pooled_count": len(pooled_reranked),
        "post_rerank_metrics": {
            **post_metrics,
            "reranked_count": len(top_reranked),
            "ground_truth_retrieved": len(reranked_ids & ground_truth_ids),
        },
        "reranked_results": [
            {
                "id": r["id"],
                "rerank_score": r[FIELD_RERANK_SCORE],
                "distance": r.get(FIELD_DISTANCE, r.get("distance", 0)),
                "situation": r.get(FIELD_SITUATION, r.get("situation", "")),
                "lesson": r.get("lesson", ""),
                "is_ground_truth": r["id"] in ground_truth_ids,
            }
            for r in pooled_reranked
        ],
        "retrieved_ground_truth_ids": sorted(list(reranked_ids & ground_truth_ids)),
        "missed_ground_truth_ids": sorted(list(ground_truth_ids - reranked_ids)),
    }


def run_experiment(
    test_case_path: str,
    db_path: str,
    results_dir: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
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

    # Load prompt template
    query_prompt = load_prompt(
        "memory-query",
        version=config.prompt_version,
        prompts_dir=config.prompts_dir,
    )

    # Truncate inputs
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "\n... (truncated)"
    if len(filtered_diff) > MAX_DIFF_LENGTH:
        filtered_diff = filtered_diff[:MAX_DIFF_LENGTH] + "\n... (truncated)"

    # Get sample memories for v2+ prompts
    sample_memories: list[dict[str, Any]] = []
    memory_examples = ""
    if config.use_sample_memories and query_prompt.version >= "2.0.0":
        sample_memories = _get_random_sample_memories(config.search_backend, db_path, n=5)
        memory_examples = "\n".join([
            f'- "{m[FIELD_SITUATION]}"'
            for m in sample_memories[:5]
        ])
        print(f"Using prompt {query_prompt.version_tag} with {len(sample_memories)} sample memories")
    else:
        print(f"Using prompt {query_prompt.version_tag}")

    messages = query_prompt.render(
        context=context,
        filtered_diff=filtered_diff,
        memory_examples=memory_examples,
    )

    # Generate queries via LLM
    print(f"Generating queries via {config.model}...")
    response = call_openrouter(
        api_key,
        config.model,
        messages,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    queries = parse_queries_robust(response)
    print(f"Generated {len(queries)} queries")

    # Execute search for each query
    all_retrieved_ids: set[str] = set()
    query_results: list[dict[str, Any]] = []
    is_vector = isinstance(config.search_backend, VectorBackend)

    for query in queries:
        results = config.search_backend.search(db_path, query, limit=config.search_limit)
        retrieved_ids = {r.id for r in results}
        all_retrieved_ids.update(retrieved_ids)

        qr_results = []
        for r in results:
            result_entry: dict[str, Any] = {
                "id": r.id,
                "situation": r.situation,
                "lesson": r.lesson,
                "is_ground_truth": r.id in ground_truth_ids,
            }
            if is_vector:
                result_entry["distance"] = r.raw_score
                result_entry["confidence"] = get_confidence_from_distance(r.raw_score)
            else:
                result_entry["rank"] = r.raw_score

            qr_results.append(result_entry)

        query_results.append({
            "query": query,
            "word_count": len(query.split()),
            "result_count": len(results),
            "results": qr_results,
        })

    # --- Build result structure ---
    result: dict[str, Any] = {
        "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "test_case_id": test_case.get("test_case_id", "unknown"),
        "source_file": test_case.get("source_file", "unknown"),
        "pr_context": f"{meta.get('sourceBranch', '?')} -> {meta.get('targetBranch', '?')}",
        "model": config.model,
        "prompt_version": query_prompt.version_tag,
        "search_limit": config.search_limit,
        "diff_stats": diff_stats,
        "ground_truth": {
            "memory_ids": sorted(list(ground_truth_ids)),
            "count": len(ground_truth_ids),
        },
        "queries": query_results,
    }

    if sample_memories:
        result["sample_memories_used"] = [m[FIELD_SITUATION] for m in sample_memories]

    if config.reranker is not None:
        # --- Reranking path ---
        result["distance_threshold"] = config.distance_threshold
        result["rerank_top_n"] = config.rerank_top_n
        result["reranker_model"] = config.reranker.model_name

        # Pre-rerank metrics
        pre_rerank_threshold_ids: set[str] = {
            r["id"]
            for qr in query_results
            for r in qr["results"]
            if r.get("distance", 0) <= config.distance_threshold
        }
        pre_rerank_metrics = compute_metrics(pre_rerank_threshold_ids, ground_truth_ids)

        result["pre_rerank_metrics"] = {
            **pre_rerank_metrics,
            "total_unique_retrieved": len(all_retrieved_ids),
            "total_within_threshold": len(pre_rerank_threshold_ids),
            "ground_truth_retrieved": len(pre_rerank_threshold_ids & ground_truth_ids),
        }

        # Per-query reranking with strategy support
        strategies = config.rerank_text_strategies or {"default": None}

        result["rerank_queries"] = queries
        strategies_results: dict[str, dict[str, Any]] = {}

        for strategy_name, text_fn in strategies.items():
            strategy = _run_rerank_strategy(
                config.reranker, query_results, ground_truth_ids,
                config.rerank_top_n, text_fn=text_fn, text_field="situation",
            )
            strategies_results[strategy_name] = strategy
            print(f"Strategy '{strategy_name}': pooled {strategy['pooled_count']} candidates")

        # Primary strategy (first one) populates top-level keys for backward compat
        primary_name = next(iter(strategies))
        primary = strategies_results[primary_name]
        result["post_rerank_metrics"] = primary["post_rerank_metrics"]
        result["reranked_results"] = primary["reranked_results"]
        result["retrieved_ground_truth_ids"] = primary["retrieved_ground_truth_ids"]
        result["missed_ground_truth_ids"] = primary["missed_ground_truth_ids"]

        if len(strategies) > 1:
            result["rerank_strategies"] = {
                name: {k: v for k, v in s.items() if k != "pooled_count"}
                for name, s in strategies_results.items()
            }

        # Print summary
        print(f"\nALL CANDIDATES within distance {config.distance_threshold} ({len(pre_rerank_threshold_ids)} results):")
        print(f"  Recall:    {pre_rerank_metrics['recall']:.1%}")
        print(f"  Precision: {pre_rerank_metrics['precision']:.1%}")
        print(f"  F1 Score:  {pre_rerank_metrics['f1']:.3f}")

        for strategy_name, strategy in strategies_results.items():
            post_metrics = strategy["post_rerank_metrics"]
            label = f"RERANKED TOP-{config.rerank_top_n} [{strategy_name}]" if len(strategies) > 1 else f"RERANKED TOP-{config.rerank_top_n}"
            print(f"\n{label} ({post_metrics['reranked_count']} results):")
            print(f"  Recall:    {post_metrics['recall']:.1%}")
            print(f"  Precision: {post_metrics['precision']:.1%}")
            print(f"  F1 Score:  {post_metrics['f1']:.3f}")
        print(f"  (Note: different N — see analysis cells for fair comparison)")

    else:
        # --- Standard (no reranking) path ---
        result["distance_threshold"] = config.distance_threshold

        if is_vector:
            filtered_retrieved_ids: set[str] = {
                r["id"]
                for qr in query_results
                for r in qr["results"]
                if r.get("distance", 0) <= config.distance_threshold
            }
        else:
            filtered_retrieved_ids = all_retrieved_ids

        metrics = compute_metrics(filtered_retrieved_ids, ground_truth_ids)
        query_analysis = analyze_query_performance(query_results)

        result["metrics"] = {
            "total_queries": len(queries),
            "total_unique_retrieved": len(all_retrieved_ids),
            "total_within_threshold": len(filtered_retrieved_ids),
            "ground_truth_retrieved": len(filtered_retrieved_ids & ground_truth_ids),
            **metrics,
        }
        result["query_analysis"] = query_analysis
        result["retrieved_ground_truth_ids"] = sorted(list(filtered_retrieved_ids & ground_truth_ids))
        result["missed_ground_truth_ids"] = sorted(list(ground_truth_ids - filtered_retrieved_ids))

        # Print summary
        print("\nMETRICS:")
        print(f"  Recall:    {metrics['recall']:.1%}")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")

    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = (
        Path(results_dir)
        / f"results_{test_case.get('test_case_id', 'unknown')}_{result['experiment_id']}.json"
    )
    save_json(result, results_path)
    print(f"\nResults saved to: {results_path}")

    return result


def run_all_experiments(
    test_cases_dir: str,
    db_path: str,
    results_dir: str,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    test_case_files = sorted(Path(test_cases_dir).glob("*.json"))
    all_results: list[dict[str, Any]] = []

    for i, tc_file in enumerate(test_case_files):
        print(f"\n[{i + 1}/{len(test_case_files)}] Processing {tc_file.name}")

        try:
            result = run_experiment(
                str(tc_file),
                db_path=db_path,
                results_dir=results_dir,
                config=config,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {tc_file.name}: {e}")
            all_results.append({"test_case_file": tc_file.name, "error": str(e)})

        if i < len(test_case_files) - 1:
            time.sleep(config.sleep_between)

    # Print summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    if config.reranker is not None:
        success_key = "post_rerank_metrics"
    else:
        success_key = "metrics"
    successful = [r for r in all_results if success_key in r]

    if config.reranker is not None:
        if successful:
            avg_pre = {
                m: sum(r["pre_rerank_metrics"][m] for r in successful) / len(successful)
                for m in ["recall", "precision", "f1"]
            }

            print(f"Experiments run: {len(successful)}")
            print(f"Rerank top-n: {config.rerank_top_n}")

            strategy_names = list((config.rerank_text_strategies or {"default": None}).keys())
            has_strategies = "rerank_strategies" in successful[0]

            avg_pre_n = sum(r["pre_rerank_metrics"]["total_within_threshold"] for r in successful) / len(successful)

            if has_strategies:
                # Multi-strategy summary
                header = f"{'Metric':<12} {'All cands':>12}"
                divider_len = 26
                for name in strategy_names:
                    header += f" {'top-' + str(config.rerank_top_n) + ' ' + name:>24}"
                    divider_len += 26
                print()
                print(header)
                print(f"{'':12} {'(~' + f'{avg_pre_n:.0f}' + ' avg)':>12}", end="")
                for _ in strategy_names:
                    print(f" {'(' + str(config.rerank_top_n) + ' results)':>24}", end="")
                print()
                print("-" * divider_len)
                for metric in ["recall", "precision", "f1"]:
                    row = f"{metric:<12} {avg_pre[metric]:>12.3f}"
                    for name in strategy_names:
                        avg_val = sum(
                            r["rerank_strategies"][name]["post_rerank_metrics"][metric]
                            for r in successful
                        ) / len(successful)
                        row += f" {avg_val:>24.3f}"
                    print(row)
                print(f"\nNote: columns use different N — see analysis cells for fair top-N comparison")
            else:
                # Single strategy summary (backward compat)
                avg_post = {
                    m: sum(r["post_rerank_metrics"][m] for r in successful) / len(successful)
                    for m in ["recall", "precision", "f1"]
                }
                print()
                print(f"{'Metric':<12} {'All cands':>12} {'Top-' + str(config.rerank_top_n):>12}")
                print(f"{'':12} {'(~' + f'{avg_pre_n:.0f}' + ' avg)':>12} {'(' + str(config.rerank_top_n) + ' results)':>12}")
                print("-" * 38)
                for metric in ["recall", "precision", "f1"]:
                    pre = avg_pre[metric]
                    post = avg_post[metric]
                    print(f"{metric:<12} {pre:>12.3f} {post:>12.3f}")
                print(f"\nNote: columns use different N — see analysis cells for fair top-N comparison")
        else:
            print("No successful experiments")
    else:
        if successful:
            avg_recall = sum(r["metrics"]["recall"] for r in successful) / len(successful)
            avg_precision = sum(r["metrics"]["precision"] for r in successful) / len(successful)
            avg_f1 = sum(r["metrics"]["f1"] for r in successful) / len(successful)
            total_gt = sum(r["ground_truth"]["count"] for r in successful)
            total_retrieved = sum(r["metrics"]["ground_truth_retrieved"] for r in successful)

            print(f"Experiments run: {len(successful)}")
            print(f"Total ground truth memories: {total_gt}")
            print(f"Total retrieved: {total_retrieved}")
            print(f"\nAGGREGATE METRICS:")
            print(f"  Average recall:    {avg_recall:.1%}")
            print(f"  Average precision: {avg_precision:.1%}")
            print(f"  Average F1:        {avg_f1:.3f}")
        else:
            print("No successful experiments")

    if len(successful) < len(all_results):
        print(f"Failed experiments: {len(all_results) - len(successful)}")

    return all_results
