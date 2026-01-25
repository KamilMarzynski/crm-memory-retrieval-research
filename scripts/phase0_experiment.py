"""
Phase 0 Experiment: Validate memory retrieval using FTS5 keyword search.

Reads raw PR data, generates search queries via LLM, and measures recall
against memories that were extracted from the same PR.

Usage:
    uv run python scripts/phase0_experiment.py data/review_data/<file>.json
    uv run python scripts/phase0_experiment.py --all
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import sys

import requests

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from phase0_sqlite_fts import search_memories, load_memories

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Files to exclude from diff (meaningless for code review patterns)
EXCLUDED_FILE_PATTERNS = [
    r"package-lock\.json$",
    r"yarn\.lock$",
    r"pnpm-lock\.yaml$",
    r"\.snap$",
    r"__snapshots__/",
    r"\.min\.js$",
    r"\.min\.css$",
    r"\.map$",
    r"\.d\.ts$",
    r"dist/",
    r"build/",
    r"node_modules/",
    r"\.generated\.",
    r"migrations/\d+",
]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_diff(full_diff: str) -> str:
    """Remove diff sections for files matching excluded patterns."""
    if not full_diff:
        return ""

    lines = full_diff.split("\n")
    filtered_lines = []
    skip_until_next_file = False
    current_file = None

    for line in lines:
        # Detect file header in diff
        if line.startswith("diff --git") or line.startswith("diff --git "):
            # Extract filename from diff header
            match = re.search(r"[ab]/(.+?)(?:\s|$)", line)
            if match:
                current_file = match.group(1)
                # Check if this file should be excluded
                skip_until_next_file = any(
                    re.search(pattern, current_file) for pattern in EXCLUDED_FILE_PATTERNS
                )

            if not skip_until_next_file:
                filtered_lines.append(line)
        elif not skip_until_next_file:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _call_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 1500,
    timeout_s: int = 120,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "memory-retrieval-research",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def _build_query_generation_prompt(context: str, filtered_diff: str) -> List[Dict[str, str]]:
    """Build prompt for LLM to generate search queries from PR context."""

    # Truncate if too long
    max_context = 3000
    max_diff = 12000

    if len(context) > max_context:
        context = context[:max_context] + "\n... (truncated)"
    if len(filtered_diff) > max_diff:
        filtered_diff = filtered_diff[:max_diff] + "\n... (truncated)"

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
    """Extract query list from LLM response."""
    # Try to find JSON array in response
    try:
        # Look for array pattern
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Fallback: split by newlines if no JSON
    queries = []
    for line in response.split("\n"):
        line = line.strip().strip("-").strip("*").strip('"').strip()
        if line and len(line) > 3:
            queries.append(line)

    return queries[:20]  # Cap at 20 queries


def _get_ground_truth_memory_ids(raw_path: str, all_memories: List[Dict[str, Any]]) -> Set[str]:
    """Get memory IDs that were extracted from this raw file."""
    raw_data = _load_json(raw_path)
    comment_ids = {c.get("id") for c in raw_data.get("code_review_comments", [])}

    ground_truth_ids = set()
    for mem in all_memories:
        source_comment_id = mem.get("metadata", {}).get("source_comment_id")
        if source_comment_id in comment_ids:
            ground_truth_ids.add(mem["id"])

    return ground_truth_ids


def run_experiment(
    raw_path: str,
    db_path: str = "data/phase0_memories/memories.db",
    phase0_dir: str = "data/phase0_memories",
    model: str = "anthropic/claude-sonnet-4.5",
    results_dir: str = "data/phase0_results",
) -> Dict[str, Any]:
    """
    Run retrieval experiment for a single raw PR file.

    Returns experiment results with queries, retrieved memories, and recall metrics.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY env var")

    # Load raw data
    raw_data = _load_json(raw_path)
    context = raw_data.get("context", "")
    full_diff = raw_data.get("full_diff", "")
    meta = raw_data.get("meta", {})

    # Filter diff
    filtered_diff = _filter_diff(full_diff)
    print(f"Diff filtered: {len(full_diff)} -> {len(filtered_diff)} chars")

    # Get ground truth
    all_memories = load_memories(phase0_dir)
    ground_truth_ids = _get_ground_truth_memory_ids(raw_path, all_memories)
    print(f"Ground truth memories from this PR: {len(ground_truth_ids)}")

    # Generate queries via LLM
    print(f"Generating queries via {model}...")
    prompt = _build_query_generation_prompt(context, filtered_diff)
    response = _call_openrouter(api_key, model, prompt)
    queries = _parse_queries_from_response(response)
    print(f"Generated {len(queries)} queries")

    # Search for each query
    all_retrieved_ids: Set[str] = set()
    query_results = []

    for query in queries:
        results = search_memories(db_path, query, limit=5)
        retrieved_ids = {r["id"] for r in results}
        all_retrieved_ids.update(retrieved_ids)

        query_results.append({
            "query": query,
            "result_count": len(results),
            "results": [
                {
                    "id": r["id"],
                    "rank": r["rank"],
                    "situation": r["situation_description"][:100],
                    "is_ground_truth": r["id"] in ground_truth_ids,
                }
                for r in results
            ],
        })

    # Calculate recall
    retrieved_ground_truth = all_retrieved_ids & ground_truth_ids
    recall = len(retrieved_ground_truth) / len(ground_truth_ids) if ground_truth_ids else 0.0

    # Calculate percentage of retrieved memories with user_note
    retrieved_with_user_note = sum(
        1 for mem_id in all_retrieved_ids
        if (mem := next((m for m in all_memories if m["id"] == mem_id), None))
        and mem.get("metadata", {}).get("user_note")
    )
    pct_with_user_note = (
        retrieved_with_user_note / len(all_retrieved_ids) if all_retrieved_ids else 0.0
    )

    # Build results
    results = {
        "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "raw_file": str(Path(raw_path).name),
        "pr_context": f"{meta.get('sourceBranch', '?')} -> {meta.get('targetBranch', '?')}",
        "model": model,
        "diff_stats": {
            "original_length": len(full_diff),
            "filtered_length": len(filtered_diff),
        },
        "ground_truth": {
            "memory_ids": list(ground_truth_ids),
            "count": len(ground_truth_ids),
        },
        "queries": query_results,
        "metrics": {
            "total_queries": len(queries),
            "total_unique_retrieved": len(all_retrieved_ids),
            "ground_truth_retrieved": len(retrieved_ground_truth),
            "recall": round(recall, 4),
            "retrieved_with_user_note": retrieved_with_user_note,
            "pct_retrieved_with_user_note": round(pct_with_user_note, 4),
        },
        "retrieved_ground_truth_ids": list(retrieved_ground_truth),
        "missed_ground_truth_ids": list(ground_truth_ids - all_retrieved_ids),
    }

    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_dir) / f"results_{Path(raw_path).stem}_{results['experiment_id']}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Console output
    print("\n" + "=" * 60)
    print(f"EXPERIMENT RESULTS: {Path(raw_path).name}")
    print("=" * 60)
    print(f"PR: {results['pr_context']}")
    print(f"Model: {model}")
    print(f"Queries generated: {len(queries)}")
    print(f"Ground truth memories: {len(ground_truth_ids)}")
    print(f"Retrieved (unique): {len(all_retrieved_ids)}")
    print(f"Ground truth retrieved: {len(retrieved_ground_truth)}")
    print(f"Retrieved with user_note: {retrieved_with_user_note} ({pct_with_user_note:.1%})")
    print(f"\nRECALL: {recall:.1%}")
    print("=" * 60)

    if ground_truth_ids - all_retrieved_ids:
        print("\nMissed memories:")
        for mid in ground_truth_ids - all_retrieved_ids:
            mem = next((m for m in all_memories if m["id"] == mid), None)
            if mem:
                print(f"  - {mid}: {mem['situation_description'][:80]}...")

    print(f"\nResults saved to: {results_path}")
    return results


def run_all_experiments(
    raw_dir: str = "data/review_data",
    db_path: str = "data/phase0_memories/memories.db",
    phase0_dir: str = "data/phase0_memories",
    model: str = "anthropic/claude-sonnet-4.5",
    results_dir: str = "data/phase0_results",
    sleep_between: float = 1.0,
) -> List[Dict[str, Any]]:
    """Run experiment for all raw files in directory."""
    raw_files = list(Path(raw_dir).glob("*.json"))
    all_results = []

    for i, raw_file in enumerate(raw_files):
        print(f"\n[{i+1}/{len(raw_files)}] Processing {raw_file.name}")
        try:
            result = run_experiment(
                str(raw_file),
                db_path=db_path,
                phase0_dir=phase0_dir,
                model=model,
                results_dir=results_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {raw_file.name}: {e}")
            all_results.append({"raw_file": raw_file.name, "error": str(e)})

        if i < len(raw_files) - 1:
            time.sleep(sleep_between)

    # Summary
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

    return all_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run python scripts/phase0_experiment.py data/review_data/<file>.json")
        print("  uv run python scripts/phase0_experiment.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        run_all_experiments()
    else:
        run_experiment(sys.argv[1])
