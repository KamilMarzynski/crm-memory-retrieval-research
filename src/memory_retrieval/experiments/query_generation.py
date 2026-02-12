import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_retrieval.infra.io import load_json, save_json
from memory_retrieval.infra.llm import OPENROUTER_API_KEY_ENV, call_openrouter
from memory_retrieval.infra.prompts import load_prompt
from memory_retrieval.memories.schema import FIELD_SITUATION
from memory_retrieval.search.base import SearchBackend
from memory_retrieval.search.vector import VectorBackend

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

# Batch execution configuration
DEFAULT_SLEEP_BETWEEN_EXPERIMENTS = 1.0


def parse_queries_robust(response: str) -> list[str]:
    """Robustly parse query list from LLM response with multiple fallback strategies."""
    # Try direct JSON parse first
    try:
        match = re.search(r"\[[\s\S]*\]", response)
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
    for line in response.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\d\.\-\*]+\s*", "", line)
        line = line.strip("\"'")
        if 20 <= len(line) <= 200:
            lines.append(line)

    return lines[:MAX_QUERIES_PER_EXPERIMENT]


@dataclass
class QueryGenerationConfig:
    """Configuration for LLM-based query generation."""

    prompts_dir: str | Path = "data/prompts/phase1"
    prompt_version: str | None = None
    model: str = DEFAULT_MODEL
    use_sample_memories: bool = True
    sleep_between: float = DEFAULT_SLEEP_BETWEEN_EXPERIMENTS


def _get_random_sample_memories(
    backend: SearchBackend, db_path: str, num_samples: int = 5
) -> list[dict[str, Any]]:
    """Get sample memories if the backend supports it."""
    if isinstance(backend, VectorBackend):
        return backend.get_random_sample_memories(db_path, n=num_samples)
    return []


def generate_queries_for_test_case(
    test_case_path: str,
    queries_dir: str,
    config: QueryGenerationConfig,
    db_path: str | None = None,
    search_backend: SearchBackend | None = None,
) -> dict[str, Any]:
    """Generate and save queries for a single test case via LLM.

    Loads the test case, generates search queries using the configured LLM,
    and saves the result to a JSON file in queries_dir.

    Returns the saved query data dict.
    """
    api_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Missing {OPENROUTER_API_KEY_ENV} environment variable")

    test_case = load_json(test_case_path)
    test_case_id = test_case.get("test_case_id", "unknown")
    source_file = test_case.get("source_file", "unknown")
    context = test_case.get("pr_context", "")
    filtered_diff = test_case.get("filtered_diff", "")

    print(f"Test case: {test_case_id}")

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
    if (
        config.use_sample_memories
        and query_prompt.version >= "2.0.0"
        and search_backend
        and db_path
    ):
        sample_memories = _get_random_sample_memories(search_backend, db_path, num_samples=5)
        memory_examples = "\n".join(
            [f'- "{memory[FIELD_SITUATION]}"' for memory in sample_memories[:5]]
        )
        print(
            f"Using prompt {query_prompt.version_tag} with {len(sample_memories)} sample memories"
        )
    else:
        print(f"Using prompt {query_prompt.version_tag}")

    messages = query_prompt.render(
        context=context,
        filtered_diff=filtered_diff,
        memory_examples=memory_examples,
    )

    # Generate queries via LLM
    print(f"Generating queries via {config.model}...")
    raw_response = call_openrouter(
        api_key,
        config.model,
        messages,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    queries = parse_queries_robust(raw_response)
    print(f"Generated {len(queries)} queries")

    # Build query data
    query_data: dict[str, Any] = {
        "test_case_id": test_case_id,
        "source_file": source_file,
        "model": config.model,
        "prompt_version": query_prompt.version_tag,
        "generated_at": datetime.now().isoformat(),
        "raw_response": raw_response,
        "queries": queries,
    }

    if sample_memories:
        query_data["sample_memories_used"] = [memory[FIELD_SITUATION] for memory in sample_memories]

    # Save to queries_dir
    Path(queries_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(test_case_path).stem
    query_file_path = Path(queries_dir) / f"{stem}.json"
    save_json(query_data, query_file_path)
    print(f"Queries saved to: {query_file_path}")

    return query_data


def generate_all_queries(
    test_cases_dir: str,
    queries_dir: str,
    config: QueryGenerationConfig,
    db_path: str | None = None,
    search_backend: SearchBackend | None = None,
) -> list[dict[str, Any]]:
    """Generate queries for all test cases. Calls LLM for each — costs money!

    Returns list of query data dicts (one per test case).
    """
    test_case_files = sorted(Path(test_cases_dir).glob("*.json"))
    all_query_data: list[dict[str, Any]] = []

    for i, test_case_file in enumerate(test_case_files):
        print(f"\n[{i + 1}/{len(test_case_files)}] Generating queries for {test_case_file.name}")

        try:
            query_data = generate_queries_for_test_case(
                str(test_case_file),
                queries_dir=queries_dir,
                config=config,
                db_path=db_path,
                search_backend=search_backend,
            )
            all_query_data.append(query_data)
        except Exception as error:
            print(f"Error generating queries for {test_case_file.name}: {error}")
            all_query_data.append({"test_case_file": test_case_file.name, "error": str(error)})

        if i < len(test_case_files) - 1:
            time.sleep(config.sleep_between)

    successful = [data for data in all_query_data if "queries" in data]
    print(f"\nGenerated queries for {len(successful)}/{len(all_query_data)} test cases")

    return all_query_data
