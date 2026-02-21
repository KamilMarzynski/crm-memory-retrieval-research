import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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

logger = logging.getLogger(__name__)

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

# Parallel execution configuration
DEFAULT_MAX_WORKERS = 5


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending a truncation notice."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _parse_json_array_queries(text: str) -> list[str] | None:
    """Try to parse a JSON array of query strings from the LLM response.

    Returns the list if found and valid, None otherwise.
    """
    try:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:MAX_QUERIES_PER_EXPERIMENT]
    except json.JSONDecodeError:
        pass
    return None


def _parse_quoted_queries(text: str) -> list[str] | None:
    """Try to extract double-quoted strings (20–200 chars) as queries.

    Returns the list if any found, None otherwise.
    """
    quoted = re.findall(r'"([^"]{20,200})"', text)
    return quoted[:MAX_QUERIES_PER_EXPERIMENT] if quoted else None


def _parse_newline_queries(text: str) -> list[str]:
    """Parse queries by splitting on newlines and stripping list formatting.

    Last-resort fallback — always returns a list (possibly empty).
    """
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\d\.\-\*]+\s*", "", line)
        line = line.strip("\"'")
        if 20 <= len(line) <= 200:
            lines.append(line)
    return lines[:MAX_QUERIES_PER_EXPERIMENT]


def parse_queries_robust(response: str) -> list[str]:
    """Robustly parse query list from LLM response with multiple fallback strategies."""
    return (
        _parse_json_array_queries(response)
        or _parse_quoted_queries(response)
        or _parse_newline_queries(response)
    )


@dataclass
class QueryGenerationConfig:
    """Configuration for LLM-based query generation."""

    prompts_dir: str | Path = "data/prompts/phase1"
    prompt_version: str | None = None
    model: str = DEFAULT_MODEL
    use_sample_memories: bool = True
    max_workers: int = DEFAULT_MAX_WORKERS

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {self.max_workers}")


def _get_random_sample_memories(
    backend: SearchBackend, db_path: str, num_samples: int = 5
) -> list[dict[str, Any]]:
    """Get sample memories if the backend supports it."""
    if isinstance(backend, VectorBackend):
        return backend.get_random_sample_memories(db_path, num_samples=num_samples)
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

    # Load prompt template
    query_prompt = load_prompt(
        "memory-query",
        version=config.prompt_version,
        prompts_dir=config.prompts_dir,
    )

    # Truncate inputs to stay within token limits
    context = _truncate_text(context, MAX_CONTEXT_LENGTH)
    filtered_diff = _truncate_text(filtered_diff, MAX_DIFF_LENGTH)

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

    messages = query_prompt.render(
        context=context,
        filtered_diff=filtered_diff,
        memory_examples=memory_examples,
    )

    # Generate queries via LLM
    raw_response = call_openrouter(
        api_key,
        config.model,
        messages,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    queries = parse_queries_robust(raw_response)

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

    return query_data


def generate_all_queries(
    test_cases_dir: str,
    queries_dir: str,
    config: QueryGenerationConfig,
    db_path: str | None = None,
    search_backend: SearchBackend | None = None,
) -> list[dict[str, Any]]:
    """Generate queries for all test cases in parallel. Calls LLM for each — costs money!

    Uses ThreadPoolExecutor to run LLM calls concurrently (I/O-bound).
    Returns list of query data dicts (one per test case), in sorted file order.
    """
    test_case_files = sorted(Path(test_cases_dir).glob("*.json"))
    total = len(test_case_files)
    results_by_file: dict[Path, dict[str, Any]] = {}
    completed_count = 0

    def process_one(test_case_file: Path) -> tuple[Path, dict[str, Any]]:
        try:
            query_data = generate_queries_for_test_case(
                str(test_case_file),
                queries_dir=queries_dir,
                config=config,
                db_path=db_path,
                search_backend=search_backend,
            )
            return test_case_file, query_data
        except Exception as error:
            return test_case_file, {"test_case_file": test_case_file.name, "error": str(error)}

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [executor.submit(process_one, f) for f in test_case_files]
        for future in as_completed(futures):
            completed_file, query_data = future.result()
            results_by_file[completed_file] = query_data

            completed_count += 1
            if "error" in query_data:
                logger.info(
                    "[%d/%d] %s — ERROR: %s",
                    completed_count,
                    total,
                    completed_file.stem,
                    query_data["error"],
                )
            else:
                num_queries = len(query_data.get("queries", []))
                logger.info(
                    "[%d/%d] %s — %d queries",
                    completed_count,
                    total,
                    completed_file.stem,
                    num_queries,
                )

    # Return in deterministic sorted order
    all_query_data = [results_by_file[f] for f in test_case_files]
    successful = [data for data in all_query_data if "queries" in data]
    logger.info("Generated queries for %d/%d test cases", len(successful), total)

    return all_query_data
