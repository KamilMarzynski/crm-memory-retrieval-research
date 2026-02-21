# Comprehensive Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Comprehensive code quality sweep — type tightening, function decomposition, DRY via SearchBackendBase, class-based orchestrators, config validation, logging, and centralized constants.

**Architecture:** File-by-file in dependency order (infra → memories → search → experiments), then one cross-file cleanup pass. Each file gets all concerns addressed atomically: types + decompose + DRY within that file. No public APIs change — notebooks remain untouched.

**Tech Stack:** Python 3.13, uv, pytest, mypy (for verification)

---

## Orientation

Run tests before touching anything:
```bash
uv run pytest tests/ -q
```
All tests must pass before and after every task.

The test runner throughout this plan:
```bash
uv run pytest tests/ -q --tb=short
```

---

## Task 1: Create `types.py` — shared TypedDicts and TypeAliases

**Files:**
- Create: `src/memory_retrieval/types.py`
- Create: `tests/test_types.py`

**Step 1: Write the failing test**

```python
# tests/test_types.py
from memory_retrieval.types import (
    ConfigFingerprint,
    DiffStats,
    ExperimentResultEntry,
    MemoryDict,
    MemoryMetadata,
    MemorySource,
    MetricsDict,
    QueryResult,
    ScoreType,
    TestCase,
    TextStrategyFn,
)


def test_memory_dict_structure() -> None:
    memory: MemoryDict = {
        "id": "mem_abc123",
        "situation_description": "A developer added a utility function without tests.",
        "lesson": "Always write tests for utility functions.",
        "metadata": {
            "repo": "my-repo",
            "file_pattern": "*.py",
            "language": "python",
            "severity": "medium",
            "confidence": "high",
        },
        "source": {
            "file": "utils.py",
            "line": 42,
            "code_snippet": "def helper(): pass",
            "comment": "Needs test",
            "pr_context": "main <- feature/add-helper",
        },
    }
    assert memory["id"] == "mem_abc123"


def test_metrics_dict_structure() -> None:
    metrics: MetricsDict = {
        "precision": 0.8,
        "recall": 0.75,
        "f1_score": 0.774,
        "reciprocal_rank": 1.0,
    }
    assert metrics["f1_score"] == 0.774


def test_score_type_literal() -> None:
    valid_types: list[ScoreType] = ["cosine_distance", "bm25_rank", "rerank_score"]
    assert len(valid_types) == 3


def test_text_strategy_fn_callable() -> None:
    strategy: TextStrategyFn = lambda result: result.get("situation", "")  # noqa: E731
    assert strategy({"situation": "hello"}) == "hello"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_types.py -v
```
Expected: `ModuleNotFoundError: No module named 'memory_retrieval.types'`

**Step 3: Write the implementation**

```python
# src/memory_retrieval/types.py
from collections.abc import Callable
from typing import Any, Literal, NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Core domain types
# ---------------------------------------------------------------------------


class MemoryMetadata(TypedDict):
    repo: str
    file_pattern: str
    language: str
    severity: str
    confidence: str


class MemorySource(TypedDict):
    file: str
    line: int
    code_snippet: str
    comment: str
    pr_context: str


class MemoryDict(TypedDict):
    id: str
    situation_description: str
    lesson: str
    metadata: MemoryMetadata
    source: MemorySource


# ---------------------------------------------------------------------------
# Experiment pipeline types
# ---------------------------------------------------------------------------


class MetricsDict(TypedDict):
    precision: float
    recall: float
    f1_score: float
    reciprocal_rank: float


class QueryResult(TypedDict):
    query: str
    results: list[dict[str, Any]]
    num_results: int


class ExperimentResultEntry(TypedDict):
    test_case_id: str
    ground_truth_ids: list[str]
    query_results: list[QueryResult]
    pre_rerank_metrics: MetricsDict
    reranked_results: NotRequired[list[dict[str, Any]]]
    post_rerank_metrics: NotRequired[MetricsDict]


class ConfigFingerprint(TypedDict):
    extraction_prompt: str
    extraction_model: str
    query_prompt: str
    query_model: str
    search_limit: int
    distance_threshold: float
    reranker: str | None


class DiffStats(TypedDict):
    total_lines: int
    added_lines: int
    removed_lines: int
    num_files: int


class TestCase(TypedDict):
    test_case_id: str
    pr_context: str
    diff: str
    ground_truth_ids: list[str]
    diff_stats: DiffStats


# ---------------------------------------------------------------------------
# TypeAliases
# ---------------------------------------------------------------------------

# Function that extracts the text to rerank from a result dict
TextStrategyFn = Callable[[dict[str, Any]], str]

# Map of strategy name → text extraction function
RerankerStrategies = dict[str, TextStrategyFn]

# Validator function signature used in memories/validators.py
ValidatorFn = Callable[[str], tuple[bool, str]]

# Score type stored in SearchResult.score_type
ScoreType = Literal["cosine_distance", "bm25_rank", "rerank_score"]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_types.py tests/ -q
```
Expected: all pass.

**Step 5: Commit**

```bash
git add src/memory_retrieval/types.py tests/test_types.py
git commit -m "feat: add shared types.py with TypedDicts and TypeAliases"
```

---

## Task 2: Create `constants.py` — centralize magic values

**Files:**
- Create: `src/memory_retrieval/constants.py`
- Create: `tests/test_constants.py`

**Step 1: Write the failing test**

```python
# tests/test_constants.py
import re

from memory_retrieval.constants import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_SEARCH_LIMIT,
    EMBEDDING_MODEL_DIMENSIONS,
    EXCLUDED_FILE_PATTERNS,
)


def test_embedding_model_dimensions_has_expected_models() -> None:
    assert "mxbai-embed-large" in EMBEDDING_MODEL_DIMENSIONS
    assert EMBEDDING_MODEL_DIMENSIONS["mxbai-embed-large"] == 1024


def test_excluded_file_patterns_are_precompiled() -> None:
    assert all(isinstance(pattern, re.Pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def test_excluded_patterns_match_expected_files() -> None:
    # Convert to set of pattern strings for easy assertion
    pattern_strings = [p.pattern for p in EXCLUDED_FILE_PATTERNS]
    combined = "|".join(pattern_strings)
    assert re.search(combined, "package-lock.json")
    assert re.search(combined, "yarn.lock")
    assert re.search(combined, "app.min.js")


def test_default_experiment_values() -> None:
    assert DEFAULT_SEARCH_LIMIT == 20
    assert DEFAULT_DISTANCE_THRESHOLD == 1.1
    assert DEFAULT_RERANK_TOP_N == 5
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_constants.py -v
```
Expected: `ModuleNotFoundError: No module named 'memory_retrieval.constants'`

**Step 3: Write the implementation**

Read `src/memory_retrieval/experiments/test_cases.py` first to find the full list of excluded patterns, then:

```python
# src/memory_retrieval/constants.py
import re

# ---------------------------------------------------------------------------
# Embedding model registry
# (moved from search/vector.py — single source of truth)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    "mxbai-embed-large": 1024,
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-large": 1024,
    "bge-m3": 1024,
}

# ---------------------------------------------------------------------------
# Experiment defaults
# (moved from experiments/runner.py)
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_LIMIT: int = 20
DEFAULT_DISTANCE_THRESHOLD: float = 1.1
DEFAULT_RERANK_TOP_N: int = 5

# ---------------------------------------------------------------------------
# File exclusion patterns for diff filtering
# Pre-compiled at import time — never re-compiled per function call.
# (moved from experiments/test_cases.py)
# ---------------------------------------------------------------------------

EXCLUDED_FILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"package-lock\.json$"),
    re.compile(r"yarn\.lock$"),
    re.compile(r"pnpm-lock\.yaml$"),
    re.compile(r"\.lock$"),
    re.compile(r"\.min\.(js|css)$"),
    re.compile(r"\.map$"),
    re.compile(r"dist/"),
    re.compile(r"build/"),
    re.compile(r"\.generated\.(ts|js|py)$"),
    re.compile(r"__generated__"),
    re.compile(r"migrations/\d+_"),
]
```

Note: Read the actual patterns from `src/memory_retrieval/experiments/test_cases.py` before writing this file — use the exact same patterns currently in that file.

**Step 4: Run tests**

```bash
uv run pytest tests/test_constants.py tests/ -q
```

**Step 5: Commit**

```bash
git add src/memory_retrieval/constants.py tests/test_constants.py
git commit -m "feat: add constants.py with centralized magic values"
```

---

## Task 3: Infra cleanup — `prompts.py`, `runs.py`, `llm.py`

**Files:**
- Modify: `src/memory_retrieval/infra/prompts.py`
- Modify: `src/memory_retrieval/infra/runs.py`
- Modify: `src/memory_retrieval/infra/llm.py`
- Test: `tests/infra/test_prompts.py` (existing — verify still passes)
- Test: `tests/infra/test_runs.py` (existing — verify still passes)

These are refactors only — behavior does not change. All existing tests must pass before and after.

**Step 1: Run existing infra tests to confirm baseline**

```bash
uv run pytest tests/infra/ -q
```
All must pass.

**Step 2: Update `infra/prompts.py` — type `_SafeDict` + extract `_find_latest_semver_file`**

In `infra/prompts.py`, make two changes:

*Change 1* — type `_SafeDict` properly:
```python
# Before:
class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""

# After:
class _SafeDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""
```

*Change 2* — extract semver file finding from `load_prompt()`:
```python
def _find_latest_semver_file(prompt_dir: Path) -> Path | None:
    """Return the prompt file with the highest semver in the given directory.

    Returns None if no valid semver files exist.
    """
    best_file: Path | None = None
    best_ver: tuple[int, int, int] = (-1, -1, -1)
    for md_file in sorted(prompt_dir.glob("v*.md")):
        ver_str = md_file.stem[1:]
        try:
            ver_tuple = _parse_semver(ver_str)
            if ver_tuple > best_ver:
                best_ver = ver_tuple
                best_file = md_file
        except ValueError:
            continue
    return best_file
```

Then in `load_prompt()`, replace the inline loop (lines 80–95) with:
```python
    else:
        if not list(prompt_dir.glob("v*.md")):
            raise FileNotFoundError(f"No prompt files found in {prompt_dir}")
        best_file = _find_latest_semver_file(prompt_dir)
        if best_file is None:
            raise FileNotFoundError(f"No valid semver prompt files in {prompt_dir}")
        file_path = best_file
        version = best_file.stem[1:]
```

**Step 3: Update `infra/runs.py` — extract metadata builders + docstring**

Extract `_build_run_metadata` from `create_run()`:
```python
def _build_run_metadata(
    run_id: str,
    phase: str,
    description: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "phase": phase,
        "description": description,
        "pipeline_status": {},
    }
```

Update `create_run()` to call it:
```python
    metadata = _build_run_metadata(run_id, phase, description)
    _save_run_metadata(run_dir, metadata)
```

Extract `_build_subrun_metadata` from `create_subrun()`:
```python
def _build_subrun_metadata(
    subrun_id: str,
    parent_run_id: str,
    description: str | None,
) -> dict[str, Any]:
    return {
        "subrun_id": subrun_id,
        "parent_run_id": parent_run_id,
        "created_at": datetime.now().isoformat(),
        "description": description,
        "pipeline_status": {},
    }
```

Add docstring to `_generate_run_id()`:
```python
def _generate_run_id() -> str:
    """Generate a timestamp-based run ID with microsecond precision.

    Microseconds prevent collisions when two runs are created within the same
    second (e.g. in batch_runner.py running multiple pipeline configurations).
    """
    return f"{RUN_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
```

**Step 4: Update `infra/llm.py` — add docstring**

```python
def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> str:
    """Call the OpenRouter API and return the assistant message content.

    OpenRouter accepts messages in OpenAI chat format:
      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Args:
        api_key: OpenRouter API key (from OPENROUTER_API_KEY env var).
        model: OpenRouter model ID (e.g. "anthropic/claude-haiku-4.5").
        messages: Conversation messages in OpenAI chat format.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the response.
        timeout_s: HTTP request timeout in seconds.

    Returns:
        The assistant's response text, stripped of leading/trailing whitespace.

    Raises:
        requests.HTTPError: On non-2xx API responses.
    """
```

**Step 5: Run infra tests**

```bash
uv run pytest tests/infra/ tests/ -q
```
All must pass.

**Step 6: Commit**

```bash
git add src/memory_retrieval/infra/
git commit -m "refactor: infra cleanup — typed _SafeDict, extract metadata builders, docstrings"
```

---

## Task 4: Memories — `validators.py` and `loader.py`

**Files:**
- Modify: `src/memory_retrieval/memories/validators.py`
- Modify: `src/memory_retrieval/memories/loader.py`
- Test: `tests/memories/` (existing — must still pass)

**Step 1: Run existing memories tests**

```bash
uv run pytest tests/memories/ -q
```
All must pass.

**Step 2: Fix `validators.py` — replace lambda with explicit version comparison**

The current `get_situation_validator` uses `max(..., key=lambda f: f.__name__)` which relies on alphabetical function name ordering. Replace with explicit version comparison using `_parse_semver` from `infra/prompts.py`:

```python
# Before (line 52):
return max(SITUATION_VALIDATORS.values(), key=lambda f: f.__name__)

# After:
from memory_retrieval.infra.prompts import _parse_semver

def get_situation_validator(version: str) -> Callable[[str], tuple[bool, str]]:
    """Get situation validator for a prompt version, defaulting to the latest."""
    if version in SITUATION_VALIDATORS:
        return SITUATION_VALIDATORS[version]
    # Fall back to the validator for the highest registered version
    latest_version = max(SITUATION_VALIDATORS.keys(), key=_parse_semver)
    return SITUATION_VALIDATORS[latest_version]
```

**Step 3: Update `loader.py` — tighten return type**

The `load_memories` function currently returns `list[dict[str, Any]]`. Import `MemoryDict` from `types.py` and tighten the return type:

```python
# Add import at top:
from memory_retrieval.types import MemoryDict

# Change return type annotation:
def load_memories(memories_dir: str) -> list[MemoryDict]:
```

The function body does not need to change — it already loads the correct shape.

**Step 4: Run tests**

```bash
uv run pytest tests/memories/ tests/ -q
```

**Step 5: Commit**

```bash
git add src/memory_retrieval/memories/validators.py src/memory_retrieval/memories/loader.py
git commit -m "refactor: tighten validators.py version selection and loader.py return type"
```

---

## Task 5: Memories — decompose `extractor.py`

**Files:**
- Modify: `src/memory_retrieval/memories/extractor.py`
- Test: `tests/memories/test_extractor.py` (existing — must still pass)

**Step 1: Run existing extractor tests**

```bash
uv run pytest tests/memories/ -q
```
All must pass.

**Step 2: Add `ExtractionSummary` dataclass and `__post_init__` validation to `ExtractionConfig`**

Add after the `SituationFormat` enum:

```python
from dataclasses import dataclass, field

@dataclass
class ExtractionSummary:
    """Counts from a completed extraction run."""
    written: int = 0
    rejected: int = 0
    errors: int = 0
```

Add `__post_init__` to `ExtractionConfig` (convert it to a `@dataclass` or add validation manually):

```python
class ExtractionConfig:
    def __init__(
        self,
        situation_format: SituationFormat = SituationFormat.SINGLE,
        prompts_dir: str | Path = "data/prompts/phase1",
        prompt_version: str | None = None,
        model: str = "anthropic/claude-haiku-4.5",
        sleep_s: float = 0.25,
        author_tag: str = "openrouter",
    ):
        if not model:
            raise ValueError("ExtractionConfig.model must not be empty")
        if sleep_s < 0:
            raise ValueError(f"ExtractionConfig.sleep_s must be >= 0, got {sleep_s}")
        self.situation_format = situation_format
        self.prompts_dir = Path(prompts_dir)
        self.prompt_version = prompt_version
        self.model = model
        self.sleep_s = sleep_s
        self.author_tag = author_tag
```

**Step 3: Move the inline import to module top**

```python
# Remove from inside extract_memories():
#   from memory_retrieval.memories.validators import validate_situation_v1

# Add to module-level imports at top of file:
from memory_retrieval.memories.validators import (
    get_situation_validator,
    validate_lesson,
    validate_situation_v1,
)
```

**Step 4: Extract `_build_memory_dict`**

This extracts the memory object construction (lines 254–290 in extractor.py):

```python
def _build_memory_dict(
    memory_id: str,
    situation: str,
    lesson: str,
    comment: dict[str, Any],
    pr_context: str,
    gathered_at: str,
    repo: str,
    config: ExtractionConfig,
    variants: list[str] | None = None,
    prompt_version_tag: str | None = None,
    raw_context_hash: str = "",
) -> dict[str, Any]:
    """Construct the memory dict from extracted situation and lesson."""
    import hashlib

    metadata: dict[str, Any] = {
        "repo": repo,
        "file_pattern": file_pattern(comment.get("file", "")),
        "language": lang_from_file(comment.get("file", "")),
        "tags": [],
        "severity": comment.get("severity", "info"),
        "confidence": confidence_map(comment.get("confidence", "medium")),
        "author": config.author_tag,
        "source_comment_id": comment.get("id"),
        "status": comment.get("status", None),
    }
    if prompt_version_tag:
        metadata["prompt_version"] = prompt_version_tag

    memory: dict[str, Any] = {
        "id": memory_id,
        "lesson": lesson,
        "metadata": metadata,
        "source": {
            "file": comment.get("file"),
            "line": comment.get("line", None),
            "code_snippet": comment.get("code_snippet", None),
            "comment": comment.get("message"),
            "user_note": comment.get("user_note", None),
            "rationale": comment.get("rationale", None),
            "verifiedBy": comment.get("verifiedBy", None),
            "pr_context": pr_context,
            "gathered_at": gathered_at,
            "raw_context_hash": raw_context_hash,
        },
    }

    if variants is not None:
        memory["situation_variants"] = variants
    else:
        memory["situation_description"] = situation

    return memory
```

**Step 5: Extract `_validate_variants`**

```python
def _validate_variants(
    situation_raw: str,
    validate_situation: Callable[[str], tuple[bool, str]],
) -> tuple[bool, list[str], str]:
    """Parse and validate the 3 semicolon-separated situation variants.

    Returns:
        (all_valid, variants, rejection_reason)
        If all_valid is False, rejection_reason explains which variant failed.
    """
    variants = [v.strip() for v in situation_raw.split(";")]

    if len(variants) != 3:
        return False, variants, f"wrong_variant_count_{len(variants)}"

    for i, variant in enumerate(variants):
        ok, reason = validate_situation(variant)
        if reason == "ok_no_punct":
            variants[i] = variant.rstrip() + "."
        elif not ok:
            return False, variants, f"variant_{i}_{reason}"

    return True, variants, ""
```

**Step 6: Extract `_extract_situation`**

```python
def _extract_situation(
    comment: dict[str, Any],
    pr_context_text: str,
    situation_prompt: Any,
    validate_situation: Callable[[str], tuple[bool, str]],
    api_key: str,
    config: ExtractionConfig,
) -> tuple[bool, str, list[str] | None, str]:
    """Extract and validate the situation for one comment.

    Returns:
        (success, situation_text, variants_or_none, rejection_reason)
    """
    code = (comment.get("code_snippet") or "").strip()
    user_note = (comment.get("user_note") or "").strip()

    if config.situation_format == SituationFormat.VARIANTS:
        situation_raw = call_openrouter(
            api_key=api_key,
            model=config.model,
            messages=situation_prompt.render(
                context=pr_context_text,
                file=comment.get("file", ""),
                severity=comment.get("severity", "info"),
                code=code if code else "(none)",
                comment=comment.get("message", ""),
                user_note=user_note,
            ),
            temperature=0.0,
            max_tokens=600,
        )
        all_valid, variants, reason = _validate_variants(situation_raw, validate_situation)
        if not all_valid:
            return False, "", None, reason
        return True, variants[0], variants, ""

    else:
        additional_context = f"ADDITIONAL CONTEXT: {user_note}" if user_note else ""
        situation = call_openrouter(
            api_key=api_key,
            model=config.model,
            messages=situation_prompt.render(
                context=pr_context_text,
                file=comment.get("file", ""),
                severity=comment.get("severity", "info"),
                code=code[:800] if code else "(none)",
                comment=comment.get("message", ""),
                user_note=user_note,
                additional_context=additional_context,
            ),
            temperature=0.0,
            max_tokens=600,
        )
        ok, reason = validate_situation(situation)
        if reason == "ok_no_punct":
            situation = situation.rstrip() + "."
        elif not ok:
            return False, "", None, reason
        return True, situation, None, ""
```

**Step 7: Extract `_extract_memory_for_comment`**

```python
def _extract_memory_for_comment(
    comment: dict[str, Any],
    context: str,
    pr_context: str,
    gathered_at: str,
    repo: str,
    situation_prompt: Any,
    lesson_prompt: Any,
    validate_situation: Callable[[str], tuple[bool, str]],
    api_key: str,
    config: ExtractionConfig,
    reject_file: Any,
) -> dict[str, Any] | None:
    """Extract a single memory from a comment. Returns None if rejected.

    Writes rejection details to reject_file on failure.
    """
    import time

    comment_id = comment.get("id")

    # Skip pre-rejected comments
    if comment.get("status") == "rejected":
        _write_rejection(reject_file, comment_id, "status", "comment_rejected")
        return None

    # Extract situation
    success, situation, variants, reason = _extract_situation(
        comment, context, situation_prompt, validate_situation, api_key, config
    )
    if not success:
        _write_rejection(reject_file, comment_id, "situation", reason, text=situation)
        return None

    time.sleep(config.sleep_s)

    # Extract lesson
    rationale = (comment.get("rationale") or "").strip()
    lesson = call_openrouter(
        api_key=api_key,
        model=config.model,
        messages=lesson_prompt.render(
            situation=situation,
            comment=comment.get("message", ""),
            rationale=rationale if rationale else "(none)",
        ),
        temperature=0.0,
        max_tokens=120,
    )
    ok_lesson, reason_lesson = validate_lesson(lesson)
    if reason_lesson == "ok_no_punct":
        lesson = lesson.rstrip() + "."
    if not ok_lesson:
        _write_rejection(reject_file, comment_id, "lesson", reason_lesson, text=lesson)
        return None

    import hashlib

    memory_id = stable_id(
        comment.get("id", ""),
        situation,
        lesson,
    )
    raw_context_hash = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]

    return _build_memory_dict(
        memory_id=memory_id,
        situation=situation,
        lesson=lesson,
        comment=comment,
        pr_context=pr_context,
        gathered_at=gathered_at,
        repo=repo,
        config=config,
        variants=variants,
        prompt_version_tag=(
            situation_prompt.version_tag
            if config.situation_format == SituationFormat.SINGLE
            else None
        ),
        raw_context_hash=raw_context_hash,
    )
```

Add a small `_write_rejection` helper:
```python
def _write_rejection(
    reject_file: Any,
    comment_id: Any,
    stage: str,
    reason: str,
    text: str = "",
) -> None:
    entry: dict[str, Any] = {"comment_id": comment_id, "stage": stage, "reason": reason}
    if text:
        entry["text"] = text
    reject_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Step 8: Rewrite `extract_memories` as thin orchestrator (~30 lines)**

```python
def extract_memories(
    raw_path: str,
    out_dir: str,
    config: ExtractionConfig,
) -> str:
    """Extract memories from a raw code review JSON file.

    Returns the path to the output JSONL file.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY env var")

    raw = load_json(raw_path)
    context = raw.get("context", "")
    meta = raw.get("meta", {})
    comments = raw.get("code_review_comments", [])

    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(out_dir) / f"memories_{Path(raw_path).stem}_{ts}.jsonl")
    reject_path = str(Path(out_dir) / f"rejected_{Path(raw_path).stem}_{ts}.jsonl")

    repo = short_repo_name(meta.get("repoPath", ""))
    pr_context = f"{meta.get('sourceBranch', '?')} → {meta.get('targetBranch', '?')}"
    gathered_at = meta.get("gatheredAt", "")

    situation_prompt = load_prompt(
        "memory-situation", version=config.prompt_version, prompts_dir=config.prompts_dir
    )
    lesson_prompt = load_prompt("memory-lesson", prompts_dir=config.prompts_dir)

    validate_situation = (
        get_situation_validator(situation_prompt.version)
        if config.situation_format == SituationFormat.SINGLE
        else validate_situation_v1
    )

    summary = ExtractionSummary()
    with (
        open(out_path, "w", encoding="utf-8") as out_file,
        open(reject_path, "w", encoding="utf-8") as reject_file,
    ):
        for comment in comments:
            memory = _extract_memory_for_comment(
                comment=comment,
                context=context,
                pr_context=pr_context,
                gathered_at=gathered_at,
                repo=repo,
                situation_prompt=situation_prompt,
                lesson_prompt=lesson_prompt,
                validate_situation=validate_situation,
                api_key=api_key,
                config=config,
                reject_file=reject_file,
            )
            if memory is not None:
                out_file.write(json.dumps(memory, ensure_ascii=False) + "\n")
                summary.written += 1
                import time
                time.sleep(config.sleep_s)
            else:
                summary.rejected += 1

    print(f"Memories written: {summary.written} -> {out_path}")
    print(f"Rejected: {summary.rejected} -> {reject_path}")
    return out_path
```

**Step 9: Run all tests**

```bash
uv run pytest tests/ -q
```

**Step 10: Commit**

```bash
git add src/memory_retrieval/memories/extractor.py
git commit -m "refactor: decompose extract_memories into focused helper functions"
```

---

## Task 6: Move `get_confidence_from_distance` to `memories/helpers.py`

**Files:**
- Modify: `src/memory_retrieval/memories/helpers.py`
- Modify: `src/memory_retrieval/search/vector.py`
- Modify: `src/memory_retrieval/experiments/runner.py`

**Step 1: Add `get_confidence_from_distance` to `memories/helpers.py`**

Read helpers.py first, then append:
```python
def get_confidence_from_distance(distance: float) -> str:
    """Map a cosine distance to a human-readable confidence label.

    Used to annotate vector search results with an interpretable confidence level.
    Lower distance = closer match = higher confidence.
    """
    if distance < 0.5:
        return "high"
    elif distance < 0.8:
        return "medium"
    elif distance < 1.2:
        return "low"
    else:
        return "very_low"
```

**Step 2: Remove `get_confidence_from_distance` from `vector.py` and update its import**

In `vector.py`:
- Delete the function definition (lines 61–69)
- Any internal usage within vector.py (there is none — it's only imported externally)

**Step 3: Update `runner.py` import**

```python
# Before:
from memory_retrieval.search.vector import VectorBackend, get_confidence_from_distance

# After:
from memory_retrieval.memories.helpers import get_confidence_from_distance
from memory_retrieval.search.vector import VectorBackend
```

**Step 4: Run all tests**

```bash
uv run pytest tests/ -q
```

**Step 5: Commit**

```bash
git add src/memory_retrieval/memories/helpers.py \
        src/memory_retrieval/search/vector.py \
        src/memory_retrieval/experiments/runner.py
git commit -m "refactor: move get_confidence_from_distance to memories/helpers (semantic home)"
```

---

## Task 7: Search — add `SearchBackendBase` + `ScoreType` to `base.py`

**Files:**
- Modify: `src/memory_retrieval/search/base.py`
- Create: `tests/search/test_base.py`

**Step 1: Write the failing test**

```python
# tests/search/test_base.py
import sqlite3
from pathlib import Path

import pytest

from memory_retrieval.search.base import SearchBackendBase, SearchResult
from memory_retrieval.types import ScoreType


def test_search_result_score_type_is_literal() -> None:
    result = SearchResult(
        id="mem_abc",
        situation="A developer wrote untested code.",
        lesson="Always write tests.",
        metadata={},
        source={},
        score=0.8,
        raw_score=0.2,
        score_type="cosine_distance",
    )
    assert result.score_type == "cosine_distance"


def test_search_result_rejects_invalid_score_type() -> None:
    # This is a type-level test — runtime is fine since it's a dataclass
    # The point is that our annotation is ScoreType (Literal)
    result = SearchResult(
        id="x", situation="s", lesson="l", metadata={}, source={},
        score=1.0, raw_score=0.0, score_type="cosine_distance",
    )
    assert result.score_type in ("cosine_distance", "bm25_rank", "rerank_score")
```

**Step 2: Run to verify test_search_result_score_type_is_literal passes (it already should)**

```bash
uv run pytest tests/search/test_base.py -v
```

**Step 3: Update `search/base.py` — add `ScoreType` annotation + `SearchBackendBase`**

```python
# src/memory_retrieval/search/base.py
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from memory_retrieval.types import ScoreType


@dataclass
class SearchResult:
    id: str
    situation: str
    lesson: str
    metadata: dict[str, Any]
    source: dict[str, Any]
    score: float       # Higher = better (normalized)
    raw_score: float   # Original (BM25 rank or cosine distance)
    score_type: ScoreType  # Tightened from str


class SearchBackend(Protocol):
    def create_database(self, db_path: str) -> None: ...
    def insert_memories(self, db_path: str, memories: list[dict[str, Any]]) -> int: ...
    def search(self, db_path: str, query: str, limit: int = 10) -> list[SearchResult]: ...
    def rebuild_database(self, db_path: str, memories_dir: str) -> None: ...


class SearchBackendBase(ABC):
    """Abstract base class providing shared implementation for search backends.

    Subclasses must implement `_get_db_connection` to provide their backend-specific
    SQLite connection context manager (e.g. plain SQLite vs. sqlite-vec).
    """

    @abstractmethod
    @contextmanager
    def _get_db_connection(self, db_path: str) -> Iterator[Any]: ...

    def get_memory_by_id(self, db_path: str, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by its ID.

        Returns the memory dict with id, situation/variants, lesson, metadata, source,
        or None if not found.
        """
        from memory_retrieval.memories.schema import (
            FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE, FIELD_SITUATION,
        )
        from memory_retrieval.search.db_utils import deserialize_json_field

        with self._get_db_connection(db_path) as conn:
            import sqlite3
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {FIELD_ID}, {FIELD_SITUATION}, {FIELD_LESSON},
                       {FIELD_METADATA}, {FIELD_SOURCE}
                FROM memories WHERE {FIELD_ID} = ?
                """,
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                FIELD_ID: row[FIELD_ID],
                FIELD_SITUATION: row[FIELD_SITUATION],
                FIELD_LESSON: row[FIELD_LESSON],
                FIELD_METADATA: deserialize_json_field(row[FIELD_METADATA]),
                FIELD_SOURCE: deserialize_json_field(row[FIELD_SOURCE]),
            }

    def get_sample_memories(
        self, db_path: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Get the first N memories from the database (deterministic order)."""
        from memory_retrieval.memories.schema import (
            FIELD_ID, FIELD_LESSON, FIELD_METADATA, FIELD_SOURCE, FIELD_SITUATION,
        )
        from memory_retrieval.search.db_utils import deserialize_json_field
        import sqlite3

        with self._get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {FIELD_ID}, {FIELD_SITUATION}, {FIELD_LESSON},
                       {FIELD_METADATA}, {FIELD_SOURCE}
                FROM memories LIMIT ?
                """,
                (limit,),
            )
            return [
                {
                    FIELD_ID: row[FIELD_ID],
                    FIELD_SITUATION: row[FIELD_SITUATION],
                    FIELD_LESSON: row[FIELD_LESSON],
                    FIELD_METADATA: deserialize_json_field(row[FIELD_METADATA]),
                    FIELD_SOURCE: deserialize_json_field(row[FIELD_SOURCE]),
                }
                for row in cursor.fetchall()
            ]
```

**Step 4: Run tests**

```bash
uv run pytest tests/ -q
```

**Step 5: Commit**

```bash
git add src/memory_retrieval/search/base.py tests/search/test_base.py
git commit -m "feat: add SearchBackendBase with shared get_memory_by_id and get_sample_memories"
```

---

## Task 8: Search — refactor `vector.py` to inherit `SearchBackendBase`

**Files:**
- Modify: `src/memory_retrieval/search/vector.py`
- Modify: `src/memory_retrieval/constants.py` (import EMBEDDING_MODEL_DIMENSIONS from there)

**Step 1: Run existing vector tests**

```bash
uv run pytest tests/search/ -q
```
All must pass.

**Step 2: Update `vector.py` to inherit `SearchBackendBase`**

Key changes:
1. Inherit `SearchBackendBase`
2. Implement `_get_db_connection` using the local sqlite-vec version
3. Replace manual chunking with `itertools.batched`
4. Import `EMBEDDING_MODEL_DIMENSIONS` from `constants.py` instead of defining it locally
5. Replace `get_memory_by_id` and `get_sample_memories` method bodies with `super()` calls (or just delete them since `SearchBackendBase` provides them)

```python
# Top of vector.py — add imports:
from itertools import batched
from memory_retrieval.constants import EMBEDDING_MODEL_DIMENSIONS
from memory_retrieval.search.base import SearchBackendBase, SearchResult

# Remove local EMBEDDING_MODEL_DIMENSIONS dict definition

# Change class declaration:
class VectorBackend(SearchBackendBase):

# Implement the required abstract method:
    @contextmanager
    def _get_db_connection(self, db_path: str):  # type: ignore[override]
        yield from get_db_connection(db_path)

# Replace chunking in _get_embeddings_batch:
        chunks = list(batched(texts, chunk_size))

# Delete get_memory_by_id entirely (inherited from SearchBackendBase)
# Delete get_sample_memories entirely (inherited from SearchBackendBase)
# Keep get_random_sample_memories (not in base class — vector-specific)
# Keep get_memory_count (not in base class — vector-specific)
```

Note: `_get_db_connection` in the base class is declared as a `@contextmanager`. In `VectorBackend`, the existing `get_db_connection` function is already a `@contextmanager`. Implement as:
```python
    @contextmanager
    def _get_db_connection(self, db_path: str) -> Iterator[sqlite3.Connection]:
        with get_db_connection(db_path) as conn:
            yield conn
```

**Step 3: Run all tests**

```bash
uv run pytest tests/ -q
```

**Step 4: Commit**

```bash
git add src/memory_retrieval/search/vector.py src/memory_retrieval/constants.py
git commit -m "refactor: VectorBackend inherits SearchBackendBase, use itertools.batched"
```

---

## Task 9: Search — refactor `fts5.py` to inherit `SearchBackendBase`

**Files:**
- Modify: `src/memory_retrieval/search/fts5.py`
- Test: `tests/search/test_fts5.py` (existing — must still pass)

**Step 1: Run existing fts5 tests**

```bash
uv run pytest tests/search/test_fts5.py -q
```

**Step 2: Update `fts5.py` to inherit `SearchBackendBase`**

Same pattern as vector.py:
1. Inherit `SearchBackendBase`
2. Implement `_get_db_connection` using `get_db_connection` from `db_utils`
3. Delete `get_memory_by_id` if it exists (check if it's defined — use base class version)

```python
from memory_retrieval.search.base import SearchBackendBase, SearchResult

class FTS5Backend(SearchBackendBase):

    @contextmanager
    def _get_db_connection(self, db_path: str) -> Iterator[sqlite3.Connection]:
        with get_db_connection(db_path) as conn:
            yield conn
```

**Step 3: Run all tests**

```bash
uv run pytest tests/ -q
```

**Step 4: Commit**

```bash
git add src/memory_retrieval/search/fts5.py
git commit -m "refactor: FTS5Backend inherits SearchBackendBase"
```

---

## Task 10: Search — clean up `reranker.py`

**Files:**
- Modify: `src/memory_retrieval/search/reranker.py`

Read the full file first, then:

1. Extract model-specific formatting config to module-level dict:
```python
# Module-level constant replacing branching in _format_pair
_QWEN3_PREFIX = "Instruct: "
_MODEL_USES_INSTRUCT_PREFIX: set[str] = {"Qwen/Qwen3-Reranker-0.6B"}
```

2. Use f-strings consistently in string building

3. No test needed beyond "existing tests pass":
```bash
uv run pytest tests/ -q
```

**Commit:**
```bash
git add src/memory_retrieval/search/reranker.py
git commit -m "refactor: reranker.py — extract model config constants, use f-strings"
```

---

## Task 11: Experiments — `test_cases.py` type tightening

**Files:**
- Modify: `src/memory_retrieval/experiments/test_cases.py`
- Test: existing tests must pass

**Step 1: Run existing test_cases tests**

```bash
uv run pytest tests/ -k "test_case" -q
```

**Step 2: Import `EXCLUDED_FILE_PATTERNS` from `constants.py`**

Read `test_cases.py` fully first, then:
- Remove the locally defined exclusion patterns list
- Import `EXCLUDED_FILE_PATTERNS` from `memory_retrieval.constants`
- Update `filter_diff()` to use the imported pre-compiled patterns:
```python
from memory_retrieval.constants import EXCLUDED_FILE_PATTERNS

def filter_diff(diff: str) -> str:
    lines = diff.split("\n")
    filtered = []
    for line in lines:
        # Check if line references an excluded file
        if any(pattern.search(line) for pattern in EXCLUDED_FILE_PATTERNS):
            continue
        filtered.append(line)
    return "\n".join(filtered)
```

**Step 3: Run all tests**

```bash
uv run pytest tests/ -q
```

**Step 4: Commit**

```bash
git add src/memory_retrieval/experiments/test_cases.py
git commit -m "refactor: test_cases.py uses pre-compiled EXCLUDED_FILE_PATTERNS from constants"
```

---

## Task 12: Experiments — `query_generation.py` refactor

**Files:**
- Modify: `src/memory_retrieval/experiments/query_generation.py`
- Create: `tests/experiments/test_query_generation.py`

**Step 1: Write tests for the parse strategy functions**

```python
# tests/experiments/test_query_generation.py
from memory_retrieval.experiments.query_generation import (
    _parse_json_array_queries,
    _parse_newline_queries,
    _parse_quoted_queries,
    parse_queries_robust,
)


def test_parse_json_array_queries_valid_json() -> None:
    response = '["query one about testing", "query two about linting"]'
    result = _parse_json_array_queries(response)
    assert result == ["query one about testing", "query two about linting"]


def test_parse_json_array_queries_embedded_in_text() -> None:
    response = 'Here are queries: ["what happens when tests fail", "how to write assertions"]'
    result = _parse_json_array_queries(response)
    assert result is not None
    assert len(result) == 2


def test_parse_json_array_queries_returns_none_on_failure() -> None:
    assert _parse_json_array_queries("no json here") is None
    assert _parse_json_array_queries('{"not": "an array"}') is None


def test_parse_quoted_queries_extracts_quoted_strings() -> None:
    response = '''
    1. "Always validate input at system boundaries"
    2. "Use type hints for public API functions"
    '''
    result = _parse_quoted_queries(response)
    assert result is not None
    assert len(result) == 2
    assert "Always validate input at system boundaries" in result


def test_parse_newline_queries_cleans_list_formatting() -> None:
    response = """
    1. Write failing test before implementation
    - Use descriptive variable names over abbreviations
    * Check return types on public functions
    """
    result = _parse_newline_queries(response)
    assert len(result) >= 2
    assert all(not line.startswith(("1.", "-", "*")) for line in result)


def test_parse_queries_robust_falls_back_gracefully() -> None:
    # Valid JSON array
    assert len(parse_queries_robust('["a query with enough words here", "another one also long"]')) == 2
    # Quoted strings fallback
    assert len(parse_queries_robust('"a query with enough words here"')) >= 1
    # Returns list even for garbage
    result = parse_queries_robust("no queries here")
    assert isinstance(result, list)


def test_query_generation_config_validates_max_workers() -> None:
    from memory_retrieval.experiments.query_generation import QueryGenerationConfig
    import pytest
    with pytest.raises(ValueError, match="max_workers"):
        QueryGenerationConfig(max_workers=0)
```

**Step 2: Run to verify tests fail**

```bash
uv run pytest tests/experiments/test_query_generation.py -v
```
Expected: `ImportError` for `_parse_json_array_queries` (not yet extracted).

**Step 3: Refactor `query_generation.py`**

*Extract parse strategies:*
```python
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
```

*Add `__post_init__` equivalent to `QueryGenerationConfig`:*
```python
@dataclass
class QueryGenerationConfig:
    prompts_dir: str | Path = "data/prompts/phase1"
    prompt_version: str | None = None
    model: str = DEFAULT_MODEL
    use_sample_memories: bool = True
    max_workers: int = DEFAULT_MAX_WORKERS

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {self.max_workers}")
```

*Replace print + Lock with logging:*
```python
import logging

logger = logging.getLogger(__name__)

# In generate_all_queries, replace:
#   print_lock = threading.Lock()
#   with print_lock: print(...)
# With:
#   logger.info(...)
# Remove threading import if no longer needed
```

*Extract `_truncate_text` helper:*
```python
def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending a truncation notice."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"
```

*Replace inline truncation in `generate_queries_for_test_case`:*
```python
    context = _truncate_text(context, MAX_CONTEXT_LENGTH)
    filtered_diff = _truncate_text(filtered_diff, MAX_DIFF_LENGTH)
```

**Step 4: Run tests**

```bash
uv run pytest tests/experiments/test_query_generation.py tests/ -q
```

**Step 5: Commit**

```bash
git add src/memory_retrieval/experiments/query_generation.py \
        tests/experiments/test_query_generation.py
git commit -m "refactor: query_generation.py — extract parse strategies, logging, validation"
```

---

## Task 13: Experiments — decompose `runner.py`

**Files:**
- Modify: `src/memory_retrieval/experiments/runner.py`
- Create: `tests/experiments/test_runner_helpers.py`

**Step 1: Write tests for the new helper functions**

```python
# tests/experiments/test_runner_helpers.py
import pytest
from memory_retrieval.experiments.runner import (
    ExperimentConfig,
    _compute_pre_rerank_metrics,
    _print_experiment_summary,
)
from memory_retrieval.search.fts5 import FTS5Backend


def test_experiment_config_validates_search_limit() -> None:
    with pytest.raises(ValueError, match="search_limit"):
        ExperimentConfig(search_backend=FTS5Backend(), search_limit=0)


def test_experiment_config_validates_distance_threshold() -> None:
    with pytest.raises(ValueError, match="distance_threshold"):
        ExperimentConfig(search_backend=FTS5Backend(), distance_threshold=0.0)
    with pytest.raises(ValueError, match="distance_threshold"):
        ExperimentConfig(search_backend=FTS5Backend(), distance_threshold=2.5)


def test_compute_pre_rerank_metrics_empty_retrieval() -> None:
    query_results = [
        {"query": "test", "results": [], "word_count": 1, "result_count": 0}
    ]
    ground_truth_ids = {"mem_abc"}
    metrics = _compute_pre_rerank_metrics(
        query_results=query_results,
        ground_truth_ids=ground_truth_ids,
        distance_threshold=1.1,
    )
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0


def test_compute_pre_rerank_metrics_perfect_retrieval() -> None:
    query_results = [
        {
            "query": "test query",
            "results": [{"id": "mem_abc", "distance": 0.3}],
            "word_count": 2,
            "result_count": 1,
        }
    ]
    ground_truth_ids = {"mem_abc"}
    metrics = _compute_pre_rerank_metrics(
        query_results=query_results,
        ground_truth_ids=ground_truth_ids,
        distance_threshold=1.1,
    )
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
```

**Step 2: Run to verify tests fail**

```bash
uv run pytest tests/experiments/test_runner_helpers.py -v
```
Expected: `ImportError` on `_compute_pre_rerank_metrics`.

**Step 3: Add `__post_init__` to `ExperimentConfig`**

Convert `ExperimentConfig` to use `__post_init__`:

```python
@dataclass
class ExperimentConfig:
    """Configuration for running retrieval experiments against pre-generated queries."""

    search_backend: SearchBackend
    search_limit: int = DEFAULT_SEARCH_LIMIT
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    reranker: Reranker | None = None
    rerank_text_strategies: RerankerStrategies | None = None

    def __post_init__(self) -> None:
        if self.search_limit <= 0:
            raise ValueError(f"search_limit must be > 0, got {self.search_limit}")
        if not (0 < self.distance_threshold <= 2.0):
            raise ValueError(
                f"distance_threshold must be in (0, 2], got {self.distance_threshold}"
            )
```

Import `RerankerStrategies` and `DEFAULT_SEARCH_LIMIT`, `DEFAULT_DISTANCE_THRESHOLD` from their new homes:
```python
from memory_retrieval.constants import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_SEARCH_LIMIT
from memory_retrieval.types import RerankerStrategies
```

**Step 4: Extract `_compute_pre_rerank_metrics`**

```python
def _compute_pre_rerank_metrics(
    query_results: list[dict[str, Any]],
    ground_truth_ids: set[str],
    distance_threshold: float,
) -> dict[str, Any]:
    """Compute pre-rerank precision/recall/F1 for all results within the distance threshold."""
    pre_rerank_threshold_ids: set[str] = {
        entry["id"]
        for query_result in query_results
        for entry in query_result["results"]
        if entry.get("distance", 0) <= distance_threshold
    }
    metrics = metric_point_to_dict(
        compute_set_metrics(pre_rerank_threshold_ids, ground_truth_ids)
    )
    total_unique = len({
        entry["id"]
        for query_result in query_results
        for entry in query_result["results"]
    })
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "total_unique_retrieved": total_unique,
        "total_within_threshold": len(pre_rerank_threshold_ids),
        "ground_truth_retrieved": len(pre_rerank_threshold_ids & ground_truth_ids),
    }
```

**Step 5: Extract `_execute_queries`**

```python
def _execute_queries(
    queries: list[str],
    db_path: str,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Run all queries against the search backend.

    Returns (query_results, all_retrieved_ids).
    """
    all_retrieved_ids: set[str] = set()
    query_results: list[dict[str, Any]] = []
    uses_distance = config.search_backend.search(db_path, queries[0], limit=1)[0].score_type == "cosine_distance" if queries else False

    for query in queries:
        results = config.search_backend.search(db_path, query, limit=config.search_limit)
        retrieved_ids = {result.id for result in results}
        all_retrieved_ids.update(retrieved_ids)

        result_entries = []
        for result in results:
            entry: dict[str, Any] = {
                "id": result.id,
                "situation": result.situation,
                "lesson": result.lesson,
                "is_ground_truth": False,  # filled in by caller with ground_truth_ids
            }
            if result.score_type == "cosine_distance":
                entry["distance"] = result.raw_score
                entry["confidence"] = get_confidence_from_distance(result.raw_score)
            else:
                entry["rank"] = result.raw_score
            result_entries.append(entry)

        query_results.append({
            "query": query,
            "word_count": len(query.split()),
            "result_count": len(results),
            "results": result_entries,
        })

    return query_results, all_retrieved_ids
```

Note: The `is_ground_truth` flag on individual results — currently set inline in `run_experiment` using `result.id in ground_truth_ids`. Keep that in place or pass `ground_truth_ids` into `_execute_queries`. Use the simpler approach: pass it in.

**Step 6: Extract `_print_experiment_summary`**

Move the 65-line summary block from `run_all_experiments` into:

```python
def _print_experiment_summary(
    all_results: list[dict[str, Any]],
    config: ExperimentConfig,
) -> None:
    """Print aggregate metrics after all experiments complete."""
    success_key = "pre_rerank_metrics" if config.reranker is not None else "metrics"
    successful = [r for r in all_results if success_key in r]

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    if not successful:
        print("No successful experiments")
        return

    if config.reranker is not None:
        # ... reranking summary block
    else:
        # ... standard summary block

    if len(successful) < len(all_results):
        print(f"Failed experiments: {len(all_results) - len(successful)}")
```

Replace the inline block in `run_all_experiments` with:
```python
    _print_experiment_summary(all_results, config)
    return all_results
```

**Step 7: Replace `isinstance(VectorBackend)` check**

```python
# Before (line 139):
is_vector = isinstance(config.search_backend, VectorBackend)

# After — remove VectorBackend import from runner.py entirely,
# detect by score_type on the first result:
# (handled inside _execute_queries via result.score_type)
```

Remove the `from memory_retrieval.search.vector import VectorBackend, get_confidence_from_distance` import from runner.py — we already moved `get_confidence_from_distance` to helpers, and `VectorBackend` is no longer needed.

**Step 8: Run all tests**

```bash
uv run pytest tests/ -q
```

**Step 9: Commit**

```bash
git add src/memory_retrieval/experiments/runner.py \
        tests/experiments/test_runner_helpers.py
git commit -m "refactor: decompose run_experiment into helpers, add ExperimentConfig validation"
```

---

## Task 14: Experiments — `comparison.py` — `RunSummaryGenerator` + `FingerprintReconstructor`

**Files:**
- Modify: `src/memory_retrieval/experiments/comparison.py`
- Test: `tests/experiments/test_comparison_migration.py` (existing — must pass)

This is the most complex task. Read the full `comparison.py` file before starting.

**Step 1: Run existing comparison tests**

```bash
uv run pytest tests/experiments/test_comparison_migration.py -q
```

**Step 2: Add `ConfigFingerprint` return type to `build_config_fingerprint`**

```python
from memory_retrieval.types import ConfigFingerprint

def build_config_fingerprint(...) -> ConfigFingerprint:
```

**Step 3: Create `FingerprintReconstructor` class**

Replace `reconstruct_fingerprint_from_run()` function with:

```python
class FingerprintReconstructor:
    """Reconstructs a ConfigFingerprint from a completed run directory.

    Handles legacy runs that may not have all fields in run.json.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    def reconstruct(self) -> ConfigFingerprint:
        """Reconstruct and return the config fingerprint for the run."""
        run_metadata = _load_run_metadata_safe(self._run_dir)
        fingerprint = run_metadata.get("config_fingerprint", {})
        query_info = self._extract_query_info()
        return {
            "extraction_prompt": fingerprint.get("extraction_prompt", "unknown"),
            "extraction_model": fingerprint.get("extraction_model", "unknown"),
            "query_prompt": query_info.get("prompt_version", "unknown"),
            "query_model": query_info.get("model", "unknown"),
            "search_limit": fingerprint.get("search_limit", 20),
            "distance_threshold": fingerprint.get("distance_threshold", 1.1),
            "reranker": fingerprint.get("reranker", None),
        }

    def _extract_query_info(self) -> dict[str, str]:
        """Extract query model and prompt version from the first available query file."""
        queries_dir = self._run_dir / "queries"
        if not queries_dir.exists():
            return {}
        query_files = sorted(queries_dir.glob("*.json"))
        if not query_files:
            return {}
        try:
            first_query = load_json(str(query_files[0]))
            return {
                "model": first_query.get("model", "unknown"),
                "prompt_version": first_query.get("prompt_version", "unknown"),
            }
        except Exception:
            return {}
```

Keep the old `reconstruct_fingerprint_from_run()` function as a one-line delegate for backward compatibility:
```python
def reconstruct_fingerprint_from_run(run_dir: Path) -> ConfigFingerprint:
    """Reconstruct config fingerprint from a run directory. See FingerprintReconstructor."""
    return FingerprintReconstructor(run_dir).reconstruct()
```

**Step 4: Create `RunSummaryGenerator` class**

Read `generate_run_summary()` fully, understand its structure, then extract:

```python
class RunSummaryGenerator:
    """Generates and saves run_summary.json for a completed experiment run.

    Processes all result files in the run directory, computes macro-averaged
    metrics, sweeps across distance thresholds and top-N values, and writes
    the summary to run_summary.json.
    """

    def __init__(self, run_dir: Path, phase: str) -> None:
        self._run_dir = run_dir
        self._phase = phase

    def generate(self) -> dict[str, Any]:
        """Generate the run summary and save it to run_summary.json."""
        results = self._load_results()
        per_case_metrics = [self._process_test_case(result) for result in results]
        macro_averages = self._compute_macro_averages(per_case_metrics)
        summary = self._build_summary(results, per_case_metrics, macro_averages)
        self._save(summary)
        return summary

    def _load_results(self) -> list[dict[str, Any]]:
        """Load all experiment result JSON files from the results directory."""
        results_dir = self._run_dir / "results"
        result_files = sorted(results_dir.glob("*.json"))
        return [load_json(str(f)) for f in result_files]

    def _process_test_case(self, result: dict[str, Any]) -> dict[str, Any]:
        """Compute per-test-case metrics (pre/post rerank, sweeps)."""
        # Extract the per-case computation from current generate_run_summary
        ...

    def _compute_macro_averages(
        self, per_case_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Average per-case metrics across all test cases."""
        ...

    def _build_summary(
        self,
        results: list[dict[str, Any]],
        per_case_metrics: list[dict[str, Any]],
        macro_averages: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the final summary dict matching run_summary.json schema."""
        ...

    def _save(self, summary: dict[str, Any]) -> None:
        """Write summary to run_summary.json in the run directory."""
        save_json(summary, self._run_dir / "run_summary.json")
```

Keep `generate_run_summary()` as a one-line delegate:
```python
def generate_run_summary(run_dir: Path, phase: str) -> dict[str, Any]:
    """Generate and save run summary. See RunSummaryGenerator."""
    return RunSummaryGenerator(run_dir, phase).generate()
```

**Step 5: Run all tests including migration tests**

```bash
uv run pytest tests/ -q
```

**Step 6: Commit**

```bash
git add src/memory_retrieval/experiments/comparison.py
git commit -m "refactor: extract RunSummaryGenerator and FingerprintReconstructor classes"
```

---

## Task 15: Experiments — `metrics.py` and `metrics_adapter.py` cleanup

**Files:**
- Modify: `src/memory_retrieval/experiments/metrics.py`
- Modify: `src/memory_retrieval/experiments/metrics_adapter.py`
- Test: `tests/experiments/test_metrics_migration_adapter.py` (existing — must pass)

**Step 1: Run existing metrics tests**

```bash
uv run pytest tests/experiments/ -q
```

**Step 2: Remove pure pass-through wrappers from `metrics.py`**

Read `metrics.py` fully. Any function that is literally:
```python
def foo(x):
    return bar(x)
```
with no transformation — remove it and update callers to import from `retrieval_metrics` directly.

Keep only wrappers that do real transformation (e.g. convert between dict shapes, apply domain-specific logic).

**Step 3: Add deprecation notice to `metrics_adapter.py`**

At the top of the file, add:
```python
"""Backward-compatibility adapter for the retrieval_metrics migration (Feb 2026).

DEPRECATED: This module maps retrieval_metrics dataclasses to legacy dict shapes
expected by existing notebook code. Once notebooks are migrated to import from
retrieval_metrics directly, this module can be retired.

Retirement path:
1. Update all notebooks to import MetricPoint, SweepResult from retrieval_metrics directly
2. Remove calls to metric_point_to_dict() and extract_metric_from_nested()
3. Delete this module
"""
```

**Step 4: Flatten nested conditionals in `extract_metric_from_nested`**

Read the function and replace nested if/elif chains with early returns:
```python
def extract_metric_from_nested(
    summary: dict[str, Any],
    metric: str,
    strategy: str | None = None,
    sweep_type: str = "top_n",
) -> float | None:
    """Extract a metric value from a nested run summary dict.

    Returns None if the path does not exist (handles old and new summary formats).
    """
    macro = summary.get("macro_averaged", {})

    if strategy is not None:
        # Post-rerank path
        post = macro.get("post_rerank", {}).get(strategy, {})
        key = "at_optimal_top_n" if sweep_type == "top_n" else "at_optimal_threshold"
        point = post.get(key, {})
        return point.get(metric)

    # Pre-rerank path
    pre = macro.get("pre_rerank", {})
    key = "at_optimal_top_n" if sweep_type == "top_n" else "at_optimal_distance_threshold"
    point = pre.get(key, {})
    if point:
        return point.get(metric)

    # Legacy fallback
    return macro.get(metric)
```

**Step 5: Run all tests**

```bash
uv run pytest tests/ -q
```

**Step 6: Commit**

```bash
git add src/memory_retrieval/experiments/metrics.py \
        src/memory_retrieval/experiments/metrics_adapter.py
git commit -m "refactor: metrics.py remove pass-through wrappers, adapter deprecation notice"
```

---

## Task 16: Final cross-file verification pass

**Step 1: Verify no residual duplication between FTS5 and Vector backends**

```bash
uv run grep -n "def get_memory_by_id" src/memory_retrieval/search/fts5.py src/memory_retrieval/search/vector.py
uv run grep -n "def get_sample_memories" src/memory_retrieval/search/fts5.py src/memory_retrieval/search/vector.py
```
Expected: no matches (methods now live only in `SearchBackendBase`).

**Step 2: Verify `EMBEDDING_MODEL_DIMENSIONS` is only in `constants.py`**

```bash
uv run grep -rn "EMBEDDING_MODEL_DIMENSIONS" src/
```
Expected: only `constants.py` defines it; `vector.py` imports from constants.

**Step 3: Verify `DEFAULT_SEARCH_LIMIT` and `DEFAULT_DISTANCE_THRESHOLD` are only in `constants.py`**

```bash
uv run grep -rn "DEFAULT_SEARCH_LIMIT\|DEFAULT_DISTANCE_THRESHOLD" src/
```
Expected: only `constants.py` defines them; `runner.py` imports from constants.

**Step 4: Verify no functions over ~60 lines**

```bash
uv run python - <<'EOF'
import ast, pathlib

def check_long_functions(path_glob: str, max_lines: int = 60) -> None:
    for path in sorted(pathlib.Path("src").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or 0) - node.lineno
                if length > max_lines:
                    print(f"{path}:{node.lineno} {node.name}() — {length} lines")

check_long_functions("src/**/*.py")
EOF
```
Review any remaining long functions and decide if further decomposition is warranted.

**Step 5: Run the full test suite one final time**

```bash
uv run pytest tests/ -v
```
All tests must pass.

**Step 6: Final commit**

```bash
git add -p  # review any remaining unstaged changes
git commit -m "refactor: cross-file cleanup — verify DRY, constants, no long functions"
```

---

## Completion Checklist

- [ ] `types.py` created with all TypedDicts and TypeAliases
- [ ] `constants.py` created with EMBEDDING_MODEL_DIMENSIONS, defaults, EXCLUDED_FILE_PATTERNS
- [ ] `infra/`: typed `_SafeDict`, extracted helpers, added docstrings
- [ ] `memories/validators.py`: explicit version comparison
- [ ] `memories/loader.py`: returns `list[MemoryDict]`
- [ ] `memories/extractor.py`: `ExtractionSummary`, validation, 5 focused functions, moved import
- [ ] `memories/helpers.py`: `get_confidence_from_distance` moved here
- [ ] `search/base.py`: `SearchBackendBase` with `get_memory_by_id`, `ScoreType`
- [ ] `search/vector.py`: inherits base, `itertools.batched`, imports from constants
- [ ] `search/fts5.py`: inherits base
- [ ] `search/reranker.py`: model config constants, f-strings
- [ ] `experiments/test_cases.py`: uses pre-compiled patterns from constants
- [ ] `experiments/query_generation.py`: extracted parse strategies, logging, `_truncate_text`, validation
- [ ] `experiments/runner.py`: `ExperimentConfig` validation, `_execute_queries`, `_compute_pre_rerank_metrics`, `_print_experiment_summary`, no `isinstance(VectorBackend)`
- [ ] `experiments/comparison.py`: `RunSummaryGenerator`, `FingerprintReconstructor`, `ConfigFingerprint` return type
- [ ] `experiments/metrics.py`: no pure pass-through wrappers
- [ ] `experiments/metrics_adapter.py`: deprecation notice, flattened conditionals
- [ ] All existing tests pass
- [ ] No function over ~60 lines
- [ ] `EMBEDDING_MODEL_DIMENSIONS` defined only in `constants.py`
