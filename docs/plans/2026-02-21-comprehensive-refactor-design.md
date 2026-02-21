# Comprehensive Refactor Design

**Date:** 2026-02-21
**Approach:** B — File-by-file in dependency order, then cross-file cleanup pass
**Scope:** Types → DRY → Decompose → Validation → Error handling (all concerns per file)

---

## Goals

1. **Type safety** — replace `dict[str, Any]` with TypedDicts, add TypeAliases, tighten all return types
2. **Large function decomposition** — no function over ~50 lines; extract focused helpers
3. **DRY via SearchBackendBase** — eliminate ~60% code duplication between FTS5 and Vector backends
4. **Class-based orchestrators** — `RunSummaryGenerator`, `FingerprintReconstructor` replace mega-functions
5. **Config validation** — `__post_init__` validators on all config dataclasses
6. **Logging over print** — replace `print()` + threading locks with `logging.getLogger(__name__)`
7. **Constants centralized** — scattered magic values gathered into `constants.py`

---

## New Files

### `src/memory_retrieval/types.py`

All shared TypedDicts, TypeAliases, and Protocols.

```python
# Core domain types
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

# Experiment pipeline types
class QueryResult(TypedDict):
    query: str
    results: list[dict[str, Any]]
    num_results: int

class MetricsDict(TypedDict):
    precision: float
    recall: float
    f1_score: float
    reciprocal_rank: float

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

# TypeAliases
TextStrategyFn = Callable[[dict[str, Any]], str]
RerankerStrategies = dict[str, TextStrategyFn]
ValidatorFn = Callable[[str], tuple[bool, str]]
ScoreType = Literal["cosine_distance", "bm25_rank", "rerank_score"]
```

### `src/memory_retrieval/constants.py`

All tunable parameters and pre-compiled patterns gathered in one place.

```python
# Embedding model registry (from search/vector.py)
EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    "mxbai-embed-large": 1024,
    "nomic-embed-text": 768,
    "all-minilm": 384,
}

# Experiment defaults (from experiments/runner.py)
DEFAULT_SEARCH_LIMIT: int = 20
DEFAULT_DISTANCE_THRESHOLD: float = 1.1
DEFAULT_RERANK_TOP_N: int = 5

# File exclusion patterns for diff filtering (from experiments/test_cases.py)
# Pre-compiled at import time — never re-compiled per call
EXCLUDED_FILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"package-lock\.json$"),
    re.compile(r"yarn\.lock$"),
    re.compile(r"\.min\.js$"),
    re.compile(r"\.generated\.(ts|js|py)$"),
    # ... full list
]
```

---

## File-by-File Changes

### Execution order (bottom-up by dependency)

```
Pass 1: infra/       (no internal dependencies)
Pass 2: memories/    (depends on infra)
Pass 3: search/      (depends on memories, infra)
Pass 4: experiments/ (depends on all)
Pass 5: Cross-file   (SearchBackendBase extraction, adapter retirement)
```

---

### Pass 1: Infra Domain

#### `infra/io.py` — unchanged

#### `infra/llm.py`
- Add `timeout: float = 30.0` parameter
- Add docstring explaining OpenRouter message format

#### `infra/prompts.py`
- `_SafeDict` typed as `class _SafeDict(dict[str, str])`
- Extract `_find_latest_semver_file(prompt_dir: Path) -> Path | None` helper (~15 lines removed from `load_prompt()`)

#### `infra/runs.py`
- Extract `_build_run_metadata(run_id: str, phase: str) -> RunMetadata`
- Extract `_build_subrun_metadata(...)` helper
- Add docstring to `_generate_run_id()` explaining microsecond precision (collision prevention)

#### `infra/figures.py`
- Type review pass: tighten `dict[str, Any]` where shapes are known

---

### Pass 2: Memories Domain

#### `memories/loader.py`
- Return type: `list[MemoryDict]` (was `list[dict[str, Any]]`)

#### `memories/helpers.py`
- Tighten return types using `MemoryDict` where applicable
- `get_confidence_from_distance()` **moves here** from `search/vector.py` (semantic home: memory concern, not backend concern)

#### `memories/validators.py`
- Replace `lambda f: f.__name__` version selection with explicit semver comparison using `_parse_semver()` from `infra/prompts.py`

#### `memories/extractor.py`
- Move `from memory_retrieval.memories.validators import validate_situation_v1` to module top
- Add `__post_init__` to `ExtractionConfig`: validates `sleep_s >= 0`, `model` non-empty
- Add `ExtractionSummary` dataclass: `written: int`, `rejected: int`, `errors: int`
- Decompose 250-line `extract_memories()` into:
  - `extract_memories()` — orchestration entry point (~30 lines)
  - `_extract_memory_for_comment(comment, pr_context, config) -> MemoryDict | None` (~40 lines)
  - `_extract_situation(comment_text, prompt, config) -> tuple[bool, str]` (~30 lines)
  - `_validate_variants(raw_text, validator) -> tuple[bool, list[str]]` (~20 lines)
  - `_build_memory_dict(situation, lesson, comment, pr_context) -> MemoryDict` (~20 lines)

---

### Pass 3: Search Domain

#### `search/base.py`
- Add `SearchBackendBase(ABC)` abstract class alongside existing Protocol
- Shared methods: `_insert_memories_with_count()`, `get_memory_by_id()`, `_build_search_result()`
- `SearchResult.score_type` tightened to `ScoreType` Literal type

#### `search/fts5.py`
- Inherit `SearchBackendBase`
- Remove ~40 lines of duplicated insertion and fetch logic
- Keep FTS5-specific query building and variant deduplication

#### `search/vector.py`
- Inherit `SearchBackendBase`
- Remove ~40 lines of duplicated insertion and fetch logic
- Replace manual chunking with `itertools.batched()` (Python 3.13)
- `EMBEDDING_MODEL_DIMENSIONS` moves to `constants.py`
- `get_confidence_from_distance()` moves to `memories/helpers.py`

#### `search/reranker.py`
- Extract `_MODEL_FORMAT_CONFIGS` dict at module level instead of branching in `_format_pair()`
- Use f-strings consistently

---

### Pass 4: Experiments Domain

#### `experiments/test_cases.py`
- Import `EXCLUDED_FILE_PATTERNS` from `constants.py` (pre-compiled, no longer re-compiled per call)
- `build_test_case()` returns `TestCase` TypedDict
- `build_test_cases()` returns `list[TestCase]`

#### `experiments/query_generation.py`
- Add return type hints to all functions
- Replace `print()` + `threading.Lock()` with `logging.getLogger(__name__)`
- Extract parse strategies:
  - `_parse_json_array_queries(text: str) -> list[str] | None`
  - `_parse_quoted_queries(text: str) -> list[str] | None`
  - `_parse_newline_queries(text: str) -> list[str] | None`
  - `parse_queries_robust(text: str) -> list[str]` — tries each in order
- Extract helpers from `generate_queries_for_test_case()`:
  - `_truncate_text(text: str, max_chars: int) -> str`
  - `_load_sample_memories(db_path, backend, num_samples) -> list[MemoryDict]`
  - `_save_query_data(queries_dir, test_case_id, data) -> None`
- Add `__post_init__` to `QueryGenerationConfig`: `max_workers > 0`

#### `experiments/runner.py`
- Add `TextStrategyFn` TypeAlias import from `types.py`
- Add `__post_init__` to `ExperimentConfig`: `search_limit > 0`, `distance_threshold` in `(0, 2]`
- Replace `isinstance(config.search_backend, VectorBackend)` with `score_type == "cosine_distance"` check
- Decompose 182-line `run_experiment()`:
  - `run_experiment()` — dispatch entry point (~20 lines)
  - `_execute_queries(queries, db_path, config) -> list[QueryResult]` (~30 lines)
  - `_run_standard_experiment(query_results, ground_truth_ids, config) -> ExperimentResultEntry` (~30 lines)
  - `_run_reranking_experiment(query_results, ground_truth_ids, config) -> ExperimentResultEntry` (~40 lines)
  - `_compute_pre_rerank_metrics(query_results, ground_truth_ids) -> MetricsDict` (~20 lines)
- Extract `_print_experiment_summary(results)` from `run_all_experiments()` (~65 lines removed from main logic)

#### `experiments/comparison.py`
- Add return types to all functions
- `build_config_fingerprint()` returns `ConfigFingerprint` TypedDict
- Replace `generate_run_summary()` (~150-line function) with `RunSummaryGenerator` class:
  ```python
  class RunSummaryGenerator:
      def __init__(self, run_dir: Path, phase: str) -> None
      def generate(self) -> RunSummary
      def _load_results(self) -> list[ExperimentResultEntry]
      def _process_test_case(self, result) -> PerCaseMetrics
      def _compute_macro_averages(self, per_case) -> MacroAveragedMetrics
      def _build_summary(self, ...) -> RunSummary
      def _save(self, summary: RunSummary) -> None
  ```
- Replace `reconstruct_fingerprint_from_run()` (~65-line function) with `FingerprintReconstructor` class:
  ```python
  class FingerprintReconstructor:
      def __init__(self, run_dir: Path) -> None
      def reconstruct(self) -> ConfigFingerprint
      def _extract_query_info(self) -> dict[str, str]
      def _validate_required_fields(self, data: dict[str, Any]) -> None
  ```
- Add `RunSummary`, `PerCaseMetrics`, `MacroAveragedMetrics` TypedDicts

#### `experiments/metrics.py`
- Remove pure 1-line pass-through wrappers; callers import from `retrieval_metrics` directly
- Keep only wrappers that perform real transformation

#### `experiments/metrics_adapter.py`
- Add `# DEPRECATED` module docstring with retirement path note
- Flatten nested conditionals in `extract_metric_from_nested()` with early returns

---

### Pass 5: Cross-file cleanup

- Verify `SearchBackendBase` is fully utilized by both backends (no residual duplication)
- Ensure all `MemoryDict` usages are consistent (loader → extractor → runner → comparison)
- Confirm `constants.py` is the single source of truth for all magic values
- Run full test suite; add tests for any newly exposed helper functions

---

## What Does NOT Change

- Public API of all modules (import paths, function signatures visible to notebooks)
- `run_summary.json` structure on disk
- Data pipeline order
- Notebook code (notebooks import from `memory_retrieval.*` — all public APIs preserved)
- Test file structure in `tests/`

---

## Success Criteria

- No function over ~50 lines (excluding docstrings)
- No `dict[str, Any]` where a TypedDict captures the known shape
- `mypy --strict` passes (or near-passes) on `src/memory_retrieval/`
- All existing tests pass unchanged
- `search/fts5.py` and `search/vector.py` share shared logic via `SearchBackendBase`
- `constants.py` is the single source for default experiment parameters
