# Unit Test Suite Design

**Date:** 2026-02-21
**Status:** Approved

## Goal

Add comprehensive unit tests to the `memory-retrieval` research package. The codebase is largely stable; tests protect against regressions as prompts, metrics, and pipeline components evolve.

## Scope

### Modules to test

| Module | Type | Why |
|---|---|---|
| `memories/helpers.py` | Pure functions | Zero deps, deterministic |
| `memories/validators.py` | Pure functions | Boundary conditions on extraction validation |
| `memories/loader.py` | File I/O | JSONL loading with error handling |
| `search/db_utils.py` | Pure functions | Serialization round-trips |
| `search/fts5.py` | SQLite integration | FTS5 backend correctness |
| `infra/prompts.py` | File I/O | Prompt loading, semver selection, rendering |
| `infra/runs.py` | File I/O + state | Run isolation system (core pipeline) |
| `experiments/test_cases.py` | Pure + File I/O | `filter_diff` is critical, ground truth matching |
| `experiments/metrics.py` | Pure functions | Pool/dedup not yet tested |

### Modules excluded (require external services)

- `memories/extractor.py` — LLM API calls
- `experiments/query_generation.py` — LLM API calls
- `search/vector.py` — Requires Ollama running
- `search/reranker.py` — Requires sentence-transformers model files
- `infra/llm.py` — HTTP API calls

### Already well-tested (leave as-is)

- `experiments/metrics.py` compute functions — `test_metrics_migration_adapter.py`
- `experiments/comparison.py` — `test_comparison_migration.py`

## Structure

Flat layout, matching existing test files:

```
tests/
  conftest.py                        # shared fixtures
  test_memories_helpers.py
  test_memories_validators.py
  test_memories_loader.py
  test_search_db_utils.py
  test_search_fts5.py
  test_infra_prompts.py
  test_infra_runs.py
  test_experiments_test_cases.py
  test_experiments_metrics.py
  # existing files (unchanged):
  test_metrics_migration_adapter.py
  test_comparison_migration.py
```

## Infrastructure changes

**`pyproject.toml`:**
- Add `pytest` to `[dependency-groups] dev`
- Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]`

**`tests/conftest.py`:**
- `sample_memory()` helper — canonical memory dict for reuse across tests
- `tmp_fts5_db` fixture — temp FTS5 database with schema, populated, auto-cleaned

## Key design decisions

- **`infra/runs.py` isolation:** `DATA_DIR` is a module-level constant. Tests use `monkeypatch.setattr` to redirect it to `tmp_path`, so no real data directory is touched.
- **FTS5 tests:** Use `tmp_path` for the SQLite file. Tests are self-contained (create schema, insert, search, assert). No mocking.
- **No mocking of business logic.** Only external services (LLM, Ollama) are skipped.

## Test inventory (by module)

### `test_memories_helpers.py`
- `stable_id`: determinism, `mem_` prefix, 12-char hash, different inputs → different IDs
- `lang_from_file`: `.py`, `.ts`, no extension, uppercase suffix
- `file_pattern`: nested path produces glob, top-level file, no-extension
- `short_repo_name`: URL-style, empty string → `"unknown"`
- `confidence_map`: all three levels (high/medium/low), unknown string → `0.5`, None/empty

### `test_memories_validators.py`
- `validate_situation_v1`: below/above char limits, with/without terminal punctuation
- `validate_situation_v2`: below/above word count limits, valid range
- `validate_lesson`: below/above char limits, valid
- `get_situation_validator`: `"1.0.0"` → v1, `"2.0.0"` → v2, unknown → latest fallback

### `test_memories_loader.py`
- Empty directory → `[]`
- Single valid JSONL → loads all records
- Multiple `memories_*.jsonl` files → all loaded sorted alphabetically
- Malformed JSON line skipped without crash
- Non-matching filenames ignored

### `test_search_db_utils.py`
- `serialize_json_field(None)` → `"{}"`
- `serialize_json_field(dict)` → valid JSON string
- Unicode preserved through round-trip
- `deserialize_json_field(None)` → `{}`
- `deserialize_json_field("")` → `{}`
- Round-trip: serialize then deserialize preserves data

### `test_infra_prompts.py`
- `_parse_semver`: valid format → tuple, invalid → raises `ValueError`
- `_parse_prompt_file`: extracts system and user sections, raises on missing sections
- `Prompt.render`: substitutes kwargs, missing key → empty string, `version_tag` property
- `load_prompt`: latest when version=None, specific version, raises `FileNotFoundError`

### `test_infra_runs.py`
- `create_run`: creates 4 subdirs, writes `run.json` with correct fields
- `get_latest_run`: returns newest by name, raises `NoRunsFoundError` when empty
- `get_run`: returns path when exists, raises `FileNotFoundError` otherwise
- `list_runs`: empty dir → `[]`, multiple runs → newest first
- `update_run_status`: persists stage to `run.json`, adds `completed_at`
- `create_subrun`: creates `subruns/<id>/results/`, writes `subrun.json`, rejects path separators in ID
- `list_subruns`: empty → `[]`, lists all subruns
- `get_subrun_db_path`: prefers subrun db if exists, falls back to parent

### `test_experiments_test_cases.py`
- `filter_diff`: empty string, filters `package-lock.json`, `yarn.lock`, `.min.js`, `.map`, keeps `.ts`, multi-file mixed diff
- `build_test_case`: `None` when no ground truth, correct dict structure when matched

### `test_experiments_metrics.py`
- `pool_and_deduplicate_by_distance`: same ID across queries → min distance wins, sorted ascending
- `pool_and_deduplicate_by_rerank_score`: same ID → max rerank score wins, sorted descending

### `test_search_fts5.py`
- Full round-trip: create schema, insert memories, search, assert results
- Multiple variants per memory → deduplication returns one result per memory
- `get_memory_count`: returns correct count after inserts
- `get_memory_by_id`: finds existing memory, returns `None` for missing ID
