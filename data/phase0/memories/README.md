# Phase 0 Memories

Contains extracted memories from code reviews and the SQLite FTS5 search database.

## How Data Gets Here

1. Place raw code review JSON files in `data/review_data/`
2. Run memory extraction:
   ```bash
   uv run python scripts/pre0_build_memories.py data/review_data/<file>.json
   ```
3. Build/update the search database:
   ```bash
   uv run python scripts/phase0_sqlite_fts.py --rebuild
   ```

## Files

- `memories_*.jsonl` - Accepted memories (one JSON object per line)
- `rejected_*.jsonl` - Rejected memories with rejection reasons
- `memories.db` - SQLite database with FTS5 full-text search index

## Memory JSONL Schema

```json
{
  "id": "mem_<12-char-hash>",
  "situation_variants": ["Variant 1 (use [0] for display)", "Variant 2", "Variant 3"],
  "lesson": "Actionable imperative guidance (max 160 chars)",
  "metadata": {
    "repo": "repository-name",
    "file_pattern": "path/to/files/*.ts",
    "language": "ts",
    "tags": [],
    "severity": "issue | suggestion | risk",
    "confidence": 0.85,
    "author": "pre0-openrouter",
    "source_comment_id": "UUID from original comment",
    "status": "accepted | rejected"
  },
  "source": {
    "file": "path/to/file.ts",
    "line": 123,
    "code_snippet": "relevant code fragment",
    "comment": "original review comment",
    "user_note": "optional user annotation",
    "rationale": "why this matters",
    "verifiedBy": "verification method",
    "pr_context": "source/branch -> target/branch",
    "gathered_at": "ISO 8601 timestamp",
    "raw_context_hash": "12-char hash"
  }
}
```

## SQLite Database Schema

```sql
-- Main table
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    situation_variants TEXT NOT NULL,  -- JSON array of 3 variants
    lesson TEXT NOT NULL,
    metadata TEXT,  -- JSON
    source TEXT     -- JSON
);

-- FTS5 virtual table for keyword search (indexes each variant separately)
CREATE VIRTUAL TABLE memories_fts USING fts5(
    situation_variant,
    memory_id UNINDEXED
);
```

## Testing Search

```bash
uv run python scripts/fetch_memories.py "optional chaining"
```
