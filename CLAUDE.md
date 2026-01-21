# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for experimenting with memory retrieval techniques for the [code-review-mentat](https://github.com/KamilMarzynski/code-review-mentat) project. Processes real-world code review data using AI to extract reusable engineering knowledge as structured memories.

## Commands

```bash
# Run memory extraction (requires OPENROUTER_API_KEY env var)
python scripts/pre0_build_memories.py data/raw/<input_file>.json

# Install dependencies (uses uv package manager)
uv sync
```

## Architecture

### Data Pipeline

1. **Input**: Raw code review JSON files in `data/raw/` containing PR context, metadata, and code review comments with fields like severity, confidence, code snippets, and user notes

2. **Processing** (`scripts/pre0_build_memories.py`):
   - Two-stage AI processing via OpenRouter API (default model: `meta-llama/llama-3.1-8b-instruct`)
   - Stage 1: Extract concrete situation description (2 sentences max, 40-450 chars)
   - Stage 2: Extract actionable lesson (imperative, max 160 chars, must start with Always/Never/Ensure/Avoid/Verify/Check/Prefer/Consider/Use)
   - Quality validation rejects generic or malformed outputs
   - Confidence filtering rejects memories with fused confidence < 0.6

3. **Output**: JSONL files in `data/phase0/` containing memories with:
   - `id`: Deterministic hash-based ID (`mem_<12-char-hash>`)
   - `situation_description`: When this knowledge applies
   - `lesson`: Actionable imperative guidance
   - `metadata`: repo, file pattern, language, severity, confidence
   - `source`: Original code snippet, comment, PR context

### Key Functions in pre0_build_memories.py

- `build_memories(raw_path, out_dir, model, sleep_s)` - Main orchestrator
- `_call_openrouter(...)` - API wrapper with configurable model/temperature
- `_prompt_situation/lesson(...)` - Prompt engineering for each extraction stage
- `_validate_situation/lesson(...)` - Quality validation rules
- `_stable_id(...)` - SHA1-based deterministic memory ID generation

## Environment

- Python 3.13 (uv package manager)
- Requires `OPENROUTER_API_KEY` environment variable
- Data files in `data/` are gitignored (contains sensitive real-world code reviews)
