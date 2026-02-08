"""
Extract memories from raw code review JSON files.

This script processes raw code review data using AI to extract reusable
engineering knowledge as structured memories.

Usage:
    # Process single file (creates new run)
    uv run python scripts/phase1/build_memories.py data/review_data/review_001.json

    # Process all JSON files in directory (creates new run)
    uv run python scripts/phase1/build_memories.py --all data/review_data

    # Use specific run (for re-processing)
    uv run python scripts/phase1/build_memories.py --all data/review_data --run-id run_20260208_143022
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import load_json, ensure_dir, call_openrouter
from common.prompts import load_prompt
from common.runs import create_run, get_run, update_run_status, PHASE1

# Phase 1 prompts directory
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _short_repo_name(repo_path: str) -> str:
    if not repo_path:
        return "unknown"
    return Path(repo_path).name


def _lang_from_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower().lstrip(".")
    return ext or "unknown"


def _file_pattern(file_path: str) -> str:
    p = Path(file_path)
    if len(p.parts) >= 2:
        return str(Path(*p.parts[:-1]) / f"*.{p.suffix.lstrip('.')}")
    return str(p.name)


def _confidence_map(c: str) -> float:
    return {"high": 1.0, "medium": 0.7, "low": 0.4}.get((c or "").lower(), 0.5)


def _stable_id(raw_comment_id: str, situation: str, lesson: str) -> str:
    h = hashlib.sha1(
        (raw_comment_id + "\n" + situation + "\n" + lesson).encode("utf-8")
    ).hexdigest()
    return f"mem_{h[:12]}"


def _validate_situation_v1(text: str) -> Tuple[bool, str]:
    """Validation for v1 prompts (40-450 chars)."""
    t = (text or "").strip()
    if len(t) < 40:
        return False, "situation_too_short"
    if len(t) > 450:
        return False, "situation_too_long"
    if not t.endswith((".", "!", "?")):
        return True, "ok_no_punct"
    return True, "ok"


def _validate_situation_v2(text: str) -> Tuple[bool, str]:
    """
    Validation for v2 prompts (25-60 words, aligned with query length).

    Word count validation ensures memory and query lengths are similar
    for better vector similarity matching.
    """
    t = (text or "").strip()
    word_count = len(t.split())

    if word_count < 20:
        return False, f"situation_too_short_{word_count}_words"
    if word_count > 70:
        return False, f"situation_too_long_{word_count}_words"
    if not t.endswith((".", "!", "?")):
        return True, "ok_no_punct"
    return True, "ok"

def _validate_lesson(text: str) -> Tuple[bool, str]:
    t = (text or "").strip()
    if len(t) < 20:
        return False, "lesson_too_short"
    if len(t) > 220:
        return False, "lesson_too_long"
    if not t.endswith((".", "!", "?")):
        return True, "ok_no_punct"
    return True, "ok"


# Map prompt versions to their situation validators
_SITUATION_VALIDATORS = {
    "1.0.0": _validate_situation_v1,
    "2.0.0": _validate_situation_v2,
}


def build_memories(
    raw_path: str,
    out_dir: str,
    model: str = "anthropic/claude-haiku-4.5",
    sleep_s: float = 0.25,
    prompt_version: Optional[str] = None,
) -> str:
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

    repo = _short_repo_name(meta.get("repoPath", ""))
    pr_context = f"{meta.get('sourceBranch', '?')} → {meta.get('targetBranch', '?')}"
    gathered_at = meta.get("gatheredAt", "")

    # Load prompts
    situation_prompt = load_prompt("memory-situation", version=prompt_version, prompts_dir=_PROMPTS_DIR)
    lesson_prompt = load_prompt("memory-lesson", prompts_dir=_PROMPTS_DIR)

    # Select validator: use version-specific if available, else latest
    validate_fn = _SITUATION_VALIDATORS.get(
        situation_prompt.version,
        max(_SITUATION_VALIDATORS.values(), key=lambda f: f.__name__),
    )

    written = 0
    rejected = 0

    with (
        open(out_path, "w", encoding="utf-8") as out_f,
        open(reject_path, "w", encoding="utf-8") as rej_f,
    ):
        for c in comments:
            if c.get("status") == "rejected":
                rej_f.write(
                    json.dumps(
                        {
                            "comment_id": c.get("id"),
                            "stage": "status",
                            "reason": "comment_rejected",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1
                continue

            # Pre-compute template variables
            code = (c.get("code_snippet") or "").strip()
            user_note = (c.get("user_note") or "").strip()
            additional_context = f"ADDITIONAL CONTEXT: {user_note}" if user_note else ""

            situation = call_openrouter(
                api_key=api_key,
                model=model,
                messages=situation_prompt.render(
                    context=context,
                    file=c.get("file", ""),
                    severity=c.get("severity", "info"),
                    code=code[:800] if code else "(none)",
                    comment=c.get("message", ""),
                    user_note=user_note,
                    additional_context=additional_context,
                ),
                temperature=0.0,
                max_tokens=600,
            )

            # Validate situation
            all_valid = True
            ok_s, reason_s = validate_fn(situation)
            if reason_s == "ok_no_punct":
                situation = situation.rstrip() + "."
            elif not ok_s:
                rej_f.write(
                    json.dumps(
                        {
                            "comment_id": c.get("id"),
                            "stage": "situation",
                            "reason": reason_s,
                            "text": situation,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                all_valid = False
                break

            if not all_valid:
                rejected += 1
                continue

            time.sleep(sleep_s)

            rationale = (c.get("rationale") or "").strip()
            lesson = call_openrouter(
                api_key=api_key,
                model=model,
                messages=lesson_prompt.render(
                    situation=situation,
                    comment=c.get("message", ""),
                    rationale=rationale if rationale else "(none)",
                ),
                temperature=0.0,
                max_tokens=120,
            )
            ok_l, reason_l = _validate_lesson(lesson)
            if reason_l == "ok_no_punct":
                lesson = lesson.rstrip() + "."
            if not ok_l:
                rej_f.write(
                    json.dumps(
                        {
                            "comment_id": c.get("id"),
                            "stage": "lesson",
                            "reason": reason_l,
                            "text": lesson,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1
                continue

            original_conf = _confidence_map(c.get("confidence", "medium"))

            memory = {
                "id": _stable_id(
                    c.get("id", str(datetime.now().timestamp())), situation, lesson
                ),
                "situation_description": situation,
                "lesson": lesson,
                "metadata": {
                    "repo": repo,
                    "file_pattern": _file_pattern(c.get("file", "")),
                    "language": _lang_from_file(c.get("file", "")),
                    "tags": [],
                    "severity": c.get("severity", "info"),
                    "confidence": original_conf,
                    "author": "phase1-openrouter",
                    "prompt_version": situation_prompt.version_tag,
                    "source_comment_id": c.get("id"),
                    "status": c.get("status", None),
                },
                "source": {
                    "file": c.get("file"),
                    "line": c.get("line", None),
                    "code_snippet": c.get("code_snippet", None),
                    "comment": c.get("message"),
                    "user_note": c.get("user_note", None),
                    "rationale": c.get("rationale", None),
                    "verifiedBy": c.get("verifiedBy", None),
                    "pr_context": pr_context,
                    "gathered_at": gathered_at,
                    "raw_context_hash": hashlib.sha1(
                        context.encode("utf-8")
                    ).hexdigest()[:12],
                },
            }

            out_f.write(json.dumps(memory, ensure_ascii=False) + "\n")
            written += 1

            time.sleep(sleep_s)

    print(f"Memories written: {written} -> {out_path}")
    print(f"Rejected: {rejected} -> {reject_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build memories from code review JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file (creates new run)
  uv run python scripts/phase1/build_memories.py data/review_data/review_001.json

  # Process all JSON files in directory (creates new run)
  uv run python scripts/phase1/build_memories.py --all data/review_data

  # Use specific run (for re-processing)
  uv run python scripts/phase1/build_memories.py --all data/review_data --run-id run_20260208_143022
        """,
    )
    parser.add_argument(
        "path",
        help="Path to a JSON file or directory (when using --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all .json files in the specified directory",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Prompt semver to use (e.g. '2.0.0'). Defaults to latest.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Use existing run instead of creating new one (e.g. 'run_20260208_143022')",
    )

    args = parser.parse_args()

    # Determine run directory
    if args.run_id:
        run_dir = get_run(PHASE1, args.run_id)
        run_id = args.run_id
        print(f"Using existing run: {run_id}")
    else:
        run_id, run_dir = create_run(PHASE1)
        print(f"Created new run: {run_id}")

    out_dir = str(run_dir / "memories")
    print(f"Output directory: {out_dir}\n")

    if args.all:
        dir_path = Path(args.path)
        if not dir_path.is_dir():
            raise SystemExit(f"Error: {args.path} is not a directory")

        json_files = sorted(dir_path.glob("*.json"))
        if not json_files:
            raise SystemExit(f"No .json files found in {args.path}")

        print(f"Found {len(json_files)} JSON file(s) to process\n")

        total_success = 0
        total_failed = 0

        for i, json_file in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] Processing: {json_file.name}")
            try:
                build_memories(str(json_file), out_dir=out_dir, prompt_version=args.prompt_version)
                total_success += 1
            except Exception as e:
                print(f"Failed to process {json_file.name}: {e}")
                total_failed += 1
            print()

        print("=== Summary ===")
        print(f"Successfully processed: {total_success}")
        print(f"Failed: {total_failed}")
        print(f"Run: {run_id}")

        # Update run status
        update_run_status(run_dir, "build_memories", {
            "count": total_success,
            "failed": total_failed,
            "prompt_version": args.prompt_version,
        })
    else:
        build_memories(args.path, out_dir=out_dir, prompt_version=args.prompt_version)
        update_run_status(run_dir, "build_memories", {
            "count": 1,
            "prompt_version": args.prompt_version,
        })
