"""
Extract memories from raw code review JSON files.

This script processes raw code review data using AI to extract reusable
engineering knowledge as structured memories.

Usage:
    # Process single file
    uv run python scripts/phase1/build_memories.py data/review_data/review_001.json

    # Process all JSON files in directory
    uv run python scripts/phase1/build_memories.py --all data/review_data
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import load_json, ensure_dir, call_openrouter

# Prompt version constants for tracking
MEMORY_PROMPT_VERSION_V1 = "situation_v1.0"
MEMORY_PROMPT_VERSION_V2 = "situation_v2.0"
TAGS_PROMPT_VERSION = "tags_v2.0"

# Current active version
ACTIVE_MEMORY_PROMPT_VERSION = MEMORY_PROMPT_VERSION_V2


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


def _prompt_situation_v1(context: str, c: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract situation description from code review comment (original prompt).

    Version: 1.0
    Output length: 40-450 chars (character-based validation)
    """
    code = (c.get("code_snippet") or "").strip()
    user_note = (c.get("user_note") or "").strip()
    file = c.get("file", "")
    severity = c.get("severity", "info")
    comment = c.get("message", "")

    system = (
        "You extract reusable engineering knowledge from code reviews. "
        "Your output will be used for retrieval. Be specific and concrete."
    )

    user = f"""
        PR CONTEXT:

        {context}

        FILE: {file}
        SEVERITY: {severity}
        CODE SNIPPET:

        {code}

        CODE REVIEW COMMENT:
        {comment}

        USER NOTE (optional)
        {user_note}

        Note: The user_note field is optional but critically important when provided. It gives you explicit guidance on what type of pattern to extract and how abstract or domain-specific your output should be.

        # Your Task

        Extract a reusable pattern from this code review comment that will be easy to retrieve by full text search query. Sentences should be short and concise.

        ## Understanding Abstraction Levels

        You need to determine the appropriate abstraction level for your output:

        **Technical/Abstract Pattern** - Focus on code structure and technical concerns, avoiding business domain terms
        - Example: "Method using optional chaining on nested objects. Redundant optional chaining."

        **Domain-Specific Pattern** - Include business context when it's essential to the pattern
        - Example: "Payment processing service handling refund requests,."

        ## Decision Process

        - If user_note is provided and indicates a need for domain context or business logic preservation → extract a domain-specific pattern
        - If user_note is provided and indicates technical focus → extract a technical/abstract pattern
        - If user_note is not provided or is ambiguous → default to technical/abstract pattern

          # RULES:
          - Be very CONCISE. Generate 1-2 sentences. 
          - Focus on the PATTERN/SITUATION
          - Describe WHEN this applies (what code pattern triggers this)
          - NEVER suggest code changes - it's not your role
          - Use technical terms (undefined, null, optional, edge case) over domain terms
          - Make it retrievable: think 'would this match similar situations in different domains?'
          - Output ONLY pattern description text (no meta text, no headers, no markdown)

         # GOOD EXAMPLES FOR TECHNICAL/ABSTRACT PATTERNS:
         - Test file for mapper method accepting optional object (Type | undefined). Missing test case for fully undefined parent object. Only tests undefined nested properties.
         - API mapper class renaming response fields. Fields consumed by external clients.
         - Service method using optional chaining on nested objects. Early returns might skip validation.
         - Validator helper processing optional config object. null vs undefined handled differently.
         - Missing tests for all possible conditional logic. Multiple if-else branches untested.
         # GOOD EXAMPLES FOR DOMAIN-SPECIFIC PATTERNS:
         - When calculating patient medication dosages in the pharmacy system. Verifying prescription object exists.
         - When updating SpecificModelMapper property called `foo` to `bar`. Database column mapping changed.
         - Payment refund processing service. Transaction settlement status check missing.

         # BAD EXAMPLES:
         - Be careful when changing code to handle edge cases. [too generic, no specifics]
         - API mapper class renaming response fields. Remember about updating tests. [suggests solution]
         - Service method using optional chaining on nested objects. Check if object is required. [suggests solution]
         - This code has a bug in the validation logic. [too vague, no pattern description]
         - Optional chaining. Nested objects. [too terse, lacks context]
         - The reviewOptionalChainingIssue function in UserService.ts needs null checks before accessing user.profile.settings.theme. [too specific, includes variable names and file names]."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _prompt_situation_v2(context: str, c: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract searchable situation description from code review comment.

    Version: 2.0
    Output length: 25-60 words (aligned with query length for vector similarity)

    Key improvements over v1:
    - Aligned length to 25-60 words (was 40-120) to match query length
    - Relaxed rigid structure - elements can appear in natural order
    - Focus on technical patterns, structure context, and gap identification
    """
    code = (c.get("code_snippet") or "").strip()
    user_note = (c.get("user_note") or "").strip()
    file = c.get("file", "")
    severity = c.get("severity", "info")
    comment = c.get("message", "")

    system = """You extract reusable code review patterns for a vector search database.

Your output will be retrieved by semantic similarity, so focus on:
1. Technical patterns (optional chaining, null checks, conditional logic)
2. Code structure context (test file, mapper, service, decorator)
3. The gap or issue (missing, lacks, inconsistent, redundant)

Write 1-2 sentences (25-60 words) that naturally incorporate these elements.
The elements can appear in any order that reads naturally.

Do NOT include:
- Solutions, advice, or "should" statements
- File names, variable names, or code identifiers
- Generic statements without specific patterns

GOOD EXAMPLES:
"Test file for mapper method accepting optional object. Missing test case for completely undefined parent object; only tests undefined nested properties."
"Service method using optional chaining on nested properties. Early return may skip downstream validation when parent object is null."
"Decorator applying configuration at multiple levels. Session setting duplicated between decorator parameter and server-level config."

BAD EXAMPLES:
"The UserMapper.ts file has a bug." (includes file name, too vague)
"Add null check before accessing user.profile." (includes solution)
"Be careful with optional properties." (too vague, no pattern)
"Authentication middleware needs better error handling." (domain term without pattern context)"""

    user = f"""FILE: {file}
SEVERITY: {severity}

CODE:
{code[:800] if code else "(none)"}

REVIEW COMMENT:
{comment}

{f"ADDITIONAL CONTEXT: {user_note}" if user_note else ""}

Extract the reusable pattern (25-60 words). Output ONLY the description text."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _prompt_lesson(situation: str, c: Dict[str, Any]) -> List[Dict[str, str]]:
    comment = c.get("message", "")
    rationale = (c.get("rationale") or "").strip()

    system = """You convert code review feedback into a single actionable lesson.
        Keep it concise and imperative."""

    user = f"""SITUATION:
        {situation}

        COMMENT:
        {comment}

        RATIONALE (optional):
        {rationale if rationale else "(none)"}
        TASK:
        Write ONE actionable lesson (imperative), max 160 characters.
        GOOD EXAMPLES:
        - Always add a deprecation period when renaming API fields to avoid breaking clients.
        - Avoid mutating reduce accumulators; return a new object to keep merging logic immutable.
        RULES:
        - One sentence.
        - Starts with an imperative cue: Always / Never / Ensure / Avoid / Verify / Check / Prefer.
        - Output ONLY the lesson text."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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


def build_memories(
    raw_path: str,
    out_dir: str = "data/phase1/memories",
    model: str = "anthropic/claude-haiku-4.5",
    sleep_s: float = 0.25,
    prompt_version: str = "v2",
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

            # Select prompt and validation based on version
            if prompt_version == "v2":
                prompt_fn = _prompt_situation_v2
                validate_fn = _validate_situation_v2
                version_tag = MEMORY_PROMPT_VERSION_V2
            else:
                prompt_fn = _prompt_situation_v1
                validate_fn = _validate_situation_v1
                version_tag = MEMORY_PROMPT_VERSION_V1

            situation = call_openrouter(
                api_key=api_key,
                model=model,
                messages=prompt_fn(context, c),
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

            lesson = call_openrouter(
                api_key=api_key,
                model=model,
                messages=_prompt_lesson(situation, c),
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
                    "prompt_version": version_tag,
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
  # Process single file
  uv run python scripts/phase1/build_memories.py data/review_data/review_001.json

  # Process all JSON files in directory
  uv run python scripts/phase1/build_memories.py --all data/review_data
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
        choices=["v1", "v2"],
        default="v2",
        help="Prompt version to use (default: v2 with aligned lengths)",
    )

    args = parser.parse_args()

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
                build_memories(str(json_file), prompt_version=args.prompt_version)
                total_success += 1
            except Exception as e:
                print(f"Failed to process {json_file.name}: {e}")
                total_failed += 1
            print()

        print("=== Summary ===")
        print(f"Successfully processed: {total_success}")
        print(f"Failed: {total_failed}")
    else:
        build_memories(args.path, prompt_version=args.prompt_version)
