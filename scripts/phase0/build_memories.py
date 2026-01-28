"""
Extract memories from raw code review JSON files.

This script processes raw code review data using AI to extract reusable
engineering knowledge as structured memories.

Usage:
    # Process single file
    uv run python scripts/phase0/build_memories.py data/review_data/review_001.json

    # Process all JSON files in directory
    uv run python scripts/phase0/build_memories.py --all data/review_data
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
    h = hashlib.sha1((raw_comment_id + "\n" + situation + "\n" + lesson).encode("utf-8")).hexdigest()
    return f"mem_{h[:12]}"


def _prompt_situation(context: str, c: Dict[str, Any]) -> List[Dict[str, str]]:
    code = (c.get("code_snippet") or "").strip()
    user_note = (c.get("user_note") or "").strip()
    file = c.get("file", "")
    severity = c.get("severity", "info")
    comment = c.get("message", "")

    system = (
        "You extract reusable engineering knowledge from code reviews. "
        "Your output will be used for retrieval. Be specific and concrete."
    )

    user = (
        f"""
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
          - Be very CONCISE. Generate exactly 3 SHORT sentences separated by semicolons (;).
          - Focus on the PATTERN/SITUATION
          - Describe WHEN this applies (what code pattern triggers this)
          - NEVER suggest code changes - it's not your role
          - Use technical terms (undefined, null, optional, edge case) over domain terms
          - Make it retrievable: think 'would this match similar situations in different domains?'
          - Output ONLY the pattern description as 3 semicolon-separated sentences (no meta text, no headers, no markdown)

         # GOOD EXAMPLES FOR TECHNICAL/ABSTRACT PATTERNS:
         - Test file for mapper method accepting optional object (Type | undefined); Missing test case for fully undefined parent object; Only tests undefined nested properties.
         - API mapper class renaming response fields; Fields consumed by external clients; Breaking change risk.
         - Service method using optional chaining on nested objects; Early returns might skip validation; Chained ?. operator.
         - Validator helper processing optional config object; null vs undefined handled differently; Optional parameter validation.
         - Missing tests for all possible conditional logic; Multiple if-else branches untested; Edge case coverage incomplete.
         # GOOD EXAMPLES FOR DOMAIN-SPECIFIC PATTERNS:
         - When calculating patient medication dosages in the pharmacy system; Verifying prescription object exists; Drug interaction fields accessed.
         - When updating SpecificModelMapper property called `foo` to `bar`; Database column mapping changed; Migration script needed.
        - Payment refund processing service; Transaction settlement status check missing; Partial refund validation.

         # BAD EXAMPLES:
         - Be careful when changing code to handle edge cases. [too generic, no specifics]
         - API mapper class renaming response fields. Remember about updating tests. [suggests solution]
         - Service method using optional chaining on nested objects. Check if object is required. [suggests solution]
         - This code has a bug in the validation logic. [too vague, no pattern description]
         - Optional chaining; Nested objects; Validation. [too terse, lacks context]
         - The reviewOptionalChainingIssue function in UserService.ts needs null checks before accessing user.profile.settings.theme. [too specific, includes variable names and file names]."""
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _prompt_lesson(situation: str, c: Dict[str, Any]) -> List[Dict[str, str]]:
    comment = c.get("message", "")
    rationale = (c.get("rationale") or "").strip()

    system = (
        """You convert code review feedback into a single actionable lesson.
        Keep it concise and imperative."""
    )

    user = (
        f"""SITUATION:
        {situation}

        COMMENT:
        {comment}

        RATIONALE (optional):
        {rationale if rationale else '(none)'}
        TASK:
        Write ONE actionable lesson (imperative), max 160 characters.
        GOOD EXAMPLES:
        - Always add a deprecation period when renaming API fields to avoid breaking clients.
        - Avoid mutating reduce accumulators; return a new object to keep merging logic immutable.
        RULES:
        - One sentence.
        - Starts with an imperative cue: Always / Never / Ensure / Avoid / Verify / Check / Prefer.
        - Output ONLY the lesson text."""
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_situation(text: str) -> Tuple[bool, str]:
    t = (text or "").strip()
    if len(t) < 40:
        return False, "situation_too_short"
    if len(t) > 450:
        return False, "situation_too_long"
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
    out_dir: str = "data/phase0/memories",
    model: str = "anthropic/claude-haiku-4.5",
    sleep_s: float = 0.25,
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
    pr_context = f"{meta.get('sourceBranch','?')} → {meta.get('targetBranch','?')}"
    gathered_at = meta.get("gatheredAt", "")

    written = 0
    rejected = 0

    with open(out_path, "w", encoding="utf-8") as out_f, open(reject_path, "w", encoding="utf-8") as rej_f:
        for c in comments:
            if c.get("status") == 'rejected':
                rej_f.write(
                    json.dumps(
                        {"comment_id": c.get("id"), "stage": "status", "reason": "comment_rejected"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1
                continue

            # Generate 3 semicolon-separated situation variants
            situation_raw = call_openrouter(
                api_key=api_key,
                model=model,
                messages=_prompt_situation(context, c),
                temperature=0.0,
                max_tokens=600,
            )

            # Parse semicolon-separated variants
            variants = [v.strip() for v in situation_raw.split(';')]

            # Validate we got exactly 3 variants
            if len(variants) != 3:
                rej_f.write(
                    json.dumps(
                        {
                            "comment_id": c.get("id"),
                            "stage": "situation",
                            "reason": f"wrong_variant_count_{len(variants)}",
                            "text": situation_raw
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1
                continue

            # Validate each variant individually
            all_valid = True
            for i, variant in enumerate(variants):
                ok_s, reason_s = _validate_situation(variant)
                if reason_s == "ok_no_punct":
                    variants[i] = variant.rstrip() + "."
                elif not ok_s:
                    rej_f.write(
                        json.dumps(
                            {
                                "comment_id": c.get("id"),
                                "stage": "situation",
                                "reason": f"variant_{i}_{reason_s}",
                                "text": variant
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

            # Use first variant as primary display
            situation = variants[0]

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
                        {"comment_id": c.get("id"), "stage": "lesson", "reason": reason_l, "text": lesson},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1
                continue

            original_conf = _confidence_map(c.get("confidence", "medium"))

            memory = {
                "id": _stable_id(c.get("id", str(datetime.now().timestamp())), situation, lesson),
                "situation_variants": variants,
                "lesson": lesson,
                "metadata": {
                    "repo": repo,
                    "file_pattern": _file_pattern(c.get("file", "")),
                    "language": _lang_from_file(c.get("file", "")),
                    "tags": [],
                    "severity": c.get("severity", "info"),
                    "confidence": original_conf,
                    "author": "pre0-openrouter",
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
                    "raw_context_hash": hashlib.sha1(context.encode("utf-8")).hexdigest()[:12],
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
  uv run python scripts/phase0/build_memories.py data/review_data/review_001.json

  # Process all JSON files in directory
  uv run python scripts/phase0/build_memories.py --all data/review_data
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
                build_memories(str(json_file))
                total_success += 1
            except Exception as e:
                print(f"Failed to process {json_file.name}: {e}")
                total_failed += 1
            print()

        print("=== Summary ===")
        print(f"Successfully processed: {total_success}")
        print(f"Failed: {total_failed}")
    else:
        build_memories(args.path)
