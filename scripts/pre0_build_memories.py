import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


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


def _call_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 220,
    timeout_s: int = 60,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "memory-retrieval-research",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


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

        Extract a reusable pattern from this code review comment that can help engineers in similar situations.
        Sentences created by you should be maximally optimized for retrieval using full text search, with limited context.

        ## Understanding Abstraction Levels

        You need to determine the appropriate abstraction level for your output:

        **Technical/Abstract Pattern** - Focus on code structure and technical concerns, avoiding business domain terms
        - Example: "Service method using optional chaining (?.) on nested objects: early returns might skip validation."

        **Domain-Specific Pattern** - Include business context when it's essential to the pattern
        - Example: "Payment processing service handling refund requests: verify transaction settlement status before allowing partial refunds to prevent double-crediting."

        ## Decision Process

        - If user_note is provided and indicates a need for domain context or business logic preservation → extract a domain-specific pattern
        - If user_note is provided and indicates technical focus → extract a technical/abstract pattern  
        - If user_note is not provided or is ambiguous → default to technical/abstract pattern

        # RULES:
         - 2 sentences max - ideally, only five words each - targeted to be easy retrieved by full text search
         - Focus on the PATTERN/SITUATION
         - Describe WHEN this applies (what code pattern triggers this)
         - NEVER suggest code changes - it's not your role
         - Use technical terms (undefined, null, optional, edge case) over domain terms
         - Make it retrievable: think 'would this match similar situations in different domains?'
         - Avoid: specific variable names, business logic, file names, generic advice
         - Output ONLY the pattern description (no meta text, no headers, no markdown)
        # GOOD EXAMPLES FOR TECHNICAL/ABSTRACT PATTERNS:
         - Test file for mapper method accepting optional object (Type | undefined): missing test case for fully undefined parent object, only tests undefined nested properties.
         - API mapper class renaming response fields: fields consumed by external clients may break dependencies.
         - Service method using optional chaining (?.) on nested objects: early returns might skip validation.
         - Validator helper processing optional config object: null vs undefined handled differently.
        # GOOD EXAMPLES FOR DOMAIN-SPECIFIC PATTERNS:
         - When calculating patient medication dosages in the pharmacy system, verify the prescription object exists before accessing drug interaction fields.
         - When updating SpecificModelMapper property called `foo` to `bar`.
        # BAD EXAMPLES:
         - Be careful when changing code to handle edge cases. [too generic]"""
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

    raw = _load_json(raw_path)
    context = raw.get("context", "")
    meta = raw.get("meta", {})
    comments = raw.get("code_review_comments", [])

    _ensure_dir(out_dir)
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
            
            situation = _call_openrouter(
                api_key=api_key,
                model=model,
                messages=_prompt_situation(context, c),
                temperature=0.0,
                max_tokens=220,
            )
            ok_s, reason_s = _validate_situation(situation)
            if reason_s == "ok_no_punct":
                situation = situation.rstrip() + "."
            if not ok_s:
                rej_f.write(
                    json.dumps(
                        {"comment_id": c.get("id"), "stage": "situation", "reason": reason_s, "text": situation},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1
                continue

            time.sleep(sleep_s)

            lesson = _call_openrouter(
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
                "situation_description": situation,
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

    print(f"✓ memories written: {written} -> {out_path}")
    print(f"✓ rejected: {rejected} -> {reject_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build memories from code review JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python scripts/pre0_build_memories.py data/review_data/review_001.json

  # Process all JSON files in directory
  python scripts/pre0_build_memories.py --all data/review_data
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
        # Process all JSON files in directory
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
                print(f"✗ Failed to process {json_file.name}: {e}")
                total_failed += 1
            print()

        print("=== Summary ===")
        print(f"Successfully processed: {total_success}")
        print(f"Failed: {total_failed}")
    else:
        # Process single file
        build_memories(args.path)
