import hashlib
import json
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from memory_retrieval.infra.io import ensure_dir, load_json
from memory_retrieval.infra.llm import call_openrouter
from memory_retrieval.infra.prompts import load_prompt
from memory_retrieval.memories.helpers import (
    confidence_map,
    file_pattern,
    lang_from_file,
    short_repo_name,
    stable_id,
)
from memory_retrieval.memories.validators import (
    get_situation_validator,
    validate_lesson,
)


class SituationFormat(Enum):
    VARIANTS = "variants"  # 3 semicolon-separated (for FTS5)
    SINGLE = "single"  # Single description (for vector)


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
        self.situation_format = situation_format
        self.prompts_dir = Path(prompts_dir)
        self.prompt_version = prompt_version
        self.model = model
        self.sleep_s = sleep_s
        self.author_tag = author_tag


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
        "memory-situation",
        version=config.prompt_version,
        prompts_dir=config.prompts_dir,
    )
    lesson_prompt = load_prompt("memory-lesson", prompts_dir=config.prompts_dir)

    if config.situation_format == SituationFormat.SINGLE:
        validate_situation = get_situation_validator(situation_prompt.version)
    else:
        from memory_retrieval.memories.validators import validate_situation_v1

        validate_situation = validate_situation_v1

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

            # --- Extract situation ---
            code = (c.get("code_snippet") or "").strip()
            user_note = (c.get("user_note") or "").strip()

            if config.situation_format == SituationFormat.VARIANTS:
                situation_raw = call_openrouter(
                    api_key=api_key,
                    model=config.model,
                    messages=situation_prompt.render(
                        context=context,
                        file=c.get("file", ""),
                        severity=c.get("severity", "info"),
                        code=code if code else "(none)",
                        comment=c.get("message", ""),
                        user_note=user_note,
                    ),
                    temperature=0.0,
                    max_tokens=600,
                )
                variants = [v.strip() for v in situation_raw.split(";")]

                if len(variants) != 3:
                    rej_f.write(
                        json.dumps(
                            {
                                "comment_id": c.get("id"),
                                "stage": "situation",
                                "reason": f"wrong_variant_count_{len(variants)}",
                                "text": situation_raw,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    rejected += 1
                    continue

                all_valid = True
                for i, variant in enumerate(variants):
                    ok_s, reason_s = validate_situation(variant)
                    if reason_s == "ok_no_punct":
                        variants[i] = variant.rstrip() + "."
                    elif not ok_s:
                        rej_f.write(
                            json.dumps(
                                {
                                    "comment_id": c.get("id"),
                                    "stage": "situation",
                                    "reason": f"variant_{i}_{reason_s}",
                                    "text": variant,
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

                situation = variants[0]

            else:
                additional_context = f"ADDITIONAL CONTEXT: {user_note}" if user_note else ""

                situation = call_openrouter(
                    api_key=api_key,
                    model=config.model,
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

                ok_s, reason_s = validate_situation(situation)
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
                    rejected += 1
                    continue

            time.sleep(config.sleep_s)

            # --- Extract lesson ---
            rationale = (c.get("rationale") or "").strip()
            lesson_messages = lesson_prompt.render(
                situation=situation,
                comment=c.get("message", ""),
                rationale=rationale if rationale else "(none)",
            )

            lesson = call_openrouter(
                api_key=api_key,
                model=config.model,
                messages=lesson_messages,
                temperature=0.0,
                max_tokens=120,
            )
            ok_l, reason_l = validate_lesson(lesson)
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

            original_conf = confidence_map(c.get("confidence", "medium"))

            # --- Build memory object ---
            memory: dict[str, Any] = {
                "id": stable_id(
                    c.get("id", str(datetime.now().timestamp())),
                    situation,
                    lesson,
                ),
                "lesson": lesson,
                "metadata": {
                    "repo": repo,
                    "file_pattern": file_pattern(c.get("file", "")),
                    "language": lang_from_file(c.get("file", "")),
                    "tags": [],
                    "severity": c.get("severity", "info"),
                    "confidence": original_conf,
                    "author": config.author_tag,
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

            if config.situation_format == SituationFormat.VARIANTS:
                memory["situation_variants"] = variants
            else:
                memory["situation_description"] = situation
                memory["metadata"]["prompt_version"] = situation_prompt.version_tag

            out_f.write(json.dumps(memory, ensure_ascii=False) + "\n")
            written += 1

            time.sleep(config.sleep_s)

    print(f"Memories written: {written} -> {out_path}")
    print(f"Rejected: {rejected} -> {reject_path}")
    return out_path
