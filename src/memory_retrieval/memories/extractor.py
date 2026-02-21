import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
    validate_situation_v1,
)


class SituationFormat(Enum):
    VARIANTS = "variants"  # 3 semicolon-separated (for FTS5)
    SINGLE = "single"  # Single description (for vector)


@dataclass
class ExtractionSummary:
    """Counts from a completed extraction run."""

    written: int = 0
    rejected: int = 0
    errors: int = 0


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


def _write_rejection(
    reject_file: Any,
    comment_id: Any,
    stage: str,
    reason: str,
    text: str = "",
) -> None:
    """Write a rejection entry to the reject JSONL file."""
    entry: dict[str, Any] = {"comment_id": comment_id, "stage": stage, "reason": reason}
    if text:
        entry["text"] = text
    reject_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


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


def _build_memory_dict(
    memory_id: str,
    situation: str,
    lesson: str,
    comment: dict[str, Any],
    pr_context: str,
    gathered_at: str,
    repo: str,
    config: ExtractionConfig,
    raw_context_hash: str = "",
    variants: list[str] | None = None,
    prompt_version_tag: str | None = None,
) -> dict[str, Any]:
    """Construct the memory dict from extracted situation and lesson."""
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
        raw_context_hash=raw_context_hash,
        variants=variants,
        prompt_version_tag=(
            situation_prompt.version_tag
            if config.situation_format == SituationFormat.SINGLE
            else None
        ),
    )


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
                time.sleep(config.sleep_s)
            else:
                summary.rejected += 1

    print(f"Memories written: {summary.written} -> {out_path}")
    print(f"Rejected: {summary.rejected} -> {reject_path}")
    return out_path
