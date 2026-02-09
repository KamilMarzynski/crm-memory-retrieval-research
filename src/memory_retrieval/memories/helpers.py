import hashlib
from pathlib import Path


def short_repo_name(repo_path: str) -> str:
    if not repo_path:
        return "unknown"
    return Path(repo_path).name


def lang_from_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower().lstrip(".")
    return ext or "unknown"


def file_pattern(file_path: str) -> str:
    p = Path(file_path)
    ext = p.suffix.lstrip(".")
    glob = f"*.{ext}" if ext else "*"
    if len(p.parts) >= 2:
        return str(Path(*p.parts[:-1]) / glob)
    return str(p.name)


def confidence_map(c: str) -> float:
    return {"high": 1.0, "medium": 0.7, "low": 0.4}.get((c or "").lower(), 0.5)


def stable_id(raw_comment_id: str, situation: str, lesson: str) -> str:
    h = hashlib.sha1(
        (raw_comment_id + "\n" + situation + "\n" + lesson).encode("utf-8")
    ).hexdigest()
    return f"mem_{h[:12]}"
