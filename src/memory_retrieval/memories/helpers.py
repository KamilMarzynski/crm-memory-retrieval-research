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
    path = Path(file_path)
    ext = path.suffix.lstrip(".")
    glob = f"*.{ext}" if ext else "*"
    if len(path.parts) >= 2:
        return str(Path(*path.parts[:-1]) / glob)
    return str(path.name)


def confidence_map(confidence: str) -> float:
    return {"high": 1.0, "medium": 0.7, "low": 0.4}.get((confidence or "").lower(), 0.5)


def stable_id(raw_comment_id: str, situation: str, lesson: str) -> str:
    hash_object = hashlib.sha1(
        (raw_comment_id + "\n" + situation + "\n" + lesson).encode("utf-8")
    ).hexdigest()
    return f"mem_{hash_object[:12]}"


def get_confidence_from_distance(distance: float) -> str:
    """Map a cosine distance to a human-readable confidence label.

    Used to annotate vector search results with an interpretable confidence level.
    Lower distance = closer match = higher confidence.
    """
    if distance < 0.5:
        return "high"
    elif distance < 0.8:
        return "medium"
    elif distance < 1.2:
        return "low"
    else:
        return "very_low"
