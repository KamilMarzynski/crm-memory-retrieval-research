from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


@dataclass(frozen=True)
class FigureExportSession:
    session_id: str
    session_dir: Path
    manifest_path: Path
    notebook_slug: str
    formats: tuple[str, ...]
    context: dict[str, Any]


def slugify_for_path(value: str, max_len: int = 120) -> str:
    if max_len <= 0:
        raise ValueError("max_len must be > 0")

    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()
    if not slug:
        slug = "na"
    return slug[:max_len].strip("-") or "na"


def create_figure_session(
    root_dir: Path,
    notebook_slug: str,
    context_key: str,
    context: dict[str, Any],
    formats: tuple[str, ...] = ("png", "svg"),
) -> FigureExportSession:
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    notebook_parts = _slugify_parts(notebook_slug)
    context_part = slugify_for_path(context_key)
    normalized_formats = _normalize_formats(formats)

    session_dir = Path(root_dir)
    for part in notebook_parts:
        session_dir = session_dir / part
    session_dir = session_dir / context_part / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "session_manifest.json"

    session = FigureExportSession(
        session_id=session_id,
        session_dir=session_dir,
        manifest_path=manifest_path,
        notebook_slug=notebook_slug,
        formats=normalized_formats,
        context=context,
    )
    _write_manifest(
        manifest_path,
        {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "notebook_slug": notebook_slug,
            "formats": list(normalized_formats),
            "context": _json_safe(context),
            "figures": [],
        },
    )
    return session


def save_figure(
    fig: Figure,
    session: FigureExportSession,
    figure_key: str,
    title: str | None = None,
    dpi: int = 220,
    close: bool = False,
) -> dict[str, Path]:
    key_slug = slugify_for_path(figure_key)
    manifest = _load_manifest(session.manifest_path)
    next_version = _next_figure_version(manifest, key_slug)
    version = _next_available_version(session.session_dir, session.formats, key_slug, next_version)
    file_stem = key_slug if version == 1 else f"{key_slug}__v{version:02d}"

    saved_paths: dict[str, Path] = {}
    for fmt in session.formats:
        output_path = session.session_dir / f"{file_stem}.{fmt}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_paths[fmt] = output_path

    manifest.setdefault("figures", []).append(
        {
            "figure_key": key_slug,
            "title": title,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "files": {fmt: str(path.resolve()) for fmt, path in saved_paths.items()},
        }
    )
    _write_manifest(session.manifest_path, manifest)

    if close:
        from matplotlib import pyplot as plt

        plt.close(fig)

    return saved_paths


def _normalize_formats(formats: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for fmt in formats:
        candidate = fmt.lower().lstrip(".")
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    if not normalized:
        raise ValueError("At least one figure format is required")
    return tuple(normalized)


def _slugify_parts(value: str) -> tuple[str, ...]:
    raw_parts = [part for part in re.split(r"[\\/]+", value) if part]
    if not raw_parts:
        return ("na",)
    return tuple(slugify_for_path(part) for part in raw_parts)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"figures": []}
    with open(path, encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)


def _next_figure_version(manifest: dict[str, Any], figure_key: str) -> int:
    versions = [
        int(entry.get("version", 1))
        for entry in manifest.get("figures", [])
        if entry.get("figure_key") == figure_key
    ]
    return max(versions, default=0) + 1


def _next_available_version(
    session_dir: Path,
    formats: tuple[str, ...],
    figure_key: str,
    requested_version: int,
) -> int:
    version = requested_version
    while True:
        file_stem = figure_key if version == 1 else f"{figure_key}__v{version:02d}"
        if all(not (session_dir / f"{file_stem}.{fmt}").exists() for fmt in formats):
            return version
        version += 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
