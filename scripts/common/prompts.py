"""
Load and render versioned prompt templates from markdown files.

Prompt files live in <prompts_dir>/<name>/v<semver>.md and use section markers:

    ---system---
    System message with {variable} placeholders...

    ---user---
    User message with {variable} placeholders...

Usage:
    from common.prompts import load_prompt

    PROMPTS_DIR = Path(__file__).parent / "prompts"
    prompt = load_prompt("memory-situation", prompts_dir=PROMPTS_DIR)          # latest
    prompt = load_prompt("memory-situation", "1.0.0", prompts_dir=PROMPTS_DIR) # specific

    messages = prompt.render(context="...", file="...")
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


class _SafeDict(dict):
    """Dict subclass that returns empty string for missing keys in str.format_map()."""

    def __missing__(self, key: str) -> str:
        return ""


def _parse_semver(version: str) -> tuple:
    """Parse a semver string into a comparable tuple."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise ValueError(f"Invalid semver: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _parse_prompt_file(text: str) -> tuple:
    """
    Parse a prompt file into (system, user) template strings.

    Expects ---system--- and ---user--- section markers.
    """
    parts = re.split(r"^---(\w+)---\s*$", text, flags=re.MULTILINE)

    system = ""
    user = ""
    i = 1
    while i < len(parts) - 1:
        section_name = parts[i].strip()
        section_body = parts[i + 1]
        if section_name == "system":
            system = section_body.strip()
        elif section_name == "user":
            user = section_body.strip()
        i += 2

    if not system and not user:
        raise ValueError("Prompt file must contain ---system--- and/or ---user--- sections")

    return system, user


@dataclass(frozen=True)
class Prompt:
    """A versioned prompt template with system and user messages."""

    name: str
    version: str
    system: str
    user: str

    @property
    def version_tag(self) -> str:
        """Return a tag like 'memory-situation/v2.0.0' for metadata tracking."""
        return f"{self.name}/v{self.version}"

    def render(self, **kwargs) -> List[Dict[str, str]]:
        """
        Render the prompt templates with the given variables.

        Missing variables resolve to empty string.
        Returns list of message dicts suitable for chat completion API.
        """
        safe = _SafeDict(kwargs)
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system.format_map(safe)})
        if self.user:
            messages.append({"role": "user", "content": self.user.format_map(safe)})
        return messages


def load_prompt(
    name: str,
    version: Optional[str] = None,
    *,
    prompts_dir: Union[str, Path],
) -> Prompt:
    """
    Load a prompt template from <prompts_dir>/<name>/v<version>.md.

    Args:
        name: Prompt name (e.g. "memory-situation").
        version: Semver string (e.g. "2.0.0"). If None, picks the highest version.
        prompts_dir: Base directory containing prompt subdirectories.

    Returns:
        Prompt dataclass ready for rendering.

    Raises:
        FileNotFoundError: If no prompt files found or requested version missing.
        ValueError: If version string is not valid semver.
    """
    prompt_dir = Path(prompts_dir) / name
    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    if version is not None:
        # Load exact version
        file_path = prompt_dir / f"v{version}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        _parse_semver(version)  # validate
    else:
        # Find highest semver
        md_files = sorted(prompt_dir.glob("v*.md"))
        if not md_files:
            raise FileNotFoundError(f"No prompt files found in {prompt_dir}")

        best_file = None
        best_ver = (-1, -1, -1)
        for f in md_files:
            ver_str = f.stem[1:]  # strip leading 'v'
            try:
                ver_tuple = _parse_semver(ver_str)
                if ver_tuple > best_ver:
                    best_ver = ver_tuple
                    best_file = f
            except ValueError:
                continue

        if best_file is None:
            raise FileNotFoundError(f"No valid semver prompt files in {prompt_dir}")

        file_path = best_file
        version = best_file.stem[1:]

    text = file_path.read_text(encoding="utf-8")
    system, user = _parse_prompt_file(text)
    return Prompt(name=name, version=version, system=system, user=user)
