import re
from dataclasses import dataclass
from pathlib import Path


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _parse_semver(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise ValueError(f"Invalid semver: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _parse_prompt_file(text: str) -> tuple[str, str]:
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
    name: str
    version: str
    system: str
    user: str

    @property
    def version_tag(self) -> str:
        return f"{self.name}/v{self.version}"

    def render(self, **kwargs: str) -> list[dict[str, str]]:
        safe = _SafeDict(kwargs)
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system.format_map(safe)})
        if self.user:
            messages.append({"role": "user", "content": self.user.format_map(safe)})
        return messages


def load_prompt(
    name: str,
    version: str | None = None,
    *,
    prompts_dir: str | Path,
) -> Prompt:
    prompt_dir = Path(prompts_dir) / name
    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    if version is not None:
        file_path = prompt_dir / f"v{version}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        _parse_semver(version)
    else:
        md_files = sorted(prompt_dir.glob("v*.md"))
        if not md_files:
            raise FileNotFoundError(f"No prompt files found in {prompt_dir}")

        best_file = None
        best_ver = (-1, -1, -1)
        for md_file in md_files:
            ver_str = md_file.stem[1:]
            try:
                ver_tuple = _parse_semver(ver_str)
                if ver_tuple > best_ver:
                    best_ver = ver_tuple
                    best_file = md_file
            except ValueError:
                continue

        if best_file is None:
            raise FileNotFoundError(f"No valid semver prompt files in {prompt_dir}")

        file_path = best_file
        version = best_file.stem[1:]

    text = file_path.read_text(encoding="utf-8")
    system, user = _parse_prompt_file(text)
    return Prompt(name=name, version=version, system=system, user=user)
