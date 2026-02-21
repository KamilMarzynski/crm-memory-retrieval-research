from pathlib import Path

import pytest

from memory_retrieval.infra.prompts import Prompt, _parse_prompt_file, _parse_semver, load_prompt


# ---------- _parse_semver ----------


def test_parse_semver_valid_string_returns_tuple() -> None:
    assert _parse_semver("1.2.3") == (1, 2, 3)
    assert _parse_semver("0.0.1") == (0, 0, 1)
    assert _parse_semver("10.20.30") == (10, 20, 30)


def test_parse_semver_too_few_parts_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _parse_semver("1.2")


def test_parse_semver_non_numeric_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _parse_semver("not.a.version")


def test_parse_semver_with_prefix_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _parse_semver("v1.2.3")


# ---------- _parse_prompt_file ----------


def test_parse_prompt_file_extracts_system_and_user_sections() -> None:
    content = "---system---\nSystem instructions here.\n---user---\nUser template here.\n"
    system, user = _parse_prompt_file(content)
    assert system == "System instructions here."
    assert user == "User template here."


def test_parse_prompt_file_no_valid_sections_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _parse_prompt_file("just plain text with no sections at all")


def test_parse_prompt_file_multiline_sections() -> None:
    content = "---system---\nLine one.\nLine two.\n---user---\nQuery: {query}\n"
    system, user = _parse_prompt_file(content)
    assert "Line one." in system
    assert "Line two." in system
    assert "{query}" in user


# ---------- Prompt.render ----------


def test_prompt_render_substitutes_all_kwargs() -> None:
    prompt = Prompt(name="test", version="1.0.0", system="Hello {name}.", user="Query: {query}.")
    messages = prompt.render(name="world", query="find bugs")
    assert messages[0]["content"] == "Hello world."
    assert messages[1]["content"] == "Query: find bugs."


def test_prompt_render_missing_key_becomes_empty_string() -> None:
    prompt = Prompt(name="test", version="1.0.0", system="Value: {missing_key}.", user="")
    messages = prompt.render()
    assert messages[0]["content"] == "Value: ."


def test_prompt_render_returns_system_and_user_messages() -> None:
    prompt = Prompt(name="test", version="1.0.0", system="System.", user="User.")
    messages = prompt.render()
    roles = [message["role"] for message in messages]
    assert "system" in roles
    assert "user" in roles


def test_prompt_version_tag_property() -> None:
    prompt = Prompt(name="my-prompt", version="2.1.0", system="", user="")
    assert prompt.version_tag == "my-prompt/v2.1.0"


# ---------- load_prompt ----------


def _write_prompt_file(directory: Path, version: str, system: str, user: str) -> None:
    content = f"---system---\n{system}\n---user---\n{user}\n"
    (directory / f"v{version}.md").write_text(content, encoding="utf-8")


def test_load_prompt_loads_specific_version(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "my-prompt"
    prompt_dir.mkdir()
    _write_prompt_file(prompt_dir, "1.0.0", "System v1.", "User v1.")
    _write_prompt_file(prompt_dir, "2.0.0", "System v2.", "User v2.")

    prompt = load_prompt("my-prompt", version="1.0.0", prompts_dir=tmp_path)
    assert prompt.system == "System v1."
    assert prompt.version == "1.0.0"


def test_load_prompt_loads_latest_when_version_is_none(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "my-prompt"
    prompt_dir.mkdir()
    _write_prompt_file(prompt_dir, "1.0.0", "System v1.", "User v1.")
    _write_prompt_file(prompt_dir, "2.0.0", "System v2.", "User v2.")
    _write_prompt_file(prompt_dir, "1.5.0", "System v1.5.", "User v1.5.")

    prompt = load_prompt("my-prompt", prompts_dir=tmp_path)
    assert prompt.system == "System v2."
    assert prompt.version == "2.0.0"


def test_load_prompt_missing_directory_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent-prompt", prompts_dir=tmp_path)


def test_load_prompt_missing_version_raises_file_not_found_error(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "my-prompt"
    prompt_dir.mkdir()
    _write_prompt_file(prompt_dir, "1.0.0", "System.", "User.")

    with pytest.raises(FileNotFoundError):
        load_prompt("my-prompt", version="9.9.9", prompts_dir=tmp_path)


def test_load_prompt_name_is_set_correctly(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "query-builder"
    prompt_dir.mkdir()
    _write_prompt_file(prompt_dir, "1.0.0", "System.", "User.")

    prompt = load_prompt("query-builder", prompts_dir=tmp_path)
    assert prompt.name == "query-builder"
