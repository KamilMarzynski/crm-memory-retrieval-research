from memory_retrieval.memories.helpers import (
    confidence_map,
    file_pattern,
    lang_from_file,
    short_repo_name,
    stable_id,
)


def test_stable_id_is_deterministic() -> None:
    id1 = stable_id("comment_1", "situation text", "lesson text")
    id2 = stable_id("comment_1", "situation text", "lesson text")
    assert id1 == id2


def test_stable_id_starts_with_mem_prefix() -> None:
    memory_id = stable_id("comment_1", "situation", "lesson")
    assert memory_id.startswith("mem_")


def test_stable_id_hash_is_12_characters() -> None:
    memory_id = stable_id("comment_1", "situation", "lesson")
    hash_part = memory_id[len("mem_") :]
    assert len(hash_part) == 12


def test_stable_id_differs_for_different_comment_ids() -> None:
    id1 = stable_id("comment_1", "same situation", "same lesson")
    id2 = stable_id("comment_2", "same situation", "same lesson")
    assert id1 != id2


def test_stable_id_differs_for_different_content() -> None:
    id1 = stable_id("c1", "situation A", "lesson A")
    id2 = stable_id("c1", "situation B", "lesson B")
    assert id1 != id2


def test_lang_from_file_returns_lowercase_extension() -> None:
    assert lang_from_file("module.py") == "py"
    assert lang_from_file("component.ts") == "ts"
    assert lang_from_file("style.CSS") == "css"


def test_lang_from_file_no_extension_returns_unknown() -> None:
    assert lang_from_file("Makefile") == "unknown"
    assert lang_from_file("DOCKERFILE") == "unknown"


def test_file_pattern_nested_path_produces_extension_glob() -> None:
    pattern = file_pattern("src/components/Button.tsx")
    assert pattern == "src/components/*.tsx"


def test_file_pattern_top_level_file_returns_filename() -> None:
    pattern = file_pattern("main.py")
    assert pattern == "main.py"


def test_file_pattern_no_extension_produces_wildcard() -> None:
    pattern = file_pattern("src/Makefile")
    assert pattern == "src/*"


def test_short_repo_name_extracts_last_path_segment() -> None:
    assert short_repo_name("/home/user/projects/my-repo") == "my-repo"
    assert short_repo_name("https://github.com/org/my-project") == "my-project"


def test_short_repo_name_empty_string_returns_unknown() -> None:
    assert short_repo_name("") == "unknown"


def test_confidence_map_high() -> None:
    assert confidence_map("high") == 1.0


def test_confidence_map_medium() -> None:
    assert confidence_map("medium") == 0.7


def test_confidence_map_low() -> None:
    assert confidence_map("low") == 0.4


def test_confidence_map_unknown_string_returns_default() -> None:
    assert confidence_map("very_high") == 0.5
    assert confidence_map("unknown") == 0.5


def test_confidence_map_none_returns_default() -> None:
    assert confidence_map(None) == 0.5


def test_confidence_map_is_case_insensitive() -> None:
    assert confidence_map("HIGH") == 1.0
    assert confidence_map("Medium") == 0.7
    assert confidence_map("LOW") == 0.4
