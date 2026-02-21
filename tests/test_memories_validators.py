from memory_retrieval.memories.validators import (
    get_situation_validator,
    validate_lesson,
    validate_situation_v1,
    validate_situation_v2,
)


# ---------- validate_situation_v1 (char-length based) ----------


def test_validate_situation_v1_too_short_fails() -> None:
    is_valid, reason = validate_situation_v1("Too short.")
    assert not is_valid
    assert reason == "situation_too_short"


def test_validate_situation_v1_too_long_fails() -> None:
    is_valid, reason = validate_situation_v1("x" * 451 + ".")
    assert not is_valid
    assert reason == "situation_too_long"


def test_validate_situation_v1_valid_with_terminal_punctuation() -> None:
    is_valid, reason = validate_situation_v1("a" * 50 + ".")
    assert is_valid
    assert reason == "ok"


def test_validate_situation_v1_valid_without_terminal_punctuation() -> None:
    is_valid, reason = validate_situation_v1("a" * 50)
    assert is_valid
    assert reason == "ok_no_punct"


def test_validate_situation_v1_empty_string_fails() -> None:
    is_valid, _ = validate_situation_v1("")
    assert not is_valid


# ---------- validate_situation_v2 (word-count based) ----------


def test_validate_situation_v2_too_few_words_fails() -> None:
    short_situation = "only three words."  # 3 words
    is_valid, reason = validate_situation_v2(short_situation)
    assert not is_valid
    assert "situation_too_short" in reason


def test_validate_situation_v2_too_many_words_fails() -> None:
    long_situation = " ".join(["word"] * 75) + "."
    is_valid, reason = validate_situation_v2(long_situation)
    assert not is_valid
    assert "situation_too_long" in reason


def test_validate_situation_v2_valid_range_with_punctuation() -> None:
    valid_situation = " ".join(["word"] * 30) + "."
    is_valid, reason = validate_situation_v2(valid_situation)
    assert is_valid
    assert reason == "ok"


def test_validate_situation_v2_valid_range_without_punctuation() -> None:
    valid_situation = " ".join(["word"] * 30)
    is_valid, reason = validate_situation_v2(valid_situation)
    assert is_valid
    assert reason == "ok_no_punct"


# ---------- validate_lesson ----------


def test_validate_lesson_too_short_fails() -> None:
    is_valid, reason = validate_lesson("Too short.")
    assert not is_valid
    assert reason == "lesson_too_short"


def test_validate_lesson_too_long_fails() -> None:
    is_valid, reason = validate_lesson("x" * 221 + ".")
    assert not is_valid
    assert reason == "lesson_too_long"


def test_validate_lesson_valid() -> None:
    lesson = "Always validate input before passing it to external APIs."
    is_valid, reason = validate_lesson(lesson)
    assert is_valid
    assert reason == "ok"


def test_validate_lesson_empty_string_fails() -> None:
    is_valid, _ = validate_lesson("")
    assert not is_valid


# ---------- get_situation_validator ----------


def test_get_situation_validator_returns_v1_for_version_1() -> None:
    validator = get_situation_validator("1.0.0")
    assert validator is validate_situation_v1


def test_get_situation_validator_returns_v2_for_version_2() -> None:
    validator = get_situation_validator("2.0.0")
    assert validator is validate_situation_v2


def test_get_situation_validator_unknown_version_returns_callable() -> None:
    validator = get_situation_validator("99.0.0")
    # Must be callable and produce the expected tuple shape
    is_valid, reason = validator("a" * 50 + ".")
    assert isinstance(is_valid, bool)
    assert isinstance(reason, str)
