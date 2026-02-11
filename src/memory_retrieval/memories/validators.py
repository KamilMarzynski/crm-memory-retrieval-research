from typing import Callable


def validate_situation_v1(text: str) -> tuple[bool, str]:
    """Validation for v1 prompts (40-450 chars)."""
    text_stripped = (text or "").strip()
    if len(text_stripped) < 40:
        return False, "situation_too_short"
    if len(text_stripped) > 450:
        return False, "situation_too_long"
    if not text_stripped.endswith((".", "!", "?")):
        return True, "ok_no_punct"
    return True, "ok"


def validate_situation_v2(text: str) -> tuple[bool, str]:
    """Validation for v2 prompts (25-60 words, aligned with query length)."""
    text_stripped = (text or "").strip()
    word_count = len(text_stripped.split())

    if word_count < 20:
        return False, f"situation_too_short_{word_count}_words"
    if word_count > 70:
        return False, f"situation_too_long_{word_count}_words"
    if not text_stripped.endswith((".", "!", "?")):
        return True, "ok_no_punct"
    return True, "ok"


def validate_lesson(text: str) -> tuple[bool, str]:
    text_stripped = (text or "").strip()
    if len(text_stripped) < 20:
        return False, "lesson_too_short"
    if len(text_stripped) > 220:
        return False, "lesson_too_long"
    if not text_stripped.endswith((".", "!", "?")):
        return True, "ok_no_punct"
    return True, "ok"


# Maps prompt versions to their situation validators
SITUATION_VALIDATORS: dict[str, Callable[[str], tuple[bool, str]]] = {
    "1.0.0": validate_situation_v1,
    "2.0.0": validate_situation_v2,
}


def get_situation_validator(version: str) -> Callable[[str], tuple[bool, str]]:
    """Get situation validator for a prompt version, defaulting to the latest."""
    if version in SITUATION_VALIDATORS:
        return SITUATION_VALIDATORS[version]
    return max(SITUATION_VALIDATORS.values(), key=lambda f: f.__name__)
