from typing import Any


def sample_memory(
    memory_id: str = "mem_test001",
    variants: list[str] | None = None,
    lesson: str = "Always validate your inputs.",
) -> dict[str, Any]:
    """Return a canonical memory dict for use in tests."""
    if variants is None:
        variants = ["A developer added code without proper input validation."]
    return {
        "id": memory_id,
        "situation_variants": variants,
        "lesson": lesson,
        "metadata": {"repo": "test-repo", "language": "py"},
        "source": {"file": "main.py"},
    }
