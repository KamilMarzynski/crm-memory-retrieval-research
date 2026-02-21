from pathlib import Path
from typing import Any

import pytest

from memory_retrieval.search.fts5 import FTS5Backend


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


@pytest.fixture
def tmp_fts5_db(tmp_path: Path) -> str:
    """Create a temporary FTS5 database with two sample memories pre-inserted."""
    db_path = str(tmp_path / "memories.db")
    backend = FTS5Backend()
    backend.create_database(db_path)
    backend.insert_memories(
        db_path,
        [
            sample_memory(
                "mem_001",
                ["Python code has type errors when types are wrong."],
                "Type check everything.",
            ),
            sample_memory(
                "mem_002",
                ["Code review missing without proper test coverage."],
                "Write tests first.",
            ),
        ],
    )
    return db_path
