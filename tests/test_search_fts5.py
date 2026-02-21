from pathlib import Path

import pytest

from memory_retrieval.search.fts5 import FTS5Backend


def _make_memory(memory_id: str, variants: list[str], lesson: str) -> dict:
    return {
        "id": memory_id,
        "situation_variants": variants,
        "lesson": lesson,
        "metadata": {"repo": "test-repo"},
        "source": {},
    }


@pytest.fixture
def empty_fts5_db(tmp_path: Path) -> str:
    """An FTS5 database with schema created but no memories inserted."""
    db_path = str(tmp_path / "test.db")
    FTS5Backend().create_database(db_path)
    return db_path


def test_search_returns_matching_memory(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [
            _make_memory(
                "mem_001",
                ["Python type errors occur when passing wrong arguments"],
                "Always type check.",
            )
        ],
    )
    results = backend.search(empty_fts5_db, "Python type", limit=5)
    assert len(results) == 1
    assert results[0].id == "mem_001"


def test_search_returns_correct_lesson(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [
            _make_memory(
                "mem_001", ["A situation involving authentication"], "Always validate tokens."
            )
        ],
    )
    results = backend.search(empty_fts5_db, "authentication")
    assert results[0].lesson == "Always validate tokens."


def test_search_deduplicates_multiple_matching_variants(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [
            _make_memory(
                "mem_001",
                ["test variant one about validation", "test variant two about validation"],
                "Lesson A.",
            )
        ],
    )
    # Both variants match "test", but only one result should be returned
    results = backend.search(empty_fts5_db, "test")
    assert len(results) == 1
    assert results[0].id == "mem_001"


def test_search_returns_no_results_for_unmatched_query(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["Python code review scenario"], "Use type hints.")],
    )
    results = backend.search(empty_fts5_db, "totally_unrelated_zzz")
    assert results == []


def test_search_score_is_positive_and_score_type_is_bm25(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["keyword search example with relevant terms"], "Lesson.")],
    )
    results = backend.search(empty_fts5_db, "keyword")
    assert results[0].score > 0
    assert results[0].score_type == "bm25_rank"


def test_search_respects_limit(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    memories = [
        _make_memory(f"mem_{index:03}", [f"test situation {index} about code"], f"Lesson {index}.")
        for index in range(10)
    ]
    backend.insert_memories(empty_fts5_db, memories)
    results = backend.search(empty_fts5_db, "test situation", limit=3)
    assert len(results) <= 3


def test_get_memory_count_returns_correct_count(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    memories = [
        _make_memory("m1", ["first situation"], "First lesson."),
        _make_memory("m2", ["second situation"], "Second lesson."),
        _make_memory("m3", ["third situation"], "Third lesson."),
    ]
    backend.insert_memories(empty_fts5_db, memories)
    assert backend.get_memory_count(empty_fts5_db) == 3


def test_get_memory_by_id_returns_full_memory(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    backend.insert_memories(
        empty_fts5_db,
        [_make_memory("mem_001", ["situation description text"], "Use this lesson.")],
    )
    memory = backend.get_memory_by_id(empty_fts5_db, "mem_001")
    assert memory is not None
    assert memory["id"] == "mem_001"
    assert "situation description text" in memory["situation_variants"]
    assert memory["lesson"] == "Use this lesson."


def test_get_memory_by_id_returns_none_for_nonexistent_id(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    memory = backend.get_memory_by_id(empty_fts5_db, "mem_nonexistent")
    assert memory is None


def test_insert_memories_returns_inserted_count(empty_fts5_db: str) -> None:
    backend = FTS5Backend()
    memories = [
        _make_memory("m1", ["first"], "First."),
        _make_memory("m2", ["second"], "Second."),
    ]
    count = backend.insert_memories(empty_fts5_db, memories)
    assert count == 2
