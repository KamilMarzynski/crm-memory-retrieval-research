"""Tests for SentenceTransformerVectorBackend.

All tests mock `sentence_transformers.SentenceTransformer` to avoid downloading
the actual model during CI. The mock returns deterministic float32 normalised
vectors of the correct dimension.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memory_retrieval.search.vector import SentenceTransformerVectorBackend


PPLX_MODEL = "perplexity-ai/pplx-embed-v1-0.6B"
PPLX_DIMENSIONS = 1024


def _make_fake_embedding(dimensions: int) -> np.ndarray:
    """Return a deterministic normalised float32 vector."""
    vector = np.ones(dimensions, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _make_mock_sentence_transformer(dimensions: int) -> MagicMock:
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kwargs: np.stack(
        [_make_fake_embedding(dimensions) for _ in texts]
    )
    return mock_model


@pytest.fixture()
def backend_and_db(tmp_path: Path):
    """Return a SentenceTransformerVectorBackend with mocked model and fresh DB path."""
    mock_model = _make_mock_sentence_transformer(PPLX_DIMENSIONS)
    with patch("memory_retrieval.search.vector.SentenceTransformer", return_value=mock_model):
        backend = SentenceTransformerVectorBackend(model_name=PPLX_MODEL)
        # Trigger lazy load so the mock is captured
        _ = backend._model
        db_path = str(tmp_path / "memories.db")
        yield backend, db_path, mock_model


def test_dimensions_resolved_from_constants() -> None:
    with patch("memory_retrieval.search.vector.SentenceTransformer"):
        backend = SentenceTransformerVectorBackend(model_name=PPLX_MODEL)
    assert backend.vector_dimensions == PPLX_DIMENSIONS


def test_dimensions_override_accepted() -> None:
    with patch("memory_retrieval.search.vector.SentenceTransformer"):
        backend = SentenceTransformerVectorBackend(model_name=PPLX_MODEL, vector_dimensions=512)
    assert backend.vector_dimensions == 512


def test_model_lazy_loaded_on_first_embedding_call() -> None:
    mock_model = _make_mock_sentence_transformer(PPLX_DIMENSIONS)
    with patch(
        "memory_retrieval.search.vector.SentenceTransformer", return_value=mock_model
    ) as mock_class:
        backend = SentenceTransformerVectorBackend(model_name=PPLX_MODEL)
        mock_class.assert_not_called()
        backend._get_embedding("test text")
        mock_class.assert_called_once_with(PPLX_MODEL, trust_remote_code=True)


def test_get_embeddings_batch_calls_encode_once(backend_and_db) -> None:
    backend, _, mock_model = backend_and_db
    texts = ["first situation", "second situation", "third situation"]
    embeddings = backend._get_embeddings_batch(texts)
    mock_model.encode.assert_called_once()
    call_args = mock_model.encode.call_args[0][0]
    assert call_args == texts
    assert len(embeddings) == 3
    assert len(embeddings[0]) == PPLX_DIMENSIONS


def test_get_embedding_returns_list_of_floats(backend_and_db) -> None:
    backend, _, mock_model = backend_and_db
    embedding = backend._get_embedding("a situation")
    assert isinstance(embedding, list)
    assert all(isinstance(value, float) for value in embedding)
    assert len(embedding) == PPLX_DIMENSIONS


def test_create_database_creates_schema(backend_and_db) -> None:
    backend, db_path, _ = backend_and_db
    backend.create_database(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = {row[0] for row in cursor.fetchall()}
    conn.close()
    assert "memories" in table_names
    assert "vec_memories" in table_names


def test_insert_and_search_round_trip(backend_and_db) -> None:
    backend, db_path, _ = backend_and_db
    backend.create_database(db_path)
    memories = [
        {
            "id": "mem_aaa",
            "situation_description": "Developer forgot to handle null input",
            "lesson": "Always validate inputs at system boundaries.",
            "metadata": {},
            "source": {},
        }
    ]
    count = backend.insert_memories(db_path, memories)
    assert count == 1
    results = backend.search(db_path, "null input validation", limit=5)
    assert len(results) == 1
    assert results[0].id == "mem_aaa"
    assert results[0].score_type == "cosine_distance"


def test_insert_memories_returns_zero_for_empty_list(backend_and_db) -> None:
    backend, db_path, _ = backend_and_db
    backend.create_database(db_path)
    count = backend.insert_memories(db_path, [])
    assert count == 0
