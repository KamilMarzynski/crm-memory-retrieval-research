import re

from memory_retrieval.constants import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_SEARCH_LIMIT,
    EMBEDDING_MODEL_DIMENSIONS,
    EXCLUDED_FILE_PATTERNS,
)


def test_embedding_model_dimensions_has_expected_models() -> None:
    assert "mxbai-embed-large" in EMBEDDING_MODEL_DIMENSIONS
    assert EMBEDDING_MODEL_DIMENSIONS["mxbai-embed-large"] == 1024


def test_excluded_file_patterns_are_precompiled() -> None:
    assert all(isinstance(pattern, re.Pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def test_excluded_patterns_match_expected_files() -> None:
    # Convert to set of pattern strings for easy assertion
    pattern_strings = [p.pattern for p in EXCLUDED_FILE_PATTERNS]
    combined = "|".join(pattern_strings)
    assert re.search(combined, "package-lock.json")
    assert re.search(combined, "yarn.lock")
    assert re.search(combined, "app.min.js")


def test_default_experiment_values() -> None:
    assert DEFAULT_SEARCH_LIMIT == 20
    assert DEFAULT_DISTANCE_THRESHOLD == 1.1
    assert DEFAULT_RERANK_TOP_N == 5


def test_embedding_model_dimensions_includes_pplx_models() -> None:
    expected_pplx_models = {
        "perplexity-ai/pplx-embed-v1-0.6B": 1024,
        "perplexity-ai/pplx-embed-v1-4B": 2560,
        "perplexity-ai/pplx-embed-context-v1-0.6B": 1024,
        "perplexity-ai/pplx-embed-context-v1-4B": 2560,
    }
    for model_name, expected_dimensions in expected_pplx_models.items():
        assert model_name in EMBEDDING_MODEL_DIMENSIONS, f"Missing: {model_name}"
        assert EMBEDDING_MODEL_DIMENSIONS[model_name] == expected_dimensions, (
            f"{model_name}: expected {expected_dimensions}, "
            f"got {EMBEDDING_MODEL_DIMENSIONS[model_name]}"
        )
