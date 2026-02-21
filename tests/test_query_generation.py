import pytest

from memory_retrieval.experiments.query_generation import (
    QueryGenerationConfig,
    _parse_json_array_queries,
    _parse_newline_queries,
    _parse_quoted_queries,
    parse_queries_robust,
)


def test_parse_json_array_queries_valid_json() -> None:
    response = '["query one about testing", "query two about linting"]'
    result = _parse_json_array_queries(response)
    assert result == ["query one about testing", "query two about linting"]


def test_parse_json_array_queries_embedded_in_text() -> None:
    response = 'Here are queries: ["what happens when tests fail", "how to write assertions"]'
    result = _parse_json_array_queries(response)
    assert result is not None
    assert len(result) == 2


def test_parse_json_array_queries_returns_none_on_failure() -> None:
    assert _parse_json_array_queries("no json here") is None
    assert _parse_json_array_queries('{"not": "an array"}') is None


def test_parse_quoted_queries_extracts_quoted_strings() -> None:
    response = """
    1. "Always validate input at system boundaries"
    2. "Use type hints for public API functions"
    """
    result = _parse_quoted_queries(response)
    assert result is not None
    assert len(result) == 2
    assert "Always validate input at system boundaries" in result


def test_parse_newline_queries_cleans_list_formatting() -> None:
    response = """
    1. Write failing test before implementation
    - Use descriptive variable names over abbreviations
    * Check return types on public functions
    """
    result = _parse_newline_queries(response)
    assert len(result) >= 2
    assert all(not line.startswith(("1.", "-", "*")) for line in result)


def test_parse_queries_robust_falls_back_gracefully() -> None:
    # Valid JSON array
    assert (
        len(parse_queries_robust('["a query with enough words here", "another one also long"]'))
        == 2
    )
    # Quoted strings fallback
    assert len(parse_queries_robust('"a query with enough words here"')) >= 1
    # Returns list even for garbage
    result = parse_queries_robust("no queries here")
    assert isinstance(result, list)


def test_query_generation_config_validates_max_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        QueryGenerationConfig(max_workers=0)
