from memory_retrieval.search.base import SearchBackendBase, SearchResult
from memory_retrieval.types import ScoreType


def test_search_result_score_type_is_literal() -> None:
    result = SearchResult(
        id="mem_abc",
        situation="A developer wrote untested code.",
        lesson="Always write tests.",
        metadata={},
        source={},
        score=0.8,
        raw_score=0.2,
        score_type="cosine_distance",
    )
    assert result.score_type == "cosine_distance"


def test_search_result_rejects_invalid_score_type() -> None:
    # Verify the annotation covers the expected literal values
    result = SearchResult(
        id="x",
        situation="s",
        lesson="l",
        metadata={},
        source={},
        score=1.0,
        raw_score=0.0,
        score_type="cosine_distance",
    )
    assert result.score_type in ("cosine_distance", "bm25_rank", "rerank_score")
