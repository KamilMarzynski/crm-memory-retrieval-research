from memory_retrieval.types import (
    ConfigFingerprint,
    DiffStats,
    ExperimentResultEntry,
    MemoryDict,
    MemoryMetadata,
    MemorySource,
    MetricsDict,
    QueryResult,
    ScoreType,
    TestCase,
    TextStrategyFn,
)


def test_memory_dict_structure() -> None:
    memory: MemoryDict = {
        "id": "mem_abc123",
        "situation_description": "A developer added a utility function without tests.",
        "lesson": "Always write tests for utility functions.",
        "metadata": {
            "repo": "my-repo",
            "file_pattern": "*.py",
            "language": "python",
            "severity": "medium",
            "confidence": "high",
        },
        "source": {
            "file": "utils.py",
            "line": 42,
            "code_snippet": "def helper(): pass",
            "comment": "Needs test",
            "pr_context": "main <- feature/add-helper",
        },
    }
    assert memory["id"] == "mem_abc123"


def test_metrics_dict_structure() -> None:
    metrics: MetricsDict = {
        "precision": 0.8,
        "recall": 0.75,
        "f1_score": 0.774,
        "reciprocal_rank": 1.0,
    }
    assert metrics["f1_score"] == 0.774


def test_score_type_literal() -> None:
    valid_types: list[ScoreType] = ["cosine_distance", "bm25_rank", "rerank_score"]
    assert len(valid_types) == 3


def test_text_strategy_fn_callable() -> None:
    strategy: TextStrategyFn = lambda result: result.get("situation", "")  # noqa: E731
    assert strategy({"situation": "hello"}) == "hello"
