from collections.abc import Callable
from typing import Any, Literal, NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Core domain types
# ---------------------------------------------------------------------------


class MemoryMetadata(TypedDict):
    repo: str
    file_pattern: str
    language: str
    severity: str
    confidence: str


class MemorySource(TypedDict):
    file: str
    line: int
    code_snippet: str
    comment: str
    pr_context: str


class MemoryDict(TypedDict):
    id: str
    situation_description: str
    lesson: str
    metadata: MemoryMetadata
    source: MemorySource


# ---------------------------------------------------------------------------
# Experiment pipeline types
# ---------------------------------------------------------------------------


class MetricsDict(TypedDict):
    precision: float
    recall: float
    f1_score: float
    reciprocal_rank: float


class QueryResult(TypedDict):
    query: str
    results: list[dict[str, Any]]
    num_results: int


class ExperimentResultEntry(TypedDict):
    test_case_id: str
    ground_truth_ids: list[str]
    query_results: list[QueryResult]
    pre_rerank_metrics: MetricsDict
    reranked_results: NotRequired[list[dict[str, Any]]]
    post_rerank_metrics: NotRequired[MetricsDict]


class ConfigFingerprint(TypedDict):
    extraction_prompt: str
    extraction_model: str
    query_prompt: str
    query_model: str
    search_limit: int
    distance_threshold: float
    reranker: str | None


class DiffStats(TypedDict):
    total_lines: int
    added_lines: int
    removed_lines: int
    num_files: int


class TestCase(TypedDict):
    test_case_id: str
    pr_context: str
    diff: str
    ground_truth_ids: list[str]
    diff_stats: DiffStats


# ---------------------------------------------------------------------------
# TypeAliases
# ---------------------------------------------------------------------------

# Function that extracts the text to rerank from a result dict
TextStrategyFn = Callable[[dict[str, Any]], str]

# Map of strategy name → text extraction function
RerankerStrategies = dict[str, TextStrategyFn]

# Validator function signature used in memories/validators.py
ValidatorFn = Callable[[str], tuple[bool, str]]

# Score type stored in SearchResult.score_type
ScoreType = Literal["cosine_distance", "bm25_rank", "rerank_score"]
