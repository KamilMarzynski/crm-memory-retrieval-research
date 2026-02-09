from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class SearchResult:
    id: str
    situation: str
    lesson: str
    metadata: dict[str, Any]
    source: dict[str, Any]
    score: float          # Higher = better (normalized)
    raw_score: float      # Original (BM25 rank or cosine distance)
    score_type: str       # "bm25_rank" | "cosine_distance" | "rerank_score"


class SearchBackend(Protocol):
    def create_database(self, db_path: str) -> None: ...
    def insert_memories(self, db_path: str, memories: list[dict[str, Any]]) -> int: ...
    def search(self, db_path: str, query: str, limit: int = 10) -> list[SearchResult]: ...
    def rebuild_database(self, db_path: str, memories_dir: str) -> None: ...
