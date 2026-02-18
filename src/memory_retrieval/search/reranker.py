import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from memory_retrieval.memories.schema import FIELD_RERANK_SCORE, FIELD_SITUATION

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            print(f"Loading reranker model: {self.model_name}...")
            start = time.time()
            self._model = CrossEncoder(self.model_name)
            elapsed = time.time() - start
            print(f"Reranker model loaded in {elapsed:.1f}s")

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Score a single query against multiple documents."""
        return self.score_all_pairs([(query, doc) for doc in documents])

    def score_all_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a heterogeneous list of (query, document) pairs in a single model call.

        More efficient than score_pairs called repeatedly when queries differ,
        since model.predict() overhead is paid once for all pairs.
        """
        self._load_model()

        if not pairs:
            return []

        scores = self._model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int | None = None,
        text_field: str = FIELD_SITUATION,
        text_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        documents = (
            [text_fn(c) for c in candidates] if text_fn else [c[text_field] for c in candidates]
        )
        scores = self.score_pairs(query, documents)

        scored = []
        for candidate, score in zip(candidates, scores, strict=True):
            enriched = dict(candidate)
            enriched[FIELD_RERANK_SCORE] = score
            scored.append(enriched)

        scored.sort(key=lambda x: x[FIELD_RERANK_SCORE], reverse=True)

        if top_n is not None:
            scored = scored[:top_n]

        return scored
