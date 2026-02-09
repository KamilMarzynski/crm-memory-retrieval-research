import time
from typing import TYPE_CHECKING, Any

from memory_retrieval.memories.schema import FIELD_SITUATION, FIELD_RERANK_SCORE

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
        self._load_model()

        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int | None = None,
        text_field: str = FIELD_SITUATION,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        documents = [c[text_field] for c in candidates]
        scores = self.score_pairs(query, documents)

        scored = []
        for candidate, score in zip(candidates, scores):
            enriched = dict(candidate)
            enriched[FIELD_RERANK_SCORE] = score
            scored.append(enriched)

        scored.sort(key=lambda x: x[FIELD_RERANK_SCORE], reverse=True)

        if top_n is not None:
            scored = scored[:top_n]

        return scored
