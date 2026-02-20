import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from memory_retrieval.memories.schema import FIELD_RERANK_SCORE, FIELD_SITUATION

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

QWEN3_RERANKER_MODELS = {
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    "tomaarsen/Qwen3-Reranker-4B-seq-cls",
    "tomaarsen/Qwen3-Reranker-8B-seq-cls",
}

DEFAULT_TASK_INSTRUCTION = (
    "Given a query based on code review data, retrieve relevant past review descriptions"
    " that match the described situation"
)

QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided."
    ' Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        task_instruction: str = DEFAULT_TASK_INSTRUCTION,
    ):
        self.model_name = model_name
        self.task_instruction = task_instruction
        self._is_qwen3 = model_name in QWEN3_RERANKER_MODELS
        self._model: CrossEncoder | None = None

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            print(f"Loading reranker model: {self.model_name}...")
            start = time.time()
            self._model = CrossEncoder(self.model_name)
            elapsed = time.time() - start
            print(f"Reranker model loaded in {elapsed:.1f}s")

    def _format_pair(self, query: str, document: str) -> tuple[str, str]:
        """Format a (query, document) pair for the model.

        BGE models use raw strings. Qwen3 models require chat-template wrapping
        with instruction, query, and document tags.
        """
        if not self._is_qwen3:
            return (query, document)

        formatted_query = f"{QWEN3_PREFIX}<Instruct>: {self.task_instruction}\n<Query>: {query}\n"
        formatted_document = f"<Document>: {document}{QWEN3_SUFFIX}"
        return (formatted_query, formatted_document)

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

        formatted_pairs = [self._format_pair(query, document) for query, document in pairs]
        scores = self._model.predict(formatted_pairs)
        return [float(score) for score in scores]

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
