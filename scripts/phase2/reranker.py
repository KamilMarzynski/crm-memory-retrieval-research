"""
Cross-encoder reranker for Phase 2 memory retrieval.

Wraps a sentence-transformers CrossEncoder model (bge-reranker-v2-m3)
to rerank vector search candidates by query-document relevance.

The reranker takes query-document pairs and produces relevance scores,
allowing more accurate ranking than bi-encoder cosine distance alone.

Usage:
    from phase2.reranker import Reranker

    reranker = Reranker()
    reranked = reranker.rerank(query="async error handling", candidates=results, top_n=4)

    # CLI test
    uv run python scripts/phase2/reranker.py --test "async error handling"
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase2.load_memories import FIELD_SITUATION, FIELD_RERANK_SCORE

# Model configuration
DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Lazily loads the model on first use to avoid startup cost when
    the reranker isn't needed.

    Attributes:
        model_name: HuggingFace model identifier for the cross-encoder.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the reranker.

        Args:
            model_name: HuggingFace model identifier. Defaults to bge-reranker-v2-m3.
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        """Load the CrossEncoder model (lazy initialization)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            print(f"Loading reranker model: {self.model_name}...")
            start = time.time()
            self._model = CrossEncoder(self.model_name)
            elapsed = time.time() - start
            print(f"Reranker model loaded in {elapsed:.1f}s")

    def score_pairs(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Score query-document pairs using the cross-encoder.

        Args:
            query: Search query string.
            documents: List of document texts to score against the query.

        Returns:
            List of relevance scores (higher = more relevant).
        """
        self._load_model()

        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        text_field: str = FIELD_SITUATION,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate memories by cross-encoder relevance.

        Takes vector search results and reranks them using query-document
        pair scoring. Each candidate gets a `rerank_score` field added.

        Args:
            query: Search query string.
            candidates: List of memory dictionaries from vector search.
                Each must contain the text_field key.
            top_n: Number of top results to return after reranking.
                None returns all candidates (sorted by rerank score).
            text_field: Key in candidate dicts containing the text to score.
                Defaults to FIELD_SITUATION ("situation_description").

        Returns:
            List of candidate dictionaries sorted by rerank_score (descending),
            truncated to top_n if specified. Each dict has `rerank_score` added.
        """
        if not candidates:
            return []

        documents = [c[text_field] for c in candidates]
        scores = self.score_pairs(query, documents)

        # Add rerank scores to candidates
        scored = []
        for candidate, score in zip(candidates, scores):
            enriched = dict(candidate)
            enriched[FIELD_RERANK_SCORE] = score
            scored.append(enriched)

        # Sort by rerank score (higher = more relevant)
        scored.sort(key=lambda x: x[FIELD_RERANK_SCORE], reverse=True)

        if top_n is not None:
            scored = scored[:top_n]

        return scored


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from common.runs import get_latest_run, get_run, PHASE1
    from phase1.db import search_memories

    parser = argparse.ArgumentParser(
        description="Phase 2 Reranker Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/phase2/reranker.py --test "async error handling"
  uv run python scripts/phase2/reranker.py --test "React component state" --top-n 5
  uv run python scripts/phase2/reranker.py --test "database query" --phase1-run-id run_20260208_143022
        """,
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Query string to test reranking with",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=4,
        help="Number of results after reranking (default: 4)",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=20,
        help="Number of vector search candidates (default: 20)",
    )
    parser.add_argument(
        "--phase1-run-id",
        default=None,
        help="Phase 1 run ID for database (default: latest)",
    )

    args = parser.parse_args()

    # Resolve Phase 1 run
    if args.phase1_run_id:
        phase1_run = get_run(PHASE1, args.phase1_run_id)
    else:
        phase1_run = get_latest_run(PHASE1)

    db_path = str(phase1_run / "memories" / "memories.db")
    print(f"Phase 1 run: {phase1_run.name}")
    print(f"Database: {db_path}")
    print(f"Query: {args.test}")
    print(f"Search limit: {args.search_limit}")
    print(f"Rerank top-n: {args.top_n}")
    print()

    # Vector search
    print("--- Vector Search Results ---")
    candidates = search_memories(db_path, args.test, limit=args.search_limit)
    for i, c in enumerate(candidates[:5], 1):
        print(f"  [{i}] dist={c['distance']:.4f} | {c['id']} | {c[FIELD_SITUATION][:80]}...")
    if len(candidates) > 5:
        print(f"  ... and {len(candidates) - 5} more")
    print()

    # Rerank
    reranker = Reranker()
    reranked = reranker.rerank(args.test, candidates, top_n=args.top_n)

    print(f"--- Reranked Results (top {args.top_n}) ---")
    for i, r in enumerate(reranked, 1):
        print(f"  [{i}] rerank={r[FIELD_RERANK_SCORE]:.4f} dist={r['distance']:.4f} | {r['id']}")
        print(f"      {r[FIELD_SITUATION][:100]}")
        print(f"      Lesson: {r['lesson'][:100]}")
        print()
