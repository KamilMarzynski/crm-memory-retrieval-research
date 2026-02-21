import re

# ---------------------------------------------------------------------------
# Embedding model registry
# (moved from search/vector.py — single source of truth)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    "mxbai-embed-large": 1024,
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-large": 1024,
    "bge-m3": 1024,
}

# ---------------------------------------------------------------------------
# Experiment defaults
# (moved from experiments/runner.py)
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_LIMIT: int = 20
DEFAULT_DISTANCE_THRESHOLD: float = 1.1
DEFAULT_RERANK_TOP_N: int = 5

# ---------------------------------------------------------------------------
# File exclusion patterns for diff filtering
# Pre-compiled at import time — never re-compiled per function call.
# (moved from experiments/test_cases.py)
# ---------------------------------------------------------------------------

EXCLUDED_FILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"package-lock\.json$"),
    re.compile(r"yarn\.lock$"),
    re.compile(r"pnpm-lock\.yaml$"),
    re.compile(r"\.snap$"),
    re.compile(r"__snapshots__/"),
    re.compile(r"\.min\.js$"),
    re.compile(r"\.min\.css$"),
    re.compile(r"\.map$"),
    re.compile(r"\.d\.ts$"),
    re.compile(r"dist/"),
    re.compile(r"build/"),
    re.compile(r"node_modules/"),
    re.compile(r"\.generated\."),
    re.compile(r"migrations/\d+"),
]
