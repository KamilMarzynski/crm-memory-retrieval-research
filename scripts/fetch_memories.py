"""Quick CLI to test memory retrieval."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_sqlite_fts import search_memories, get_memory_count

DB_PATH = "data/phase0_memories/memories.db"


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/fetch_memories.py <query>")
        print("Example: uv run python scripts/fetch_memories.py 'optional chaining'")
        sys.exit(1)

    query = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    total = get_memory_count(DB_PATH)
    print(f"Database: {total} memories")
    print(f"Query: {query!r}")
    print("-" * 60)

    results = search_memories(DB_PATH, query, limit=limit)

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['id']} (rank: {r['rank']:.2f})")
        print(f"    Situation: {r['situation_description']}")
        print(f"    Lesson: {r['lesson']}")


if __name__ == "__main__":
    main()
