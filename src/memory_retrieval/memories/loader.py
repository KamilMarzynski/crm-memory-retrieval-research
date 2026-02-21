import json
from pathlib import Path

from memory_retrieval.types import MemoryDict


def load_memories(memories_dir: str | Path) -> list[MemoryDict]:
    memories: list[MemoryDict] = []
    memories_path = Path(memories_dir)

    for jsonl_file in sorted(memories_path.glob("memories_*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        memories.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Skipping malformed JSON in {jsonl_file.name}:{line_num}: {e}"
                        )

    return memories
