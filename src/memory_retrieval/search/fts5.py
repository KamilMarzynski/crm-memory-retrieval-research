import json
import sqlite3
from typing import Any

from memory_retrieval.memories.loader import load_memories
from memory_retrieval.memories.schema import (
    FIELD_ID,
    FIELD_LESSON,
    FIELD_METADATA,
    FIELD_SOURCE,
    FIELD_VARIANTS,
    FIELD_RANK,
)
from memory_retrieval.search.base import SearchResult
from memory_retrieval.search.db_utils import (
    get_db_connection,
    serialize_json_field,
    deserialize_json_field,
)


class FTS5Backend:
    def create_database(self, db_path: str) -> None:
        with get_db_connection(db_path) as conn:
            cur = conn.cursor()

            cur.execute("DROP TABLE IF EXISTS memories_fts")
            cur.execute("DROP TABLE IF EXISTS memories")

            cur.execute(f"""
                CREATE TABLE memories (
                    {FIELD_ID} TEXT PRIMARY KEY,
                    {FIELD_VARIANTS} TEXT NOT NULL,
                    {FIELD_LESSON} TEXT NOT NULL,
                    {FIELD_METADATA} TEXT,
                    {FIELD_SOURCE} TEXT
                )
            """)

            cur.execute("""
                CREATE VIRTUAL TABLE memories_fts USING fts5(
                    situation_variant,
                    memory_id UNINDEXED
                )
            """)

            cur.execute(f"""
                CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(situation_variant, memory_id)
                    SELECT value, new.{FIELD_ID}
                    FROM json_each(new.{FIELD_VARIANTS});
                END
            """)

            cur.execute(f"""
                CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE memory_id = old.{FIELD_ID};
                END
            """)

            cur.execute(f"""
                CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                    DELETE FROM memories_fts WHERE memory_id = old.{FIELD_ID};
                    INSERT INTO memories_fts(situation_variant, memory_id)
                    SELECT value, new.{FIELD_ID}
                    FROM json_each(new.{FIELD_VARIANTS});
                END
            """)

    def insert_memories(self, db_path: str, memories: list[dict[str, Any]]) -> int:
        with get_db_connection(db_path) as conn:
            cur = conn.cursor()
            inserted = 0

            for mem in memories:
                try:
                    cur.execute(
                        f"""
                        INSERT OR REPLACE INTO memories
                        ({FIELD_ID}, {FIELD_VARIANTS}, {FIELD_LESSON}, {FIELD_METADATA}, {FIELD_SOURCE})
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            mem[FIELD_ID],
                            serialize_json_field(mem.get(FIELD_VARIANTS)),
                            mem[FIELD_LESSON],
                            serialize_json_field(mem.get(FIELD_METADATA)),
                            serialize_json_field(mem.get(FIELD_SOURCE)),
                        ),
                    )
                    inserted += 1
                except (KeyError, sqlite3.Error) as e:
                    print(f"Warning: Skipping memory {mem.get(FIELD_ID, '?')}: {e}")

        return inserted

    def search(self, db_path: str, query: str, limit: int = 10) -> list[SearchResult]:
        with get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT m.{FIELD_ID}, m.{FIELD_VARIANTS}, m.{FIELD_LESSON},
                       m.{FIELD_METADATA}, m.{FIELD_SOURCE},
                       bm25(memories_fts) as {FIELD_RANK}
                FROM memories_fts fts
                JOIN memories m ON m.{FIELD_ID} = fts.memory_id
                WHERE memories_fts MATCH ?
                ORDER BY {FIELD_RANK}
                """,
                (query,),
            )

            seen_ids: set[str] = set()
            results: list[SearchResult] = []
            for row in cur.fetchall():
                mem_id = row[FIELD_ID]
                if mem_id in seen_ids:
                    continue
                seen_ids.add(mem_id)

                variants = json.loads(row[FIELD_VARIANTS]) if row[FIELD_VARIANTS] else []
                rank = row[FIELD_RANK]

                results.append(SearchResult(
                    id=mem_id,
                    situation=variants[0] if variants else "",
                    lesson=row[FIELD_LESSON],
                    metadata=deserialize_json_field(row[FIELD_METADATA]),
                    source=deserialize_json_field(row[FIELD_SOURCE]),
                    score=-rank,  # BM25 rank is negative; higher = better
                    raw_score=rank,
                    score_type="bm25_rank",
                ))

                if len(results) >= limit:
                    break

            return results

    def rebuild_database(self, db_path: str, memories_dir: str) -> None:
        print(f"Creating database at {db_path}...")
        self.create_database(db_path)

        print(f"Loading memories from {memories_dir}...")
        memories = load_memories(memories_dir)
        print(f"Found {len(memories)} memories")

        print("Inserting memories...")
        count = self.insert_memories(db_path, memories)
        print(f"Inserted {count} memories into database")

    def get_memory_count(self, db_path: str) -> int:
        with get_db_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories")
            return cur.fetchone()[0]

    def get_memory_by_id(self, db_path: str, memory_id: str) -> dict[str, Any] | None:
        with get_db_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT {FIELD_ID}, {FIELD_VARIANTS}, {FIELD_LESSON},
                       {FIELD_METADATA}, {FIELD_SOURCE}
                FROM memories WHERE {FIELD_ID} = ?
                """,
                (memory_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            variants = json.loads(row[FIELD_VARIANTS]) if row[FIELD_VARIANTS] else []
            return {
                FIELD_ID: row[FIELD_ID],
                FIELD_VARIANTS: variants,
                FIELD_LESSON: row[FIELD_LESSON],
                FIELD_METADATA: deserialize_json_field(row[FIELD_METADATA]),
                FIELD_SOURCE: deserialize_json_field(row[FIELD_SOURCE]),
            }
