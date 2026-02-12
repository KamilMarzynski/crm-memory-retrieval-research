import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def get_db_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def serialize_json_field(data: dict[str, Any] | None) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def deserialize_json_field(json_str: str | None) -> dict[str, Any]:
    if not json_str:
        return {}
    return json.loads(json_str)
