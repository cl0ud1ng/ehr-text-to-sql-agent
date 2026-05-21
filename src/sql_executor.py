"""Read-only SQLite execution helper with timeout and row limits."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

try:
    from .schema_index import DB_PATHS
    from .sql_guard import validate_sql
except ImportError:  # pragma: no cover
    from schema_index import DB_PATHS
    from sql_guard import validate_sql


def _resolve_db_path(db_id: str, db_path: Optional[str] = None) -> Path:
    path = Path(db_path) if db_path else Path(DB_PATHS[db_id])
    if not path.exists():
        raise FileNotFoundError(f"database not found: {path}")
    return path


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def execute_sql(
    sql: str,
    db_id: str,
    db_path: Optional[str] = None,
    max_rows: int = 100,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Validate and execute a read-only SQLite query."""

    started = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    guard = validate_sql(sql, db_id=db_id, db_path=db_path)
    if not guard["ok"]:
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "elapsed_ms": elapsed_ms(),
            "error": {"type": "validation_error", "message": guard["error"]},
        }

    try:
        path = _resolve_db_path(db_id, db_path)
        deadline = time.perf_counter() + max(0.0, float(timeout_seconds))
        conn = _connect_readonly(path)
        try:
            conn.execute("PRAGMA query_only = ON")

            def should_abort() -> int:
                return 1 if time.perf_counter() > deadline else 0

            conn.set_progress_handler(should_abort, 1000)
            cursor = conn.execute(sql)
            columns = [description[0] for description in (cursor.description or [])]
            limit = max(0, int(max_rows))
            fetched = cursor.fetchmany(limit + 1)
            truncated = len(fetched) > limit
            visible_rows = fetched[:limit]
            rows = [[row[column] for column in columns] for row in visible_rows]
            conn.set_progress_handler(None, 0)
            return {
                "ok": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
                "elapsed_ms": elapsed_ms(),
                "error": None,
            }
        finally:
            conn.close()
    except Exception as exc:
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "elapsed_ms": elapsed_ms(),
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
        }
