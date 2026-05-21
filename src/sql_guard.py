"""Static guardrails for read-only SQLite SQL."""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

try:  # Optional dependency; the fallback below is used in the base environment.
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover - covered indirectly when sqlglot is absent.
    sqlglot = None
    exp = None

try:
    from .schema_index import load_schema
except ImportError:  # pragma: no cover
    from schema_index import load_schema


DANGEROUS_KEYWORDS = {
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
}

_WORD_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\"|'[^']+'|`[^`]+`|\[[^\]]+\])",
    re.IGNORECASE,
)


def _strip_comments_and_strings(sql: str, keep_string_space: bool = True) -> str:
    out: list[str] = []
    i = 0
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if ch == "-" and nxt == "-":
            i += 2
            while i < len(sql) and sql[i] not in "\r\n":
                i += 1
            out.append(" ")
        elif ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(sql) and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2
            out.append(" ")
        elif ch in {"'", '"', "`"}:
            quote = ch
            out.append(" " if keep_string_space else ch)
            i += 1
            while i < len(sql):
                if sql[i] == quote:
                    if i + 1 < len(sql) and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                if sql[i] == "\\":
                    i += 2
                else:
                    i += 1
            if not keep_string_space:
                out.append(quote)
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _split_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    quote: Optional[str] = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if quote:
            current.append(ch)
            if ch == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    current.append(sql[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
        elif ch in {"'", '"', "`"}:
            quote = ch
            current.append(ch)
            i += 1
        elif ch == "-" and nxt == "-":
            while i < len(sql) and sql[i] not in "\r\n":
                current.append(sql[i])
                i += 1
        elif ch == "/" and nxt == "*":
            current.append(ch)
            current.append(nxt)
            i += 2
            while i + 1 < len(sql) and not (sql[i] == "*" and sql[i + 1] == "/"):
                current.append(sql[i])
                i += 1
            if i + 1 < len(sql):
                current.append(sql[i])
                current.append(sql[i + 1])
                i += 2
        elif ch == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            i += 1
        else:
            current.append(ch)
            i += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _unquote_identifier(name: str) -> str:
    name = name.strip()
    if len(name) >= 2 and ((name[0], name[-1]) in {('"', '"'), ("'", "'"), ("`", "`"), ("[", "]")}):
        return name[1:-1]
    return name


def _schema_table_names(
    schema: Optional[dict[str, Any]],
    db_id: Optional[str],
    db_path: Optional[str],
) -> Optional[set[str]]:
    if schema is None and db_id:
        schema = load_schema(db_id, db_path=db_path)
    if schema is None:
        return None
    return {table["name"].lower() for table in schema.get("tables", [])}


def _extract_cte_names(clean_sql: str) -> set[str]:
    stripped = clean_sql.lstrip()
    if not stripped[:4].upper() == "WITH":
        return set()
    names: set[str] = set()
    depth = 0
    pos = 4
    while pos < len(stripped):
        char = stripped[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", stripped[pos:], re.IGNORECASE)
            if match:
                names.add(match.group(1).lower())
                pos += match.end() - 1
        pos += 1
    return names


def _fallback_referenced_tables(statement: str) -> set[str]:
    clean_sql = _strip_comments_and_strings(statement)
    ctes = _extract_cte_names(clean_sql)
    tables = {_unquote_identifier(match.group(1)).lower() for match in _REF_RE.finditer(clean_sql)}
    return {table for table in tables if table not in ctes}


def _fallback_validate(statement: str) -> Tuple[bool, Optional[str], set[str]]:
    clean_sql = _strip_comments_and_strings(statement)
    words = [word.upper() for word in _WORD_RE.findall(clean_sql)]
    if any(word in DANGEROUS_KEYWORDS for word in words):
        return False, "only read-only SELECT statements are allowed", set()
    if not words:
        return False, "empty SQL", set()
    if words[0] == "SELECT":
        return True, None, _fallback_referenced_tables(statement)
    if words[0] == "WITH" and "SELECT" in words:
        return True, None, _fallback_referenced_tables(statement)
    return False, "SQL must be a single SELECT or WITH ... SELECT statement", set()


def _sqlglot_validate(statement: str) -> Tuple[bool, Optional[str], set[str]]:
    if sqlglot is None:
        return _fallback_validate(statement)
    try:
        parsed = sqlglot.parse_one(statement, read="sqlite")
    except Exception as exc:
        return False, f"SQL parse error: {exc}", set()

    allowed_roots = (exp.Select, exp.With, exp.Union, exp.Subquery)
    if not isinstance(parsed, allowed_roots):
        return False, "SQL must be a single SELECT or WITH ... SELECT statement", set()

    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Command,
    )
    if any(isinstance(node, forbidden) for node in parsed.walk()):
        return False, "only read-only SELECT statements are allowed", set()

    tables = {table.name.lower() for table in parsed.find_all(exp.Table)}
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    return True, None, tables - ctes


def validate_sql(
    sql: str,
    db_id: Optional[str] = None,
    schema: Optional[dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Validate that SQL is one read-only query and references known tables."""

    if not sql or not sql.strip():
        return {"ok": False, "error": "empty SQL", "tables": []}

    statements = _split_statements(sql)
    if len(statements) != 1:
        return {"ok": False, "error": "multiple SQL statements are not allowed", "tables": []}

    statement = statements[0]
    ok, error, tables = _sqlglot_validate(statement)
    if not ok:
        return {"ok": False, "error": error, "tables": sorted(tables)}

    known_tables = _schema_table_names(schema, db_id, db_path)
    if known_tables is not None:
        missing = sorted(table for table in tables if table.lower() not in known_tables)
        if missing:
            return {"ok": False, "error": f"unknown table(s): {', '.join(missing)}", "tables": sorted(tables)}

    return {"ok": True, "error": None, "tables": sorted(tables)}
