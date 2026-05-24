"""SQLite schema loading and deterministic retrieval for EHRSQL databases."""

from __future__ import annotations

import re
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EHRSQL_ROOT = PROJECT_ROOT / "data" / "EHRSQL"

DB_PATHS = {
    "mimic_iii": str(EHRSQL_ROOT / "mimic_iii.sqlite"),
    "eicu": str(EHRSQL_ROOT / "eicu.sqlite"),
}

_SCHEMA_CACHE: dict[Tuple[str, str, float], dict[str, Any]] = {}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ALIAS_BOOSTS = [
    ({"intake", "route", "administer", "administered", "drug", "medicine", "medication", "medications"}, {"prescriptions", "medication"}, 30.0),
    ({"diagnosis", "diagnoses", "diagnostic", "icd"}, {"diagnoses_icd", "d_icd_diagnoses", "diagnosis"}, 30.0),
    ({"lab", "labs", "test", "tests", "laboratory"}, {"labevents", "d_labitems", "lab"}, 30.0),
    ({"heart", "rate", "vital", "vitals"}, {"chartevents", "d_items", "vitalperiodic"}, 30.0),
    ({"cost", "costs", "price", "prices", "charge", "charges"}, {"cost"}, 30.0),
    (
        {"cost", "costs", "price", "prices", "charge", "charges"},
        {"procedures_icd", "d_icd_procedures", "treatment", "labevents", "d_labitems", "lab", "prescriptions", "medication"},
        12.0,
    ),
    ({"procedure", "procedures", "treatment", "treatments"}, {"cost", "procedures_icd", "d_icd_procedures", "treatment"}, 24.0),
    ({"patient", "patients"}, {"patients", "patient"}, 18.0),
    ({"admission", "admissions", "hospital", "hosp"}, {"admissions", "patient"}, 24.0),
    ({"icu", "icustay", "icustays"}, {"icustays", "patient"}, 24.0),
]

_IMPORTANT_COLUMNS = {
    "admissions": {
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "admission_type",
        "admission_location",
        "discharge_location",
    },
    "icustays": {"subject_id", "hadm_id", "icustay_id", "intime", "outtime", "first_careunit", "last_careunit"},
    "patients": {"subject_id", "gender", "dob", "dod"},
    "prescriptions": {"subject_id", "hadm_id", "drug", "route", "startdate", "enddate"},
    "patient": {
        "uniquepid",
        "patienthealthsystemstayid",
        "patientunitstayid",
        "hospitaladmittime",
        "hospitaldischargetime",
        "unitadmittime",
        "unitdischargetime",
        "hospitaldischargestatus",
        "unitdischargestatus",
    },
    "medication": {"patientunitstayid", "drugname", "routeadmin", "drugstartoffset", "drugstopoffset"},
    "lab": {"patientunitstayid", "labname", "labresult", "labmeasurenamesystem", "labresultoffset"},
    "diagnosis": {"patientunitstayid", "diagnosisname", "icd9code", "diagnosistime"},
    "treatment": {"treatmentid", "patientunitstayid", "treatmentname", "treatmenttime"},
    "cost": {"row_id", "subject_id", "hadm_id", "event_type", "event_id", "eventtype", "eventid", "cost"},
    "procedures_icd": {"row_id", "subject_id", "hadm_id", "icd9_code", "charttime"},
    "d_icd_procedures": {"icd9_code", "short_title", "long_title"},
    "vitalperiodic": {
        "patientunitstayid",
        "observationoffset",
        "temperature",
        "sao2",
        "heartrate",
        "respiration",
        "systemicsystolic",
        "systemicdiastolic",
    },
    "labevents": {"subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"},
    "chartevents": {"subject_id", "hadm_id", "icustay_id", "itemid", "charttime", "valuenum", "valueuom"},
    "d_labitems": {"itemid", "label", "fluid", "category"},
    "d_items": {"itemid", "label", "linksto"},
}


def _resolve_db_path(db_id: str, db_path: Optional[str] = None) -> Path:
    if db_path:
        path = Path(db_path)
    else:
        if db_id not in DB_PATHS:
            raise ValueError(f"unknown db_id: {db_id}")
        path = Path(DB_PATHS[db_id])
    if not path.exists():
        raise FileNotFoundError(f"database not found: {path}")
    return path


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _tokenize(value: Any) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(str(value).replace("_", " "))}


def _sample_values(conn: sqlite3.Connection, table: str, column: str, limit: int = 10) -> list[Any]:
    sql = (
        f"SELECT DISTINCT {_quote_identifier(column)} "
        f"FROM {_quote_identifier(table)} "
        f"WHERE {_quote_identifier(column)} IS NOT NULL "
        f"LIMIT {int(limit)}"
    )
    try:
        return [row[0] for row in conn.execute(sql).fetchall()]
    except sqlite3.Error:
        return []


def load_schema(db_id: str, db_path: Optional[str] = None, refresh: bool = False) -> dict[str, Any]:
    """Load table/column metadata from a supported SQLite database.

    The result is cached by database path and mtime unless ``refresh`` is true.
    It is intentionally JSON-like so prompts and UI code can consume it directly.
    """

    path = _resolve_db_path(db_id, db_path)
    mtime = path.stat().st_mtime
    cache_key = (db_id, str(path.resolve()), mtime)
    if not refresh and cache_key in _SCHEMA_CACHE:
        return deepcopy(_SCHEMA_CACHE[cache_key])

    tables: list[dict[str, Any]] = []
    conn = _connect_readonly(path)
    try:
        table_names = [
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY lower(name)
                """
            ).fetchall()
        ]

        for table_name in table_names:
            count_sql = f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}"
            row_count = int(conn.execute(count_sql).fetchone()[0])
            columns: list[dict[str, Any]] = []
            for cid, name, col_type, notnull, default_value, pk in conn.execute(
                f"PRAGMA table_info({_quote_identifier(table_name)})"
            ).fetchall():
                columns.append(
                    {
                        "name": name,
                        "type": col_type or "",
                        "notnull": bool(notnull),
                        "default": default_value,
                        "primary_key": bool(pk),
                        "ordinal": int(cid),
                        "sample_values": _sample_values(conn, table_name, name),
                    }
                )
            tables.append(
                {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": columns,
                    "score": 0.0,
                    "matched_terms": [],
                }
            )
    finally:
        conn.close()

    schema = {
        "db_id": db_id,
        "db_path": str(path),
        "tables": tables,
        "table_count": len(tables),
    }
    _SCHEMA_CACHE[cache_key] = deepcopy(schema)
    return schema


def _score_table(table: dict[str, Any], query_terms: set[str]) -> Tuple[float, list[str], list[dict[str, Any]]]:
    table_terms = _tokenize(table["name"])
    matched = sorted(query_terms & table_terms)
    score = float(len(matched) * 8)
    scored_columns: list[dict[str, Any]] = []

    table_name = table["name"].lower()
    for aliases, target_tables, boost in _ALIAS_BOOSTS:
        alias_matches = sorted(query_terms & aliases)
        if alias_matches and table_name in target_tables:
            score += boost
            matched.extend(alias_matches)

    for column in table["columns"]:
        column_terms = _tokenize(column["name"]) | _tokenize(column.get("type", ""))
        sample_terms: set[str] = set()
        for value in column.get("sample_values", []):
            sample_terms |= _tokenize(value)

        column_matches = sorted(query_terms & column_terms)
        sample_matches = sorted(query_terms & sample_terms)
        column_score = float(len(column_matches) * 5 + len(sample_matches) * 2)
        if column_score:
            matched.extend(column_matches)
            matched.extend(sample_matches)
        enriched = deepcopy(column)
        enriched["score"] = column_score
        enriched["matched_terms"] = sorted(set(column_matches + sample_matches))
        scored_columns.append(enriched)
        score += column_score

    return score, sorted(set(matched)), scored_columns


def retrieve_schema(
    question: str,
    db_id: str,
    top_k_tables: int = 6,
    top_k_columns: int = 15,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return a deterministic top-k schema slice relevant to ``question``."""

    schema = load_schema(db_id, db_path=db_path)
    query_terms = _tokenize(question)
    ranked_tables: list[dict[str, Any]] = []

    for table in schema["tables"]:
        score, matched_terms, scored_columns = _score_table(table, query_terms)
        ranked_columns = sorted(
            scored_columns,
            key=lambda col: (-float(col["score"]), col["ordinal"], col["name"].lower()),
        )
        if top_k_columns > 0:
            important_names = _IMPORTANT_COLUMNS.get(table["name"].lower(), set())
            important_columns = [col for col in ranked_columns if col["name"].lower() in important_names]
            other_columns = [col for col in ranked_columns if col["name"].lower() not in important_names]
            ranked_columns = (important_columns + other_columns)[:top_k_columns]

        enriched_table = deepcopy(table)
        enriched_table["columns"] = ranked_columns
        enriched_table["score"] = score
        enriched_table["matched_terms"] = matched_terms
        ranked_tables.append(enriched_table)

    ranked_tables.sort(key=lambda tbl: (-float(tbl["score"]), tbl["name"].lower()))
    if top_k_tables > 0:
        ranked_tables = ranked_tables[:top_k_tables]

    return {
        "db_id": db_id,
        "db_path": schema["db_path"],
        "question": question,
        "tables": ranked_tables,
        "table_count": len(ranked_tables),
    }


def format_schema_context(retrieved: dict[str, Any]) -> str:
    """Format retrieved schema as compact text for prompts or UI previews."""

    lines = [f"Database: {retrieved.get('db_id', '')}"]
    for table in retrieved.get("tables", []):
        score = float(table.get("score", 0.0))
        row_count = table.get("row_count", "?")
        matches = ", ".join(table.get("matched_terms", []))
        suffix = f"; matches: {matches}" if matches else ""
        lines.append(f"Table {table['name']} ({row_count} rows; score: {score:.1f}{suffix})")
        for column in table.get("columns", []):
            samples = column.get("sample_values", [])
            sample_text = f"; samples: {samples}" if samples else ""
            lines.append(f"  - {column['name']} {column.get('type', '')}{sample_text}")
    return "\n".join(lines)
