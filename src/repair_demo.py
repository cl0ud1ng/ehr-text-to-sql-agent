from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from src.agent.planner import run_agent
from src.config import ROOT_DIR


REPAIR_CASES: list[dict[str, str]] = [
    {
        "id": "wrong_table_admission",
        "db_id": "mimic_iii",
        "title": "Wrong table name",
        "error_type": "wrong_table_name",
        "question": "How many hospital admissions are recorded?",
        "broken_sql": "select count(*) as admission_count from admission",
    },
    {
        "id": "wrong_column_discharge_time",
        "db_id": "mimic_iii",
        "title": "Wrong column name",
        "error_type": "wrong_column_name",
        "question": "What is the discharge time for the first admission of patient 75581?",
        "broken_sql": (
            "select admissions.discharge_time from admissions "
            "where admissions.subject_id = 75581 "
            "order by admissions.admittime asc limit 1"
        ),
    },
    {
        "id": "unquoted_filter_literal",
        "db_id": "mimic_iii",
        "title": "SQLite filter error",
        "error_type": "sqlite_filter_error",
        "question": "How many patients have gender f?",
        "broken_sql": "select count(*) as female_count from patients where patients.gender = f",
    },
]


def run_repair_cases(
    *,
    case_ids: Optional[Iterable[str]] = None,
    output_path: Optional[Path] = None,
    max_rows: int = 20,
    use_cache: bool = True,
    save_logs: bool = True,
) -> dict[str, Any]:
    selected_ids = set(case_ids or [])
    cases = [case for case in REPAIR_CASES if not selected_ids or case["id"] in selected_ids]
    results = []
    for case in cases:
        result = run_agent(
            case["question"],
            db_id=case["db_id"],
            max_repairs=2,
            max_rows=max_rows,
            use_cache=use_cache,
            save_log=save_logs,
            initial_sql=case["broken_sql"],
        )
        results.append(_summarize_case(case, result))

    payload = {
        "total_cases": len(results),
        "successful_cases": sum(1 for item in results if item["execution_ok"] and item["repair_count"] > 0),
        "cases": results,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload


def _summarize_case(case: dict[str, str], result: dict[str, Any]) -> dict[str, Any]:
    repairs = result.get("repairs") or []
    first_repair = repairs[0] if repairs else {}
    repair_payload = first_repair.get("repair") or {}
    execution = result.get("execution") or {}
    return {
        "case_id": case["id"],
        "db_id": case["db_id"],
        "title": case["title"],
        "error_type": case["error_type"],
        "question": case["question"],
        "before_sql": case["broken_sql"],
        "error_info": first_repair.get("error") or _fallback_error(result),
        "after_sql": repair_payload.get("sql") or "",
        "final_sql": result.get("generated_sql") or "",
        "repair_count": len(repairs),
        "repair_source": repair_payload.get("source"),
        "repair_reason": repair_payload.get("reason"),
        "execution_ok": bool(execution.get("ok")),
        "result_columns": execution.get("columns") or [],
        "result_rows_preview": (execution.get("rows") or [])[:5],
        "row_count": execution.get("row_count", 0),
        "final_answer": result.get("final_answer") or "",
        "log_path": result.get("log_path"),
    }


def _fallback_error(result: dict[str, Any]) -> str:
    errors = result.get("errors") or []
    if errors:
        return errors[-1].get("message") or errors[-1].get("type") or ""
    execution = result.get("execution") or {}
    error = execution.get("error") or {}
    if isinstance(error, dict):
        return error.get("message") or error.get("type") or ""
    return str(error or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic SQL repair demo cases.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Case id to run; repeat for multiple cases.")
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "outputs" / "evaluation" / "repair_cases.json"),
        help="Path for the repair-case summary JSON.",
    )
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the summary JSON.")
    args = parser.parse_args()

    payload = run_repair_cases(
        case_ids=args.case_ids,
        output_path=Path(args.output),
        max_rows=args.max_rows,
        use_cache=not args.no_cache,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"Repair cases: {payload['successful_cases']}/{payload['total_cases']} succeeded. "
            f"Summary: {args.output}"
        )


if __name__ == "__main__":
    main()
