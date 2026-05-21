from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from src.agent.planner import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight EHRSQL Text-to-SQL agent")
    parser.add_argument("--db", default="mimic_iii", choices=["mimic_iii", "eicu"])
    parser.add_argument("--question", help="Natural language question")
    parser.add_argument("--sql", help="Execute a raw SQL query instead of calling the agent")
    parser.add_argument("--schema", action="store_true", help="Show retrieved schema for --question")
    parser.add_argument("--model", default=None, help="DeepSeek model id")
    parser.add_argument("--prompt-version", default="schema", choices=["base", "schema", "fewshot", "reflection"])
    parser.add_argument("--mode", default="new_query", choices=["new_query", "followup"])
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    if args.schema:
        _print_schema(args)
        return
    if args.sql:
        result = _execute_raw_sql(args)
        _print_result(result, as_json=args.json)
        return
    if not args.question:
        parser.error("--question is required unless --sql or --schema is provided")

    result = run_agent(
        args.question,
        db_id=args.db,
        model=args.model,
        prompt_version=args.prompt_version,
        mode=args.mode,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout,
        use_cache=not args.no_cache,
    )
    _print_result(result, as_json=args.json)


def _print_schema(args: argparse.Namespace) -> None:
    from src.schema_index import format_schema_context, retrieve_schema

    question = args.question or ""
    retrieved = retrieve_schema(question, args.db)
    print(format_schema_context(retrieved))


def _execute_raw_sql(args: argparse.Namespace) -> Dict[str, Any]:
    from src.sql_executor import execute_sql
    from src.sql_guard import validate_sql

    validation = validate_sql(args.sql, db_id=args.db)
    if not validation.get("ok"):
        return {"validation": validation, "execution": None}
    execution = execute_sql(args.sql, db_id=args.db, max_rows=args.max_rows, timeout_seconds=args.timeout)
    return {"validation": validation, "execution": execution}


def _print_result(result: Dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    print(f"Final answer: {result.get('final_answer', '')}")
    if result.get("generated_sql"):
        print("\nSQL:")
        print(result["generated_sql"])
    execution = result.get("execution")
    if execution:
        print("\nExecution:")
        print(json.dumps(execution, ensure_ascii=False, indent=2, default=str))
    validation = result.get("validation")
    if validation and not result.get("generated_sql"):
        print("\nValidation:")
        print(json.dumps(validation, ensure_ascii=False, indent=2, default=str))
    if result.get("errors"):
        print("\nErrors:")
        print(json.dumps(result["errors"], ensure_ascii=False, indent=2, default=str))
    if result.get("log_path"):
        print(f"\nRun log: {result['log_path']}")


if __name__ == "__main__":
    main()

