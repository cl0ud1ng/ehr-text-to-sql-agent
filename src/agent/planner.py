from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import ROOT_DIR
from src.llm_client import DeepSeekClient, LLMNotConfigured
from .answerability import judge_answerability
from .generator import generate_sql
from .reflector import repair_sql, summarize_result


def run_agent(
    question: str,
    db_id: str = "mimic_iii",
    *,
    model: Optional[str] = None,
    prompt_version: str = "schema",
    mode: str = "new_query",
    followup_context: Optional[str] = None,
    max_repairs: int = 2,
    max_rows: int = 100,
    timeout_seconds: float = 5.0,
    use_cache: bool = True,
    db_path: Optional[str] = None,
    save_log: bool = True,
) -> Dict[str, Any]:
    from src.schema_index import format_schema_context, retrieve_schema
    from src.sql_executor import execute_sql
    from src.sql_guard import validate_sql

    started = time.perf_counter()
    client = DeepSeekClient(model=model, use_cache=use_cache)
    run_log: Dict[str, Any] = {
        "run_id": _run_id(db_id, question),
        "db_id": db_id,
        "mode": mode,
        "model_id": model or client.model,
        "prompt_version": prompt_version,
        "question": question,
        "schema_candidates": {},
        "answerability": {},
        "generated_sql": "",
        "validation": {},
        "execution": {},
        "repairs": [],
        "final_answer": "",
        "timing": {},
        "errors": [],
        "cache_hit": False,
    }

    try:
        retrieved = retrieve_schema(question, db_id, db_path=db_path)
        schema_context = format_schema_context(retrieved)
        run_log["schema_candidates"] = retrieved

        answerability = judge_answerability(
            question,
            schema_context,
            llm_client=client,
            model=model,
            followup_context=followup_context if mode != "new_query" else None,
        )
        run_log["answerability"] = answerability
        if answerability.get("answerable") is False:
            run_log["final_answer"] = answerability.get("reason", "Question is not answerable from this database.")
            return _finish(run_log, started, save_log)

        generation = generate_sql(
            question,
            db_id,
            schema_context,
            prompt_version=prompt_version,
            llm_client=client,
            model=model,
            followup_context=followup_context if mode != "new_query" else None,
        )
        if generation.get("answerable") is False:
            run_log["answerability"] = {
                "answerable": False,
                "decision": "not_answerable",
                "reason": generation.get("reason", "Generator marked the question as not answerable."),
                "source": generation.get("source", "generator"),
            }
            run_log["final_answer"] = run_log["answerability"]["reason"]
            return _finish(run_log, started, save_log)

        sql = generation.get("sql", "").strip()
        run_log["generated_sql"] = sql
        run_log["generation"] = generation
        if not sql:
            run_log["errors"].append({"type": "empty_sql", "message": "The generator did not return SQL."})
            run_log["final_answer"] = "No SQL was generated."
            return _finish(run_log, started, save_log)

        for attempt in range(max_repairs + 1):
            validation = validate_sql(sql, db_id=db_id, db_path=db_path)
            run_log["validation"] = validation
            if validation.get("ok"):
                execution = execute_sql(sql, db_id=db_id, db_path=db_path, max_rows=max_rows, timeout_seconds=timeout_seconds)
                run_log["execution"] = execution
                if execution.get("ok"):
                    run_log["final_answer"] = summarize_result(question, sql, execution)
                    run_log["generated_sql"] = sql
                    return _finish(run_log, started, save_log)
                error_message = _error_message(execution)
            else:
                error_message = validation.get("error") or "SQL validation failed."

            if attempt >= max_repairs:
                run_log["errors"].append({"type": "repair_exhausted", "message": error_message})
                run_log["final_answer"] = f"SQL failed after repair attempts: {error_message}"
                return _finish(run_log, started, save_log)

            try:
                repaired = repair_sql(question, db_id, schema_context, sql, error_message, llm_client=client, model=model)
            except LLMNotConfigured as exc:
                run_log["errors"].append({"type": "llm_not_configured", "message": str(exc)})
                run_log["final_answer"] = str(exc)
                return _finish(run_log, started, save_log)

            new_sql = repaired.get("sql", "").strip()
            run_log["repairs"].append(
                {"attempt": attempt + 1, "failed_sql": sql, "error": error_message, "repair": repaired}
            )
            if not new_sql or new_sql == sql:
                run_log["errors"].append({"type": "repair_no_change", "message": "Repair returned empty or unchanged SQL."})
                run_log["final_answer"] = "SQL repair did not produce a usable query."
                return _finish(run_log, started, save_log)
            sql = new_sql

    except Exception as exc:
        run_log["errors"].append({"type": exc.__class__.__name__, "message": str(exc)})
        run_log["final_answer"] = f"Agent failed: {exc}"
        return _finish(run_log, started, save_log)

    return _finish(run_log, started, save_log)


def build_followup_context(turn_history: list[Dict[str, Any]], max_full_turns: int = 3) -> str:
    if not turn_history:
        return ""
    older = turn_history[:-max_full_turns]
    recent = turn_history[-max_full_turns:]
    parts = []
    if older:
        parts.append(
            "Earlier session summary: "
            + "; ".join(
                f"turn {item.get('turn_id')}: {item.get('question')} -> {item.get('result_summary')}"
                for item in older
            )
        )
    for item in recent:
        parts.append(
            "\n".join(
                [
                    f"Turn {item.get('turn_id')}",
                    f"Question: {item.get('question')}",
                    f"SQL: {item.get('sql')}",
                    f"Result summary: {item.get('result_summary')}",
                    f"Used tables: {item.get('used_tables')}",
                ]
            )
        )
    return "\n\n".join(parts)


def _finish(run_log: Dict[str, Any], started: float, save_log: bool) -> Dict[str, Any]:
    run_log["timing"]["total_elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    if save_log:
        path = _save_run_log(run_log)
        run_log["log_path"] = str(path)
    return run_log


def _save_run_log(run_log: Dict[str, Any]) -> Path:
    out_dir = ROOT_DIR / "outputs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.sha1(run_log["run_id"].encode("utf-8")).hexdigest()[:8]
    path = out_dir / f"{timestamp}_{run_log['db_id']}_{short_hash}.json"
    path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _run_id(db_id: str, question: str) -> str:
    digest = hashlib.sha1(f"{db_id}:{question}:{time.time()}".encode("utf-8")).hexdigest()[:12]
    return f"{db_id}_{digest}"


def _error_message(execution: Dict[str, Any]) -> str:
    error = execution.get("error") or {}
    if isinstance(error, dict):
        return error.get("message") or error.get("type") or "Execution failed."
    return str(error)
