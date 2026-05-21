from __future__ import annotations

from typing import Any, Dict, Optional

from src.llm_client import DeepSeekClient, LLMNotConfigured
from .generator import _read_prompt


def repair_sql(
    question: str,
    db_id: str,
    schema_context: str,
    failed_sql: str,
    error_message: str,
    *,
    llm_client: Optional[DeepSeekClient] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    client = llm_client or DeepSeekClient(model=model)
    if not client.available:
        raise LLMNotConfigured("DeepSeek API is not configured; SQL repair is unavailable.")

    messages = [
        {"role": "system", "content": _read_prompt("reflection")},
        {
            "role": "user",
            "content": (
                f"Database id: {db_id}\n"
                f"Question:\n{question}\n\n"
                f"Schema context:\n{schema_context}\n\n"
                f"Failed SQL:\n{failed_sql}\n\n"
                f"Validation or execution error:\n{error_message}\n\n"
                "Return a corrected JSON object."
            ),
        },
    ]
    response = client.json_chat(
        messages,
        cache_parts=[db_id, question, schema_context, failed_sql, error_message],
        model=model,
        task_name="repair_sql",
    )
    parsed = response.parsed or {}
    return {
        "sql": (parsed.get("sql") or "").strip(),
        "reason": parsed.get("reason") or "",
        "confidence": parsed.get("confidence"),
        "source": "llm",
        "cache_hit": response.cache_hit,
        "raw": response.content,
    }


def summarize_result(question: str, sql: str, execution: Dict[str, Any]) -> str:
    if not execution.get("ok"):
        err = execution.get("error") or {}
        return f"Query failed: {err.get('message', 'unknown error')}"
    row_count = execution.get("row_count", 0)
    if row_count == 0:
        return "The query executed successfully but returned no rows."
    columns = execution.get("columns") or []
    rows = execution.get("rows") or []
    preview = rows[:3]
    return f"Returned {row_count} row(s) with columns {columns}. Preview: {preview}"
