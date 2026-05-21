from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.agent.planner import run_agent
from src.sql_executor import execute_sql


def evaluate_file(
    data_path: Union[str, Path],
    *,
    db_id: str,
    limit: Optional[int] = None,
    model: Optional[str] = None,
    prompt_version: str = "schema",
    max_rows: int = 1000,
    timeout_seconds: float = 5.0,
    workers: int = 1,
) -> Dict[str, Any]:
    samples = json.loads(Path(data_path).read_text(encoding="utf-8"))
    if limit is not None:
        samples = samples[:limit]

    if workers <= 1:
        rows = [
            _evaluate_sample(
                index,
                sample,
                db_id=db_id,
                model=model,
                prompt_version=prompt_version,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
            )
            for index, sample in enumerate(samples)
        ]
    else:
        rows_by_index: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_sample,
                    index,
                    sample,
                    db_id=db_id,
                    model=model,
                    prompt_version=prompt_version,
                    max_rows=max_rows,
                    timeout_seconds=timeout_seconds,
                ): index
                for index, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                row = future.result()
                rows_by_index[row["index"]] = row
        rows = [rows_by_index[index] for index in range(len(samples))]

    counts = _counts(rows)
    metrics = _metrics(counts)
    return {"counts": counts, "metrics": metrics, "rows": rows}


def _evaluate_sample(
    index: int,
    sample: Dict[str, Any],
    *,
    db_id: str,
    model: Optional[str],
    prompt_version: str,
    max_rows: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    question = sample.get("question", "")
    expected_unanswerable = _is_unanswerable(sample)
    result = run_agent(
        question,
        db_id=db_id,
        model=model,
        prompt_version=prompt_version,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
    )
    predicted_unanswerable = result.get("answerability", {}).get("answerable") is False or not result.get("generated_sql")

    match = False
    gold_execution: Optional[Dict[str, Any]] = None
    if expected_unanswerable:
        match = predicted_unanswerable
    elif result.get("execution", {}).get("ok"):
        gold_sql = sample.get("query")
        gold_execution = execute_sql(gold_sql, db_id=db_id, max_rows=max_rows, timeout_seconds=timeout_seconds)
        if gold_execution.get("ok"):
            match = results_equal(gold_execution, result["execution"], order_sensitive=_order_sensitive(gold_sql, question))

    return {
        "index": index,
        "id": sample.get("id"),
        "question": question,
        "expected_unanswerable": expected_unanswerable,
        "predicted_unanswerable": predicted_unanswerable,
        "generated_sql": result.get("generated_sql"),
        "ok": result.get("execution", {}).get("ok"),
        "match": match,
        "repair_count": len(result.get("repairs", [])),
        "json_or_model_error": any(
            str(err.get("type", "")).lower() in {"runtimeerror", "llmnotconfigured"}
            or "json" in str(err.get("message", "")).lower()
            for err in result.get("errors", [])
        ),
        "errors": result.get("errors", []),
        "log_path": result.get("log_path"),
        "gold_error": None if not gold_execution else gold_execution.get("error"),
    }


def _counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "total": len(rows),
        "expected_answerable": 0,
        "expected_unanswerable": 0,
        "predicted_unanswerable": 0,
        "true_unanswerable": 0,
        "false_unanswerable": 0,
        "missed_unanswerable": 0,
        "execution_success": 0,
        "execution_match": 0,
        "answerable_execution_success": 0,
        "answerable_execution_match": 0,
        "gold_unavailable": 0,
        "total_repairs": 0,
        "json_or_model_errors": 0,
    }
    for row in rows:
        expected_unanswerable = bool(row["expected_unanswerable"])
        predicted_unanswerable = bool(row["predicted_unanswerable"])
        if expected_unanswerable:
            counts["expected_unanswerable"] += 1
        else:
            counts["expected_answerable"] += 1
        if predicted_unanswerable:
            counts["predicted_unanswerable"] += 1
        if expected_unanswerable and predicted_unanswerable:
            counts["true_unanswerable"] += 1
        if not expected_unanswerable and predicted_unanswerable:
            counts["false_unanswerable"] += 1
        if expected_unanswerable and not predicted_unanswerable:
            counts["missed_unanswerable"] += 1
        if row.get("ok"):
            counts["execution_success"] += 1
            if not expected_unanswerable:
                counts["answerable_execution_success"] += 1
        if row.get("match"):
            counts["execution_match"] += 1
            if not expected_unanswerable:
                counts["answerable_execution_match"] += 1
        if row.get("gold_error"):
            counts["gold_unavailable"] += 1
        counts["total_repairs"] += int(row.get("repair_count", 0))
        if row.get("json_or_model_error"):
            counts["json_or_model_errors"] += 1
    return counts

def results_equal(left: Dict[str, Any], right: Dict[str, Any], *, order_sensitive: bool = False) -> bool:
    left_rows = [_normalize_row(row) for row in left.get("rows", [])]
    right_rows = [_normalize_row(row) for row in right.get("rows", [])]
    if order_sensitive:
        return left_rows == right_rows
    return sorted(left_rows, key=repr) == sorted(right_rows, key=repr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the EHRSQL agent on a JSON test file")
    parser.add_argument("--db", required=True, choices=["mimic_iii", "eicu"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--prompt-version", default="schema", choices=["base", "schema", "fewshot", "reflection"])
    parser.add_argument("--workers", type=int, default=1, help="Number of samples to evaluate concurrently")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_file(
        args.data,
        db_id=args.db,
        limit=args.limit,
        model=args.model,
        prompt_version=args.prompt_version,
        workers=args.workers,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload)


def _is_unanswerable(sample: Dict[str, Any]) -> bool:
    query = sample.get("query")
    if sample.get("is_impossible") is True:
        return True
    return query is None or str(query).strip().lower() in {"", "nan", "null", "none"}


def _normalize_row(row: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(_normalize_cell(cell) for cell in row)


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 4)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return " ".join(text.split())
    if number.is_integer():
        return int(number)
    return round(number, 4)


def _order_sensitive(sql: str, question: str) -> bool:
    text = f"{sql or ''} {question or ''}".lower()
    return any(term in text for term in ["order by", "first", "last", "latest", "top", "earliest", "most recent"])


def _metrics(counts: Dict[str, int]) -> Dict[str, float]:
    total = max(counts["total"], 1)
    expected_answerable = max(counts["expected_answerable"], 1)
    predicted_unanswerable = max(counts["predicted_unanswerable"], 1)
    expected_unanswerable = max(counts["expected_unanswerable"], 1)
    precision = counts["true_unanswerable"] / predicted_unanswerable
    recall = counts["true_unanswerable"] / expected_unanswerable
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "execution_success_rate": counts["execution_success"] / total,
        "execution_match_rate": counts["execution_match"] / total,
        "answerable_execution_success_rate": counts["answerable_execution_success"] / expected_answerable,
        "answerable_execution_match_rate": counts["answerable_execution_match"] / expected_answerable,
        "predicted_unanswerable_rate": counts["predicted_unanswerable"] / total,
        "unanswerable_precision": precision,
        "unanswerable_recall": recall,
        "unanswerable_f1": f1,
        "average_repair_count": counts["total_repairs"] / total,
        "json_or_model_error_rate": counts["json_or_model_errors"] / total,
    }


if __name__ == "__main__":
    main()
