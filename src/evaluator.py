from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from src.agent.planner import run_agent
from src.sql_executor import execute_sql


COUNT_KEYS = [
    "total",
    "expected_answerable",
    "expected_unanswerable",
    "predicted_unanswerable",
    "true_unanswerable",
    "false_unanswerable",
    "missed_unanswerable",
    "execution_success",
    "execution_match",
    "answerable_execution_success",
    "answerable_execution_match",
    "gold_unavailable",
    "total_repairs",
    "json_or_model_errors",
]

SUMMARY_FIELDS = [
    "db_id",
    "total",
    "expected_answerable",
    "expected_unanswerable",
    "predicted_unanswerable",
    "true_unanswerable",
    "false_unanswerable",
    "missed_unanswerable",
    "execution_success",
    "execution_match",
    "answerable_execution_success",
    "answerable_execution_match",
    "gold_unavailable",
    "total_repairs",
    "json_or_model_errors",
    "execution_success_rate",
    "execution_match_rate",
    "answerable_execution_success_rate",
    "answerable_execution_match_rate",
    "predicted_unanswerable_rate",
    "unanswerable_precision",
    "unanswerable_recall",
    "unanswerable_f1",
    "average_repair_count",
    "json_or_model_error_rate",
]

ROW_FIELDS = [
    "index",
    "id",
    "question",
    "expected_unanswerable",
    "predicted_unanswerable",
    "ok",
    "match",
    "repair_count",
    "json_or_model_error",
    "generated_sql",
    "log_path",
    "gold_error",
]


def evaluate_file(
    data_path: Union[str, Path],
    *,
    db_id: str,
    limit: Optional[int] = None,
    model: Optional[str] = None,
    prompt_version: str = "fewshot",
    max_rows: int = 1000,
    timeout_seconds: float = 5.0,
    workers: int = 1,
    example_type: str = "auto",
    use_cache: bool = True,
    use_heuristics: bool = True,
    use_llm_answerability: bool = True,
) -> Dict[str, Any]:
    samples = json.loads(Path(data_path).read_text(encoding="utf-8"))
    if limit is not None:
        samples = samples[:limit]
    resolved_example_type = _infer_example_type_from_path(data_path, example_type)

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
                example_type=resolved_example_type,
                use_cache=use_cache,
                use_heuristics=use_heuristics,
                use_llm_answerability=use_llm_answerability,
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
                    example_type=resolved_example_type,
                    use_cache=use_cache,
                    use_heuristics=use_heuristics,
                    use_llm_answerability=use_llm_answerability,
                ): index
                for index, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                row = future.result()
                rows_by_index[row["index"]] = row
        rows = [rows_by_index[index] for index in range(len(samples))]

    counts = _counts(rows)
    metrics = _metrics(counts)
    return {
        "metadata": {
            "db_id": db_id,
            "data_path": str(data_path),
            "limit": limit,
            "sample_count": len(samples),
            "model": model,
            "prompt_version": prompt_version,
            "example_type": resolved_example_type,
            "max_rows": max_rows,
            "timeout_seconds": timeout_seconds,
            "workers": workers,
            "use_cache": use_cache,
            "use_heuristics": use_heuristics,
            "use_llm_answerability": use_llm_answerability,
        },
        "counts": counts,
        "metrics": metrics,
        "rows": rows,
    }


def _evaluate_sample(
    index: int,
    sample: Dict[str, Any],
    *,
    db_id: str,
    model: Optional[str],
    prompt_version: str,
    max_rows: int,
    timeout_seconds: float,
    example_type: str,
    use_cache: bool,
    use_heuristics: bool,
    use_llm_answerability: bool,
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
        example_type=example_type,
        sample_metadata=_sample_prompt_metadata(sample, example_type),
        use_cache=use_cache,
        use_heuristics=use_heuristics,
        use_llm_answerability=use_llm_answerability,
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


def _infer_example_type_from_path(data_path: Union[str, Path], requested: str) -> str:
    if requested != "auto":
        return requested
    name = str(data_path).lower()
    if "not_empty" in name:
        return "time"
    if "empty" in name:
        return "basic"
    return "auto"


def _sample_prompt_metadata(sample: Dict[str, Any], example_type: str) -> Dict[str, Any]:
    return {
        "example_type": example_type if example_type in {"basic", "time"} else "auto",
        "t_tag": sample.get("t_tag"),
        "tag": sample.get("tag"),
        "id": sample.get("id"),
    }


def _counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = _empty_counts()
    counts["total"] = len(rows)
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


def build_grouped_summary(results_by_db: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    databases: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    overall_counts = _empty_counts()

    for db_id in sorted(results_by_db):
        result = results_by_db[db_id]
        counts = _normalized_counts(result.get("counts", {}))
        metrics = result.get("metrics") or _metrics(counts)
        databases[db_id] = {"counts": counts, "metrics": metrics}
        summary_rows.append(_summary_row(db_id, counts, metrics))
        overall_counts = _add_counts(overall_counts, counts)

    overall_metrics = _metrics(overall_counts)
    summary_rows.append(_summary_row("overall", overall_counts, overall_metrics))
    return {
        "databases": databases,
        "overall": {"counts": overall_counts, "metrics": overall_metrics},
        "summary_rows": summary_rows,
    }


def load_results_by_db(paths: Iterable[Union[str, Path]]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        result = json.loads(path.read_text(encoding="utf-8"))
        db_id = _infer_db_id(path, result)
        key = db_id
        duplicate_index = 2
        while key in results:
            key = f"{db_id}_{duplicate_index}"
            duplicate_index += 1
        results[key] = result
    return results


def write_summary_csv(summary: Dict[str, Any], output_path: Union[str, Path]) -> None:
    _write_csv(output_path, SUMMARY_FIELDS, summary.get("summary_rows", []))


def write_rows_csv(result: Dict[str, Any], output_path: Union[str, Path]) -> None:
    rows = []
    for row in result.get("rows", []):
        rows.append({field: _csv_value(row.get(field)) for field in ROW_FIELDS})
    _write_csv(output_path, ROW_FIELDS, rows)


def _empty_counts() -> Dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def _normalized_counts(raw_counts: Dict[str, Any]) -> Dict[str, int]:
    counts = _empty_counts()
    for key in COUNT_KEYS:
        counts[key] = int(raw_counts.get(key, 0) or 0)
    return counts


def _add_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in COUNT_KEYS}


def _summary_row(db_id: str, counts: Dict[str, int], metrics: Dict[str, float]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"db_id": db_id}
    row.update({key: counts.get(key, 0) for key in COUNT_KEYS})
    row.update({key: metrics.get(key, 0.0) for key in SUMMARY_FIELDS if key not in row and key != "db_id"})
    return row


def results_equal(left: Dict[str, Any], right: Dict[str, Any], *, order_sensitive: bool = False) -> bool:
    left_rows = [_normalize_row(row) for row in left.get("rows", [])]
    right_rows = [_normalize_row(row) for row in right.get("rows", [])]
    if order_sensitive:
        return left_rows == right_rows
    return sorted(left_rows, key=repr) == sorted(right_rows, key=repr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the EHRSQL agent on a JSON test file")
    parser.add_argument("--db", choices=["mimic_iii", "eicu"])
    parser.add_argument("--data")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--prompt-version", default="fewshot", choices=["base", "schema", "fewshot", "reflection"])
    parser.add_argument("--example-type", default="auto", choices=["auto", "basic", "time"])
    parser.add_argument("--workers", type=int, default=1, help="Number of samples to evaluate concurrently")
    parser.add_argument("--no-cache", action="store_true", help="Disable model cache and force live API calls")
    parser.add_argument("--no-heuristics", action="store_true", help="Disable deterministic SQL templates")
    parser.add_argument(
        "--rule-only-answerability",
        action="store_true",
        help="Skip LLM answerability and use only deterministic answerability rules",
    )
    parser.add_argument("--output")
    parser.add_argument("--rows-csv", help="Write per-sample evaluation rows to CSV")
    parser.add_argument("--summary-output", help="Write grouped summary JSON")
    parser.add_argument("--summary-csv", help="Write grouped summary table to CSV")
    parser.add_argument(
        "--summarize",
        nargs="+",
        help="Summarize existing evaluation JSON files without running the agent",
    )
    args = parser.parse_args()

    if args.summarize:
        summary = build_grouped_summary(load_results_by_db(args.summarize))
        _write_json_if_requested(summary, args.summary_output or args.output)
        if args.summary_csv:
            write_summary_csv(summary, args.summary_csv)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return

    if not args.db or not args.data:
        parser.error("--db and --data are required unless --summarize is used")

    result = evaluate_file(
        args.data,
        db_id=args.db,
        limit=args.limit,
        model=args.model,
        prompt_version=args.prompt_version,
        workers=args.workers,
        example_type=args.example_type,
        use_cache=not args.no_cache,
        use_heuristics=not args.no_heuristics,
        use_llm_answerability=not args.rule_only_answerability,
    )
    _write_json_if_requested(result, args.output)
    if args.rows_csv:
        write_rows_csv(result, args.rows_csv)
    if args.summary_output or args.summary_csv:
        summary = build_grouped_summary({args.db: result})
        _write_json_if_requested(summary, args.summary_output)
        if args.summary_csv:
            write_summary_csv(summary, args.summary_csv)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


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


def _infer_db_id(path: Path, result: Dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    if metadata.get("db_id"):
        return str(metadata["db_id"])
    stem = path.stem.lower()
    if "mimic_iii" in stem:
        return "mimic_iii"
    if "eicu" in stem:
        return "eicu"
    return stem


def _write_json_if_requested(payload: Dict[str, Any], output_path: Optional[str]) -> None:
    if not output_path:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(output_path: Union[str, Path], fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


if __name__ == "__main__":
    main()
