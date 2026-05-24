from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config import ROOT_DIR


DEFAULT_FEWSHOT_SEED = 20260524
EXAMPLE_TYPES = {"auto", "basic", "time"}
EXAMPLE_DATA_DIR = ROOT_DIR / "data" / "EHRSQL" / "示例数据"
_DB_FILE_PREFIX = {
    "mimic_iii": "mimic_iii",
    "eicu": "eicu",
}
_TIME_CUES = {
    "first",
    "last",
    "latest",
    "earliest",
    "current",
    "currently",
    "since",
    "until",
    "within",
    "during",
    "before",
    "after",
    "ago",
    "this year",
    "this month",
    "last year",
    "last month",
    "hospital visit",
    "icu stay",
}


def build_fewshot_context(
    db_id: str,
    question: str,
    *,
    example_type: str = "auto",
    sample_metadata: Optional[Dict[str, Any]] = None,
    seed: int = DEFAULT_FEWSHOT_SEED,
    max_basic_possible: int = 1,
    max_basic_impossible: int = 2,
    max_time_patterns: int = 5,
    max_time_sql_examples: int = 1,
) -> Dict[str, Any]:
    """Build a deterministic few-shot block from EHRSQL representative examples."""

    resolved_type = infer_example_type(question, example_type=example_type, sample_metadata=sample_metadata)
    if resolved_type == "time":
        examples = _time_examples(
            db_id,
            sample_metadata=sample_metadata,
            seed=seed,
            max_patterns=max_time_patterns,
            max_sql_examples=max_time_sql_examples,
        )
    else:
        examples = _basic_examples(
            db_id,
            seed=seed,
            max_possible=max_basic_possible,
            max_impossible=max_basic_impossible,
        )

    prompt_block = format_fewshot_block(resolved_type, examples, seed=seed)
    return {
        "example_type": resolved_type,
        "seed": seed,
        "examples": examples,
        "prompt_block": prompt_block,
        "cache_key": _cache_key(resolved_type, seed, examples),
    }


def infer_example_type(
    question: str,
    *,
    example_type: str = "auto",
    sample_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    if example_type not in EXAMPLE_TYPES:
        raise ValueError(f"Unknown few-shot example type: {example_type}")
    if example_type != "auto":
        return example_type

    metadata = sample_metadata or {}
    explicit = metadata.get("example_type")
    if explicit in {"basic", "time"}:
        return str(explicit)
    if _time_tags(metadata):
        return "time"

    text = re.sub(r"\s+", " ", question.lower())
    return "time" if any(cue in text for cue in _TIME_CUES) else "basic"


def format_fewshot_block(example_type: str, examples: Sequence[Dict[str, Any]], *, seed: int) -> str:
    lines = [
        "Few-shot examples loaded from data/EHRSQL/示例数据.",
        f"Example type: {example_type}; deterministic seed: {seed}.",
        "Use these examples as patterns only. Do not copy patient IDs, entity values, or SQL literals unless they appear in the current question.",
    ]
    if example_type == "time":
        lines.append("")
        lines.append("Time reasoning pattern examples:")
        pattern_index = 1
        sql_index = 1
        for example in examples:
            if example["kind"] == "time_pattern":
                lines.append(f"{pattern_index}. Tag: {example['tag']}")
                lines.append(f"   Question: {example['question']}")
                pattern_index += 1
        sql_examples = [example for example in examples if example["kind"] == "sql_example"]
        if sql_examples:
            lines.append("")
            lines.append("Representative SQL example:")
            for example in sql_examples:
                lines.append(f"{sql_index}. Question: {example['question']}")
                lines.append(f"   SQL: {example['sql']}")
                sql_index += 1
        lines.append("For tag-only examples, infer SQL from the current schema and the active question.")
    else:
        lines.append("")
        lines.append("Basic answerability examples:")
        for index, example in enumerate(examples, start=1):
            lines.append(f"{index}. Question: {example['question']}")
            if example.get("answerable"):
                lines.append(f"   SQL: {example['sql']}")
            else:
                lines.append("   JSON: {\"answerable\": false, \"sql\": \"\", \"reason\": \"not supported by the schema\"}")
    return "\n".join(lines)


def _time_examples(
    db_id: str,
    *,
    sample_metadata: Optional[Dict[str, Any]],
    seed: int,
    max_patterns: int,
    max_sql_examples: int,
) -> List[Dict[str, Any]]:
    representatives = _load_time_representatives(db_id)
    selected_tags = [tag for tag in _time_tags(sample_metadata or {}) if tag in representatives]
    examples: List[Dict[str, Any]] = [_time_pattern_example(db_id, tag, representatives[tag]) for tag in selected_tags]

    remaining = [(tag, representatives[tag]) for tag in sorted(representatives) if tag not in selected_tags]
    shuffled = _stable_shuffle(remaining, seed, db_id, "time-patterns", *selected_tags)
    for tag, question in shuffled:
        if len([example for example in examples if example["kind"] == "time_pattern"]) >= max_patterns:
            break
        examples.append(_time_pattern_example(db_id, tag, question))

    possible_examples = _load_answerability_representatives(db_id).get("possible_examples", [])
    for item in _stable_sample(possible_examples, max_sql_examples, seed, db_id, "time-sql"):
        examples.append(
            {
                "kind": "sql_example",
                "answerable": True,
                "question": str(item.get("question", "")),
                "sql": str(item.get("query", "")),
                "source_file": _representative_file(db_id, "impossible").name,
                "source_group": "possible_examples",
            }
        )
    return examples


def _basic_examples(
    db_id: str,
    *,
    seed: int,
    max_possible: int,
    max_impossible: int,
) -> List[Dict[str, Any]]:
    representatives = _load_answerability_representatives(db_id)
    examples: List[Dict[str, Any]] = []
    for item in _stable_sample(representatives.get("possible_examples", []), max_possible, seed, db_id, "basic-possible"):
        examples.append(
            {
                "kind": "answerable",
                "answerable": True,
                "question": str(item.get("question", "")),
                "sql": str(item.get("query", "")),
                "source_file": _representative_file(db_id, "impossible").name,
                "source_group": "possible_examples",
            }
        )
    for item in _stable_sample(representatives.get("impossible_examples", []), max_impossible, seed, db_id, "basic-impossible"):
        examples.append(
            {
                "kind": "unanswerable",
                "answerable": False,
                "question": str(item.get("question", "")),
                "sql": "",
                "source_file": _representative_file(db_id, "impossible").name,
                "source_group": "impossible_examples",
            }
        )
    return examples


def _time_pattern_example(db_id: str, tag: str, question: str) -> Dict[str, Any]:
    return {
        "kind": "time_pattern",
        "tag": tag,
        "question": question,
        "source_file": _representative_file(db_id, "tag").name,
    }


def _time_tags(metadata: Dict[str, Any]) -> List[str]:
    raw = metadata.get("t_tag") or metadata.get("time_tags") or metadata.get("time_tag")
    if raw is None:
        return []
    if isinstance(raw, str):
        values: Iterable[Any] = [raw]
    elif isinstance(raw, Iterable):
        values = raw
    else:
        values = [raw]

    tags: List[str] = []
    for value in values:
        tag = str(value or "").strip()
        if tag and tag not in tags:
            tags.append(tag)
    return sorted(tags)


@lru_cache(maxsize=8)
def _load_time_representatives(db_id: str) -> Dict[str, str]:
    path = _representative_file(db_id, "tag")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object in {path}")
    return {str(key): str(value) for key, value in raw.items()}


@lru_cache(maxsize=8)
def _load_answerability_representatives(db_id: str) -> Dict[str, List[Dict[str, Any]]]:
    path = _representative_file(db_id, "impossible")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object in {path}")
    return {
        "possible_examples": [item for item in raw.get("possible_examples", []) if isinstance(item, dict)],
        "impossible_examples": [item for item in raw.get("impossible_examples", []) if isinstance(item, dict)],
    }


def _representative_file(db_id: str, kind: str) -> Path:
    prefix = _DB_FILE_PREFIX.get(db_id)
    if not prefix:
        raise ValueError(f"Unsupported EHRSQL database for few-shot examples: {db_id}")
    suffix = "test_split_tag_representatives.json" if kind == "tag" else "test_split_impossible_representatives.json"
    return EXAMPLE_DATA_DIR / f"{prefix}_{suffix}"


def _stable_sample(items: Sequence[Any], limit: int, seed: int, *parts: Any) -> List[Any]:
    if limit <= 0:
        return []
    return _stable_shuffle(list(items), seed, *parts)[:limit]


def _stable_shuffle(items: Sequence[Any], seed: int, *parts: Any) -> List[Any]:
    shuffled = list(items)
    rng = random.Random(_stable_seed(seed, *parts))
    rng.shuffle(shuffled)
    return shuffled


def _stable_seed(seed: int, *parts: Any) -> int:
    raw = json.dumps([seed, *parts], ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _cache_key(example_type: str, seed: int, examples: Sequence[Dict[str, Any]]) -> str:
    payload = {
        "example_type": example_type,
        "seed": seed,
        "examples": [
            {
                "kind": example.get("kind"),
                "tag": example.get("tag"),
                "question": example.get("question"),
                "sql": example.get("sql"),
                "answerable": example.get("answerable"),
                "source_file": example.get("source_file"),
                "source_group": example.get("source_group"),
            }
            for example in examples
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
