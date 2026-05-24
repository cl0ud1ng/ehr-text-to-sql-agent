from __future__ import annotations

from functools import lru_cache
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Set

from src.config import ROOT_DIR, database_path
from src.fewshot_examples import build_fewshot_context
from src.llm_client import DeepSeekClient, LLMNotConfigured


PROMPT_FILES = {
    "base": ROOT_DIR / "prompts" / "base_prompt.md",
    "schema": ROOT_DIR / "prompts" / "schema_prompt.md",
    "fewshot": ROOT_DIR / "prompts" / "fewshot_prompt.md",
    "reflection": ROOT_DIR / "prompts" / "reflection_prompt.md",
}


def generate_sql(
    question: str,
    db_id: str,
    schema_context: str,
    *,
    prompt_version: str = "fewshot",
    llm_client: Optional[DeepSeekClient] = None,
    model: Optional[str] = None,
    followup_context: Optional[str] = None,
    example_type: str = "auto",
    sample_metadata: Optional[Dict[str, Any]] = None,
    use_heuristics: bool = True,
) -> Dict[str, Any]:
    if use_heuristics:
        heuristic = heuristic_generate_sql(question, db_id)
        if heuristic:
            return heuristic

    client = llm_client or DeepSeekClient(model=model)
    if not client.available:
        raise LLMNotConfigured(
            "No heuristic SQL matched and DeepSeek API is not configured. Provide --sql for executor smoke tests or set DEEPSEEK_API_KEY."
        )

    prompt = _read_prompt(prompt_version)
    fewshot_context: Optional[Dict[str, Any]] = None
    if prompt_version == "fewshot":
        fewshot_context = build_fewshot_context(
            db_id,
            question,
            example_type=example_type,
            sample_metadata=sample_metadata,
        )
        prompt = f"{prompt.rstrip()}\n\n{fewshot_context['prompt_block']}"
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Database id: {db_id}\n"
                f"Question:\n{question}\n\n"
                f"Follow-up context:\n{followup_context or '(none)'}\n\n"
                f"Schema context:\n{schema_context}\n\n"
                "Return JSON now."
            ),
        },
    ]
    response = client.json_chat(
        messages,
        cache_parts=[
            db_id,
            question,
            schema_context,
            followup_context,
            prompt_version,
            fewshot_context.get("cache_key") if fewshot_context else None,
        ],
        model=model,
        task_name="generate_sql",
    )
    parsed = response.parsed or {}
    generation: Dict[str, Any] = {
        "answerable": bool(parsed.get("answerable", True)),
        "sql": (parsed.get("sql") or "").strip(),
        "reason": parsed.get("reason") or "",
        "used_tables": parsed.get("used_tables") or [],
        "confidence": parsed.get("confidence"),
        "source": "llm",
        "cache_hit": response.cache_hit,
        "raw": response.content,
    }
    if fewshot_context:
        generation["fewshot"] = _fewshot_metadata(fewshot_context)
    return generation


def _fewshot_metadata(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "example_type": context.get("example_type"),
        "seed": context.get("seed"),
        "cache_key": context.get("cache_key"),
        "examples": [
            {
                key: example.get(key)
                for key in ("kind", "tag", "question", "answerable", "source_file", "source_group")
                if example.get(key) is not None
            }
            for example in context.get("examples", [])
        ],
    }


def heuristic_generate_sql(question: str, db_id: str) -> Optional[Dict[str, Any]]:
    """Small offline fallback for common smoke-test templates."""
    text = re.sub(r"\s+", " ", question.strip().lower())
    temporal_sql = _temporal_sql(text, db_id)
    if temporal_sql:
        return temporal_sql
    stay_sql = _hospital_stay_length_sql(text, db_id)
    if stay_sql:
        return stay_sql
    cost_sql = _cost_sql(text, db_id)
    if cost_sql:
        return cost_sql
    route_sql = _medication_route_sql(text, db_id)
    if route_sql:
        return route_sql

    return None


def _temporal_sql(text: str, db_id: str) -> Optional[Dict[str, Any]]:
    if db_id == "mimic_iii":
        patient = re.search(r"patient\s+(\d+)", text)
        subject_id = patient.group(1) if patient else None
        if not subject_id:
            return None

        if "admitted" in text and "hospital" in text and "this year" in text:
            return _heuristic_sql(
                "select count(*)>0 from admissions "
                f"where admissions.subject_id = {subject_id} "
                "and datetime(admissions.admittime,'start of year') = datetime(current_time,'start of year','-0 year')",
                "Matched MIMIC-III this-year admission template.",
                ["ADMISSIONS"],
            )

        year_until = re.search(r"\buntil\s+(\d{4})\b", text)
        if year_until and any(term in text for term in ("pay", "cost", "bill")) and "hospital stay" in text:
            year = year_until.group(1)
            return _heuristic_sql(
                "select sum(cost.cost) from cost "
                "where cost.hadm_id in ( "
                f"select admissions.hadm_id from admissions where admissions.subject_id = {subject_id} "
                f") and strftime('%y',cost.chargetime) <= '{year}'",
                "Matched MIMIC-III hospital cost until-year template.",
                ["COST", "ADMISSIONS"],
            )

        year_since = re.search(r"\bsince\s+(\d{4})\b", text)
        if year_since and any(term in text for term in ("pay", "cost", "bill")) and "hospital stay" in text:
            year = year_since.group(1)
            return _heuristic_sql(
                "select sum(cost.cost) from cost "
                "where cost.hadm_id in ( "
                f"select admissions.hadm_id from admissions where admissions.subject_id = {subject_id} "
                f") and strftime('%y',cost.chargetime) >= '{year}'",
                "Matched MIMIC-III hospital cost since-year template.",
                ["COST", "ADMISSIONS"],
            )

        if "age" in text and "current hospital encounter" in text:
            return _heuristic_sql(
                "select admissions.age from admissions "
                f"where admissions.subject_id = {subject_id} and admissions.dischtime is null",
                "Matched MIMIC-III current hospital encounter age template.",
                ["ADMISSIONS"],
            )

        careunit = re.search(r"\bin the\s+([a-z0-9]+)\s+during this hospital encounter", text)
        if careunit and "how many hours" in text and "since the last stay" in text:
            unit = _sql_string(careunit.group(1))
            return _heuristic_sql(
                "select 24 * ( strftime('%j',current_time) - strftime('%j',transfers.intime) ) "
                "from transfers where transfers.hadm_id in ( "
                f"select admissions.hadm_id from admissions where admissions.subject_id = {subject_id} "
                "and admissions.dischtime is null ) "
                f"and transfers.careunit = '{unit}' order by transfers.intime desc limit 1",
                "Matched MIMIC-III current encounter last careunit stay template.",
                ["TRANSFERS", "ADMISSIONS"],
            )

    if db_id == "eicu":
        patient = re.search(r"patient\s+([0-9]{3}-[0-9]+)", text)
        uniquepid = _sql_string(patient.group(1)) if patient else None
        if not uniquepid:
            return None

        if any(term in text for term in ("total hospital cost", "hospital cost")) and "this year" in text:
            return _heuristic_sql(
                "select sum(cost.cost) from cost "
                "where cost.patienthealthsystemstayid in ( "
                f"select patient.patienthealthsystemstayid from patient where patient.uniquepid = '{uniquepid}' "
                ") and datetime(cost.chargetime,'start of year') = datetime(current_time,'start of year','-0 year')",
                "Matched eICU this-year hospital cost template.",
                ["cost", "patient"],
            )

        year_until = re.search(r"\buntil\s+(\d{4})\b", text)
        if year_until and "admitted" in text and "hospital" in text:
            year = year_until.group(1)
            return _heuristic_sql(
                "select count(*)>0 from patient "
                f"where patient.uniquepid = '{uniquepid}' "
                f"and strftime('%y',patient.hospitaladmittime) <= '{year}'",
                "Matched eICU hospital admission until-year template.",
                ["patient"],
            )

        year_since = re.search(r"\bsince\s+(\d{4})\b", text)
        if year_since and any(term in text for term in ("hospital bill", "hospital cost", "cost")):
            year = year_since.group(1)
            return _heuristic_sql(
                "select sum(cost.cost) from cost "
                "where cost.patienthealthsystemstayid in ( "
                f"select patient.patienthealthsystemstayid from patient where patient.uniquepid = '{uniquepid}' "
                f") and strftime('%y',cost.chargetime) >= '{year}'",
                "Matched eICU hospital cost since-year template.",
                ["cost", "patient"],
            )

        if "age" in text and "current hospital encounter" in text:
            return _heuristic_sql(
                f"select patient.age from patient where patient.uniquepid = '{uniquepid}' "
                "and patient.hospitaldischargetime is null",
                "Matched eICU current hospital encounter age template.",
                ["patient"],
            )

        ward = re.search(r"\bward\s+(\d+)\b", text)
        if ward and "how many hours" in text and "since the first time" in text:
            return _heuristic_sql(
                "select 24 * ( strftime('%j',current_time) - strftime('%j',patient.unitadmittime) ) "
                f"from patient where patient.uniquepid = '{uniquepid}' "
                f"and patient.wardid = {ward.group(1)} and patient.hospitaldischargetime is null "
                "order by patient.unitadmittime asc limit 1",
                "Matched eICU current encounter first ward stay template.",
                ["patient"],
            )

    return None


def _hospital_stay_length_sql(text: str, db_id: str) -> Optional[Dict[str, Any]]:
    if "hospital" not in text or "stay" not in text:
        return None
    if not any(term in text for term in ("length", "duration", "how long")):
        return None

    order = None
    if any(term in text for term in ("first", "earliest", "initial")):
        order = "asc"
    elif any(term in text for term in ("last", "latest", "most recent")):
        order = "desc"
    if order is None:
        return None

    if db_id == "mimic_iii":
        match = re.search(r"patient\s+(\d+)", text)
        if not match:
            return None
        subject_id = match.group(1)
        sql = (
            "select strftime('%j', admissions.dischtime) - strftime('%j', admissions.admittime) "
            "from admissions "
            f"where admissions.subject_id = {subject_id} and admissions.dischtime is not null "
            f"order by admissions.admittime {order} limit 1"
        )
        return {
            "answerable": True,
            "sql": sql,
            "reason": "Matched EHRSQL hospital stay length template.",
            "used_tables": ["ADMISSIONS"],
            "confidence": 0.7,
            "source": "heuristic",
            "cache_hit": False,
        }

    if db_id == "eicu":
        match = re.search(r"patient\s+([0-9]{3}-[0-9]+)", text)
        if not match:
            return None
        uniquepid = _sql_string(match.group(1))
        sql = (
            "select strftime('%j', patient.hospitaldischargetime) - strftime('%j', patient.hospitaladmittime) "
            "from patient "
            f"where patient.uniquepid = '{uniquepid}' and patient.hospitaladmittime is not null "
            f"order by patient.hospitaladmittime {order} limit 1"
        )
        return {
            "answerable": True,
            "sql": sql,
            "reason": "Matched EHRSQL eICU hospital stay length template.",
            "used_tables": ["patient"],
            "confidence": 0.7,
            "source": "heuristic",
            "cache_hit": False,
        }
    return None


def _heuristic_sql(sql: str, reason: str, used_tables: list[str], confidence: float = 0.74) -> Dict[str, Any]:
    return {
        "answerable": True,
        "sql": sql,
        "reason": reason,
        "used_tables": used_tables,
        "confidence": confidence,
        "source": "heuristic",
        "cache_hit": False,
    }


def _medication_route_sql(text: str, db_id: str) -> Optional[Dict[str, Any]]:
    if not any(term in text for term in ("intake", "administer", "administered", "delivered")):
        return None

    patterns = [
        r"methods? of intake for (.+?)\??$",
        r"method for administering (.+?) intake\??$",
        r"method for administering (.+?)\??$",
        r"method of (.+?) intake\??$",
        r"intake method of (.+?)\??$",
        r"ingesting method of (.+?)\??$",
        r"how (?:is|are) (.+?) delivered\??$",
        r"how (?:is|are) (.+?) administered\??$",
    ]
    drug = None
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            drug = _normalize_drug_name(match.group(1))
            break
    if not drug:
        return None

    escaped = _sql_string(drug)
    if db_id == "mimic_iii":
        return {
            "answerable": True,
            "sql": f"select distinct prescriptions.route from prescriptions where lower(prescriptions.drug) = '{escaped}'",
            "reason": "Matched EHRSQL medication route template.",
            "used_tables": ["PRESCRIPTIONS"],
            "confidence": 0.72,
            "source": "heuristic",
            "cache_hit": False,
        }
    if db_id == "eicu":
        return {
            "answerable": True,
            "sql": f"select distinct medication.routeadmin from medication where lower(medication.drugname) = '{escaped}'",
            "reason": "Matched EHRSQL eICU medication route template.",
            "used_tables": ["medication"],
            "confidence": 0.72,
            "source": "heuristic",
            "cache_hit": False,
        }
    return None


def _cost_sql(text: str, db_id: str) -> Optional[Dict[str, Any]]:
    extracted = _extract_cost_entity(text)
    if extracted is None:
        extracted = _extract_lab_lookup_entity(text, db_id)
    if extracted is None:
        return None

    entity, cues = extracted
    if not entity:
        return None

    kind = _classify_cost_entity(db_id, entity, cues)
    if not kind:
        return None
    resolved = _resolve_cost_entity(db_id, kind, entity) or entity
    escaped = _sql_string(resolved)

    if db_id == "mimic_iii":
        sql_by_kind = {
            "procedure": (
                "select distinct cost.cost from cost "
                "where cost.event_type = 'procedures_icd' and cost.event_id in ( "
                "select procedures_icd.row_id from procedures_icd "
                "where procedures_icd.icd9_code = ( "
                "select d_icd_procedures.icd9_code from d_icd_procedures "
                f"where lower(d_icd_procedures.short_title) = '{escaped}' ) )"
            ),
            "lab": (
                "select distinct cost.cost from cost "
                "where cost.event_type = 'labevents' and cost.event_id in ( "
                "select labevents.row_id from labevents "
                "where labevents.itemid in ( "
                "select d_labitems.itemid from d_labitems "
                f"where lower(d_labitems.label) = '{escaped}' ) )"
            ),
            "drug": (
                "select distinct cost.cost from cost "
                "where cost.event_type = 'prescriptions' and cost.event_id in ( "
                "select prescriptions.row_id from prescriptions "
                f"where lower(prescriptions.drug) = '{escaped}' )"
            ),
            "diagnosis": (
                "select distinct cost.cost from cost "
                "where cost.event_type = 'diagnoses_icd' and cost.event_id in ( "
                "select diagnoses_icd.row_id from diagnoses_icd "
                "where diagnoses_icd.icd9_code = ( "
                "select d_icd_diagnoses.icd9_code from d_icd_diagnoses "
                f"where lower(d_icd_diagnoses.short_title) = '{escaped}' ) )"
            ),
        }
        sql = sql_by_kind.get(kind)
        if not sql:
            return None
        return {
            "answerable": True,
            "sql": sql,
            "reason": f"Matched EHRSQL MIMIC-III {kind} cost template.",
            "used_tables": _cost_used_tables(db_id, kind),
            "confidence": 0.72,
            "source": "heuristic",
            "cache_hit": False,
        }
    if db_id == "eicu":
        sql_by_kind = {
            "procedure": (
                "select distinct cost.cost from cost "
                "where cost.eventtype = 'treatment' and cost.eventid in ( "
                "select treatment.treatmentid from treatment "
                f"where lower(treatment.treatmentname) = '{escaped}' )"
            ),
            "lab": (
                "select distinct cost.cost from cost "
                "where cost.eventtype = 'lab' and cost.eventid in ( "
                "select lab.labid from lab "
                f"where lower(lab.labname) = '{escaped}' )"
            ),
            "drug": (
                "select distinct cost.cost from cost "
                "where cost.eventtype = 'medication' and cost.eventid in ( "
                "select medication.medicationid from medication "
                f"where lower(medication.drugname) = '{escaped}' )"
            ),
            "diagnosis": (
                "select distinct cost.cost from cost "
                "where cost.eventtype = 'diagnosis' and cost.eventid in ( "
                "select diagnosis.diagnosisid from diagnosis "
                f"where lower(diagnosis.diagnosisname) = '{escaped}' )"
            ),
        }
        sql = sql_by_kind.get(kind)
        if not sql:
            return None
        return {
            "answerable": True,
            "sql": sql,
            "reason": f"Matched EHRSQL eICU {kind} cost template.",
            "used_tables": _cost_used_tables(db_id, kind),
            "confidence": 0.72,
            "source": "heuristic",
            "cache_hit": False,
        }
    return None


def _extract_cost_entity(text: str) -> Optional[tuple[str, Set[str]]]:
    patterns = [
        (r"cost for the procedure (?:known as |called )?(.+?)\??$", {"procedure"}),
        (r"cost of the procedure (?:known as |called )?(.+?)\??$", {"procedure"}),
        (r"cost of a procedure (?:known as |called )?(.+?)\??$", {"procedure"}),
        (r"cost of a treatment (?:known as |called )?(.+?)\??$", {"procedure"}),
        (r"cost of a diagnostic (.+?)\??$", {"procedure"}),
        (r"cost of (?:a |an |the )?lab (.+?) tests?\??$", {"lab"}),
        (r"cost for (.+?) lab tests?\??$", {"lab"}),
        (r"cost of a drug named (.+?)\??$", {"drug"}),
        (r"price of a drug named (.+?)\??$", {"drug"}),
        (r"cost to take (.+?)\??$", {"drug"}),
        (r"cost for (?:an |a |the )?(.+?)\??$", {"general"}),
        (r"cost of (?:an |a |the )?(.+?)\??$", {"general"}),
        (r"costs of (?:an |a |the )?(.+?)\??$", {"general"}),
        (r"cost to have (?:an |a |the )?(.+?)\??$", {"procedure"}),
        (r"cost to undergo (?:an |a |the )?(.+?)\??$", {"procedure"}),
        (r"prices? for (?:an |a |the )?(.+?)\??$", {"general"}),
        (r"price of (?:an |a |the )?(.+?)\??$", {"general"}),
        (r"how much does it cost to take (.+?)\??$", {"drug"}),
        (r"how much does it cost for (.+?) lab tests?\??$", {"lab"}),
        (r"how much does (.+?) cost\??$", {"general"}),
        (r"how much do (.+?) cost\??$", {"general"}),
        (r"how much is the cost of (?:an |a |the )?(.+?)\??$", {"general"}),
        (r"how much is (?:an |a |the )?(.+?)\??$", {"general"}),
    ]
    for pattern, cues in patterns:
        match = re.search(pattern, text)
        if match:
            entity = match.group(1)
            return _clean_entity(entity), cues
    return None


def _extract_lab_lookup_entity(text: str, db_id: str) -> Optional[tuple[str, Set[str]]]:
    if db_id not in {"mimic_iii", "eicu"}:
        return None
    match = re.search(r"what is (?:the )?(.+?)\??$", text)
    if not match:
        return None
    entity = _clean_entity(match.group(1))
    if _resolve_cost_entity(db_id, "lab", entity):
        return entity, {"lab"}
    return None


def _classify_cost_entity(db_id: str, entity: str, cues: Set[str]) -> Optional[str]:
    if db_id not in {"mimic_iii", "eicu"}:
        return None
    cue_priority = []
    if "drug" in cues:
        cue_priority.append("drug")
    if "lab" in cues:
        cue_priority.append("lab")
    if "procedure" in cues:
        cue_priority.append("procedure")
    if "diagnosis" in cues:
        cue_priority.append("diagnosis")
    for kind in cue_priority:
        if _resolve_cost_entity(db_id, kind, entity):
            return kind
    if cue_priority:
        return cue_priority[0]

    for kind in ("procedure", "lab", "drug", "diagnosis"):
        if _resolve_cost_entity(db_id, kind, entity):
            return kind
    return None


def _cost_used_tables(db_id: str, kind: str) -> list[str]:
    if db_id == "mimic_iii":
        return {
            "procedure": ["COST", "PROCEDURES_ICD", "D_ICD_PROCEDURES"],
            "lab": ["COST", "LABEVENTS", "D_LABITEMS"],
            "drug": ["COST", "PRESCRIPTIONS"],
            "diagnosis": ["COST", "DIAGNOSES_ICD", "D_ICD_DIAGNOSES"],
        }.get(kind, ["COST"])
    return {
        "procedure": ["cost", "treatment"],
        "lab": ["cost", "lab"],
        "drug": ["cost", "medication"],
        "diagnosis": ["cost", "diagnosis"],
    }.get(kind, ["cost"])


_COST_ENTITY_FIELDS = {
    "mimic_iii": {
        "procedure": ("d_icd_procedures", "short_title"),
        "lab": ("d_labitems", "label"),
        "drug": ("prescriptions", "drug"),
        "diagnosis": ("d_icd_diagnoses", "short_title"),
    },
    "eicu": {
        "procedure": ("treatment", "treatmentname"),
        "lab": ("lab", "labname"),
        "drug": ("medication", "drugname"),
        "diagnosis": ("diagnosis", "diagnosisname"),
    },
}


@lru_cache(maxsize=4096)
def _resolve_cost_entity(db_id: str, kind: str, entity: str) -> Optional[str]:
    table_column = _COST_ENTITY_FIELDS.get(db_id, {}).get(kind)
    if not table_column:
        return None
    table, column = table_column
    try:
        path = database_path(db_id)
    except Exception:
        return None
    if not path.exists():
        return None

    candidates = _entity_candidates(entity)
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            for candidate in candidates:
                row = conn.execute(
                    f"select {column} from {table} where lower({column}) = ? limit 1",
                    (candidate,),
                ).fetchone()
                if row and row[0] is not None:
                    return str(row[0]).strip().lower()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return None


def _entity_candidates(entity: str) -> tuple[str, ...]:
    candidates: list[str] = []

    def add(value: str) -> None:
        cleaned = _clean_entity(value)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    add(entity)
    for suffix in (" lab tests", " lab test", " tests", " test", " suspension"):
        if entity.endswith(suffix):
            add(entity[: -len(suffix)])
    if entity.startswith("diagnostic "):
        add(entity[len("diagnostic ") :])
    return tuple(candidates)


def _normalize_drug_name(value: str) -> str:
    drug = _clean_entity(value)
    if drug.startswith("chemo ") and "(chemo)" in drug:
        drug = drug[len("chemo ") :]
    return drug


def _clean_entity(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip(" ?;:'\"")).lower()
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned)
    cleaned = re.sub(r"^(?:procedure|treatment|drug named|known as|called)\s+", "", cleaned)
    cleaned = re.sub(r"^diagnostic\s+", "", cleaned)
    cleaned = re.sub(r"^lab\s+", "", cleaned)
    cleaned = re.sub(r"\s+lab tests?$", "", cleaned)
    return cleaned


def _read_prompt(prompt_version: str) -> str:
    path = PROMPT_FILES.get(prompt_version, PROMPT_FILES["schema"])
    return Path(path).read_text(encoding="utf-8")


def _sql_string(value: str) -> str:
    return value.strip(" ?;:'\"").replace("'", "''")
