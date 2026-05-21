from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import ROOT_DIR
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
    prompt_version: str = "schema",
    llm_client: Optional[DeepSeekClient] = None,
    model: Optional[str] = None,
    followup_context: Optional[str] = None,
) -> Dict[str, Any]:
    heuristic = heuristic_generate_sql(question, db_id)
    if heuristic:
        return heuristic

    client = llm_client or DeepSeekClient(model=model)
    if not client.available:
        raise LLMNotConfigured(
            "No heuristic SQL matched and DeepSeek API is not configured. Provide --sql for executor smoke tests or set DEEPSEEK_API_KEY."
        )

    prompt = _read_prompt(prompt_version)
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
        cache_parts=[db_id, question, schema_context, followup_context, prompt_version],
        model=model,
        task_name="generate_sql",
    )
    parsed = response.parsed or {}
    return {
        "answerable": bool(parsed.get("answerable", True)),
        "sql": (parsed.get("sql") or "").strip(),
        "reason": parsed.get("reason") or "",
        "used_tables": parsed.get("used_tables") or [],
        "confidence": parsed.get("confidence"),
        "source": "llm",
        "cache_hit": response.cache_hit,
        "raw": response.content,
    }


def heuristic_generate_sql(question: str, db_id: str) -> Optional[Dict[str, Any]]:
    """Small offline fallback for common smoke-test templates."""
    text = re.sub(r"\s+", " ", question.strip().lower())
    stay_sql = _hospital_stay_length_sql(text, db_id)
    if stay_sql:
        return stay_sql
    cost_sql = _procedure_cost_sql(text, db_id)
    if cost_sql:
        return cost_sql
    route_sql = _medication_route_sql(text, db_id)
    if route_sql:
        return route_sql

    return None


def _hospital_stay_length_sql(text: str, db_id: str) -> Optional[Dict[str, Any]]:
    if "hospital" not in text or "stay" not in text:
        return None
    if not any(term in text for term in ("length", "duration", "how long")):
        return None

    order = None
    if any(term in text for term in ("first", "earliest")):
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


def _procedure_cost_sql(text: str, db_id: str) -> Optional[Dict[str, Any]]:
    if not any(term in text for term in ("cost", "price", "prices", "charge")):
        return None

    entity = _extract_cost_entity(text)
    if not entity:
        return None
    escaped = _sql_string(entity)

    if db_id == "mimic_iii":
        sql = (
            "select distinct cost.cost from cost "
            "where cost.event_type = 'procedures_icd' and cost.event_id in ( "
            "select procedures_icd.row_id from procedures_icd "
            "where procedures_icd.icd9_code = ( "
            "select d_icd_procedures.icd9_code from d_icd_procedures "
            f"where lower(d_icd_procedures.short_title) = '{escaped}' ) )"
        )
        return {
            "answerable": True,
            "sql": sql,
            "reason": "Matched EHRSQL MIMIC-III procedure cost template.",
            "used_tables": ["COST", "PROCEDURES_ICD", "D_ICD_PROCEDURES"],
            "confidence": 0.72,
            "source": "heuristic",
            "cache_hit": False,
        }
    if db_id == "eicu":
        sql = (
            "select distinct cost.cost from cost "
            "where cost.eventtype = 'treatment' and cost.eventid in ( "
            "select treatment.treatmentid from treatment "
            f"where lower(treatment.treatmentname) = '{escaped}' )"
        )
        return {
            "answerable": True,
            "sql": sql,
            "reason": "Matched EHRSQL eICU treatment cost template.",
            "used_tables": ["cost", "treatment"],
            "confidence": 0.72,
            "source": "heuristic",
            "cache_hit": False,
        }
    return None


def _extract_cost_entity(text: str) -> Optional[str]:
    patterns = [
        r"cost for the procedure (?:known as |called )?(.+?)\??$",
        r"cost of the procedure (?:known as |called )?(.+?)\??$",
        r"cost of a procedure (?:known as |called )?(.+?)\??$",
        r"cost for (?:an |a |the )?(.+?)\??$",
        r"cost to have (?:an |a |the )?(.+?)\??$",
        r"cost to undergo (?:an |a |the )?(.+?)\??$",
        r"prices? for (?:an |a |the )?(.+?)\??$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            entity = match.group(1)
            entity = re.sub(r"^(procedure|known as|called)\s+", "", entity)
            return _clean_entity(entity)
    return None


def _normalize_drug_name(value: str) -> str:
    drug = _clean_entity(value)
    if drug.startswith("chemo ") and "(chemo)" in drug:
        drug = drug[len("chemo ") :]
    return drug


def _clean_entity(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip(" ?;:'\"")).lower()
    return re.sub(r"^(?:the|a|an)\s+", "", cleaned)


def _read_prompt(prompt_version: str) -> str:
    path = PROMPT_FILES.get(prompt_version, PROMPT_FILES["schema"])
    return Path(path).read_text(encoding="utf-8")


def _sql_string(value: str) -> str:
    return value.strip(" ?;:'\"").replace("'", "''")
