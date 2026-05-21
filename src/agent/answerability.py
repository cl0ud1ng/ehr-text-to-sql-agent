from __future__ import annotations

import re
from typing import Any, Dict, Optional

from src.llm_client import DeepSeekClient, LLMNotConfigured


UNSUPPORTED_TERMS = {
    "genetic",
    "genome",
    "gene",
    "mutation",
    "imaging image",
    "x-ray image",
    "mri image",
    "ct image",
    "pathology slide",
    "recommend",
    "should i",
    "what medicine to use",
    "plan to visit",
    "next hospital visit",
    "earliest next hospital visit",
    "future visit",
    "follow-up appointment",
    "relieve",
    "clinical guideline",
    "causal",
    "why did",
}


def judge_answerability(
    question: str,
    schema_context: str,
    *,
    llm_client: Optional[DeepSeekClient] = None,
    model: Optional[str] = None,
    followup_context: Optional[str] = None,
) -> Dict[str, Any]:
    rule = rule_based_answerability(question, schema_context)
    if rule["decision"] in {"answerable", "not_answerable"}:
        return rule

    client = llm_client or DeepSeekClient(model=model)
    if not client.available:
        return rule

    messages = [
        {
            "role": "system",
            "content": (
                "Decide whether a natural language question is answerable from the provided "
                "EHR SQLite schema. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Follow-up context:\n{followup_context or '(none)'}\n\n"
                f"Schema context:\n{schema_context}\n\n"
                "Return JSON with keys: answerable, reason, required_tables, missing_evidence."
            ),
        },
    ]
    try:
        response = client.json_chat(messages, cache_parts=[question, schema_context, followup_context], model=model, task_name="answerability")
    except LLMNotConfigured:
        return rule

    parsed = response.parsed or {}
    answerable = bool(parsed.get("answerable"))
    return {
        "decision": "answerable" if answerable else "not_answerable",
        "answerable": answerable,
        "reason": parsed.get("reason") or parsed.get("missing_evidence") or "Model answerability decision.",
        "required_tables": parsed.get("required_tables") or [],
        "missing_evidence": parsed.get("missing_evidence") or "",
        "source": "llm",
        "cache_hit": response.cache_hit,
    }


def rule_based_answerability(question: str, schema_context: str) -> Dict[str, Any]:
    text = re.sub(r"\s+", " ", question.lower()).strip()
    for term in UNSUPPORTED_TERMS:
        if term in text:
            return {
                "decision": "not_answerable",
                "answerable": False,
                "reason": f"Question appears to require unsupported information or medical advice: {term}.",
                "required_tables": [],
                "missing_evidence": term,
                "source": "rules",
            }
    schema_text = schema_context.lower()
    if _looks_like_cost_procedure_question(text) and "cost" in schema_text:
        if "d_icd_procedures" in schema_text or "procedures_icd" in schema_text or "treatment" in schema_text:
            return {
                "decision": "answerable",
                "answerable": True,
                "reason": "Cost/procedure questions are answerable from COST plus procedure/treatment tables in this schema.",
                "required_tables": ["COST", "PROCEDURES_ICD/D_ICD_PROCEDURES or treatment"],
                "missing_evidence": "",
                "source": "rules",
            }
    if _looks_like_medication_route_question(text):
        if "prescriptions" in schema_text or "medication" in schema_text:
            return {
                "decision": "answerable",
                "answerable": True,
                "reason": "Medication route/intake questions are answerable from prescription or medication route columns.",
                "required_tables": ["PRESCRIPTIONS or medication"],
                "missing_evidence": "",
                "source": "rules",
            }
    if _looks_like_hospital_stay_length_question(text):
        if ("admissions" in schema_text and "dischtime" in schema_text) or (
            "patient" in schema_text and "hospitaldischargetime" in schema_text
        ):
            return {
                "decision": "answerable",
                "answerable": True,
                "reason": "Hospital stay length questions are answerable from admission and discharge time columns.",
                "required_tables": ["ADMISSIONS or patient"],
                "missing_evidence": "",
                "source": "rules",
            }
    if not schema_context.strip():
        return {
            "decision": "uncertain",
            "answerable": True,
            "reason": "No schema context was available; deferring to SQL generation.",
            "required_tables": [],
            "missing_evidence": "",
            "source": "rules",
        }
    return {
        "decision": "uncertain",
        "answerable": True,
        "reason": "No rule-level reason to refuse; schema-constrained generation may proceed.",
        "required_tables": [],
        "missing_evidence": "",
        "source": "rules",
    }


def _looks_like_cost_procedure_question(text: str) -> bool:
    return any(term in text for term in ("cost", "price", "prices", "charge")) and any(
        term in text
        for term in (
            "procedure",
            "procedures",
            "transfusion",
            "lobectomy",
            "valvuloplasty",
            "fixation",
            "angiography",
            "catheter",
            "resection",
        )
    )


def _looks_like_medication_route_question(text: str) -> bool:
    return any(term in text for term in ("intake", "administer", "administered", "delivered")) and any(
        term in text for term in ("method", "methods", "how", "route", "intake")
    )


def _looks_like_hospital_stay_length_question(text: str) -> bool:
    return "hospital" in text and "stay" in text and any(term in text for term in ("length", "duration", "long"))
