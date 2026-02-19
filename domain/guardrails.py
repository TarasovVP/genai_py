from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def mask_pii(text: str) -> Tuple[str, Dict[str, int]]:
    if not text:
        return text, {"email": 0, "phone": 0, "card": 0, "ssn": 0}

    counts: Dict[str, int] = {"email": 0, "phone": 0, "card": 0, "ssn": 0}
    out = text

    def _sub(regex: re.Pattern, label: str, token: str) -> None:
        nonlocal out
        matches = list(regex.finditer(out))
        if matches:
            counts[label] += len(matches)
            out = regex.sub(token, out)

    _sub(_EMAIL_RE, "email", "[EMAIL]")
    _sub(_SSN_RE, "ssn", "[SSN]")
    _sub(_CARD_RE, "card", "[CARD]")
    _sub(_PHONE_RE, "phone", "[PHONE]")

    return out, counts


_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|bypass)\b.*\b(previous|earlier|above)\b.*\b(instruction|rules)\b",
    r"\b(system prompt|developer message|hidden instructions)\b",
    r"\b(reveal|show|leak)\b.*\b(system|prompt|instructions)\b",
    r"\b(act as|roleplay)\b.*\b(system|developer|admin)\b",
    r"\b(do anything now|dan)\b",
    r"\b(jailbreak|prompt injection)\b",
    r"\bdisable\b.*\b(guardrails|safety|filters)\b",
    r"\bexfiltrate\b|\bsteal\b|\bdata leak\b",
    r"\bexecute\b.*\b(drop|truncate|delete|update|insert|alter|create)\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(user_text: str) -> Tuple[bool, str]:
    if not user_text:
        return False, ""

    if _INJECTION_RE.search(user_text):
        return True, "This looks like a prompt-injection / jailbreak attempt."

    secret_words = re.search(r"\b(api key|secret key|password|token)\b", user_text, flags=re.IGNORECASE)
    exfil_words = re.search(r"\b(show|reveal|print|leak|dump)\b", user_text, flags=re.IGNORECASE)
    if secret_words and exfil_words:
        return True, "This looks like an attempt to obtain secrets (keys/tokens/passwords)."

    return False, ""


_ANALYTICS_KEYWORDS = {
    "select",
    "sql",
    "query",
    "table",
    "schema",
    "column",
    "columns",
    "rows",
    "join",
    "group",
    "group by",
    "aggregate",
    "sum",
    "avg",
    "average",
    "count",
    "min",
    "max",
    "distinct",
    "filter",
    "where",
    "order",
    "limit",
    "top",
    "histogram",
    "chart",
    "plot",
    "bar",
    "line",
    "graph",
    "visual",
    "visualize",
    "show",
    "list",
    "how many",
    "distribution",
    "trend",
}

_OFFTOPIC_PATTERNS = [
    r"\b(write|compose|poem|lyrics|story)\b",
    r"\b(romance|dating)\b",
    r"\b(recipe|cook|cooking)\b",
    r"\b(homework|essay)\b",
    r"\b(hack|exploit|malware)\b",
]
_OFFTOPIC_RE = re.compile("|".join(f"(?:{p})" for p in _OFFTOPIC_PATTERNS), re.IGNORECASE)


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def is_on_topic(user_text: str, schema: Optional[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
    text = _normalize_text(user_text)
    meta: Dict[str, Any] = {"matched_keywords": [], "matched_schema_tokens": []}

    if not text:
        return False, "Empty question.", meta

    if _OFFTOPIC_RE.search(text):
        return (
            False,
            "This does not look like a data/SQL/visualization question. Ask about the loaded dataset instead.",
            meta,
        )

    for kw in _ANALYTICS_KEYWORDS:
        if kw in text:
            meta["matched_keywords"].append(kw)
    if meta["matched_keywords"]:
        return True, "OK", meta

    tables: List[str] = []
    cols: List[str] = []
    try:
        t = (schema or {}).get("tables") or {}
        for tname, tmeta in t.items():
            if tname:
                tables.append(str(tname).lower())
            columns = (tmeta or {}).get("columns") or {}
            for cname in columns.keys():
                if cname:
                    cols.append(str(cname).lower())
    except Exception:
        pass

    for tok in set(tables + cols):
        if tok and tok in text:
            meta["matched_schema_tokens"].append(tok)

    if meta["matched_schema_tokens"]:
        return True, "OK", meta

    return (
        False,
        "I can answer questions about the loaded data and schema (SQL/tables/charts). Try asking for counts, top-N, filters, joins, or charts.",
        meta,
    )


@dataclass
class GuardrailsResult:
    allowed: bool
    reason: str
    kind: str
    masked_text: str
    pii_counts: Dict[str, int]
    meta: Dict[str, Any]


def run_guardrails(user_text: str, schema: Optional[Dict[str, Any]]) -> GuardrailsResult:
    masked_text, pii_counts = mask_pii(user_text)

    inj, inj_reason = detect_prompt_injection(masked_text)
    if inj:
        return GuardrailsResult(
            allowed=False,
            reason=inj_reason,
            kind="injection",
            masked_text=masked_text,
            pii_counts=pii_counts,
            meta={},
        )

    ok, why, meta = is_on_topic(masked_text, schema)
    if not ok:
        return GuardrailsResult(
            allowed=False,
            reason=why,
            kind="off_topic",
            masked_text=masked_text,
            pii_counts=pii_counts,
            meta=meta or {},
        )

    return GuardrailsResult(
        allowed=True,
        reason="OK",
        kind="ok",
        masked_text=masked_text,
        pii_counts=pii_counts,
        meta=meta or {},
    )
