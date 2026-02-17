from __future__ import annotations

import re
from typing import List

_ENUM_COL_RE = re.compile(
    r"""
    (?P<col>"?[A-Za-z_][A-Za-z0-9_]*"?)
    \s+
    ENUM
    \s*\(
        (?P<vals>[^)]*)
    \)
    (?P<rest>[^,\n]*)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _split_enum_vals(vals_raw: str) -> List[str]:
    vals: List[str] = []
    for m in re.finditer(r"'((?:[^'\\]|\\.)*)'\s*(?:,|$)", (vals_raw or "").strip()):
        v = m.group(1)
        v = v.replace("\\'", "'")
        v = v.replace("\\\\", "\\")
        vals.append(v)
    return vals


def _escape_sql_literal(s: str) -> str:
    return (s or "").replace("'", "''")


def normalize_ddl_for_postgres(ddl: str) -> str:
    """
    Нормализация MySQL-подобного DDL под PostgreSQL:
    - убираем AUTO_INCREMENT / ENGINE / CHARSET / COLLATE / UNSIGNED
    - DATETIME -> TIMESTAMP
    - ENUM(...) -> TEXT + CHECK(col IN (...))
    """
    s = ddl or ""
    s = re.sub(r"\bAUTO_INCREMENT\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\)\s*ENGINE\s*=\s*\w+\s*;?", ");", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDEFAULT\s+CHARSET\s*=\s*\w+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCHARSET\s*=\s*\w+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCOLLATE\s*=\s*[\w_]+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bUNSIGNED\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDATETIME\b", "TIMESTAMP", s, flags=re.IGNORECASE)

    def repl_enum(m: re.Match) -> str:
        col = m.group("col")
        vals_raw = m.group("vals") or ""
        rest = (m.group("rest") or "").strip()

        vals = _split_enum_vals(vals_raw)
        if not vals:
            return f'{col} TEXT {rest}'.rstrip()

        in_list = ", ".join(f"'{_escape_sql_literal(v)}'" for v in vals)
        check = f'CHECK ({col} IN ({in_list}))'
        out = f"{col} TEXT {rest} {check}"
        out = re.sub(r"\s+", " ", out).strip()
        return out

    s = _ENUM_COL_RE.sub(repl_enum, s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s
