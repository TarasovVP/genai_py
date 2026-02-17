from __future__ import annotations

import re
from typing import Tuple

_SQL_BLOCKLIST = re.compile(
    r"\b(drop|truncate|alter|create|grant|revoke|comment|vacuum|analyze|insert|update|delete|merge)\b",
    re.IGNORECASE,
)


def is_sql_safe_readonly(sql: str) -> Tuple[bool, str]:
    if not sql or not sql.strip():
        return False, "Empty SQL"
    s = sql.strip().strip(";").strip()
    if not (re.match(r"^(with\b[\s\S]+?\bselect\b|select\b)", s, flags=re.IGNORECASE)):
        return False, "Only SELECT (or WITH ... SELECT) is allowed"
    if _SQL_BLOCKLIST.search(s):
        return False, "Only read-only queries are allowed"
    if ";" in s:
        return False, "Multiple statements are not allowed"
    return True, "OK"
