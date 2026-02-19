from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

from domain.sql_guard import is_sql_safe_readonly
from services.vertex_logged import VertexLogged
from services.postgres_logged import PostgresLogged


def sql_gen_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "sql": {"type": "string"},
            "explanation": {"type": "string"},
            "result_kind": {"type": "string", "enum": ["table", "scalar", "empty"]},
            "chart": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["none", "bar", "line", "hist"]},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "title": {"type": "string"},
                    "bins": {"type": "integer"},
                },
                "required": ["type"],
                "additionalProperties": False,
            },
        },
        "required": ["sql", "explanation", "result_kind", "chart"],
        "additionalProperties": False,
    }


def _format_chat_context(messages: List[Dict[str, Any]], limit: int = 10, max_chars: int = 5000) -> str:
    if not messages:
        return "No previous messages."

    tail = messages[-limit:]
    lines: List[str] = []
    for m in tail:
        role = (m.get("role") or "").strip().lower()
        if role not in ("user", "assistant", "system"):
            role = "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        lines.append(f"{role.upper()}: {content}")

    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out or "No previous messages."


def build_nl2sql_prompt(
    *,
    question: str,
    schema_text: str,
    chat_messages: Optional[List[Dict[str, Any]]] = None,
    last_sql: Optional[str] = None,
) -> str:
    ctx = _format_chat_context(chat_messages or [], limit=10, max_chars=5000)
    last_sql_block = (last_sql or "").strip()
    if not last_sql_block:
        last_sql_block = "None"

    return f"""
You are a data analyst assistant that converts user messages into a single PostgreSQL SELECT query.

You are chatting with the user. The user may ask follow-up questions like:
- "now group by department"
- "make it a histogram"
- "filter only last month"
You MUST use the conversation context and the previous SQL (if any) to answer correctly.

Hard rules:
- Output MUST be valid PostgreSQL.
- Only SELECT or WITH ... SELECT is allowed. No INSERT/UPDATE/DELETE/DDL.
- Single statement only.
- Prefer standard PostgreSQL functions.
- Use JOINs and aggregation when appropriate.
- If user asks for "top", use ORDER BY + LIMIT.

Visualization rules (if the user asks for a chart):
- bar/line: return aggregated results with columns that match chart.x and chart.y.
- hist: return the raw numeric column (one row per entity) and set chart.type="hist" and chart.y to that column name.
  If user specifies bins, set chart.bins (integer). If they say "bins of 1000", choose bins heuristically
  (you can set bins to 20-50 if unsure) OR set bins to a reasonable number based on data size.

Database schema:
{schema_text}

Conversation (most recent last):
{ctx}

Previous SQL (if any):
{last_sql_block}

Current user message:
{question}

Return JSON with:
- sql: string
- explanation: short explanation
- result_kind: "table"|"scalar"|"empty"
- chart: object with "type": "none"|"bar"|"line"|"hist" and optional x/y/title/bins (only if it makes sense).
""".strip()


def _build_sql_repair_prompt(
    *,
    question: str,
    schema_text: str,
    chat_messages: Optional[List[Dict[str, Any]]],
    bad_sql: str,
    error_text: str,
    attempt: int,
    last_sql: Optional[str] = None,
) -> str:
    ctx = _format_chat_context(chat_messages or [], limit=10, max_chars=5000)
    last_sql_block = (last_sql or "").strip() or "None"

    return f"""
You previously generated a PostgreSQL query, but it failed at execution time.

Database schema:
{schema_text}

Conversation (most recent last):
{ctx}

Previous SQL (if any):
{last_sql_block}

User message:
{question}

Failed SQL:
{bad_sql}

PostgreSQL error:
{error_text}

Fix the SQL so it runs successfully and still answers the user's message.

Rules:
- Output MUST be valid PostgreSQL.
- Only SELECT or WITH ... SELECT is allowed. No INSERT/UPDATE/DELETE/DDL.
- Single statement only.
- Use explicit casts if needed.
- Prefer standard PostgreSQL functions.
- Keep it simple and correct.

Return JSON with:
- sql: string
- explanation: short explanation of what you changed
- result_kind: "table"|"scalar"|"empty"
- chart: object with "type": "none"|"bar"|"line"|"hist" and optional x/y/title/bins
Attempt: {attempt}
""".strip()


def execute_sql_with_repairs(
    *,
    vertex_logged: VertexLogged,
    pg_logged: PostgresLogged,
    question: str,
    schema_text: str,
    initial_out: dict,
    max_repairs: int,
    chat_messages: Optional[List[Dict[str, Any]]] = None,
    last_sql: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict, List[dict]]:
    repairs: List[dict] = []
    out = dict(initial_out or {})
    resp_schema = sql_gen_schema()

    for attempt in range(0, max_repairs + 1):
        sql = (out.get("sql") or "").strip()
        ok, why = is_sql_safe_readonly(sql)
        if not ok:
            raise RuntimeError(f"Generated SQL was rejected: {why}")

        try:
            df = pg_logged.run_select_to_df(sql)
            return df, out, repairs
        except Exception as e:
            if attempt >= max_repairs:
                raise

            error_text = str(e)
            prompt = _build_sql_repair_prompt(
                question=question,
                schema_text=schema_text,
                chat_messages=chat_messages,
                bad_sql=sql,
                error_text=error_text,
                attempt=attempt + 1,
                last_sql=last_sql,
            )

            out2 = vertex_logged.generate_json(
                phase="sql_repair",
                prompt=prompt,
                response_schema=resp_schema,
                temperature=0.0,
                max_output_tokens=1024,
                repair_attempts=1,
                token_expand_attempts=1,
                max_output_tokens_cap=2048,
                metadata={
                    "attempt": attempt + 1,
                    "prev_sql": sql[:5000],
                    "pg_error": error_text[:5000],
                },
            )

            repairs.append(
                {
                    "attempt": attempt + 1,
                    "prev_sql": sql,
                    "error": error_text,
                    "new_sql": (out2 or {}).get("sql") or "",
                }
            )
            out = dict(out2 or {})

    raise RuntimeError("Unexpected control flow in execute_sql_with_repairs")
