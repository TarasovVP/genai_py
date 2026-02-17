from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from domain.sql_guard import is_sql_safe_readonly


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


def build_nl2sql_prompt(question: str, schema_text: str) -> str:
    return f"""
You are a data analyst. Convert the user's natural-language question into a single PostgreSQL SELECT query.

Rules:
- Output MUST be valid PostgreSQL.
- Only SELECT or WITH ... SELECT is allowed. No INSERT/UPDATE/DELETE/DDL.
- Do not use functions that require unusual extensions. Prefer standard PostgreSQL functions.
- Use explicit casts when needed (e.g., ::numeric, ::int).
- Use double quotes for identifiers ONLY if needed; otherwise prefer unquoted lowercase identifiers.
- If user asks for "top", use ORDER BY + LIMIT.
- If multiple tables needed, use correct JOINs based on schema.
- Prefer simple, readable SQL.

If the user asks for a chart:
- For line/bar: return aggregated results with columns that match chart.x and chart.y.
- For histogram: prefer returning the raw numeric column (one row per entity) and set chart.type="hist" and chart.y to that column name; include chart.bins if user specifies.

Database schema:
{schema_text}

User question:
{question}

Return JSON with:
- sql: string
- explanation: short explanation
- result_kind: "table"|"scalar"|"empty"
- chart: object with "type": "none"|"bar"|"line"|"hist" and optional x/y/title/bins (only if it makes sense).
""".strip()


def _build_sql_repair_prompt(
    question: str,
    schema_text: str,
    bad_sql: str,
    error_text: str,
    attempt: int,
) -> str:
    return f"""
You previously generated a PostgreSQL query, but it failed at execution time.

User question:
{question}

Database schema:
{schema_text}

Failed SQL:
{bad_sql}

PostgreSQL error:
{error_text}

Fix the SQL so it runs successfully and still answers the user's question.

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
    vertex_logged: Any,
    pg_logged: Any,
    question: str,
    schema_text: str,
    initial_out: dict,
    max_repairs: int,
) -> Tuple[pd.DataFrame, dict, List[dict]]:
    """
    Важно: никакого st.session_state тут нет.
    vertex_logged: VertexLogged (generate_json)
    pg_logged: PostgresLogged (run_select_to_df)
    """
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
                bad_sql=sql,
                error_text=error_text,
                attempt=attempt + 1,
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
