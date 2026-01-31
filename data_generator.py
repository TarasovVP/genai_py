# data_generator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import random
import pandas as pd

from vertex_client import VertexGenAIClient


# --------- type mapping ---------
def _map_sql_type_to_json(sql_type: str) -> str:
    t = (sql_type or "").strip().lower()
    base = t.split("(")[0].strip()

    if base in ("int", "integer", "bigint", "smallint", "serial", "bigserial"):
        return "integer"
    if base in ("float", "double", "real", "numeric", "decimal"):
        return "number"
    if base in ("bool", "boolean"):
        return "boolean"
    # date/time/uuid пусть будут string
    return "string"


# --------- schema/prompt builders ---------
def build_table_response_schema(table_name: str, table_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    table_meta: schema["tables"][table_name] из ddl_parser
      {
        "columns": { "col": { "type_pg": "...", "nullable": bool, ... }, ... },
        "primary_key": [...],
        "foreign_keys": [...],
        ...
      }
    """
    props: Dict[str, Any] = {}
    required: List[str] = []

    for col_name, col_info in table_meta["columns"].items():
        sql_type = col_info.get("type_pg") or col_info.get("type") or col_info.get("type_raw") or ""
        json_type = _map_sql_type_to_json(sql_type)

        props[col_name] = {"type": json_type}

        # optional hints
        st = (sql_type or "").lower()
        if json_type == "string" and (st.startswith("date") or "timestamp" in st):
            props[col_name]["description"] = "ISO 8601 string"

        if col_info.get("nullable") is False:
            required.append(col_name)

    return {
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": props,
                    "additionalProperties": False,
                    **({"required": required} if required else {}),
                },
            },
        },
        "required": ["table", "rows"],
        "additionalProperties": False,
    }


def build_prompt_for_table(
    table_name: str,
    table_meta: Dict[str, Any],
    rows_count: int,
    fk_allowed_values: Dict[str, List[Any]],
) -> str:
    cols_lines = []
    for col_name, col_info in table_meta["columns"].items():
        t = col_info.get("type_pg") or col_info.get("type") or col_info.get("type_raw") or ""
        nn = "NOT NULL" if col_info.get("nullable") is False else ""
        cols_lines.append(f"- {col_name}: {t} {nn}".rstrip())

    pk = table_meta.get("primary_key") or []
    fks = table_meta.get("foreign_keys") or []

    fk_lines = []
    for fk in fks:
        child_cols = fk.get("columns") or []
        ref_table = fk.get("ref_table")
        ref_cols = fk.get("ref_columns") or []
        fk_lines.append(f"- {child_cols} -> {ref_table}.{ref_cols}")

    allowed_text = ""
    for col, values in (fk_allowed_values or {}).items():
        sample = values[:50]
        allowed_text += f"\n- {col} MUST be one of: {sample}"

    allowed_section = allowed_text if allowed_text else "\n- none"

    # Важно: максимально явно просим ONLY JSON без markdown/объяснений.
    return f"""
You generate synthetic data for ONE SQL table.

Return ONLY valid JSON. Do not wrap in markdown. Do not add any commentary.

Table: {table_name}

Columns:
{chr(10).join(cols_lines)}

Primary key columns: {pk if pk else "none/unknown"}
Foreign keys:
{chr(10).join(fk_lines) if fk_lines else "none"}

Rules:
- Output MUST be valid JSON and MUST match the provided JSON schema exactly.
- Generate exactly {rows_count} rows.
- Use realistic values.
- Respect NOT NULL.
- If there is a primary key, ensure it is unique.
- For foreign keys use only allowed values:{allowed_section}

Output format:
{{ "table": "{table_name}", "rows": [ ... ] }}
""".strip()


def _build_repair_prompt(table_name: str, bad_text: str) -> str:
    head = bad_text[:2500]
    return f"""
The previous output for table "{table_name}" was NOT valid JSON.

Fix it and return ONLY valid JSON (no markdown, no comments), preserving the same structure:
{{ "table": "{table_name}", "rows": [ ... ] }}

Here is the invalid output (truncated):
{head}
""".strip()


# --------- dependency order ---------
def _build_dependency_order(schema: Dict[str, Any]) -> List[str]:
    tables = schema["tables"]
    deps = {t: set() for t in tables.keys()}

    for t, meta in tables.items():
        for fk in meta.get("foreign_keys", []) or []:
            parent = fk.get("ref_table")
            if parent and parent in tables:
                deps[t].add(parent)

    order: List[str] = []
    ready = [t for t, d in deps.items() if not d]

    while ready:
        cur = ready.pop()
        order.append(cur)
        for t in deps:
            if cur in deps[t]:
                deps[t].remove(cur)
                if not deps[t]:
                    ready.append(t)

    # если есть цикл/остаток
    rest = [t for t in tables.keys() if t not in order]
    order.extend(rest)
    return order


# --------- postprocess helpers (чтобы FK/PK точно совпали) ---------
def _postprocess_primary_key(rows: List[Dict[str, Any]], table_meta: Dict[str, Any]) -> None:
    pk = table_meta.get("primary_key") or []
    if len(pk) != 1:
        return
    pk_col = pk[0]
    if not rows:
        return

    # если PK выглядит как integer (по ddl) — сделаем 1..N
    col_info = table_meta["columns"].get(pk_col, {})
    sql_type = (col_info.get("type_pg") or "").lower()
    if any(k in sql_type for k in ["int", "serial", "bigint", "smallint"]):
        for i, r in enumerate(rows, start=1):
            r[pk_col] = i


def _postprocess_foreign_keys(
    rows: List[Dict[str, Any]],
    fk_allowed_values: Dict[str, List[Any]],
) -> None:
    if not rows:
        return
    for fk_col, allowed in (fk_allowed_values or {}).items():
        if not allowed:
            continue
        for r in rows:
            if fk_col not in r or r[fk_col] not in allowed:
                r[fk_col] = random.choice(allowed)


# --------- main entry ---------
def generate_all_tables(
    vertex: VertexGenAIClient,
    ddl_schema: Dict[str, Any],
    rows_per_table: int,
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, pd.DataFrame]:
    tables = ddl_schema["tables"]
    order = _build_dependency_order(ddl_schema)

    generated_rows: Dict[str, List[Dict[str, Any]]] = {}
    dfs: Dict[str, pd.DataFrame] = {}

    for table_name in order:
        meta = tables[table_name]
        fks = meta.get("foreign_keys") or []

        # собрать allowed значения для FK из уже сгенерированных родителей
        fk_allowed_values: Dict[str, List[Any]] = {}
        for fk in fks:
            child_cols = fk.get("columns") or []
            parent = fk.get("ref_table")
            ref_cols = fk.get("ref_columns") or []

            if not child_cols or not parent:
                continue

            child_fk_col = child_cols[0]

            # ref_col: если не указан — попробуем взять PK родителя
            if ref_cols:
                parent_ref_col = ref_cols[0]
            else:
                parent_pk = (tables.get(parent, {}) or {}).get("primary_key") or []
                parent_ref_col = parent_pk[0] if parent_pk else None

            if not parent_ref_col:
                continue

            parent_rows = generated_rows.get(parent, [])
            allowed = [r.get(parent_ref_col) for r in parent_rows if parent_ref_col in r]
            fk_allowed_values[child_fk_col] = [v for v in allowed if v is not None]

        response_schema = build_table_response_schema(table_name, meta)
        prompt = build_prompt_for_table(table_name, meta, rows_per_table, fk_allowed_values)

        # 1) основной вызов
        try:
            out = vertex.generate_json(
                prompt=prompt,
                response_schema=response_schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as e:
            # чтобы Streamlit показывал понятную ошибку + можно было быстро понять table_name
            raise RuntimeError(f"Generation failed for table '{table_name}': {e}")

        # 2) иногда модель всё равно может вернуть поломанную структуру (редко, но бывает):
        rows = out.get("rows", []) if isinstance(out, dict) else []
        if not isinstance(rows, list):
            rows = []

        # 3) постпроцессинг для надежности
        _postprocess_primary_key(rows, meta)
        _postprocess_foreign_keys(rows, fk_allowed_values)

        df = pd.DataFrame(rows)
        dfs[table_name] = df
        generated_rows[table_name] = rows

    return dfs