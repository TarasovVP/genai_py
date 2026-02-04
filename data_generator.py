from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
import random
import pandas as pd

from vertex_client import VertexGenAIClient


def _map_sql_type_to_json(sql_type: str) -> str:
    t = (sql_type or "").strip().lower()
    base = t.split("(")[0].strip()

    if base in ("int", "integer", "bigint", "smallint", "serial", "bigserial"):
        return "integer"
    if base in ("float", "double", "real", "numeric", "decimal"):
        return "number"
    if base in ("bool", "boolean"):
        return "boolean"
    return "string"


def build_table_response_schema(table_name: str, table_meta: Dict[str, Any]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required: List[str] = []

    for col_name, col_info in table_meta["columns"].items():
        sql_type = col_info.get("type_pg") or col_info.get("type") or col_info.get("type_raw") or ""
        json_type = _map_sql_type_to_json(sql_type)

        props[col_name] = {"type": json_type}

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
    dataset_prompt: str = "",
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

    dataset_prompt = (dataset_prompt or "").strip()
    dataset_section = ""
    if dataset_prompt:
        dataset_section = (
            "\n\nGlobal dataset instructions (apply across ALL tables):\n"
            f"{dataset_prompt}\n"
            "Rules for applying these instructions:\n"
            "- Follow them as long as they do NOT violate the schema constraints (types, NOT NULL, PK/FK).\n"
            "- If an instruction conflicts with schema constraints, prefer the schema.\n"
        )

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
- For foreign keys use only allowed values:{allowed_section}{dataset_section}

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

    rest = [t for t in tables.keys() if t not in order]
    order.extend(rest)
    return order


def _postprocess_primary_key(rows: List[Dict[str, Any]], table_meta: Dict[str, Any]) -> None:
    pk = table_meta.get("primary_key") or []
    if len(pk) != 1:
        return
    pk_col = pk[0]
    if not rows:
        return

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


def _safe_progress_call(
    cb: Optional[Callable[[int, int, str], None]],
    done: int,
    total: int,
    table_name: str,
) -> None:
    if cb is None:
        return
    try:
        cb(done, total, table_name)
    except Exception:
        return


def _estimate_cols_count(table_meta: Dict[str, Any]) -> int:
    cols = table_meta.get("columns") or {}
    return max(1, len(cols))


def _choose_batch_size_smart(expected_rows: int, max_output_tokens: int, table_meta: Dict[str, Any]) -> int:
    mot = int(max_output_tokens)
    cols = _estimate_cols_count(table_meta)

    if mot <= 512:
        cap = 20
    elif mot <= 1024:
        cap = 50
    elif mot <= 2048:
        cap = 100
    elif mot <= 4096:
        cap = 200
    else:
        cap = 400

    factor = max(1.0, cols / 10.0)
    batch = int(cap / factor)

    return max(10, min(batch, expected_rows))


def _min_tokens_for_batch(table_meta: Dict[str, Any], batch_rows: int) -> int:
    cols = _estimate_cols_count(table_meta)
    est = int(batch_rows * cols * 8)
    return max(800, min(est, 8192))


def _progress_table_label(
    table_name: str,
    collected: int,
    expected_total: int,
    request_count: int,
    cur_batch: int,
    batch_size: int,
) -> str:
    return f"{table_name} ({collected}/{expected_total} rows, batch {request_count}, +{cur_batch}, bs={batch_size})"


def generate_all_tables(
    vertex: VertexGenAIClient,
    ddl_schema: Dict[str, Any],
    rows_per_table: int,
    temperature: float,
    max_output_tokens: int,
    dataset_prompt: str = "",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    tables = ddl_schema["tables"]
    order = _build_dependency_order(ddl_schema)

    generated_rows: Dict[str, List[Dict[str, Any]]] = {}
    dfs: Dict[str, pd.DataFrame] = {}

    total_tables = len(order)
    expected_total = int(rows_per_table)

    max_batch_requests_per_table = 80

    for idx, table_name in enumerate(order, start=1):
        meta = tables[table_name]
        fks = meta.get("foreign_keys") or []

        fk_allowed_values: Dict[str, List[Any]] = {}
        for fk in fks:
            child_cols = fk.get("columns") or []
            parent = fk.get("ref_table")
            ref_cols = fk.get("ref_columns") or []

            if not child_cols or not parent:
                continue

            child_fk_col = child_cols[0]

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

        all_rows: List[Dict[str, Any]] = []

        effective_tokens_for_batching = min(max(int(max_output_tokens), 2048), 8192)
        batch_size = _choose_batch_size_smart(expected_total, effective_tokens_for_batching, meta)

        if expected_total >= 50:
            batch_size = max(batch_size, 50)

        request_count = 0
        last_len = -1
        consecutive_underfilled = 0

        _safe_progress_call(
            on_progress,
            idx - 1,
            total_tables,
            _progress_table_label(table_name, 0, expected_total, 0, 0, batch_size),
        )

        while len(all_rows) < expected_total:
            request_count += 1
            if request_count > max_batch_requests_per_table:
                raise RuntimeError(
                    f"Table '{table_name}': too many batch requests ({request_count}). "
                    f"Collected {len(all_rows)}/{expected_total} rows. "
                    f"Consider increasing max_output_tokens or decreasing rows_per_table."
                )

            remaining = expected_total - len(all_rows)
            cur_batch = min(batch_size, remaining)

            _safe_progress_call(
                on_progress,
                idx - 1,
                total_tables,
                _progress_table_label(table_name, len(all_rows), expected_total, request_count, cur_batch, batch_size),
            )

            response_schema = build_table_response_schema(table_name, meta)
            prompt = build_prompt_for_table(
                table_name=table_name,
                table_meta=meta,
                rows_count=cur_batch,
                fk_allowed_values=fk_allowed_values,
                dataset_prompt=dataset_prompt,
            )
            prompt += (
                "\n\nIMPORTANT:\n"
                f"- This is a batch request. You MUST return exactly {cur_batch} rows in this response.\n"
                "- Return ONLY JSON.\n"
                "- Do NOT stop early.\n"
                "- Ensure rows are diverse and not duplicates.\n"
            )

            req_tokens_min = _min_tokens_for_batch(meta, cur_batch)
            req_tokens = max(int(max_output_tokens), req_tokens_min)
            req_tokens = min(req_tokens, 8192)

            attempts = 2
            out: Dict[str, Any] = {}
            rows: List[Dict[str, Any]] = []

            for attempt in range(1, attempts + 1):
                try:
                    out = vertex.generate_json(
                        prompt=prompt,
                        response_schema=response_schema,
                        temperature=float(temperature) if attempt == 1 else min(float(temperature), 0.2),
                        max_output_tokens=int(req_tokens),
                        repair_attempts=1,
                        token_expand_attempts=2,
                        max_output_tokens_cap=8192,
                    )
                except Exception as e:
                    raise RuntimeError(f"Generation failed for table '{table_name}': {e}")

                candidate_rows = out.get("rows", []) if isinstance(out, dict) else []
                rows = candidate_rows if isinstance(candidate_rows, list) else []

                if len(rows) == cur_batch:
                    break

            if len(rows) == 0:
                raise RuntimeError(
                    f"Table '{table_name}': model returned 0 rows.\n"
                    f"Finish reason: {getattr(vertex, 'last_finish_reason', '')}\n"
                    f"Raw head:\n{getattr(vertex, 'last_raw', '')[:800]}"
                )

            if len(rows) < cur_batch:
                consecutive_underfilled += 1
            else:
                consecutive_underfilled = 0

            all_rows.extend(rows)

            if len(all_rows) == last_len:
                raise RuntimeError(
                    f"Table '{table_name}': no progress while collecting rows.\n"
                    f"Collected {len(all_rows)}/{expected_total}.\n"
                    f"Finish reason: {getattr(vertex, 'last_finish_reason', '')}\n"
                    f"Raw head:\n{getattr(vertex, 'last_raw', '')[:800]}"
                )
            last_len = len(all_rows)

            if consecutive_underfilled >= 2 and batch_size > 10:
                batch_size = max(10, batch_size // 2)
                consecutive_underfilled = 0

            _safe_progress_call(
                on_progress,
                idx - 1,
                total_tables,
                _progress_table_label(table_name, len(all_rows), expected_total, request_count, 0, batch_size),
            )

        if len(all_rows) > expected_total:
            all_rows = all_rows[:expected_total]

        _postprocess_primary_key(all_rows, meta)
        _postprocess_foreign_keys(all_rows, fk_allowed_values)

        df = pd.DataFrame(all_rows)
        dfs[table_name] = df
        generated_rows[table_name] = all_rows

        _safe_progress_call(
            on_progress,
            idx,
            total_tables,
            _progress_table_label(table_name, expected_total, expected_total, request_count, 0, batch_size),
        )

    return dfs