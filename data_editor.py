# data_editor.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import random
import pandas as pd


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
    return "string"


def _table_columns_props(table_meta: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    props: Dict[str, Any] = {}
    required: List[str] = []

    for col_name, col_info in (table_meta.get("columns") or {}).items():
        sql_type = col_info.get("type_pg") or col_info.get("type") or col_info.get("type_raw") or ""
        json_type = _map_sql_type_to_json(sql_type)
        props[col_name] = {"type": json_type}

        st = (sql_type or "").lower()
        if json_type == "string" and (st.startswith("date") or "timestamp" in st):
            props[col_name]["description"] = "ISO 8601 string"

        if col_info.get("nullable") is False:
            required.append(col_name)

    return props, required


# --------- patch schema / prompt ---------
def build_table_patch_schema(table_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    СУПЕР-ЛЁГКАЯ схема для Vertex, чтобы избежать ошибки:
    "schema produces a constraint that has too many states for serving"

    Ключевая идея:
    - НЕ вшиваем список колонок в JSON schema (это взрывает количество состояний).
    - Разрешаем свободные объекты в where/set/rows, а строгость обеспечиваем в apply_patch_to_df.
    """

    op_item_schema = {
        "type": "object",
        "properties": {
            # ожидаем: update_where | delete_where | add_rows (но не ограничиваем enum-ом ради простоты)
            "op": {"type": "string"},

            # where — свободный объект (мы потом валидируем "column"/equals/contains)
            "where": {
                "type": "object",
                "additionalProperties": True,
            },

            # set — свободный объект с произвольными ключами (мы потом отфильтруем по schema columns)
            "set": {
                "type": "object",
                "additionalProperties": True,
            },

            # rows — массив свободных объектов (мы потом выкинем лишние колонки и приведём типы)
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },

            # limit — просто integer без min/max, чтобы не усложнять constraints
            "limit": {"type": "integer"},
        },
        "required": ["op"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "ops": {
                "type": "array",
                "items": op_item_schema,
            },
        },
        "required": ["table", "ops"],
        "additionalProperties": False,
    }


def build_prompt_for_table_patch(
    table_name: str,
    table_meta: Dict[str, Any],
    user_instruction: str,
    sample_rows: List[Dict[str, Any]],
    fk_allowed_values: Dict[str, List[Any]],
    max_ops: int = 20,
) -> str:
    cols_lines = []
    for col_name, col_info in (table_meta.get("columns") or {}).items():
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

    return f"""
You are editing an existing synthetic dataset table.

Return ONLY valid JSON. No markdown. No commentary.

Table: {table_name}

Columns:
{chr(10).join(cols_lines)}

Primary key columns: {pk if pk else "none/unknown"}
Foreign keys:
{chr(10).join(fk_lines) if fk_lines else "none"}

Allowed values for FK columns:
{allowed_section}

User edit instruction:
{user_instruction}

You must output a PATCH (operations list). Do NOT output full table rows unless you are adding rows.

Patch rules:
- Keep ops <= {max_ops}.
- Use only these op types: update_where, delete_where, add_rows.
- For update_where/delete_where, "where" uses:
  - column: column name
  - equals: exact match
  - contains: substring match (string columns only)
- For update_where, do NOT modify primary key columns.
- Respect NOT NULL and column types.
- Respect FK constraints: for FK columns, use only allowed values.

Here is a small sample of current rows (for context, not exhaustive):
{sample_rows}

Output format:
{{
  "table": "{table_name}",
  "ops": [
    {{
      "op": "update_where",
      "where": {{ "column": "status", "equals": "inactive" }},
      "set": {{ "status": "active" }},
      "limit": 1000
    }},
    {{
      "op": "add_rows",
      "rows": [ ... ]
    }}
  ]
}}
""".strip()


# --------- apply patch to dataframe ---------
def apply_patch_to_df(
    df: pd.DataFrame,
    patch: Dict[str, Any],
    table_meta: Dict[str, Any],
    fk_allowed_values: Dict[str, List[Any]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Возвращает (new_df, warnings).
    Делает best-effort приведение типов и соблюдение PK/FK.
    """
    warnings: List[str] = []

    if not isinstance(patch, dict) or "ops" not in patch:
        raise ValueError("Patch must be an object with 'ops'.")

    ops = patch.get("ops") or []
    if not isinstance(ops, list):
        raise ValueError("'ops' must be a list.")

    meta_cols: Dict[str, Any] = table_meta.get("columns") or {}
    col_names = list(meta_cols.keys())
    pk_cols = table_meta.get("primary_key") or []
    pk_cols = list(pk_cols)

    # Ensure df has all schema columns (if model added missing columns earlier)
    for c in col_names:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.copy()

    def _coerce_value(col: str, value: Any) -> Any:
        """Try to coerce to schema type."""
        info = meta_cols.get(col, {}) or {}
        sql_type = (info.get("type_pg") or info.get("type") or info.get("type_raw") or "").lower()
        jt = _map_sql_type_to_json(sql_type)

        if value is None:
            return None

        try:
            if jt == "integer":
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, (int, float)) and not pd.isna(value):
                    return int(value)
                return int(str(value).strip())
            if jt == "number":
                if isinstance(value, bool):
                    return float(int(value))
                if isinstance(value, (int, float)) and not pd.isna(value):
                    return float(value)
                return float(str(value).strip().replace(",", "."))
            if jt == "boolean":
                if isinstance(value, bool):
                    return value
                s = str(value).strip().lower()
                if s in ("true", "1", "yes", "y"):
                    return True
                if s in ("false", "0", "no", "n"):
                    return False
                return bool(value)
            # string
            return str(value)
        except Exception:
            warnings.append(f"Could not coerce value for column '{col}': {value!r}. Keeping raw.")
            return value

    def _build_mask(where: Dict[str, Any]) -> pd.Series:
        if not isinstance(where, dict):
            raise ValueError("'where' must be an object.")
        col = where.get("column")
        if not col or col not in df.columns:
            raise ValueError(f"Unknown where.column: {col!r}")

        series = df[col]

        contains = where.get("contains")
        equals = where.get("equals")

        if contains is not None and str(contains).strip() != "":
            needle = str(contains)
            return series.astype(str).str.contains(needle, case=False, na=False)

        if equals is None:
            # only column provided => treat as "not null"
            return series.notna()

        eq_str = str(equals)

        if pd.api.types.is_numeric_dtype(series):
            try:
                eq_num = float(eq_str)
                return series.astype(float) == eq_num
            except Exception:
                return series.astype(str) == eq_str

        if pd.api.types.is_bool_dtype(series):
            s = eq_str.strip().lower()
            if s in ("true", "1", "yes", "y"):
                return series.astype(bool) == True
            if s in ("false", "0", "no", "n"):
                return series.astype(bool) == False
            return series.astype(str) == eq_str

        return series.astype(str) == eq_str

    def _enforce_not_null(row: Dict[str, Any]) -> None:
        for c, info in meta_cols.items():
            if info.get("nullable") is False:
                if c not in row or row[c] is None:
                    raise ValueError(f"NOT NULL column '{c}' is missing or null in added row.")

    def _fix_fk_in_row(row: Dict[str, Any]) -> None:
        for fk_col, allowed in (fk_allowed_values or {}).items():
            if not allowed:
                continue
            if fk_col in row:
                if row[fk_col] not in allowed:
                    row[fk_col] = random.choice(allowed)

    def _assign_pk_if_missing(row: Dict[str, Any]) -> None:
        if len(pk_cols) != 1:
            return
        pk = pk_cols[0]
        if pk not in df.columns:
            return

        info = meta_cols.get(pk, {}) or {}
        sql_type = (info.get("type_pg") or "").lower()
        if not any(k in sql_type for k in ["int", "serial", "bigint", "smallint"]):
            return

        if pk not in row or row[pk] in (None, "", pd.NA):
            try:
                cur_max = pd.to_numeric(df[pk], errors="coerce").max()
                cur_max = 0 if pd.isna(cur_max) else int(cur_max)
                row[pk] = cur_max + 1
            except Exception:
                row[pk] = None

    # Apply operations
    for i, op in enumerate(ops, start=1):
        if not isinstance(op, dict):
            warnings.append(f"Op #{i} is not an object. Skipped.")
            continue

        op_type = op.get("op")
        if op_type == "update_where":
            where = op.get("where") or {}
            set_map = op.get("set") or {}
            limit = op.get("limit")

            if not isinstance(set_map, dict) or not set_map:
                warnings.append(f"update_where op #{i} has empty 'set'. Skipped.")
                continue

            # Disallow PK modifications
            for pk in pk_cols:
                if pk in set_map:
                    warnings.append(f"update_where op #{i}: attempt to modify PK '{pk}' is ignored.")
                    set_map.pop(pk, None)

            if not set_map:
                warnings.append(f"update_where op #{i}: nothing to set after PK filtering. Skipped.")
                continue

            mask = _build_mask(where)
            idxs = df.index[mask].tolist()

            if limit is not None:
                try:
                    lim = int(limit)
                    idxs = idxs[: max(0, lim)]
                except Exception:
                    pass

            if not idxs:
                warnings.append(f"update_where op #{i}: matched 0 rows.")
                continue

            for col, val in set_map.items():
                if col not in df.columns:
                    warnings.append(f"update_where op #{i}: unknown column '{col}'. Ignored.")
                    continue

                coerced = _coerce_value(col, val)
                df.loc[idxs, col] = coerced

            # FK columns: force allowed
            for fk_col, allowed in (fk_allowed_values or {}).items():
                if not allowed or fk_col not in df.columns:
                    continue
                bad_mask = df.index.isin(idxs) & (~df[fk_col].isin(allowed))
                bad_idxs = df.index[bad_mask].tolist()
                if bad_idxs:
                    for bi in bad_idxs:
                        df.at[bi, fk_col] = random.choice(allowed)

        elif op_type == "delete_where":
            where = op.get("where") or {}
            limit = op.get("limit")

            mask = _build_mask(where)
            idxs = df.index[mask].tolist()

            if limit is not None:
                try:
                    lim = int(limit)
                    idxs = idxs[: max(0, lim)]
                except Exception:
                    pass

            if not idxs:
                warnings.append(f"delete_where op #{i}: matched 0 rows.")
                continue

            df = df.drop(index=idxs)

        elif op_type == "add_rows":
            rows = op.get("rows") or []
            if not isinstance(rows, list) or not rows:
                warnings.append(f"add_rows op #{i} has empty 'rows'. Skipped.")
                continue

            new_rows: List[Dict[str, Any]] = []
            for r in rows:
                if not isinstance(r, dict):
                    warnings.append(f"add_rows op #{i}: row is not an object. Skipped.")
                    continue

                # Ensure no extra cols
                clean = {k: r.get(k) for k in col_names if k in r}

                # Coerce
                for col in list(clean.keys()):
                    clean[col] = _coerce_value(col, clean[col])

                _assign_pk_if_missing(clean)
                _fix_fk_in_row(clean)
                _enforce_not_null(clean)

                new_rows.append(clean)

            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        else:
            warnings.append(f"Unknown op type '{op_type}' in op #{i}. Skipped.")

    # Ensure column order = schema order (plus any extra columns at the end)
    ordered = [c for c in col_names if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    return df, warnings


def fix_fk_for_table(
    df_child: pd.DataFrame,
    child_table_meta: Dict[str, Any],
    parent_tables: Dict[str, pd.DataFrame],
    schema_tables: Dict[str, Any],
) -> Tuple[pd.DataFrame, int]:
    """
    Пробегается по foreign_keys child таблицы и заменяет значения,
    которых нет в parent таблице, на случайные допустимые.
    Возвращает (new_df, fixes_count).
    """
    fixes = 0
    fks = child_table_meta.get("foreign_keys") or []
    if not fks:
        return df_child, fixes

    df_child = df_child.copy()

    for fk in fks:
        child_cols = fk.get("columns") or []
        parent = fk.get("ref_table")
        ref_cols = fk.get("ref_columns") or []

        if not child_cols or not parent:
            continue

        child_fk_col = child_cols[0]
        if child_fk_col not in df_child.columns:
            continue

        if parent not in parent_tables:
            continue

        df_parent = parent_tables[parent]
        if ref_cols:
            parent_ref_col = ref_cols[0]
        else:
            parent_pk = (schema_tables.get(parent, {}) or {}).get("primary_key") or []
            parent_ref_col = parent_pk[0] if parent_pk else None

        if not parent_ref_col or parent_ref_col not in df_parent.columns:
            continue

        allowed = df_parent[parent_ref_col].dropna().tolist()
        if not allowed:
            continue

        bad_mask = ~df_child[child_fk_col].isin(allowed)
        bad_idxs = df_child.index[bad_mask].tolist()
        for bi in bad_idxs:
            df_child.at[bi, child_fk_col] = random.choice(allowed)
            fixes += 1

    return df_child, fixes