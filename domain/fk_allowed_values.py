from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _schema_tables(schema: dict | None) -> dict:
    return ((schema or {}).get("tables") or {}) if isinstance(schema, dict) else {}


def get_table_meta(schema: dict | None, table_name: str) -> dict:
    return (_schema_tables(schema).get(table_name) or {}) if table_name else {}


def compute_fk_allowed_values_for_table(
    schema: dict | None,
    tables: Dict[str, pd.DataFrame],
    table_name: str,
) -> Dict[str, List[Any]]:
    schema_tables = _schema_tables(schema)
    meta = schema_tables.get(table_name, {}) or {}
    fks = meta.get("foreign_keys") or []
    allowed: Dict[str, List[Any]] = {}

    for fk in fks:
        child_cols = fk.get("columns") or []
        parent = fk.get("ref_table")
        ref_cols = fk.get("ref_columns") or []

        if not child_cols or not parent:
            continue

        child_fk_col = child_cols[0]

        if parent not in tables:
            continue

        df_parent = tables[parent]

        if ref_cols:
            parent_ref_col = ref_cols[0]
        else:
            parent_pk = (schema_tables.get(parent, {}) or {}).get("primary_key") or []
            parent_ref_col = parent_pk[0] if parent_pk else None

        if not parent_ref_col or parent_ref_col not in df_parent.columns:
            continue

        vals = df_parent[parent_ref_col].dropna().tolist()
        allowed[child_fk_col] = vals

    return allowed
