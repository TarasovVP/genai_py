from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _schema_tables(schema: dict | None) -> dict:
    return ((schema or {}).get("tables") or {}) if isinstance(schema, dict) else {}


def _schema_allowed_for_table(schema: dict | None, table_name: str) -> Dict[str, List[Any]]:
    tables = _schema_tables(schema)
    meta = tables.get(table_name, {}) or {}
    allowed = meta.get("allowed_values") or {}
    if isinstance(allowed, dict):
        return allowed
    return {}


def normalize_df_to_allowed_values(schema: dict | None, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    allowed_map = _schema_allowed_for_table(schema, table_name)
    if not allowed_map:
        return df

    out = df.copy()

    for col, allowed in allowed_map.items():
        if not allowed or col not in out.columns:
            continue

        canon = {str(v).strip().lower(): v for v in allowed if v is not None}
        default_val = allowed[0] if len(allowed) > 0 else None

        def coerce(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return x
            s = str(x).strip()
            if s == "":
                return x
            key = s.lower()
            if key in canon:
                return canon[key]
            return default_val

        out[col] = out[col].apply(coerce)

    return out


def normalize_all_tables_to_allowed_values(
    schema: dict | None,
    tables: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    if not tables:
        return tables

    fixed: Dict[str, pd.DataFrame] = {}
    for tname, df in tables.items():
        fixed[tname] = normalize_df_to_allowed_values(schema, tname, df)
    return fixed
