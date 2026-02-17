from __future__ import annotations


def schema_text_for_prompt(schema: dict) -> str:
    if not schema:
        return "Schema is empty."
    tables = (schema.get("tables") or {})
    lines = []
    for tname, tmeta in tables.items():
        cols = (tmeta.get("columns") or {})
        pk = tmeta.get("primary_key") or []
        fks = tmeta.get("foreign_keys") or []

        lines.append(f"TABLE {tname}:")
        for cname, cinfo in cols.items():
            ctype = cinfo.get("type_pg") or cinfo.get("type") or cinfo.get("type_raw") or "UNKNOWN"
            nn = "" if cinfo.get("nullable", True) else " NOT NULL"
            lines.append(f"  - {cname}: {ctype}{nn}")

        if pk:
            lines.append(f"  PK: ({', '.join(pk)})")

        for fk in fks:
            ccols = fk.get("columns") or []
            rt = fk.get("ref_table")
            rcols = fk.get("ref_columns") or []
            if ccols and rt:
                if rcols:
                    lines.append(f"  FK: ({', '.join(ccols)}) -> {rt}({', '.join(rcols)})")
                else:
                    lines.append(f"  FK: ({', '.join(ccols)}) -> {rt}")

        lines.append("")

    return "\n".join(lines).strip()
