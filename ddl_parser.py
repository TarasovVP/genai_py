import re
from typing import Dict, Any, List, Tuple, Optional


_CONSTRAINT_KEYWORDS = {
    "not", "null", "primary", "references", "unique", "check", "default",
    "constraint", "collate", "generated", "identity", "auto_increment", "auto"
}

_TABLE_CONSTRAINT_PREFIXES = (
    "constraint", "primary key", "foreign key", "unique", "check"
)


def _is_quoted_ident(s: str) -> bool:
    s = s.strip()
    return (
        (s.startswith('"') and s.endswith('"') and len(s) >= 2) or
        (s.startswith("`") and s.endswith("`") and len(s) >= 2) or
        (s.startswith("[") and s.endswith("]") and len(s) >= 2)
    )


def _strip_quotes(identifier: str) -> str:
    identifier = identifier.strip()
    if _is_quoted_ident(identifier):
        return identifier[1:-1]
    return identifier


def _normalize_ident(identifier: str) -> str:
    """
    Postgres behavior:
    - unquoted identifiers are folded to lower-case
    - quoted identifiers preserve case
    We emulate this so that generated table/column names match actual DDL execution.
    """
    raw = (identifier or "").strip()
    stripped = _strip_quotes(raw)
    if _is_quoted_ident(raw):
        return stripped
    return stripped.lower()


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _remove_sql_comments(ddl: str) -> str:
    out: List[str] = []
    i = 0
    in_single = False
    in_double = False

    while i < len(ddl):
        ch = ddl[i]
        nxt = ddl[i + 1] if i + 1 < len(ddl) else ""

        if ch == "'" and not in_double:
            if in_single and nxt == "'":
                out.append("''")
                i += 2
                continue
            in_single = not in_single
            out.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            if ch == "-" and nxt == "-":
                i += 2
                while i < len(ddl) and ddl[i] not in ("\n", "\r"):
                    i += 1
                continue

            if ch == "/" and nxt == "*":
                i += 2
                while i + 1 < len(ddl) and not (ddl[i] == "*" and ddl[i + 1] == "/"):
                    i += 1
                i += 2
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_single = False
    in_double = False

    i = 0
    while i < len(s):
        ch = s[i]

        if ch == "'" and not in_double:
            if in_single and i + 1 < len(s) and s[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_create_table_blocks(ddl: str) -> List[Tuple[str, str]]:
    text = ddl
    lower = ddl.lower()

    results: List[Tuple[str, str]] = []

    idx = 0
    while True:
        m = re.search(r"\bcreate\s+table\b", lower[idx:])
        if not m:
            break
        start = idx + m.start()

        after = text[start:]
        mname = re.search(
            r"create\s+table\s+(if\s+not\s+exists\s+)?(?P<name>(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)(\.(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))?)",
            after,
            flags=re.IGNORECASE
        )
        if not mname:
            idx = start + 10
            continue

        raw_name = mname.group("name")
        name_parts = [p for p in raw_name.split(".")]
        table_name = _normalize_ident(name_parts[-1])

        pos = start + mname.end()
        while pos < len(text) and text[pos] != "(":
            pos += 1
        if pos >= len(text) or text[pos] != "(":
            idx = start + 10
            continue

        open_pos = pos
        depth = 0
        in_single = False
        in_double = False
        i = open_pos
        while i < len(text):
            ch = text[i]

            if ch == "'" and not in_double:
                if in_single and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double

            if not in_single and not in_double:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_pos = i
                        body = text[open_pos + 1:close_pos]
                        results.append((table_name, body))
                        idx = close_pos + 1
                        break
            i += 1
        else:
            idx = start + 10

    return results


def _to_postgres_type(type_raw: str) -> Dict[str, Any]:
    t = _normalize_ws(type_raw)
    tl = t.lower()

    if re.fullmatch(r"(int|integer)", tl):
        return {"type_pg": "INTEGER"}

    if tl == "bigint":
        return {"type_pg": "BIGINT"}

    m = re.fullmatch(r"varchar\s*\(\s*(\d+)\s*\)", tl, flags=re.IGNORECASE)
    if m:
        return {"type_pg": f"VARCHAR({m.group(1)})"}

    m = re.fullmatch(r"char\s*\(\s*(\d+)\s*\)", tl, flags=re.IGNORECASE)
    if m:
        return {"type_pg": f"CHAR({m.group(1)})"}

    if tl == "text":
        return {"type_pg": "TEXT"}

    if tl in ("bool", "boolean"):
        return {"type_pg": "BOOLEAN"}

    if tl == "date":
        return {"type_pg": "DATE"}

    if tl == "datetime":
        return {"type_pg": "TIMESTAMP"}

    if tl.startswith("timestamp"):
        return {"type_pg": "TIMESTAMP"}

    m = re.fullmatch(r"decimal\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", tl, flags=re.IGNORECASE)
    if m:
        return {"type_pg": f"NUMERIC({m.group(1)}, {m.group(2)})"}

    m = re.fullmatch(r"numeric\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", tl, flags=re.IGNORECASE)
    if m:
        return {"type_pg": f"NUMERIC({m.group(1)}, {m.group(2)})"}

    m = re.fullmatch(r"enum\s*\((.+)\)", t, flags=re.IGNORECASE)
    if m:
        inside = m.group(1).strip()
        vals = []
        buf = []
        in_single = False
        i = 0
        while i < len(inside):
            ch = inside[i]
            if ch == "'" and (i + 1 < len(inside) and inside[i + 1] == "'"):
                buf.append("'")
                i += 2
                continue
            if ch == "'":
                in_single = not in_single
                i += 1
                continue
            if ch == "," and not in_single:
                v = "".join(buf).strip()
                if v:
                    vals.append(v)
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            vals.append(tail)

        enum_values = [v.strip().strip("'").strip('"') for v in vals if v.strip()]
        return {
            "type_pg": "TEXT",
            "enum_values": enum_values,
            "notes": "ENUM converted to TEXT; consider CREATE TYPE + enum in Postgres"
        }

    return {"type_pg": t.upper(), "notes": "Unmapped type; kept as-is"}


def _extract_check_expressions(s: str) -> List[str]:
    """
    Extract CHECK(...) expressions from a column definition tail.
    Returns list of raw expressions inside CHECK( ... ).
    Supports multiple CHECKs in one line.
    """
    exprs: List[str] = []
    i = 0
    low = s.lower()
    while True:
        m = re.search(r"\bcheck\s*\(", low[i:])
        if not m:
            break
        start = i + m.end()
        depth = 1
        in_single = False
        in_double = False
        j = start
        while j < len(s):
            ch = s[j]
            if ch == "'" and not in_double:
                if in_single and j + 1 < len(s) and s[j + 1] == "'":
                    j += 2
                    continue
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double

            if not in_single and not in_double:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        expr = s[start:j].strip()
                        if expr:
                            exprs.append(_normalize_ws(expr))
                        i = j + 1
                        break
            j += 1
        else:
            break
    return exprs


def _parse_check_in_list(expr: str) -> Optional[Dict[str, Any]]:
    """
    Parse very common pattern:
      <col> IN ('a','b','c')
    Returns {"column": "<col>", "allowed": [...] } or None
    """
    if not expr:
        return None

    m = re.match(r'^(?P<col>"[^"]+"|`[^`]+`|\[[^\]]+\]|\w+)\s+in\s*\((?P<vals>.+)\)\s*$', expr, flags=re.IGNORECASE)
    if not m:
        return None

    col = _normalize_ident(m.group("col"))
    inside = m.group("vals").strip()

    vals = []
    buf = []
    in_single = False
    in_double = False
    i = 0
    while i < len(inside):
        ch = inside[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(inside) and inside[i + 1] == "'":
                buf.append("'")
                i += 2
                continue
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if ch == "," and not in_single and not in_double:
            v = "".join(buf).strip()
            if v:
                vals.append(v)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        vals.append(tail)

    allowed = [v.strip().strip("'").strip('"') for v in vals if v.strip()]
    return {"column": col, "allowed": allowed}


def _parse_column_def(item: str) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]], Optional[List[str]]]:
    s = _normalize_ws(item)

    s = re.sub(r"^constraint\s+(\w+|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])\s+", "", s, flags=re.IGNORECASE)

    mcol = re.match(r'^(?P<col>"[^"]+"|`[^`]+`|\[[^\]]+\]|\w+)\s+(?P<rest>.+)$', s)
    if not mcol:
        raise ValueError(f"Cannot parse column definition: {item}")

    col_raw = mcol.group("col")
    col_name = _normalize_ident(col_raw)
    rest = mcol.group("rest")

    auto_increment = bool(re.search(r"\bauto_increment\b", rest, flags=re.IGNORECASE))

    tokens = rest.split(" ")
    type_tokens: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        tl = t.lower()
        if tl in _CONSTRAINT_KEYWORDS:
            break
        type_tokens.append(t)
        i += 1
    col_type_raw = " ".join(type_tokens).strip()

    constraints_tail = " ".join(tokens[i:]).strip()
    constraints_low = constraints_tail.lower()

    nullable = True
    if "not null" in constraints_low:
        nullable = False

    default_expr = None
    mdef = re.search(
        r"\bdefault\b\s+(.+?)(?=(\bnot\b|\bnull\b|\bprimary\b|\breferences\b|\bunique\b|\bcheck\b|$))",
        constraints_tail,
        flags=re.IGNORECASE
    )
    if mdef:
        default_expr = _normalize_ws(mdef.group(1))

    is_pk = bool(re.search(r"\bprimary\s+key\b", constraints_tail, flags=re.IGNORECASE))
    is_unique = bool(re.search(r"\bunique\b", constraints_tail, flags=re.IGNORECASE))

    check_exprs = _extract_check_expressions(constraints_tail)
    parsed_checks: List[Dict[str, Any]] = []
    for ce in check_exprs:
        chk: Dict[str, Any] = {"expression": ce}
        in_list = _parse_check_in_list(ce)
        if in_list:
            chk["in_list"] = in_list
        parsed_checks.append(chk)

    fk = None
    mref = re.search(
        r"\breferences\b\s+(?P<table>(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)(\.(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))?)\s*(\((?P<cols>[^)]+)\))?",
        constraints_tail,
        flags=re.IGNORECASE
    )
    if mref:
        ref_raw = mref.group("table")
        ref_table = _normalize_ident(ref_raw.split(".")[-1])
        ref_cols = []
        if mref.group("cols"):
            ref_cols = [_normalize_ident(x.strip()) for x in mref.group("cols").split(",")]

        on_delete = None
        on_update = None
        mod = constraints_tail[mref.end():]
        mdel = re.search(
            r"\bon\s+delete\b\s+(cascade|restrict|set\s+null|set\s+default|no\s+action)",
            mod, flags=re.IGNORECASE
        )
        mupd = re.search(
            r"\bon\s+update\b\s+(cascade|restrict|set\s+null|set\s+default|no\s+action)",
            mod, flags=re.IGNORECASE
        )
        if mdel:
            on_delete = _normalize_ws(mdel.group(1)).upper()
        if mupd:
            on_update = _normalize_ws(mupd.group(1)).upper()

        fk = {
            "columns": [col_name],
            "ref_table": ref_table,
            "ref_columns": ref_cols,
            "on_delete": on_delete,
            "on_update": on_update
        }

    pg = _to_postgres_type(col_type_raw)

    col_info: Dict[str, Any] = {
        "type_raw": col_type_raw,
        "type": pg["type_pg"],
        "type_pg": pg["type_pg"],
        "enum_values": pg.get("enum_values"),
        "type_notes": pg.get("notes"),
        "nullable": nullable,
        "default": default_expr,
        "primary_key": is_pk,
        "unique": is_unique,
        "auto_increment": auto_increment,
        "checks": parsed_checks,
    }

    if auto_increment and col_info["type_pg"] in ("INTEGER", "BIGINT"):
        col_info["identity"] = True

    pk_cols = [col_name] if is_pk else None
    return col_name, col_info, fk, pk_cols


def _parse_table_constraint(item: str) -> Dict[str, Any]:
    s = _normalize_ws(item)

    mcon = re.match(
        r'^constraint\s+(?P<cname>"[^"]+"|`[^`]+`|\[[^\]]+\]|\w+)\s+(?P<rest>.+)$',
        s,
        flags=re.IGNORECASE
    )
    constraint_name = None
    rest = s
    if mcon:
        constraint_name = _strip_quotes(mcon.group("cname"))
        rest = mcon.group("rest")

    mpk = re.match(r"primary\s+key\s*\((?P<cols>[^)]+)\)", rest, flags=re.IGNORECASE)
    if mpk:
        cols = [_normalize_ident(x.strip()) for x in mpk.group("cols").split(",")]
        return {"type": "primary_key", "name": constraint_name, "columns": cols}

    muq = re.match(r"unique\s*\((?P<cols>[^)]+)\)", rest, flags=re.IGNORECASE)
    if muq:
        cols = [_normalize_ident(x.strip()) for x in muq.group("cols").split(",")]
        return {"type": "unique", "name": constraint_name, "columns": cols}

    mfk = re.match(
        r"foreign\s+key\s*\((?P<cols>[^)]+)\)\s+references\s+(?P<table>(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)(\.(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))?)\s*(\((?P<refcols>[^)]+)\))?(?P<mods>.*)$",
        rest, flags=re.IGNORECASE
    )
    if mfk:
        cols = [_normalize_ident(x.strip()) for x in mfk.group("cols").split(",")]
        ref_table = _normalize_ident(mfk.group("table").split(".")[-1])
        ref_cols = []
        if mfk.group("refcols"):
            ref_cols = [_normalize_ident(x.strip()) for x in mfk.group("refcols").split(",")]

        mods = mfk.group("mods") or ""
        on_delete = None
        on_update = None
        mdel = re.search(
            r"\bon\s+delete\b\s+(cascade|restrict|set\s+null|set\s+default|no\s+action)",
            mods, flags=re.IGNORECASE
        )
        mupd = re.search(
            r"\bon\s+update\b\s+(cascade|restrict|set\s+null|set\s+default|no\s+action)",
            mods, flags=re.IGNORECASE
        )
        if mdel:
            on_delete = _normalize_ws(mdel.group(1)).upper()
        if mupd:
            on_update = _normalize_ws(mupd.group(1)).upper()

        return {
            "type": "foreign_key",
            "name": constraint_name,
            "columns": cols,
            "ref_table": ref_table,
            "ref_columns": ref_cols,
            "on_delete": on_delete,
            "on_update": on_update
        }

    mchk = re.match(r"check\s*\((?P<expr>.+)\)", rest, flags=re.IGNORECASE)
    if mchk:
        expr = _normalize_ws(mchk.group("expr"))
        out: Dict[str, Any] = {"type": "check", "name": constraint_name, "expression": expr}
        in_list = _parse_check_in_list(expr)
        if in_list:
            out["in_list"] = in_list
        return out

    return {"type": "unknown", "name": constraint_name, "raw": s}


def parse_ddl_to_schema(ddl_text: str) -> Dict[str, Any]:
    ddl_text = _remove_sql_comments(ddl_text)
    blocks = _extract_create_table_blocks(ddl_text)

    schema: Dict[str, Any] = {
        "dialect": "postgresql",
        "tables": {},
        "errors": []
    }

    for table_name, body in blocks:
        items = _split_top_level_commas(body)

        table = {
            "columns": {},
            "primary_key": [],
            "foreign_keys": [],
            "unique": [],
            "checks": [],
            "raw_items_count": len(items),
            "allowed_values": {},
        }

        for item in items:
            item_norm = _normalize_ws(item)
            low = item_norm.lower()

            try:
                if low.startswith(_TABLE_CONSTRAINT_PREFIXES):
                    c = _parse_table_constraint(item_norm)
                    if c["type"] == "primary_key":
                        table["primary_key"] = c["columns"]
                    elif c["type"] == "foreign_key":
                        table["foreign_keys"].append({
                            "columns": c["columns"],
                            "ref_table": c["ref_table"],
                            "ref_columns": c["ref_columns"],
                            "on_delete": c.get("on_delete"),
                            "on_update": c.get("on_update"),
                            "name": c.get("name"),
                        })
                    elif c["type"] == "unique":
                        table["unique"].append(c["columns"])
                    elif c["type"] == "check":
                        table["checks"].append({"name": c.get("name"), "expression": c["expression"], "in_list": c.get("in_list")})
                        in_list = c.get("in_list")
                        if in_list and in_list.get("column") and in_list.get("allowed"):
                            table["allowed_values"][in_list["column"]] = in_list["allowed"]
                    else:
                        pass
                else:
                    col_name, col_info, fk, pk_inline = _parse_column_def(item_norm)
                    table["columns"][col_name] = col_info

                    if pk_inline:
                        if not table["primary_key"]:
                            table["primary_key"].extend(pk_inline)

                    if fk:
                        table["foreign_keys"].append(fk)

                    if col_info.get("unique"):
                        table["unique"].append([col_name])

                    ev = col_info.get("enum_values") or None
                    if ev:
                        table["allowed_values"][col_name] = ev

                    for chk in (col_info.get("checks") or []):
                        in_list = chk.get("in_list")
                        if in_list and in_list.get("column") and in_list.get("allowed"):
                            table["allowed_values"][in_list["column"]] = in_list["allowed"]

            except Exception as e:
                schema["errors"].append({
                    "table": table_name,
                    "item": item,
                    "error": str(e)
                })

        schema["tables"][table_name] = table

    return schema
