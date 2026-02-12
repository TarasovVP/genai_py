import streamlit as st
import pandas as pd
import json
import random
import inspect
import time
import os
import re
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from io import BytesIO
import zipfile

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

try:
    from langfuse import Langfuse
    _HAS_LANGFUSE = True
except Exception:
    Langfuse = None
    _HAS_LANGFUSE = False

from ddl_parser import parse_ddl_to_schema
from vertex_client import VertexGenAIClient
from data_generator import generate_all_tables
from data_editor import (
    build_table_patch_schema,
    build_prompt_for_table_patch,
    apply_patch_to_df,
)
from postgres_client import PostgresClient, PostgresConfig

DEFAULT_ROWS_PER_TABLE = 10
DEFAULT_SEED = 0

DEFAULT_VERTEX_PROJECT = "gd-gcp-gridu-genai"
DEFAULT_VERTEX_LOCATION = "europe-west1"
DEFAULT_VERTEX_MODEL = "gemini-2.0-flash-001"

DATASETS_ROOT = Path("datasets")
DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Data Assistant", layout="wide")

DEFAULT_PG_HOST = os.getenv("PG_HOST", "localhost")
DEFAULT_PG_PORT = int(os.getenv("PG_PORT", "55432"))
DEFAULT_PG_DB = os.getenv("PG_DB", "data_assistant")
DEFAULT_PG_USER = os.getenv("PG_USER", "data_assistant")
DEFAULT_PG_PASSWORD = os.getenv("PG_PASSWORD", "data_assistant")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if "tables" not in st.session_state:
    st.session_state.tables = {}

if "ddl_text" not in st.session_state:
    st.session_state.ddl_text = ""

if "schema" not in st.session_state:
    st.session_state.schema = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "dataset_prompt" not in st.session_state:
    st.session_state.dataset_prompt = ""

if "datasets" not in st.session_state:
    st.session_state.datasets = {}

if "current_dataset_id" not in st.session_state:
    st.session_state.current_dataset_id = None

if "pg" not in st.session_state:
    st.session_state.pg = PostgresClient(
        PostgresConfig(
            host=DEFAULT_PG_HOST,
            port=DEFAULT_PG_PORT,
            dbname=DEFAULT_PG_DB,
            user=DEFAULT_PG_USER,
            password=DEFAULT_PG_PASSWORD,
        )
    )

if "langfuse" not in st.session_state:
    st.session_state.langfuse = None
    if _HAS_LANGFUSE and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            st.session_state.langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
        except Exception:
            st.session_state.langfuse = None

if "trace_id" not in st.session_state:
    st.session_state.trace_id = None

st.sidebar.title("Data Assistant")
page = st.sidebar.radio(
    label="Navigation",
    options=["Data Generation", "Talk to your data"],
    index=0,
    label_visibility="collapsed",
)

def _new_trace_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:10]

def _ensure_trace(name: str, metadata: dict | None = None):
    lf = st.session_state.langfuse
    if lf is None:
        return None

    if not st.session_state.trace_id:
        st.session_state.trace_id = _new_trace_id()

    try:
        return lf.trace(
            id=st.session_state.trace_id,
            name=name,
            user_id="streamlit_user",
            metadata=metadata or {},
        )
    except Exception:
        return None

def _safe_preview(obj, limit: int = 2000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "…"
    return s

def _log_event(name: str, level: str, message: str, metadata: dict | None = None):
    lf = st.session_state.langfuse
    if lf is None:
        return

    tr = _ensure_trace(name="data_assistant", metadata={"page": page, "dataset_id": st.session_state.current_dataset_id})
    if tr is None:
        return

    try:
        lf.event(
            trace_id=st.session_state.trace_id,
            name=name,
            level=level,
            message=message,
            metadata=metadata or {},
        )
    except Exception:
        pass

def _log_generation(
    phase: str,
    prompt: str,
    response_schema: dict | None,
    model: str,
    temperature: float | None,
    max_output_tokens: int | None,
    start_ts: float,
    end_ts: float,
    output: dict | None,
    error: str | None,
    metadata: dict | None = None,
):
    lf = st.session_state.langfuse
    if lf is None:
        return

    tr = _ensure_trace(
        name="data_assistant",
        metadata={
            "page": page,
            "phase": phase,
            "dataset_id": st.session_state.current_dataset_id,
        },
    )
    if tr is None:
        return

    meta = dict(metadata or {})
    meta.update(
        {
            "phase": phase,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_schema_present": bool(response_schema),
            "output_preview": _safe_preview(output, 1500) if output is not None else "",
            "error": error or "",
        }
    )

    try:
        lf.generation(
            trace_id=st.session_state.trace_id,
            name=phase,
            model=model,
            input=prompt,
            output=_safe_preview(output, 3000) if output is not None else "",
            metadata=meta,
            start_time=datetime.utcfromtimestamp(start_ts).isoformat(timespec="seconds") + "Z",
            end_time=datetime.utcfromtimestamp(end_ts).isoformat(timespec="seconds") + "Z",
        )
    except Exception:
        pass

def _log_span(
    name: str,
    start_ts: float,
    end_ts: float,
    metadata: dict | None = None,
    status: str = "ok",
):
    lf = st.session_state.langfuse
    if lf is None:
        return

    tr = _ensure_trace(
        name="data_assistant",
        metadata={
            "page": page,
            "dataset_id": st.session_state.current_dataset_id,
        },
    )
    if tr is None:
        return

    meta = dict(metadata or {})
    meta["status"] = status

    try:
        lf.span(
            trace_id=st.session_state.trace_id,
            name=name,
            start_time=datetime.utcfromtimestamp(start_ts).isoformat(timespec="seconds") + "Z",
            end_time=datetime.utcfromtimestamp(end_ts).isoformat(timespec="seconds") + "Z",
            metadata=meta,
        )
    except Exception:
        pass

def _vertex_generate_json_logged(
    vertex: VertexGenAIClient,
    phase: str,
    prompt: str,
    response_schema: dict,
    temperature: float,
    max_output_tokens: int,
    repair_attempts: int,
    token_expand_attempts: int,
    max_output_tokens_cap: int,
    metadata: dict | None = None,
):
    t0 = time.time()
    err = None
    out = None
    try:
        out = vertex.generate_json(
            prompt=prompt,
            response_schema=response_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            repair_attempts=repair_attempts,
            token_expand_attempts=token_expand_attempts,
            max_output_tokens_cap=max_output_tokens_cap,
        )
        return out
    except Exception as e:
        err = str(e)
        raise
    finally:
        t1 = time.time()
        _log_generation(
            phase=phase,
            prompt=prompt,
            response_schema=response_schema,
            model=getattr(vertex, "model", None) or DEFAULT_VERTEX_MODEL,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            start_ts=t0,
            end_ts=t1,
            output=out if isinstance(out, dict) else {"output": out},
            error=err,
            metadata=metadata or {},
        )

_ENUM_COL_RE = re.compile(
    r"""
    (?P<col>"?[A-Za-z_][A-Za-z0-9_]*"?)
    \s+
    ENUM
    \s*\(
        (?P<vals>[^)]*)
    \)
    (?P<rest>[^,\n]*)
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _split_enum_vals(vals_raw: str) -> list[str]:
    vals = []
    for m in re.finditer(r"'((?:[^'\\]|\\.)*)'\s*(?:,|$)", vals_raw.strip()):
        v = m.group(1)
        v = v.replace("\\'", "'")
        v = v.replace("\\\\", "\\")
        vals.append(v)
    return vals

def _escape_sql_literal(s: str) -> str:
    return s.replace("'", "''")

def normalize_ddl_for_postgres(ddl: str) -> str:
    s = ddl or ""
    s = re.sub(r"\bAUTO_INCREMENT\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\)\s*ENGINE\s*=\s*\w+\s*;?", ");", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDEFAULT\s+CHARSET\s*=\s*\w+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCHARSET\s*=\s*\w+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCOLLATE\s*=\s*[\w_]+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bUNSIGNED\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDATETIME\b", "TIMESTAMP", s, flags=re.IGNORECASE)

    def repl_enum(m: re.Match) -> str:
        col = m.group("col")
        vals_raw = m.group("vals") or ""
        rest = (m.group("rest") or "").strip()
        vals = _split_enum_vals(vals_raw)
        if not vals:
            return f'{col} TEXT {rest}'.rstrip()
        in_list = ", ".join(f"'{_escape_sql_literal(v)}'" for v in vals)
        check = f'CHECK ({col} IN ({in_list}))'
        out = f"{col} TEXT {rest} {check}"
        out = re.sub(r"\s+", " ", out).strip()
        return out

    s = _ENUM_COL_RE.sub(repl_enum, s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s

def _schema_allowed_for_table(table_name: str) -> dict[str, list]:
    schema_tables = (st.session_state.schema or {}).get("tables", {}) or {}
    meta = schema_tables.get(table_name, {}) or {}
    allowed = meta.get("allowed_values") or {}
    if isinstance(allowed, dict):
        return allowed
    return {}

def _normalize_df_to_allowed_values(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    allowed_map = _schema_allowed_for_table(table_name)
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

def _normalize_all_tables_to_allowed_values(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not tables:
        return tables
    fixed: dict[str, pd.DataFrame] = {}
    for tname, df in tables.items():
        fixed[tname] = _normalize_df_to_allowed_values(tname, df)
    return fixed

def _pg_full_reload(ddl_text: str, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    t0 = time.time()
    status = "ok"
    err = None
    try:
        pg: PostgresClient = st.session_state.pg
        pg.reset_public_schema()
        pg.apply_ddl(ddl_text)
        inserted = pg.insert_tables(tables)
        return inserted
    except Exception as e:
        status = "error"
        err = str(e)
        raise
    finally:
        _log_span(
            name="postgres_full_reload",
            start_ts=t0,
            end_ts=time.time(),
            metadata={
                "tables": list((tables or {}).keys()),
                "ddl_chars": len(ddl_text or ""),
                "error": err or "",
            },
            status=status,
        )

def _pg_reload_table(table_name: str, df: pd.DataFrame) -> int:
    t0 = time.time()
    status = "ok"
    err = None
    try:
        pg: PostgresClient = st.session_state.pg
        with pg.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE;')
            conn.commit()
        return pg.insert_df(table_name, df)
    except Exception as e:
        status = "error"
        err = str(e)
        raise
    finally:
        _log_span(
            name="postgres_reload_table",
            start_ts=t0,
            end_ts=time.time(),
            metadata={
                "table": table_name,
                "rows": int(len(df)) if df is not None else 0,
                "cols": int(len(df.columns)) if df is not None else 0,
                "error": err or "",
            },
            status=status,
        )

def seed_demo_tables():
    st.session_state.tables = {
        "users": pd.DataFrame(
            {
                "ID": ["001", "002", "003"],
                "Name": ["Sample Data 1", "Sample Data 2", "Sample Data 3"],
                "Category": ["Category A", "Category B", "Category A"],
                "Value": [245.50, 127.80, 389.20],
            }
        ),
        "orders": pd.DataFrame(
            {
                "ID": ["101", "102", "103"],
                "UserID": ["001", "002", "001"],
                "Total": [19.99, 54.10, 7.50],
            }
        ),
    }

if not st.session_state.tables:
    seed_demo_tables()

def _supports_on_progress(func) -> bool:
    try:
        sig = inspect.signature(func)
        return "on_progress" in sig.parameters
    except Exception:
        return False

def _format_elapsed(seconds: float) -> str:
    sec = max(0, int(seconds))
    mm = sec // 60
    ss = sec % 60
    hh = mm // 60
    mm = mm % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"

def _get_schema_tables() -> dict:
    return (st.session_state.schema or {}).get("tables", {}) or {}

def _get_table_meta(table_name: str) -> dict:
    return _get_schema_tables().get(table_name, {}) or {}

def _compute_fk_allowed_values_for_table(table_name: str) -> dict[str, list]:
    schema_tables = _get_schema_tables()
    meta = schema_tables.get(table_name, {}) or {}
    fks = meta.get("foreign_keys") or []
    allowed: dict[str, list] = {}

    for fk in fks:
        child_cols = fk.get("columns") or []
        parent = fk.get("ref_table")
        ref_cols = fk.get("ref_columns") or []

        if not child_cols or not parent:
            continue

        child_fk_col = child_cols[0]

        if parent not in st.session_state.tables:
            continue

        df_parent = st.session_state.tables[parent]

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

def _new_dataset_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]

def _dataset_dir(dataset_id: str) -> Path:
    d = DATASETS_ROOT / dataset_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_text(path: Path, text: str):
    path.write_text(text or "", encoding="utf-8")

def _save_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj or {}, ensure_ascii=False, indent=2), encoding="utf-8")

def _save_table_csv(dataset_id: str, table_name: str, df: pd.DataFrame):
    d = _dataset_dir(dataset_id)
    csv_path = d / f"{table_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

def _save_dataset_to_disk(
    dataset_id: str,
    ddl_text: str,
    schema: dict,
    tables: dict[str, pd.DataFrame],
    dataset_prompt: str,
):
    d = _dataset_dir(dataset_id)
    _save_text(d / "ddl.sql", ddl_text or "")
    _save_json(d / "schema.json", schema or {})
    _save_json(
        d / "meta.json",
        {
            "dataset_id": dataset_id,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tables": list((tables or {}).keys()),
            "dataset_prompt": dataset_prompt or "",
        },
    )
    for tname, tdf in (tables or {}).items():
        _save_table_csv(dataset_id, tname, tdf)

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _tables_to_zip_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in (tables or {}).items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
    bio.seek(0)
    return bio.read()

_SQL_BLOCKLIST = re.compile(
    r"\b(drop|truncate|alter|create|grant|revoke|comment|vacuum|analyze|insert|update|delete|merge)\b",
    re.IGNORECASE,
)

def _is_sql_safe_readonly(sql: str) -> tuple[bool, str]:
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

def _schema_text_for_prompt(schema: dict) -> str:
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

def _sql_gen_schema() -> dict:
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

def _build_nl2sql_prompt(question: str, schema_text: str) -> str:
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

def _run_sql_to_df(sql: str) -> pd.DataFrame:
    t0 = time.time()
    status = "ok"
    err = None
    try:
        pg: PostgresClient = st.session_state.pg
        with pg.connect() as conn:
            df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        status = "error"
        err = str(e)
        raise
    finally:
        _log_span(
            name="postgres_select_query",
            start_ts=t0,
            end_ts=time.time(),
            metadata={
                "sql": (sql or "")[:5000],
                "error": err or "",
            },
            status=status,
        )

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

def _execute_sql_with_repairs(
    vertex: VertexGenAIClient,
    question: str,
    schema_text: str,
    initial_out: dict,
    max_repairs: int,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    repairs: list[dict] = []
    out = dict(initial_out or {})
    resp_schema = _sql_gen_schema()

    for attempt in range(0, max_repairs + 1):
        sql = (out.get("sql") or "").strip()
        ok, why = _is_sql_safe_readonly(sql)
        if not ok:
            raise RuntimeError(f"Generated SQL was rejected: {why}")

        try:
            df = _run_sql_to_df(sql)
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
            out2 = _vertex_generate_json_logged(
                vertex=vertex,
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

    raise RuntimeError("Unexpected control flow in _execute_sql_with_repairs")

def _maybe_render_chart(df: pd.DataFrame, chart_spec: dict):
    if df is None or df.empty:
        return
    if not chart_spec or chart_spec.get("type") in (None, "", "none"):
        return

    ctype = chart_spec.get("type")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    title = chart_spec.get("title") or ""
    bins = chart_spec.get("bins")

    if ctype not in ("bar", "line", "hist"):
        return

    if ctype in ("bar", "line"):
        if not x or not y:
            return
        if x not in df.columns or y not in df.columns:
            return

        if _HAS_MPL:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if ctype == "bar":
                ax.bar(df[x].astype(str), df[y])
            else:
                ax.plot(df[x], df[y])
            if title:
                ax.set_title(title)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            st.pyplot(fig)
        else:
            try:
                series = df.set_index(df[x].astype(str))[y]
                if ctype == "bar":
                    st.bar_chart(series)
                else:
                    st.line_chart(series)
            except Exception:
                st.info("Chart rendering is unavailable (matplotlib is not installed).")
        return

    if ctype == "hist":
        if not y or y not in df.columns:
            return
        series = pd.to_numeric(df[y], errors="coerce").dropna()
        if series.empty:
            return

        if _HAS_MPL:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if isinstance(bins, int) and bins > 0:
                ax.hist(series, bins=bins)
            else:
                ax.hist(series)
            if title:
                ax.set_title(title)
            ax.set_xlabel(y)
            ax.set_ylabel("count")
            st.pyplot(fig)
        else:
            st.info("Histogram rendering is unavailable (matplotlib is not installed).")

if page == "Data Generation":
    st.markdown("###")

    dataset_prompt = st.text_input(
        "Prompt",
        placeholder="Optional: global instructions for the whole dataset (e.g., 'E-commerce dataset for Germany, realistic names, EUR prices')",
        key="dataset_prompt",
    )

    col_upload, col_formats = st.columns([1.2, 2.8], vertical_alignment="center")
    with col_upload:
        ddl_file = st.file_uploader(
            label="Upload DDL Schema",
            type=["sql", "ddl", "txt", "json"],
            accept_multiple_files=False,
        )
    with col_formats:
        st.caption("Supported formats: SQL, JSON")

    st.markdown("---")
    st.subheader("Advanced Parameters")

    col_left, col_right = st.columns(2, vertical_alignment="center")
    with col_left:
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    with col_right:
        max_tokens = st.number_input("Max Tokens", min_value=1, value=100, step=10)

    st.markdown("###")
    generate_clicked = st.button("Generate", type="primary")

    if generate_clicked:
        st.session_state.last_error = None
        st.session_state.trace_id = _new_trace_id()

        progress = None
        status_ctx = None

        if ddl_file is None:
            st.error("Please upload a DDL schema file first.")
        else:
            ddl_text = ddl_file.read().decode("utf-8", errors="ignore")
            st.session_state.ddl_text = ddl_text

            ddl_for_pg = normalize_ddl_for_postgres(ddl_text)

            try:
                schema = parse_ddl_to_schema(ddl_for_pg)
                st.session_state.schema = schema
                st.success("DDL parsed → schema JSON is ready.")
                with st.expander("Show parsed schema (JSON)"):
                    st.code(json.dumps(schema, ensure_ascii=False, indent=2), language="json")
            except Exception as e:
                st.error(f"Failed to parse DDL: {e}")
                _log_event("ddl_parse_error", "ERROR", "DDL parsing failed", {"error": str(e)})
                st.stop()

            st.success("DDL schema uploaded.")
            with st.expander("DDL preview"):
                st.code(ddl_for_pg, language="sql")

            try:
                seed = int(DEFAULT_SEED)
                rows_per_table = int(DEFAULT_ROWS_PER_TABLE)

                if seed != 0:
                    random.seed(seed)

                vertex = VertexGenAIClient(
                    project=DEFAULT_VERTEX_PROJECT,
                    location=DEFAULT_VERTEX_LOCATION,
                    model=DEFAULT_VERTEX_MODEL,
                )

                progress = st.progress(0)
                started_at = time.time()

                with st.status("Generating data in Vertex AI…", expanded=True) as status_ctx:
                    status_line = status_ctx.empty()
                    status_line.info("Preparing…")

                    def on_progress(done: int, total: int, table_label: str):
                        shown_total = max(1, int(total))
                        shown_done = min(int(done) + 1, shown_total)

                        if total == 0:
                            pct = 0
                        else:
                            safe_done = min(max(int(done), 0), int(total))
                            pct = int(safe_done * 100 / int(total))
                        progress.progress(pct)

                        elapsed = _format_elapsed(time.time() - started_at)
                        status_line.info(f"Generating: {shown_done}/{shown_total} — {table_label} | ⏱ {elapsed}")

                    t_span0 = time.time()
                    span_status = "ok"
                    span_err = None
                    try:
                        if _supports_on_progress(generate_all_tables):
                            dfs = generate_all_tables(
                                vertex=vertex,
                                ddl_schema=schema,
                                rows_per_table=rows_per_table,
                                temperature=float(temperature),
                                max_output_tokens=int(max_tokens),
                                dataset_prompt=str(dataset_prompt or ""),
                                on_progress=on_progress,
                            )
                        else:
                            elapsed = _format_elapsed(time.time() - started_at)
                            status_line.info(f"Generation in progress… | ⏱ {elapsed}")
                            dfs = generate_all_tables(
                                vertex=vertex,
                                ddl_schema=schema,
                                rows_per_table=rows_per_table,
                                temperature=float(temperature),
                                max_output_tokens=int(max_tokens),
                                dataset_prompt=str(dataset_prompt or ""),
                            )
                    except Exception as e:
                        span_status = "error"
                        span_err = str(e)
                        raise
                    finally:
                        _log_span(
                            name="data_generation",
                            start_ts=t_span0,
                            end_ts=time.time(),
                            metadata={
                                "rows_per_table": rows_per_table,
                                "temperature": float(temperature),
                                "max_output_tokens": int(max_tokens),
                                "dataset_prompt_chars": len(str(dataset_prompt or "")),
                                "error": span_err or "",
                            },
                            status=span_status,
                        )

                    status_ctx.update(label="Generation completed ✅", state="complete", expanded=False)

                progress.progress(100)

                dfs = _normalize_all_tables_to_allowed_values(dfs)

                st.success("Done. Tables generated.")
                st.session_state.tables = dfs

                dataset_id = _new_dataset_id()
                st.session_state.current_dataset_id = dataset_id

                _save_dataset_to_disk(
                    dataset_id=dataset_id,
                    ddl_text=ddl_for_pg,
                    schema=st.session_state.schema,
                    tables=st.session_state.tables,
                    dataset_prompt=str(dataset_prompt or ""),
                )

                st.session_state.datasets[dataset_id] = {
                    "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "path": str(_dataset_dir(dataset_id)),
                    "tables": list(st.session_state.tables.keys()),
                }

                st.caption(f"Saved dataset: {dataset_id}")

                with st.status("Loading dataset into PostgreSQL…", expanded=True) as pgctx:
                    line = pgctx.empty()
                    t0 = time.time()
                    try:
                        line.info("Reset schema → apply DDL → insert tables…")

                        inserted = _pg_full_reload(
                            ddl_text=ddl_for_pg,
                            tables=st.session_state.tables,
                        )

                        total_rows = sum(inserted.values()) if inserted else 0
                        with st.expander("PostgreSQL load summary"):
                            st.json(inserted)
                            st.caption(f"Total rows inserted: {total_rows}")

                        pgctx.update(
                            label=f"PostgreSQL load completed ✅ ({_format_elapsed(time.time() - t0)})",
                            state="complete",
                            expanded=False,
                        )
                        st.success("Dataset is now stored in PostgreSQL.")
                    except Exception as e:
                        pgctx.update(label="PostgreSQL load failed ❌", state="error", expanded=True)
                        st.error(f"PostgreSQL load failed: {e}")

            except Exception as e:
                st.session_state.last_error = f"{e}"
                st.error(f"Generation failed: {st.session_state.last_error}")

                if status_ctx is not None:
                    status_ctx.update(label="Generation stopped due to an error ❌", state="error", expanded=True)

                if st.session_state.schema and st.session_state.schema.get("errors"):
                    with st.expander("DDL parser issues"):
                        st.code(
                            json.dumps(st.session_state.schema["errors"], ensure_ascii=False, indent=2),
                            language="json",
                        )

    st.markdown("###")
    st.subheader("Data Preview")

    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_right:
        table_names = list(st.session_state.tables.keys()) or ["(no tables)"]
        selected_table = st.selectbox("Table", options=table_names, label_visibility="collapsed")
    with header_left:
        st.write("")

    df = st.session_state.tables.get(selected_table)
    if df is None:
        st.info("No data yet. Click Generate after uploading a schema.")
        st.stop()

    st.markdown("###")
    export_left, export_right, export_info = st.columns([1.2, 1.2, 3.6], vertical_alignment="center")

    with export_left:
        st.download_button(
            label="Download CSV (selected table)",
            data=_df_to_csv_bytes(df),
            file_name=f"{selected_table}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_right:
        st.download_button(
            label="Download ZIP (all tables)",
            data=_tables_to_zip_bytes(st.session_state.tables),
            file_name="dataset_tables.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with export_info:
        cur = st.session_state.current_dataset_id
        if cur:
            st.caption(f"Current dataset_id: {cur} (saved on disk)")
        else:
            st.caption("Current dataset_id: not saved yet (generate to create one)")

    df_placeholder = st.empty()
    df_placeholder.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Edit selected table (LLM patch)")

    edit_prompt = st.text_input(
        "Edit instructions",
        placeholder="e.g., 'Set status=active for all inactive users, delete rows with invalid emails, add 5 new VIP users'",
        key=f"edit_prompt__{selected_table}",
    )

    col_b1, col_b2 = st.columns([1, 5], vertical_alignment="center")
    with col_b1:
        apply_edit_clicked = st.button("Apply edit", type="primary", key=f"apply_edit__{selected_table}")
    with col_b2:
        st.caption("Edits are applied via patch-operations (update/delete/add).")

    if apply_edit_clicked:
        if not edit_prompt.strip():
            st.warning("Please enter edit instructions first.")
        else:
            if st.session_state.schema is None:
                st.error("Schema is not loaded. Upload and parse DDL first.")
            else:
                vertex = None
                try:
                    st.session_state.trace_id = _new_trace_id()

                    vertex = VertexGenAIClient(
                        project=DEFAULT_VERTEX_PROJECT,
                        location=DEFAULT_VERTEX_LOCATION,
                        model=DEFAULT_VERTEX_MODEL,
                    )

                    table_meta = _get_table_meta(selected_table)
                    if not table_meta:
                        st.error(f"Table '{selected_table}' not found in schema.")
                        st.stop()

                    DEFAULT_SAMPLE_ROWS = 20
                    DEFAULT_MAX_OPS = 20

                    sample_rows = df.head(min(DEFAULT_SAMPLE_ROWS, len(df))).to_dict(orient="records")
                    fk_allowed = _compute_fk_allowed_values_for_table(selected_table)

                    patch_schema = build_table_patch_schema(table_meta)
                    patch_prompt = build_prompt_for_table_patch(
                        table_name=selected_table,
                        table_meta=table_meta,
                        user_instruction=edit_prompt,
                        sample_rows=sample_rows,
                        fk_allowed_values=fk_allowed,
                        max_ops=DEFAULT_MAX_OPS,
                    )

                    started_at = time.time()
                    with st.status("Applying edit via Gemini…", expanded=True) as sctx:
                        line = sctx.empty()
                        line.info("Requesting patch…")

                        patch = _vertex_generate_json_logged(
                            vertex=vertex,
                            phase="table_edit",
                            prompt=patch_prompt,
                            response_schema=patch_schema,
                            temperature=0.2,
                            max_output_tokens=2048,
                            repair_attempts=1,
                            token_expand_attempts=2,
                            max_output_tokens_cap=8192,
                            metadata={
                                "table": selected_table,
                                "user_instruction": edit_prompt[:2000],
                                "sample_rows_count": len(sample_rows),
                                "max_ops": DEFAULT_MAX_OPS,
                            },
                        )

                        line.info("Applying patch to dataframe…")
                        new_df, warnings = apply_patch_to_df(
                            df=df,
                            patch=patch,
                            table_meta=table_meta,
                            fk_allowed_values=fk_allowed,
                        )

                        new_df = _normalize_df_to_allowed_values(selected_table, new_df)

                        st.session_state.tables[selected_table] = new_df

                        cur_id = st.session_state.current_dataset_id
                        if cur_id:
                            _save_table_csv(cur_id, selected_table, new_df)

                        df = new_df
                        df_placeholder.dataframe(df, use_container_width=True, hide_index=True)

                        line.info("Updating table in PostgreSQL…")
                        try:
                            inserted = _pg_reload_table(selected_table, new_df)
                            line.success(f"PostgreSQL updated ✅ (inserted {inserted} rows)")
                        except Exception as e:
                            line.error(f"PostgreSQL update failed ❌: {e}")

                        sctx.update(label="Edit applied ✅", state="complete", expanded=False)

                    st.success(f"Edit applied to '{selected_table}'.")
                    st.caption(f"Time: {_format_elapsed(time.time() - started_at)}")

                    if warnings:
                        with st.expander(f"Warnings ({len(warnings)})"):
                            for w in warnings[:200]:
                                st.warning(w)

                except Exception as e:
                    st.session_state.last_error = f"{e}"
                    st.error(f"Edit failed: {st.session_state.last_error}")

                    if vertex is not None:
                        raw = getattr(vertex, "last_raw", None)
                        fr = getattr(vertex, "last_finish_reason", None)
                        if raw:
                            with st.expander("Vertex raw head"):
                                st.code(str(raw)[:2000])
                        if fr:
                            st.caption(f"Finish reason: {fr}")

else:
    st.subheader("Talk to your data")

    if not st.session_state.schema:
        st.warning("Schema is not loaded yet. Go to 'Data Generation', upload DDL and generate/load dataset first.")
        st.stop()

    schema_text = _schema_text_for_prompt(st.session_state.schema)

    with st.expander("Schema (for reference)", expanded=False):
        st.code(schema_text)

    st.markdown("###")
    question = st.text_area(
        "Question",
        placeholder="Ask a question in natural language (e.g., 'Show top 10 users by total order amount')...",
        height=120,
    )

    col_run, col_opts = st.columns([1, 3], vertical_alignment="center")
    with col_opts:
        show_sql = st.checkbox("Show SQL", value=True)
        show_expl = st.checkbox("Show explanation", value=True)
        allow_repairs = st.checkbox("Auto-repair SQL on error", value=True)
        max_repairs = st.selectbox("Max repairs", options=[0, 1, 2, 3], index=2, disabled=not allow_repairs)

    run = col_run.button("Run query", type="primary")

    st.markdown("### Result")

    if run:
        if not question.strip():
            st.warning("Please enter a question first.")
            st.stop()

        st.session_state.trace_id = _new_trace_id()

        vertex = VertexGenAIClient(
            project=DEFAULT_VERTEX_PROJECT,
            location=DEFAULT_VERTEX_LOCATION,
            model=DEFAULT_VERTEX_MODEL,
        )

        started_at = time.time()
        with st.status("Generating SQL via Gemini…", expanded=True) as sctx:
            line = sctx.empty()
            line.info("Building prompt…")

            prompt = _build_nl2sql_prompt(question=question.strip(), schema_text=schema_text)
            resp_schema = _sql_gen_schema()

            line.info("Requesting structured JSON…")
            out = _vertex_generate_json_logged(
                vertex=vertex,
                phase="nl2sql",
                prompt=prompt,
                response_schema=resp_schema,
                temperature=0.0,
                max_output_tokens=1024,
                repair_attempts=1,
                token_expand_attempts=1,
                max_output_tokens_cap=2048,
                metadata={
                    "question": question.strip()[:2000],
                },
            )

            sctx.update(label="SQL generated ✅", state="complete", expanded=False)

        repairs = []
        with st.status("Executing SQL in PostgreSQL…", expanded=True) as ectx:
            eline = ectx.empty()
            try:
                eline.info("Running query…")
                dfq, final_out, repairs = _execute_sql_with_repairs(
                    vertex=vertex,
                    question=question.strip(),
                    schema_text=schema_text,
                    initial_out=out,
                    max_repairs=int(max_repairs) if allow_repairs else 0,
                )
                ectx.update(
                    label=f"Query completed ✅ ({_format_elapsed(time.time() - started_at)})",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                ectx.update(label="Query failed ❌", state="error", expanded=True)
                st.error(f"PostgreSQL error: {e}")
                if show_sql:
                    sql_bad = ((out or {}).get("sql") or "").strip()
                    if sql_bad:
                        st.subheader("SQL")
                        st.code(sql_bad, language="sql")
                st.stop()

        sql = (final_out or {}).get("sql") or ""
        explanation = (final_out or {}).get("explanation") or ""
        chart_spec = (final_out or {}).get("chart") or {"type": "none"}

        ok, why = _is_sql_safe_readonly(sql)
        if not ok:
            st.error(f"Generated SQL was rejected: {why}")
            if show_sql and sql:
                st.code(sql, language="sql")
            st.stop()

        if show_sql:
            st.subheader("SQL")
            st.code(sql.strip(), language="sql")

        if show_expl and explanation.strip():
            st.caption(explanation.strip())

        if repairs:
            with st.expander(f"Repairs applied ({len(repairs)})", expanded=False):
                st.json(repairs)

        if dfq is None or dfq.empty:
            st.info("No rows returned.")
        else:
            st.dataframe(dfq, use_container_width=True, hide_index=True)

        try:
            _maybe_render_chart(dfq, chart_spec)
        except Exception as e:
            st.caption(f"Chart skipped: {e}")
