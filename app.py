import streamlit as st
import pandas as pd
import json
import random
import inspect
import time

import os
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from io import BytesIO
import zipfile

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

st.sidebar.title("Data Assistant")
page = st.sidebar.radio(
    label="Navigation",
    options=["Data Generation", "Talk to your data"],
    index=0,
    label_visibility="collapsed",
)

def _pg_full_reload(ddl_text: str, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    pg: PostgresClient = st.session_state.pg
    pg.reset_public_schema()
    pg.apply_ddl(ddl_text)
    inserted = pg.insert_tables(tables)
    return inserted

def _pg_reload_table(table_name: str, df: pd.DataFrame) -> int:
    pg: PostgresClient = st.session_state.pg
    with pg.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE;')
        conn.commit()
    return pg.insert_df(table_name, df)

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

        progress = None
        status_ctx = None

        if ddl_file is None:
            st.error("Please upload a DDL schema file first.")
        else:
            ddl_text = ddl_file.read().decode("utf-8", errors="ignore")
            st.session_state.ddl_text = ddl_text

            try:
                schema = parse_ddl_to_schema(ddl_text)
                st.session_state.schema = schema

                st.success("DDL parsed → schema JSON is ready.")
                with st.expander("Show parsed schema (JSON)"):
                    st.code(json.dumps(schema, ensure_ascii=False, indent=2), language="json")
            except Exception as e:
                st.error(f"Failed to parse DDL: {e}")
                st.stop()

            st.success("DDL schema uploaded.")
            with st.expander("DDL preview"):
                st.code(st.session_state.ddl_text, language="sql")

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

                    status_ctx.update(label="Generation completed ✅", state="complete", expanded=False)

                progress.progress(100)
                st.success("Done. Tables generated.")

                st.session_state.tables = dfs

                dataset_id = _new_dataset_id()
                st.session_state.current_dataset_id = dataset_id

                _save_dataset_to_disk(
                    dataset_id=dataset_id,
                    ddl_text=st.session_state.ddl_text,
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
                            ddl_text=st.session_state.ddl_text,
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

                        patch = vertex.generate_json(
                            prompt=patch_prompt,
                            response_schema=patch_schema,
                            temperature=0.2,
                            max_output_tokens=2048,
                            repair_attempts=1,
                            token_expand_attempts=2,
                            max_output_tokens_cap=8192,
                        )

                        line.info("Applying patch to dataframe…")
                        new_df, warnings = apply_patch_to_df(
                            df=df,
                            patch=patch,
                            table_meta=table_meta,
                            fk_allowed_values=fk_allowed,
                        )

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

    saved_ids = list(st.session_state.datasets.keys())
    options = ["(current session)"] + saved_ids
    dataset = st.selectbox("Dataset", options=options, index=0)

    if dataset == "(current session)":
        tables_for_view = st.session_state.tables
        st.caption("Using tables from current session_state.")
    else:
        meta = st.session_state.datasets.get(dataset, {})
        st.caption(f"Dataset path: {meta.get('path')}")
        st.caption(f"Tables: {', '.join(meta.get('tables', []))}")
        tables_for_view = st.session_state.tables

    st.markdown("### Tables preview")
    if not tables_for_view:
        st.info("No tables available yet.")
    else:
        tname = st.selectbox("Table", options=list(tables_for_view.keys()))
        st.dataframe(tables_for_view[tname], use_container_width=True, hide_index=True)

    st.markdown("###")
    question = st.text_area(
        "Question",
        placeholder="Ask a question in natural language (e.g., 'Show top 10 users by total order amount')...",
        height=120,
    )

    run = st.button("Run query", type="primary")

    if run:
        st.info("Next step: Gemini → SQL generation → execute in PostgreSQL → show text/table/plot result.")

    st.markdown("### Result")
    st.write("Result output will appear here (text, table, or chart).")
