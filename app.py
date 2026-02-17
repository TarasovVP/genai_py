from datetime import datetime
import streamlit as st
import pandas as pd
import json
import random
import inspect
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

from ddl_parser import parse_ddl_to_schema
from vertex_client import VertexGenAIClient
from data_generator import generate_all_tables
from data_editor import (
    build_table_patch_schema,
    build_prompt_for_table_patch,
    apply_patch_to_df,
)
from postgres_client import PostgresClient, PostgresConfig

from config import get_settings
from state import init_session_state
from tracing.langfuse_tracer import LangfuseTracer

from services.vertex_logged import VertexLogged
from services.postgres_logged import PostgresLogged

from storage.datasets import (
    new_dataset_id,
    save_dataset_to_disk,
    save_table_csv,
    df_to_csv_bytes,
    tables_to_zip_bytes,
    dataset_dir,
)

from domain.ddl_normalizer import normalize_ddl_for_postgres
from domain.allowed_values import (
    normalize_all_tables_to_allowed_values,
    normalize_df_to_allowed_values,
)
from domain.fk_allowed_values import (
    compute_fk_allowed_values_for_table,
    get_table_meta,
)
from domain.schema_render import schema_text_for_prompt
from domain.sql_guard import is_sql_safe_readonly
from domain.nl2sql import (
    sql_gen_schema,
    build_nl2sql_prompt,
    execute_sql_with_repairs,
)
from domain.charts import maybe_render_chart


settings = get_settings()
DATASETS_ROOT: Path = settings.datasets_root

st.set_page_config(page_title="Data Assistant", layout="wide")

init_session_state(st, settings)
tracer = LangfuseTracer(st)

DEFAULT_ROWS_PER_TABLE = settings.default_rows_per_table
DEFAULT_SEED = settings.default_seed

DEFAULT_VERTEX_PROJECT = settings.vertex_project
DEFAULT_VERTEX_LOCATION = settings.vertex_location
DEFAULT_VERTEX_MODEL = settings.vertex_model

st.sidebar.title("Data Assistant")
page = st.sidebar.radio(
    label="Navigation",
    options=["Data Generation", "Talk to your data"],
    index=0,
    label_visibility="collapsed",
)


def _new_trace_id() -> str:
    return tracer.new_trace_id()


def _log_event(name: str, level: str, message: str, metadata: Optional[Dict[str, Any]] = None):
    tracer.event(
        page=page,
        dataset_id=st.session_state.current_dataset_id,
        name=name,
        level=level,
        message=message,
        metadata=metadata or {},
    )


def _log_span(
    name: str,
    start_ts: float,
    end_ts: float,
    metadata: Optional[Dict[str, Any]] = None,
    status: str = "ok",
):
    tracer.span(
        page=page,
        dataset_id=st.session_state.current_dataset_id,
        name=name,
        start_ts=start_ts,
        end_ts=end_ts,
        metadata=metadata or {},
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
                vertex_logged = VertexLogged(
                    vertex=vertex,
                    tracer=tracer,
                    page=page,
                    dataset_id=st.session_state.current_dataset_id,
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

                dfs = normalize_all_tables_to_allowed_values(st.session_state.schema, dfs)

                st.success("Done. Tables generated.")
                st.session_state.tables = dfs

                dataset_id = new_dataset_id()
                st.session_state.current_dataset_id = dataset_id

                save_dataset_to_disk(
                    root=DATASETS_ROOT,
                    dataset_id=dataset_id,
                    ddl_text=ddl_for_pg,
                    schema=st.session_state.schema,
                    tables=st.session_state.tables,
                    dataset_prompt=str(dataset_prompt or ""),
                )

                st.session_state.datasets[dataset_id] = {
                    "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "path": str(dataset_dir(DATASETS_ROOT, dataset_id)),
                    "tables": list(st.session_state.tables.keys()),
                }

                st.caption(f"Saved dataset: {dataset_id}")

                pg_logged = PostgresLogged(
                    pg=st.session_state.pg,
                    tracer=tracer,
                    page=page,
                    dataset_id=dataset_id,
                )

                with st.status("Loading dataset into PostgreSQL…", expanded=True) as pgctx:
                    line = pgctx.empty()
                    t0 = time.time()
                    try:
                        line.info("Reset schema → apply DDL → insert tables…")

                        inserted = pg_logged.full_reload(
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
            data=df_to_csv_bytes(df),
            file_name=f"{selected_table}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_right:
        st.download_button(
            label="Download ZIP (all tables)",
            data=tables_to_zip_bytes(st.session_state.tables),
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
                    vertex_logged = VertexLogged(
                        vertex=vertex,
                        tracer=tracer,
                        page=page,
                        dataset_id=st.session_state.current_dataset_id,
                    )

                    table_meta = get_table_meta(st.session_state.schema, selected_table)
                    if not table_meta:
                        st.error(f"Table '{selected_table}' not found in schema.")
                        st.stop()

                    DEFAULT_SAMPLE_ROWS = 20
                    DEFAULT_MAX_OPS = 20

                    sample_rows = df.head(min(DEFAULT_SAMPLE_ROWS, len(df))).to_dict(orient="records")
                    fk_allowed = compute_fk_allowed_values_for_table(
                        schema=st.session_state.schema,
                        tables=st.session_state.tables,
                        table_name=selected_table,
                    )

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

                        patch = vertex_logged.generate_json(
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

                        new_df = normalize_df_to_allowed_values(st.session_state.schema, selected_table, new_df)
                        st.session_state.tables[selected_table] = new_df

                        cur_id = st.session_state.current_dataset_id
                        if cur_id:
                            save_table_csv(DATASETS_ROOT, cur_id, selected_table, new_df)

                        df = new_df
                        df_placeholder.dataframe(df, use_container_width=True, hide_index=True)

                        line.info("Updating table in PostgreSQL…")
                        pg_logged = PostgresLogged(
                            pg=st.session_state.pg,
                            tracer=tracer,
                            page=page,
                            dataset_id=cur_id,
                        )
                        try:
                            inserted = pg_logged.reload_table(selected_table, new_df)
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

    schema_text = schema_text_for_prompt(st.session_state.schema)

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
        vertex_logged = VertexLogged(
            vertex=vertex,
            tracer=tracer,
            page=page,
            dataset_id=st.session_state.current_dataset_id,
        )

        pg_logged = PostgresLogged(
            pg=st.session_state.pg,
            tracer=tracer,
            page=page,
            dataset_id=st.session_state.current_dataset_id,
        )

        started_at = time.time()
        with st.status("Generating SQL via Gemini…", expanded=True) as sctx:
            line = sctx.empty()
            line.info("Building prompt…")

            prompt = build_nl2sql_prompt(question=question.strip(), schema_text=schema_text)
            resp_schema = sql_gen_schema()

            line.info("Requesting structured JSON…")
            out = vertex_logged.generate_json(
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
                dfq, final_out, repairs = execute_sql_with_repairs(
                    vertex_logged=vertex_logged,
                    pg_logged=pg_logged,
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

        ok, why = is_sql_safe_readonly(sql)
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
            maybe_render_chart(st, dfq, chart_spec)
        except Exception as e:
            st.caption(f"Chart skipped: {e}")
