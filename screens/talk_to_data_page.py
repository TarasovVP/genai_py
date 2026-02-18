from __future__ import annotations

from typing import Optional, Dict, Any

import streamlit as st
import time

from vertex_client import VertexGenAIClient

from services.vertex_logged import VertexLogged
from services.postgres_logged import PostgresLogged

from domain.schema_render import schema_text_for_prompt
from domain.sql_guard import is_sql_safe_readonly
from domain.nl2sql import (
    sql_gen_schema,
    build_nl2sql_prompt,
    execute_sql_with_repairs,
)
from domain.charts import maybe_render_chart


def render(
    st,
    *,
    settings,
    tracer,
    page: str,
):
    def _new_trace_id() -> str:
        return tracer.new_trace_id()

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

    if not run:
        return

    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    st.session_state.trace_id = _new_trace_id()

    vertex = VertexGenAIClient(
        project=settings.vertex_project,
        location=settings.vertex_location,
        model=settings.vertex_model,
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
            metadata={"question": question.strip()[:2000]},
        )

        sctx.update(label="SQL generated ✅", state="complete", expanded=False)

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


def _format_elapsed(seconds: float) -> str:
    sec = max(0, int(seconds))
    mm = sec // 60
    ss = sec % 60
    hh = mm // 60
    mm = mm % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"
