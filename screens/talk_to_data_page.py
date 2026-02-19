from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

import pandas as pd
import streamlit as st

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
from domain.guardrails import run_guardrails

_MAX_HISTORY_ROWS = 1000


def render(
    st,
    *,
    settings,
    tracer,
    page: str,
):
    def _new_trace_id() -> str:
        return tracer.new_trace_id()

    def _log_event(name: str, level: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            tracer.event(
                page=page,
                dataset_id=st.session_state.current_dataset_id,
                name=name,
                level=level,
                message=message,
                metadata=metadata or {},
            )
        except Exception:
            pass

    st.subheader("Talk to your data")

    if not st.session_state.schema:
        st.warning("Schema is not loaded yet. Go to 'Data Generation', upload DDL and generate/load dataset first.")
        st.stop()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    schema_text = schema_text_for_prompt(st.session_state.schema)

    with st.expander("Schema (for reference)", expanded=False):
        st.code(schema_text)

    _render_chat_history(st, st.session_state.chat_messages)

    user_text = st.chat_input("Ask a question about your data…")
    if not user_text:
        return

    user_text = user_text.strip()
    if not user_text:
        return

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": user_text,
            "ts": time.time(),
        }
    )

    with st.chat_message("user"):
        st.markdown(user_text)

    gr = run_guardrails(user_text, st.session_state.schema)

    _log_event(
        name="guardrails_check",
        level="INFO" if gr.allowed else "WARN",
        message=f"guardrails={gr.kind}",
        metadata={
            "allowed": gr.allowed,
            "kind": gr.kind,
            "reason": gr.reason,
            "pii_counts": gr.pii_counts,
            "meta": gr.meta,
        },
    )

    if not gr.allowed:
        with st.chat_message("assistant"):
            if gr.kind == "injection":
                st.warning(gr.reason)
                st.caption("Please rephrase as a question about the loaded dataset (tables/SQL/charts).")
            else:
                st.info(gr.reason)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "kind": gr.kind,
                "content": gr.reason,
                "ts": time.time(),
            }
        )
        return

    prompt_question = gr.masked_text

    with st.chat_message("assistant"):
        st.session_state.trace_id = _new_trace_id()

        with st.status("Thinking…", expanded=True) as sctx:
            line = sctx.empty()
            line.info("Generating SQL…")

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

            prompt = build_nl2sql_prompt(question=prompt_question, schema_text=schema_text)
            resp_schema = sql_gen_schema()

            out = vertex_logged.generate_json(
                phase="nl2sql",
                prompt=prompt,
                response_schema=resp_schema,
                temperature=0.0,
                max_output_tokens=1024,
                repair_attempts=1,
                token_expand_attempts=1,
                max_output_tokens_cap=2048,
                metadata={"question": prompt_question[:2000]},
            )

            sctx.update(label="SQL generated ✅", state="running", expanded=True)

            line.info("Executing SQL…")
            try:
                dfq, final_out, repairs = execute_sql_with_repairs(
                    vertex_logged=vertex_logged,
                    pg_logged=pg_logged,
                    question=prompt_question,
                    schema_text=schema_text,
                    initial_out=out,
                    max_repairs=0,
                )
            except Exception as e:
                sctx.update(label="Query failed ❌", state="error", expanded=True)
                st.error(f"PostgreSQL error: {e}")

                sql_bad = ((out or {}).get("sql") or "").strip()
                if sql_bad:
                    st.subheader("SQL")
                    st.code(sql_bad, language="sql")

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "kind": "error",
                        "content": str(e),
                        "sql": sql_bad,
                        "ts": time.time(),
                    }
                )
                return

            sctx.update(label="Query completed ✅", state="complete", expanded=False)

        sql = (final_out or {}).get("sql") or ""
        explanation = (final_out or {}).get("explanation") or ""
        chart_spec = (final_out or {}).get("chart") or {"type": "none"}

        ok, why = is_sql_safe_readonly(sql)
        if not ok:
            st.error(f"Generated SQL was rejected: {why}")
            if sql:
                st.subheader("SQL")
                st.code(sql.strip(), language="sql")

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "kind": "rejected",
                    "content": why,
                    "sql": sql,
                    "ts": time.time(),
                }
            )
            return

        st.subheader("SQL")
        st.code(sql.strip(), language="sql")

        if explanation.strip():
            _stream_text(explanation.strip())

        if dfq is None or dfq.empty:
            st.info("No rows returned.")
        else:
            st.dataframe(dfq, use_container_width=True, hide_index=True)
            try:
                maybe_render_chart(st, dfq, chart_spec)
            except Exception as e:
                st.caption(f"Chart skipped: {e}")

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "kind": "result",
                "ts": time.time(),
                "sql": sql,
                "explanation": explanation,
                "chart_spec": chart_spec,
                "df": _serialize_df(dfq, max_rows=_MAX_HISTORY_ROWS),
            }
        )


def _render_chat_history(st, messages: List[Dict[str, Any]]) -> None:
    for m in messages:
        role = m.get("role", "assistant")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(m.get("content", ""))
            continue

        kind = m.get("kind", "result")
        with st.chat_message("assistant"):
            if kind in ("error", "rejected", "injection", "off_topic"):
                msg = m.get("content", "")
                if kind == "injection":
                    st.warning(msg)
                elif kind == "off_topic":
                    st.info(msg)
                else:
                    st.error(msg)

                sql = (m.get("sql") or "").strip()
                if sql:
                    st.subheader("SQL")
                    st.code(sql, language="sql")
                continue

            sql = (m.get("sql") or "").strip()
            explanation = (m.get("explanation") or "").strip()
            chart_spec = m.get("chart_spec") or {"type": "none"}
            df_obj = m.get("df")

            if sql:
                st.subheader("SQL")
                st.code(sql, language="sql")

            if explanation:
                st.caption(explanation)

            df = _deserialize_df(df_obj)
            if df is None or df.empty:
                st.info("No rows returned.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                try:
                    maybe_render_chart(st, df, chart_spec)
                except Exception:
                    pass


def _serialize_df(df: Optional[pd.DataFrame], max_rows: int) -> Optional[Dict[str, Any]]:
    if df is None:
        return None
    if df.empty:
        return {"columns": list(df.columns), "rows": [], "truncated": False}

    truncated = False
    out_df = df
    if len(df) > max_rows:
        out_df = df.head(max_rows).copy()
        truncated = True

    return {
        "columns": list(out_df.columns),
        "rows": out_df.to_dict(orient="records"),
        "truncated": truncated,
        "total_rows": int(len(df)),
    }


def _deserialize_df(obj: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not obj:
        return None
    cols = obj.get("columns") or []
    rows = obj.get("rows") or []
    try:
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return None


def _stream_text(text: str, chunk: int = 6, delay_s: float = 0.01) -> None:
    placeholder = st.empty()
    acc = ""
    for i in range(0, len(text), chunk):
        acc += text[i : i + chunk]
        placeholder.markdown(acc)
        time.sleep(delay_s)
