import streamlit as st
import pandas as pd
import json
import random
import inspect
import time

from ddl_parser import parse_ddl_to_schema
from vertex_client import VertexGenAIClient
from data_generator import generate_all_tables
from data_editor import (
    build_table_patch_schema,
    build_prompt_for_table_patch,
    apply_patch_to_df,
)

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Data Assistant", layout="wide")

# ----------------------------
# Session state
# ----------------------------
if "tables" not in st.session_state:
    st.session_state.tables = {}  # dict[str, pd.DataFrame]

if "ddl_text" not in st.session_state:
    st.session_state.ddl_text = ""

if "schema" not in st.session_state:
    st.session_state.schema = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "dataset_prompt" not in st.session_state:
    st.session_state.dataset_prompt = ""


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Data Assistant")
page = st.sidebar.radio(
    label="",
    options=["Data Generation", "Talk to your data"],
    index=0,
)

# ----------------------------
# Helpers
# ----------------------------
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
    """
    Для таблицы table_name находим все FK (child_col -> parent_table.parent_col)
    и строим allowed значения на основании уже имеющихся parent DataFrame в session_state.tables.
    """
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


# ----------------------------
# Page: Data Generation
# ----------------------------
if page == "Data Generation":
    st.markdown("###")

    # глобальная инструкция для датасета (используется при генерации)
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

    with st.expander("More parameters (optional)"):
        rows_per_table = st.number_input("Rows per table", min_value=1, value=1000, step=100)
        seed = st.number_input("Seed", min_value=0, value=0, step=1)

        st.markdown("**Vertex settings**")
        project = st.text_input("Project", value="gd-gcp-gridu-genai")
        location = st.text_input("Location", value="europe-west1")
        model = st.text_input("Model", value="gemini-2.0-flash-001")

        st.caption("These will be used for real data generation.")

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

            # 1) Parse DDL -> schema JSON
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

            # 2) Generate data via Vertex with progress UI
            try:
                if int(seed) != 0:
                    random.seed(int(seed))

                vertex = VertexGenAIClient(
                    project=project.strip(),
                    location=location.strip(),
                    model=model.strip(),
                )

                tables_dict = (schema or {}).get("tables", {}) or {}
                total_tables = len(tables_dict)

                progress = st.progress(0)
                started_at = time.time()

                with st.status("Генерация данных в Vertex AI…", expanded=True) as status_ctx:
                    status_line = status_ctx.empty()
                    status_line.info("Подготовка…")

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
                        status_line.info(f"Генерация: {shown_done}/{shown_total} — {table_label} | ⏱ {elapsed}")

                    if _supports_on_progress(generate_all_tables):
                        dfs = generate_all_tables(
                            vertex=vertex,
                            ddl_schema=schema,
                            rows_per_table=int(rows_per_table),
                            temperature=float(temperature),
                            max_output_tokens=int(max_tokens),
                            dataset_prompt=str(dataset_prompt or ""),
                            on_progress=on_progress,
                        )
                    else:
                        elapsed = _format_elapsed(time.time() - started_at)
                        status_line.info(f"Генерация выполняется… | ⏱ {elapsed}")
                        dfs = generate_all_tables(
                            vertex=vertex,
                            ddl_schema=schema,
                            rows_per_table=int(rows_per_table),
                            temperature=float(temperature),
                            max_output_tokens=int(max_tokens),
                            dataset_prompt=str(dataset_prompt or ""),
                        )

                    status_ctx.update(label="Генерация завершена ✅", state="complete", expanded=False)

                progress.progress(100)
                st.success("Готово. Таблицы сгенерированы.")

                st.session_state.tables = dfs

            except Exception as e:
                st.session_state.last_error = f"{e}"
                st.error(f"Generation failed: {st.session_state.last_error}")

                if status_ctx is not None:
                    status_ctx.update(label="Генерация остановлена из-за ошибки ❌", state="error", expanded=True)

                if st.session_state.schema and st.session_state.schema.get("errors"):
                    with st.expander("DDL parser issues (schema['errors'])"):
                        st.code(
                            json.dumps(st.session_state.schema["errors"], ensure_ascii=False, indent=2),
                            language="json",
                        )

    st.markdown("###")
    st.subheader("Data Preview")

    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_right:
        table_names = list(st.session_state.tables.keys()) or ["(no tables)"]
        selected_table = st.selectbox("", options=table_names, label_visibility="collapsed")
    with header_left:
        st.write("")

    df = st.session_state.tables.get(selected_table)
    if df is None:
        st.info("No data yet. Click Generate after uploading a schema.")
        st.stop()

    # Плейсхолдер для "живого" обновления таблицы
    df_placeholder = st.empty()
    df_placeholder.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Edit selected table (LLM patch)")

    # Оставляем только prompt
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
                    # Vertex settings: берём из UI (expander "More parameters")
                    vertex = VertexGenAIClient(
                        project=project.strip(),
                        location=location.strip(),
                        model=model.strip(),
                    )

                    table_meta = _get_table_meta(selected_table)
                    if not table_meta:
                        st.error(f"Table '{selected_table}' not found in schema.")
                        st.stop()

                    # фиксируем параметры "по умолчанию", раз UI убрали
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

                        # сохраняем и сразу "вживую" обновляем таблицу
                        st.session_state.tables[selected_table] = new_df
                        df = new_df
                        df_placeholder.dataframe(df, use_container_width=True, hide_index=True)

                        sctx.update(label="Edit applied ✅", state="complete", expanded=False)

                    st.success(f"Edit applied to '{selected_table}'.")
                    st.caption(f"Time: {_format_elapsed(time.time() - started_at)}")

                    # Patch JSON НЕ показываем (по твоему требованию)

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

# ----------------------------
# Page: Talk to your data
# ----------------------------
else:
    st.subheader("Talk to your data")

    dataset = st.selectbox("Dataset", options=["(demo) current session tables"], index=0)

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