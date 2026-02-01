import streamlit as st
import pandas as pd
import json
import random
import inspect
import time

from ddl_parser import parse_ddl_to_schema
from vertex_client import VertexGenAIClient
from data_generator import generate_all_tables

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Data Assistant", layout="wide")

# ----------------------------
# Session state (demo storage)
# ----------------------------
if "tables" not in st.session_state:
    st.session_state.tables = {}  # dict[str, pd.DataFrame]

if "ddl_text" not in st.session_state:
    st.session_state.ddl_text = ""

if "schema" not in st.session_state:
    st.session_state.schema = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None

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
    # Demo tables to show the "Data Preview" UI even before real generation is connected
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


# ----------------------------
# Page: Data Generation
# ----------------------------
if page == "Data Generation":
    st.markdown("###")

    prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

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

        # placeholders (важно: доступны и в try, и в except)
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

                # UI widgets
                progress = st.progress(0)

                started_at = time.time()

                with st.status("Генерация данных в Vertex AI…", expanded=True) as status_ctx:
                    # одна “живая” строка внутри status
                    status_line = status_ctx.empty()
                    status_line.info("Подготовка…")

                    def on_progress(done: int, total: int, table_label: str):
                        # FIX 2: показываем 1-based счётчик таблиц (1/7, 2/7 ...)
                        shown_total = max(1, int(total))
                        shown_done = min(int(done) + 1, shown_total)

                        # прогресс-бар оставляем по done (0-based), чтобы 100% было только в конце
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
                            on_progress=on_progress,
                        )
                    else:
                        # fallback: без колбеков показываем только время
                        elapsed = _format_elapsed(time.time() - started_at)
                        status_line.info(f"Генерация выполняется… | ⏱ {elapsed}")
                        dfs = generate_all_tables(
                            vertex=vertex,
                            ddl_schema=schema,
                            rows_per_table=int(rows_per_table),
                            temperature=float(temperature),
                            max_output_tokens=int(max_tokens),
                        )

                    status_ctx.update(label="Генерация завершена ✅", state="complete", expanded=False)

                progress.progress(100)
                st.success("Готово. Таблицы сгенерированы.")

                st.session_state.tables = dfs

            except Exception as e:
                st.session_state.last_error = f"{e}"

                # 1) главный блок ошибки (с деталями)
                st.error(f"Generation failed: {st.session_state.last_error}")

                # 2) статус в UI: один источник правды
                if status_ctx is not None:
                    status_ctx.update(label="Генерация остановлена из-за ошибки ❌", state="error", expanded=True)

                # 3) показать проблемы парсинга DDL (если есть)
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
    if df is not None:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data yet. Click Generate after uploading a schema.")

    edit_col, btn_col = st.columns([6, 1], vertical_alignment="center")
    with edit_col:
        edit_prompt = st.text_input(
            "",
            placeholder="Enter quick edit instructions...",
            label_visibility="collapsed",
        )
    with btn_col:
        submit_clicked = st.button("Submit")

    if submit_clicked:
        if not edit_prompt.strip():
            st.warning("Please enter edit instructions first.")
        else:
            st.success("Edit request received. Next step: apply edits via Gemini and update the table.")

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