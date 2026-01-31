import streamlit as st
import pandas as pd

from ddl_parser import parse_ddl_to_schema
import json

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

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Data Assistant")
page = st.sidebar.radio(
    label="",
    options=["Data Generation", "Talk to your data"],
    index=0
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

# ----------------------------
# Page: Data Generation
# ----------------------------
if page == "Data Generation":
    # Top card-like block
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

    # Optional extra params (useful later, but not in the sample screenshot)
    with st.expander("More parameters (optional)"):
        rows_per_table = st.number_input("Rows per table", min_value=1, value=1000, step=100)
        seed = st.number_input("Seed", min_value=0, value=0, step=1)
        st.caption("These will be used later for real data generation.")

    st.markdown("###")
    generate_clicked = st.button("Generate", type="primary")

    # Handle "Generate" (still a stub; just shows that DDL was received)
    if generate_clicked:
        if ddl_file is None:
            st.error("Please upload a DDL schema file first.")
        else:
            ddl_text = ddl_file.read().decode("utf-8", errors="ignore")
            st.session_state.ddl_text = ddl_text

            # --- NEW: parse DDL -> schema JSON ---
            try:
                schema = parse_ddl_to_schema(ddl_text)
                st.session_state.schema = schema

                st.success("DDL parsed → schema JSON is ready.")
                with st.expander("Show parsed schema (JSON)"):
                    st.code(json.dumps(schema, ensure_ascii=False, indent=2), language="json")
            except Exception as e:
                st.error(f"Failed to parse DDL: {e}")
                st.stop()
            st.success("DDL schema uploaded. Next step: connect Gemini-based generation.")
            with st.expander("DDL preview"):
                st.code(st.session_state.ddl_text, language="sql")

            # For now, keep demo tables as “generated”
            st.info("Demo data is shown below. Later, this will be real generated data.")

    st.markdown("###")
    st.subheader("Data Preview")

    # Table picker on the right (like in the sample)
    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_right:
        table_names = list(st.session_state.tables.keys()) or ["(no tables)"]
        selected_table = st.selectbox("", options=table_names, label_visibility="collapsed")
    with header_left:
        st.write("")

    # Show selected table
    df = st.session_state.tables.get(selected_table)
    if df is not None:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data yet. Click Generate after uploading a schema.")

    # Quick edit row (input + submit button)
    edit_col, btn_col = st.columns([6, 1], vertical_alignment="center")
    with edit_col:
        edit_prompt = st.text_input("",
                                    placeholder="Enter quick edit instructions...",
                                    label_visibility="collapsed")
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