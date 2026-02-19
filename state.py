from __future__ import annotations

from typing import Any

from postgres_client import PostgresClient, PostgresConfig


def init_session_state(st: Any, settings: Any) -> None:
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

    if "trace_id" not in st.session_state:
        st.session_state.trace_id = None

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "chat_max_history" not in st.session_state:
        st.session_state.chat_max_history = 50

    if "pg" not in st.session_state:
        st.session_state.pg = PostgresClient(
            PostgresConfig(
                host=settings.pg_host,
                port=int(settings.pg_port),
                dbname=settings.pg_db,
                user=settings.pg_user,
                password=settings.pg_password,
            )
        )

    if "langfuse" not in st.session_state:
        st.session_state.langfuse = None
        try:
            from langfuse import Langfuse

            if settings.langfuse_public_key and settings.langfuse_secret_key:
                st.session_state.langfuse = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
        except Exception:
            st.session_state.langfuse = None
