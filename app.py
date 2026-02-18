from pathlib import Path

import streamlit as st

from config import get_settings
from state import init_session_state
from tracing.langfuse_tracer import LangfuseTracer

from screens.data_generation_page import render as render_data_generation
from screens.talk_to_data_page import render as render_talk_to_data


settings = get_settings()
DATASETS_ROOT: Path = settings.datasets_root

st.set_page_config(page_title="Data Assistant", layout="wide")

init_session_state(st, settings)
tracer = LangfuseTracer(st)

st.sidebar.title("Data Assistant")
page = st.sidebar.radio(
    label="Navigation",
    options=["Data Generation", "Talk to your data"],
    index=0,
    label_visibility="collapsed",
)

if page == "Data Generation":
    render_data_generation(
        st,
        settings=settings,
        tracer=tracer,
        page=page,
        datasets_root=DATASETS_ROOT,
    )
else:
    render_talk_to_data(
        st,
        settings=settings,
        tracer=tracer,
        page=page,
    )
