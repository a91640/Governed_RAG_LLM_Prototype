import streamlit as st

from services.client import ServiceError, get_client
from services.ui import show_table, sidebar

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")
sidebar()
st.title("📊 Evaluation: plain RAG vs agentic RAG")

try:
    ev = get_client().evaluation()
except ServiceError as exc:
    st.error(str(exc))
    st.stop()

if not ev.get("file"):
    st.info("No comparison found yet. Run section 8 of `Agentic_RAG_Demo.ipynb`; it saves a CSV in `data/`.")
    st.stop()
st.caption(f"Latest comparison: `{ev['file']}` ({ev['n_rows']} rows), created by the notebook.")
st.subheader("By system")
show_table(ev["by_system"])
st.subheader("By question type")
show_table(ev["by_category"])
with st.expander("All results"):
    show_table(ev["rows"])
st.caption("abstain_ok: answered or refused correctly · source_hit: expected source retrieved · has_citation: answer cites [n] · "
           "clarify_ok: asked for details when it should · seconds / llm_calls: the cost of control.")
