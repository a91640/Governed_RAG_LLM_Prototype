"""Streamlit interface: chat with the governed, agentic RAG assistant.

Run:  streamlit run app.py          (local, in-process backend)
  or: RAG_API_URL=http://localhost:8000 streamlit run app.py   (talks to the FastAPI backend, see README_APP.md)
"""
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from services.client import ServiceError, get_client
from services.ui import render_result, sidebar

st.set_page_config(page_title="Governed RAG: Network Operations", page_icon="🛰️", layout="wide")
cfg = sidebar()

st.title("Network Operations Knowledge Assistant")
st.caption("Answers come only from approved, public content, with sources. Use the pages on the left for the knowledge base, audit log and evaluation.")

if cfg["health"].get("status") != "ok":
    st.error(f"The backend cannot reach the knowledge base ({cfg['health'].get('detail', 'unknown')}). "
             "Start Qdrant (`docker start qdrant`) and Ollama, then refresh.")
    st.stop()
if cfg["health"].get("chunks", 0) == 0:
    st.warning("The collection is empty. Run the ingestion section of the notebook first.")

EXAMPLES = ["How can I explicitly configure the BGP router ID in SONiC?",
            "How is BGP modelled in OpenConfig, and how does SONiC choose the BGP router ID?",
            "My BGP session is down. Please help.",
            "What is the admin password of our production core router?"]

if not st.session_state.messages:
    st.info("Ask a question, or describe a problem. The agent may ask you for missing details first.")
    for col, ex in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if col.button(ex, key=f"ex_{ex[:24]}", width="stretch"):
            st.session_state.pending = ex
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["result"], cfg["show_steps"])
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

with st.sidebar:
    st.markdown("## Suggested questions")
    st.caption("Choose an example or ask your own question.")

    with st.expander("OpenConfig and automation", expanded=True):
        if st.button("What is OpenConfig and why is it useful?"):
            st.session_state.selected_question = (
                "What is OpenConfig and why is it useful for network automation?"
            )
        if st.button("Explain the OpenConfig interfaces model"):
            st.session_state.selected_question = (
                "Explain the OpenConfig interfaces model."
            )
        if st.button("What telemetry can OpenConfig provide?"):
            st.session_state.selected_question = (
                "What network telemetry data can OpenConfig model?"
            )

    with st.expander("SONiC and network operations"):
        if st.button("What is SONiC?"):
            st.session_state.selected_question = (
                "What is SONiC and where is it used?"
            )
        if st.button("How does SONiC support switch operations?"):
            st.session_state.selected_question = (
                "How can SONiC support switch operations?"
            )
        if st.button("OpenConfig versus vendor YANG"):
            st.session_state.selected_question = (
                "What is the difference between OpenConfig and vendor-specific YANG models?"
            )

    with st.expander("Protocols and standards"):
        if st.button("Explain BGP"):
            st.session_state.selected_question = (
                "Explain the role of BGP in modern networks."
            )
        if st.button("BGP versus OSPF"):
            st.session_state.selected_question = (
                "How does OSPF differ from BGP?"
            )
        if st.button("What is NETCONF?"):
            st.session_state.selected_question = (
                "Explain the purpose of NETCONF and YANG."
            )
        if st.button("What does RFC 4271 define?"):
            st.session_state.selected_question = (
                "What does RFC 4271 define?"
            )
        if "selected_question" not in st.session_state:
            st.session_state.selected_question = None
    with st.expander("4G and 5G core networks"):
        if st.button("What is Open5GS?"):
            st.session_state.selected_question = (
                "What is Open5GS and which 4G/5G core functions does it provide?"
            )
        if st.button("Explain the 5G core"):
            st.session_state.selected_question = (
                "What are the main components of a 5G core network?"
            )
typed_question = st.chat_input(
    "Ask a question about OpenConfig, SONiC, Open5GS, RFCs, or network protocols..."
)

prompt = st.session_state.selected_question or typed_question
st.session_state.selected_question = None
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching approved knowledge…"):
                result = get_client().ask(prompt, cfg["role"], cfg["source"], cfg["session_id"], cfg["mode"])
            render_result(result, cfg["show_steps"])
            st.session_state.messages.append({"role": "assistant", "result": result})
        except ServiceError as exc:
            st.error(str(exc))
