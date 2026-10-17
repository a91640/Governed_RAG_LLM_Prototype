import streamlit as st

from services.client import ServiceError, get_client
from services.ui import show_table, sidebar

st.set_page_config(page_title="Audit log", page_icon="🧾", layout="wide")
cfg = sidebar()
st.title("🧾 Audit log")
st.caption("Every request: who (role), what, which steps, which chunks, which decision. Answer text is not stored. Restricted to the network_engineer role.")

try:
    rows = get_client().audit(cfg["role"], 100)
    stats = get_client().feedback_stats()
except ServiceError as exc:
    st.warning(f"{exc}  Switch the role in the sidebar to `network_engineer` to view it.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Requests shown", len(rows))
c2.metric("👍 helpful", stats["up"])
c3.metric("👎 not helpful", stats["down"])
show_table([{"ts": r["ts"], "system": r["system"], "role": r["role"], "question": r["question"], "decision": r["decision"],
             "path": " > ".join(r["path"]), "chunks": len(r["chunk_ids"]), "seconds": r["seconds"]} for r in rows])
