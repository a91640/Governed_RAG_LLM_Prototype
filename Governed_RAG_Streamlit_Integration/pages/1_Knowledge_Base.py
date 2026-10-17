import streamlit as st

from services.client import ServiceError, get_client
from services.ui import show_table, sidebar

st.set_page_config(page_title="Knowledge base", page_icon="📚", layout="wide")
sidebar()
st.title("📚 Knowledge base")


@st.cache_data(ttl=30, show_spinner=False)
def _kb(cache_key: str) -> dict:                   # non-sensitive inventory only; safe to cache with an expiry
    return get_client().kb()


try:
    kb = _kb(get_client().name)
except ServiceError as exc:
    st.error(str(exc))
    st.stop()

vec = kb["vector"]
c1, c2, c3 = st.columns(3)
c1.metric("Chunks stored", kb["total"])
c2.metric("Documents", len(kb["documents"]))
c3.metric("Vectors", f"{vec.get('size') or '?'} · {vec.get('distance') or '?'}")
st.subheader("Chunks per source")
for col, (src, n) in zip(st.columns(len(kb["per_source"])), kb["per_source"].items()):
    col.metric(src, n)
st.subheader("Documents and governance metadata")
show_table(kb["documents"])
st.caption("Every chunk carries: source, product, source_url, license, version, last_reviewed, classification, status, allowed_roles.")
