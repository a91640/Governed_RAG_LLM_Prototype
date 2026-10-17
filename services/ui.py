"""Shared Streamlit helpers: sidebar, tables (with a no-pandas fallback), result rendering, feedback buttons."""
from __future__ import annotations

import uuid

import streamlit as st

from services.client import ServiceError, get_client

ROLES = ["network_operator", "network_engineer", "guest"]
SOURCE_CHOICES = ["Auto (agent decides)", "openconfig", "sonic", "open5gs", "ietf"]
MODES = {"Agentic RAG": "agentic", "Plain RAG (baseline)": "plain"}
BADGES = {"answered": "🟢 answered", "needs_clarification": "🟡 needs details",
          "refused_policy": "🔴 refused (policy)", "refused_no_evidence": "🟠 not in the approved corpus"}


# ----------------------------------------------------------------------------- tables
def md_table(rows: list[dict], columns: list[str] | None = None) -> str:
    cols = columns or list(rows[0].keys())
    esc = lambda v: str(v if v is not None else "").replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(esc(r.get(c)) for c in cols) + " |" for r in rows])


def show_table(rows: list[dict], columns: list[str] | None = None) -> None:
    """st.dataframe when pandas works; a Markdown table otherwise (for example when the OS blocks pandas' DLLs)."""
    if not rows:
        st.caption("No data yet.")
        return
    try:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows, columns=columns), width="stretch", hide_index=True)
    except Exception:
        st.markdown(md_table(rows, columns))


# ----------------------------------------------------------------------------- state and sidebar
@st.cache_data(ttl=15, show_spinner=False)
def _health(cache_key: str) -> dict:
    return get_client().health()


def backend_health() -> dict:
    client = get_client()
    return _health(client.name)


def sidebar() -> dict:
    ss = st.session_state
    ss.setdefault("role", ROLES[0])
    ss.setdefault("source_label", SOURCE_CHOICES[0])
    ss.setdefault("mode_label", "Agentic RAG")
    ss.setdefault("show_steps", True)
    ss.setdefault("messages", [])
    ss.setdefault("session_id", uuid.uuid4().hex[:12])
    with st.sidebar:
        st.title("🛰️ Governed RAG")
        st.caption("Read-only assistant over an approved, public network-operations corpus.")
        ss.role = st.selectbox("Your role (demo identity)", ROLES, index=ROLES.index(ss.role),
                               help="The backend only returns chunks this role may see. 'guest' has no access. Real deployments use SSO.")
        ss.source_label = st.selectbox("Source", SOURCE_CHOICES, index=SOURCE_CHOICES.index(ss.source_label),
                                       help="Auto lets the agent route each question to the right source.")
        ss.mode_label = st.radio("Mode", list(MODES), index=list(MODES).index(ss.mode_label),
                                 help="Plain RAG = one search, one answer. Agentic adds guard, clarification, routing, evidence grading and one retry.")
        ss.show_steps = st.toggle("Show how the agent decided", value=ss.show_steps)
        st.divider()
        h = backend_health()
        st.markdown(f"**Backend:** {'🟢 ok' if h.get('status') == 'ok' else '🔴 unavailable'}  \n**Connection:** {get_client().name}")
        st.markdown(f"**Model:** `{h.get('model', '?')}`  \n**Collection:** `{h.get('collection', '?')}`  \n"
                    f"**LangSmith tracing:** {'🟢 on' if h.get('tracing') else '⚪ off'}")
        st.caption("Public corpus only. Do not enter passwords, keys or customer data.")
        if st.button("Clear conversation", width="stretch"):
            ss.messages, ss.session_id = [], uuid.uuid4().hex[:12]
            st.rerun()
    ctx = (ss.role, ss.source_label)
    if ss.get("ctx") != ctx:                                      # a new role or source starts a new conversation
        ss.ctx, ss.session_id = ctx, uuid.uuid4().hex[:12]
    source = None if ss.source_label.startswith("Auto") else ss.source_label
    return {"role": ss.role, "source": source, "mode": MODES[ss.mode_label], "show_steps": ss.show_steps,
            "session_id": ss.session_id, "health": h}


# ----------------------------------------------------------------------------- rendering
def render_result(res: dict, steps: bool) -> None:
    st.markdown(res["answer"])
    st.caption(f"{BADGES.get(res['decision'], res['decision'])} · {res['seconds']}s · {res['llm_calls']} LLM calls · "
               f"{res['search_calls']} searches · {res['system']} RAG")
    if res["sources"]:
        with st.expander(f"Sources ({len(res['sources'])})", expanded=True):
            for s in res["sources"]:
                st.markdown(f"**[{s['n']}] {s['product']}** · `{s['file']}` · score {s['score']}  \n{s['url']}")
    if steps:
        with st.expander("How the agent decided"):
            st.write("**Path:** " + " → ".join(res["path"]))
            for sq in res.get("subqueries", []):
                st.write(f"Search `{sq['source']}`: {sq['query']}")
            if res.get("evidence"):
                st.write("**Evidence sent to the model**")
                show_table([{"product": h["product"], "file": h["file"], "score": h["score"], "text": h["text"][:160]}
                            for h in res["evidence"]])
    if res["decision"] in ("answered", "refused_no_evidence"):
        _feedback_buttons(res["run_id"])


def _feedback_buttons(run_id: str) -> None:
    done = st.session_state.setdefault("feedback_done", {})
    c1, c2, c3 = st.columns([1, 1, 8])
    if run_id in done:
        c1.caption("Thanks ✓")
        return
    for col, rating, label in ((c1, "up", "👍"), (c2, "down", "👎")):
        if col.button(label, key=f"fb_{rating}_{run_id}", help="Was this answer helpful?"):
            try:
                get_client().feedback(run_id, rating)
                done[run_id] = rating
                st.rerun()
            except ServiceError as exc:
                c3.warning(str(exc))
