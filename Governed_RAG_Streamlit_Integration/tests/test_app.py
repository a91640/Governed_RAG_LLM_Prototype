import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api
from services import rag_service as svc

ROOT = Path(__file__).resolve().parents[1]


def app(page="app.py"):
    return AppTest.from_file(str(ROOT / page), default_timeout=90)


def captions(at):
    return [c.value for c in at.caption]


def test_chat_clarification_refusal_and_feedback():
    at = app().run()
    assert not at.exception
    at.chat_input[0].set_value("How can I explicitly configure the BGP router ID in SONiC?").run()
    assert not at.exception and any("answered" in c for c in captions(at))
    at.chat_input[0].set_value("My BGP session is down. Please help.").run()
    assert "I need a few details" in " ".join(m.value for m in at.markdown)
    at.chat_input[0].set_value("SONiC switch, neighbor stuck in Active").run()
    at.chat_input[0].set_value("What is the admin password of our production core router?").run()
    assert any("refused (policy)" in c for c in captions(at))
    before = svc.feedback_stats()["up"]
    [b for b in at.button if b.label == "👍"][0].click().run()
    assert svc.feedback_stats()["up"] == before + 1


def test_guest_role_is_refused_by_the_backend():
    at = app()
    at.session_state["role"] = "guest"
    at.run()
    at.chat_input[0].set_value("How can I explicitly configure the BGP router ID in SONiC?").run()
    assert not at.exception and any("not in the approved corpus" in c for c in captions(at))


def test_plain_mode_runs():
    at = app()
    at.session_state["mode_label"] = "Plain RAG (baseline)"
    at.run()
    at.chat_input[0].set_value("What is BMP?").run()
    assert not at.exception and any("plain RAG" in c for c in captions(at))


def test_pages_render_and_audit_is_restricted():
    kb = app("pages/1_Knowledge_Base.py").run()
    assert not kb.exception and [m.value for m in kb.metric][0] == "122"
    ev = app("pages/3_Evaluation.py").run()
    assert not ev.exception
    op = app("pages/2_Audit_Log.py").run()
    assert not op.exception and any("network_engineer" in w.value for w in op.warning)
    eng = app("pages/2_Audit_Log.py")
    eng.session_state["role"] = "network_engineer"
    eng.run()
    assert not eng.exception and not eng.warning


def test_everything_renders_without_pandas(monkeypatch):
    """Windows policies can block pandas' DLLs. The UI must fall back to Markdown tables."""
    monkeypatch.setitem(sys.modules, "pandas", None)
    kb = app("pages/1_Knowledge_Base.py").run()
    assert not kb.exception and any("| source |" in m.value for m in kb.markdown)
    at = app().run()
    at.chat_input[0].set_value("What is BMP?").run()
    assert not at.exception


def test_ui_works_through_the_http_api(monkeypatch):
    """Same UI, but talking to the FastAPI backend instead of running in-process."""
    http = TestClient(api.app)

    class R:
        def __init__(self, r): self._r, self.status_code, self.ok = r, r.status_code, r.is_success
        def json(self): return self._r.json()

    monkeypatch.setattr("services.client.requests.request", lambda method, url, **kw: R(http.request(
        method, url.replace("http://rag.test", ""), headers=kw.get("headers"), json=kw.get("json"), params=kw.get("params"))))
    monkeypatch.setenv("RAG_API_URL", "http://rag.test")
    at = app().run()
    assert not at.exception and any("API http://rag.test" in m.value for m in at.sidebar.markdown)
    at.chat_input[0].set_value("How can I explicitly configure the BGP router ID in SONiC?").run()
    assert not at.exception and any("answered" in c for c in captions(at))
