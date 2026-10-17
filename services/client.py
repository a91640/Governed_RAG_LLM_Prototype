"""Client used by the Streamlit pages.

RAG_API_URL set  -> HttpClient: the UI talks to the secured backend API (recommended, production pattern).
RAG_API_URL empty -> LocalClient: the same backend functions run inside the Streamlit process (one-command local demo).
Both expose the same methods, so the pages do not care which one is active.
"""
from __future__ import annotations

import os

import requests

from services import rag_service as svc


class ServiceError(Exception):
    """A readable error for the UI (never contains secrets or stack traces)."""


class LocalClient:
    name = "in-process (local)"

    def _wrap(self, fn, *a, **k):
        try:
            return fn(*a, **k)
        except svc.AccessDenied as exc:
            raise ServiceError(str(exc))
        except ValueError as exc:
            raise ServiceError(str(exc))
        except Exception as exc:
            raise ServiceError(f"Backend error ({type(exc).__name__}). Is Qdrant running (docker start qdrant) and Ollama open?")

    def health(self): return self._wrap(svc.health)
    def ask(self, question, role, source=None, session_id=None, mode="agentic"): return self._wrap(svc.ask, question, role, source, session_id, mode)
    def kb(self): return self._wrap(svc.kb_overview)
    def audit(self, role, limit=100): return self._wrap(svc.audit_tail, role, limit)
    def feedback(self, run_id, rating, comment=""): return self._wrap(svc.save_feedback, run_id, rating, comment)
    def feedback_stats(self): return self._wrap(svc.feedback_stats)
    def evaluation(self): return self._wrap(svc.latest_evaluation)


class HttpClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 300):
        self.base, self.key, self.timeout = base_url.rstrip("/"), api_key, timeout       # CPU models can be slow
        self.name = f"API {self.base}"

    def _req(self, method: str, path: str, **kw):
        headers = {"X-API-Key": self.key} if self.key else {}
        try:
            r = requests.request(method, self.base + path, headers=headers, timeout=self.timeout, **kw)
        except requests.RequestException as exc:
            raise ServiceError(f"Cannot reach the RAG API at {self.base} ({type(exc).__name__}).")
        if r.status_code == 401:
            raise ServiceError("The API rejected the key (RAG_API_KEY).")
        if r.status_code in (403, 422, 429, 503):
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = ""
            raise ServiceError(str(detail) or f"API returned {r.status_code}.")
        if not r.ok:
            raise ServiceError(f"API returned {r.status_code}.")
        return r.json()

    def health(self): return self._req("GET", "/health")
    def ask(self, question, role, source=None, session_id=None, mode="agentic"):
        return self._req("POST", "/ask", json={"question": question, "role": role, "source": source, "session_id": session_id, "mode": mode})
    def kb(self): return self._req("GET", "/kb")
    def audit(self, role, limit=100): return self._req("GET", "/audit", params={"role": role, "limit": limit})
    def feedback(self, run_id, rating, comment=""): return self._req("POST", "/feedback", json={"run_id": run_id, "rating": rating, "comment": comment})
    def feedback_stats(self): return self._req("GET", "/feedback/stats")
    def evaluation(self): return self._req("GET", "/evaluation")


def _setting(name: str) -> str:
    """Environment variable first, then Streamlit secrets (.streamlit/secrets.toml). Never hard-coded."""
    val = os.getenv(name, "")
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def get_client():
    url = _setting("RAG_API_URL")
    return HttpClient(url, _setting("RAG_API_KEY")) if url else LocalClient()
