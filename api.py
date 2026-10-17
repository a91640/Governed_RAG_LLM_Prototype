"""FastAPI backend for the Streamlit interface.   Run:  uvicorn api:app --port 8000

The API owns the RAG workflow, authorization and logging; the Streamlit UI only displays results.
Settings come from environment variables (see README_APP.md), never from the UI code.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:                                   # python-dotenv is optional
    pass

import rag_demo

rag_demo.QDRANT_URL = os.getenv("QDRANT_URL", rag_demo.QDRANT_URL)        # lets Docker point at the qdrant container
if os.getenv("LANGSMITH_API_KEY"):                                       # tracing is on only if a key is configured
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_PROJECT", "network-ops-agentic-rag")

from services import rag_service as svc  # noqa: E402

log = logging.getLogger("rag_api")
API_KEY = os.getenv("RAG_API_KEY", "")
RATE_LIMIT_PER_MIN = int(os.getenv("RAG_RATE_LIMIT_PER_MIN", "30"))
_hits: dict[str, deque] = defaultdict(deque)

app = FastAPI(title="Governed RAG API", version="1.0")


def guard(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
    """Shared-secret check plus a simple per-client rate limit."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    who = x_api_key or (request.client.host if request.client else "unknown")
    now, q = time.time(), _hits[who]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    q.append(now)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=svc.MAX_QUESTION_CHARS)
    role: Literal["network_operator", "network_engineer", "guest"]
    source: Optional[Literal["openconfig", "sonic", "open5gs", "ietf"]] = None
    session_id: Optional[str] = Field(default=None, max_length=64)
    mode: Literal["agentic", "plain"] = "agentic"


class FeedbackRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=32)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=500)


@app.get("/health")
def health():
    return svc.health()                             # open on purpose: load balancers and probes use it


@app.post("/ask", dependencies=[Depends(guard)])
def ask(req: AskRequest):
    try:
        return svc.ask(req.question, req.role, req.source, req.session_id, req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:                        # Qdrant / Ollama unavailable: log the type, do not leak internals
        log.exception("ask failed")
        raise HTTPException(status_code=503, detail=f"Backend dependency unavailable ({type(exc).__name__}).")


@app.get("/kb", dependencies=[Depends(guard)])
def kb():
    try:
        return svc.kb_overview()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge base unavailable ({type(exc).__name__}).")


@app.get("/audit", dependencies=[Depends(guard)])
def audit(role: str, limit: int = 100):
    try:
        return svc.audit_tail(role, limit)
    except svc.AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@app.post("/feedback", dependencies=[Depends(guard)])
def feedback(req: FeedbackRequest):
    svc.save_feedback(req.run_id, req.rating, req.comment)
    return {"ok": True}


@app.get("/feedback/stats", dependencies=[Depends(guard)])
def feedback_stats():
    return svc.feedback_stats()


@app.get("/evaluation", dependencies=[Depends(guard)])
def evaluation():
    return svc.latest_evaluation()
