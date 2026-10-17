"""Backend logic for the governed RAG assistant.

Shared by the FastAPI service (api.py) and the in-process client (services/client.py),
so the Streamlit UI behaves the same whether it talks to the API or runs everything locally.
The UI never touches Qdrant, the LLM or the audit files directly: it only calls these functions.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import agentic_rag as ag
import rag_demo
from qdrant_client import models

ROLES = ["network_operator", "network_engineer", "guest"]
ADMIN_ROLES = {"network_engineer"}              # may open the audit log (prototype: role is asserted by the client)
SOURCES = ["openconfig", "sonic", "open5gs", "ietf"]
MAX_QUESTION_CHARS = 2000
SESSION_TTL_S = 3600
FEEDBACK_PATH = Path("data/feedback.jsonl")
EVAL_GLOB = "data/agentic_vs_plain_*.csv"
METRICS = ["abstain_ok", "source_hit", "has_citation", "keyword_hit", "clarify_ok", "seconds", "llm_calls"]

_sessions: dict[str, tuple[float, ag.Session]] = {}
_lock = threading.Lock()


class AccessDenied(Exception):
    pass


def _client():
    return ag._vector_store().client


# ----------------------------------------------------------------------------- health
def health() -> dict[str, Any]:
    base = {"collection": rag_demo.COLLECTION, "model": ag.CHAT_MODEL,
            "tracing": os.getenv("LANGSMITH_TRACING", "").lower() == "true"}
    try:
        return {"status": "ok", "chunks": _client().count(rag_demo.COLLECTION).count, **base}
    except Exception as exc:                                    # Qdrant down, collection missing...
        return {"status": "error", "detail": type(exc).__name__, **base}


# ----------------------------------------------------------------------------- ask
def _session(session_id: str, role: str, source: str | None) -> ag.Session:
    key = f"{session_id}|{role}|{source}"
    now = time.time()
    with _lock:
        for k in [k for k, (t, _) in _sessions.items() if now - t > SESSION_TTL_S]:
            _sessions.pop(k, None)                              # expire old conversations
        sess = _sessions.get(key, (now, None))[1] or ag.Session(role, source)
        _sessions[key] = (now, sess)
    return sess


def ask(question: str, role: str, source: str | None = None, session_id: str | None = None,
        mode: str = "agentic") -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("The question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"The question is longer than {MAX_QUESTION_CHARS} characters.")
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    if source in ("", "auto", "Auto"):
        source = None
    if source is not None and source not in SOURCES:
        raise ValueError(f"Unknown source: {source}")
    if mode == "plain":
        res = ag.plain_rag(question, role, source, audit=True)
    else:
        res = _session(session_id or uuid.uuid4().hex[:12], role, source).say(question)
    return {k: res.get(k) for k in ("run_id", "system", "role", "answer", "decision", "path", "questions", "seconds",
                                    "llm_calls", "search_calls", "top_score", "sources", "subqueries")} | {
        "evidence": [{"product": h["product"], "file": os.path.basename(h["file"]), "score": h["score"], "text": h["text"][:300]}
                     for h in res.get("evidence", [])]}


# ----------------------------------------------------------------------------- knowledge base inventory
def kb_overview() -> dict[str, Any]:
    client = _client()
    total = client.count(rag_demo.COLLECTION).count
    per_source = {}
    for src in SOURCES:
        flt = models.Filter(must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=src))])
        per_source[src] = client.count(rag_demo.COLLECTION, count_filter=flt).count
    docs, offset = {}, None
    while len(docs) < 2000:
        points, offset = client.scroll(rag_demo.COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False)
        for p in points:
            m = (p.payload or {}).get("metadata", {})
            key = m.get("source_file", "?")
            d = docs.setdefault(key, {"source": m.get("source"), "file": os.path.basename(key), "version": m.get("version"),
                                      "classification": m.get("classification"), "status": m.get("status"),
                                      "allowed_roles": ", ".join(m.get("allowed_roles", [])),
                                      "last_reviewed": m.get("last_reviewed"), "chunks": 0})
            d["chunks"] += 1
        if offset is None:
            break
    vec = {"size": None, "distance": None}
    try:
        v = client.get_collection(rag_demo.COLLECTION).config.params.vectors
        vec = {"size": getattr(v, "size", None), "distance": str(getattr(v, "distance", "")).split(".")[-1].lower()}
    except Exception:
        pass
    return {"total": total, "per_source": per_source, "documents": list(docs.values()), "vector": vec}


# ----------------------------------------------------------------------------- audit log (admin only)
def audit_tail(role: str, limit: int = 100) -> list[dict[str, Any]]:
    if role not in ADMIN_ROLES:
        raise AccessDenied("The audit log is restricted to the network_engineer role.")
    path = Path(ag.AUDIT_PATH)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 500)):]
    return [json.loads(x) for x in reversed(lines) if x.strip()]


# ----------------------------------------------------------------------------- user feedback
def save_feedback(run_id: str, rating: str, comment: str = "") -> None:
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": str(run_id)[:32],
           "rating": rating, "comment": (comment or "")[:500]}
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def feedback_stats() -> dict[str, int]:
    up = down = 0
    if FEEDBACK_PATH.exists():
        for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line).get("rating")
                up += r == "up"
                down += r == "down"
    return {"up": up, "down": down, "total": up + down}


# ----------------------------------------------------------------------------- evaluation results (latest notebook comparison)
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean_table(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, items in sorted(groups.items()):
        row = dict(zip(keys, g))
        row["n"] = len(items)
        for m in METRICS:
            vals = [v for v in (_num(i.get(m)) for i in items) if v is not None]
            row[m] = round(sum(vals) / len(vals), 2) if vals else None
        out.append(row)
    return out


def latest_evaluation() -> dict[str, Any]:
    files = sorted(glob.glob(EVAL_GLOB), key=os.path.getmtime)
    if not files:
        return {"file": None}
    with open(files[-1], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {"file": os.path.basename(files[-1]), "n_rows": len(rows),
            "by_system": _mean_table(rows, ["system"]), "by_category": _mean_table(rows, ["category", "system"]),
            "rows": [{k: r.get(k) for k in ("system", "id", "category", "decision", "abstain_ok", "source_hit", "question")}
                     for r in rows[:300]]}
