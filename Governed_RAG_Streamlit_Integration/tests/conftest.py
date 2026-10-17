"""Offline test environment: in-memory Qdrant, hashed bag-of-words embeddings and a scripted fake LLM.
These tests check wiring, access control, safety rules and the UI. They do NOT measure answer quality."""
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                          # the corpus lives in data/raw (relative path)

from langchain_core.embeddings import Embeddings          # noqa: E402
from langchain_qdrant import QdrantVectorStore            # noqa: E402
from qdrant_client import QdrantClient, models            # noqa: E402

import agentic_rag as ag                                  # noqa: E402
import rag_demo                                           # noqa: E402
from services import rag_service as svc                   # noqa: E402


class BoW(Embeddings):
    def _v(self, t):
        v = np.zeros(768)
        for w in re.findall(r"[a-z0-9_]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 768] += 1
        return (v / (np.linalg.norm(v) or 1)).tolist()

    def embed_documents(self, ts): return [self._v(t) for t in ts]
    def embed_query(self, t): return self._v(t)


class _Msg:
    def __init__(self, c): self.content = c


class FakeLLM:
    def invoke(self, p):
        if "intake step" in p:
            if "Details from the user" in p:
                return _Msg(json.dumps({"needs_clarification": False, "questions": []}))
            return _Msg(json.dumps({"needs_clarification": True, "questions": ["Which platform: SONiC or another?", "What state does the neighbor show?"]}))
        if "route questions" in p: return _Msg("not json")
        if "check search results" in p: return _Msg(json.dumps({"sufficient": True, "reason": "test"}))
        if "Rewrite the question" in p: return _Msg("bgp router id")
        return _Msg("The context describes this [1]. A second detail follows [2].")


@pytest.fixture(scope="session", autouse=True)
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rag")
    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    store = QdrantVectorStore(client=client, collection_name="t", embedding=BoW())
    docs, _ = rag_demo.load_and_validate()
    chunks, ids = rag_demo.make_chunks(docs)
    store.add_documents(chunks, ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, i)) for i in ids])
    rag_demo.COLLECTION = "t"
    ag._store, ag._LLM_OVERRIDE, ag.MIN_SCORE = store, FakeLLM(), 0.0
    ag.AUDIT_PATH = tmp / "audit.jsonl"
    svc.FEEDBACK_PATH = tmp / "feedback.jsonl"
    svc.EVAL_GLOB = str(tmp / "agentic_vs_plain_*.csv")
    yield tmp


@pytest.fixture
def tmp_data(env):
    return env


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    import streamlit as st
    st.cache_data.clear()
