import pytest
from fastapi.testclient import TestClient

import api
from services import rag_service as svc

client = TestClient(api.app)
BODY = {"question": "How can I explicitly configure the BGP router ID in SONiC?", "role": "network_operator"}


@pytest.fixture(autouse=True)
def _reset():
    api._hits.clear()
    yield


def test_health_is_open():
    assert client.get("/health").json()["status"] == "ok"


def test_ask_ok():
    r = client.post("/ask", json=BODY)
    assert r.status_code == 200 and r.json()["decision"] == "answered"


@pytest.mark.parametrize("patch", [{"role": "admin"}, {"source": "cisco"}, {"question": ""}, {"mode": "other"}])
def test_bad_input_is_rejected(patch):
    assert client.post("/ask", json={**BODY, **patch}).status_code == 422


def test_audit_needs_admin_role():
    assert client.get("/audit", params={"role": "network_operator"}).status_code == 403
    r = client.get("/audit", params={"role": "network_engineer"})
    assert r.status_code == 200 and isinstance(r.json(), list)


def test_feedback_and_stats():
    assert client.post("/feedback", json={"run_id": "r1", "rating": "up"}).json() == {"ok": True}
    assert client.get("/feedback/stats").json()["total"] >= 1
    assert client.post("/feedback", json={"run_id": "r1", "rating": "maybe"}).status_code == 422


def test_api_key_is_enforced(monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "secret")
    assert client.post("/ask", json=BODY).status_code == 401
    assert client.post("/ask", json=BODY, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json=BODY, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200       # probes stay open


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 2)
    codes = [client.get("/kb").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_backend_failure_returns_503_without_internals(monkeypatch):
    def boom(*a, **k): raise RuntimeError("secret internal detail")
    monkeypatch.setattr(svc, "ask", boom)
    r = client.post("/ask", json=BODY)
    assert r.status_code == 503 and "secret internal detail" not in r.text
