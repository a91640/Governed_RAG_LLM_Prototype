import pytest

from services import rag_service as svc


def test_health():
    h = svc.health()
    assert h["status"] == "ok" and h["chunks"] == 122


def test_answer_has_sources_and_trimmed_evidence():
    r = svc.ask("How can I explicitly configure the BGP router ID in SONiC?", "network_operator")
    assert r["decision"] == "answered" and r["sources"]
    assert all(len(e["text"]) <= 300 for e in r["evidence"])


def test_policy_refusal_uses_no_model_and_no_search():
    r = svc.ask("What is the admin password of our production core router?", "network_operator")
    assert r["decision"] == "refused_policy" and r["llm_calls"] == 0 and r["search_calls"] == 0


@pytest.mark.parametrize("q,role,source", [("", "network_operator", None), ("x" * 2001, "network_operator", None),
                                           ("What is BGP?", "admin", None), ("What is BGP?", "network_operator", "cisco")])
def test_input_validation(q, role, source):
    with pytest.raises(ValueError):
        svc.ask(q, role, source)


def test_role_without_access_gets_no_evidence():
    r = svc.ask("How can I explicitly configure the BGP router ID in SONiC?", "guest")
    assert r["decision"] == "refused_no_evidence" and r["evidence"] == []


def test_clarification_conversation():
    r1 = svc.ask("My BGP session is down. Please help.", "network_operator", None, "s1")
    assert r1["decision"] == "needs_clarification" and r1["questions"]
    r2 = svc.ask("SONiC switch, neighbor stuck in Active", "network_operator", None, "s1")
    assert r2["decision"] == "answered"


def test_plain_mode():
    r = svc.ask("What is BMP?", "network_operator", None, None, "plain")
    assert r["system"] == "plain"


def test_kb_overview():
    kb = svc.kb_overview()
    assert kb["total"] == 122 and sum(kb["per_source"].values()) == 122
    assert kb["documents"] and all(d["classification"] == "public" for d in kb["documents"])


def test_audit_is_admin_only():
    svc.ask("What is BMP?", "network_operator")
    with pytest.raises(svc.AccessDenied):
        svc.audit_tail("network_operator")
    rows = svc.audit_tail("network_engineer", 10)
    assert rows and {"ts", "role", "decision", "path", "chunk_ids"} <= set(rows[0])
    assert "answer" not in rows[0]                      # answer text is never stored in the audit log


def test_feedback_round_trip():
    before = svc.feedback_stats()
    svc.save_feedback("abc123", "up")
    svc.save_feedback("abc124", "down", "wrong version")
    after = svc.feedback_stats()
    assert after["up"] == before["up"] + 1 and after["down"] == before["down"] + 1
    with pytest.raises(ValueError):
        svc.save_feedback("x", "maybe")


def test_latest_evaluation(tmp_data):
    assert svc.latest_evaluation() == {"file": None}
    (tmp_data / "agentic_vs_plain_test.csv").write_text(
        "system,id,category,decision,abstain_ok,source_hit,has_citation,keyword_hit,clarify_ok,seconds,llm_calls,question\n"
        "plain,1,single,answered,1,1,1,1,,10,1,q1\nplain,2,help,answered,0,,,,0,12,1,q2\n"
        "agentic,1,single,answered,1,1,1,1,,30,3,q1\nagentic,2,help,needs_clarification,1,,,,1,20,1,q2\n", encoding="utf-8")
    ev = svc.latest_evaluation()
    by = {r["system"]: r for r in ev["by_system"]}
    assert by["plain"]["abstain_ok"] == 0.5 and by["agentic"]["clarify_ok"] == 1.0 and by["agentic"]["llm_calls"] == 2.0


def test_definition_questions_are_not_sent_to_the_clarifier():
    r = svc.ask("What is BMP?", "network_operator")
    assert r["decision"] == "answered" and "clarify:not-needed" in r["path"]
    assert svc.ask("Tell me about the network.", "network_operator")["decision"] == "needs_clarification"
