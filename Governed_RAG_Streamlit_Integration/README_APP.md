# Streamlit interface for the governed RAG assistant

Follows the implementation manual: **the UI displays; the backend decides.** Streamlit never touches Qdrant, the model or the audit files.

```
Streamlit pages ──► services/client.py ──► (A) in-process  services/rag_service.py          (one command, local demo)
                                       └─► (B) HTTP         api.py (FastAPI) ► rag_service  (production pattern)
                                                                 └─► agentic_rag.py ► Qdrant + Ollama
```

## Files
| Path | Role |
|---|---|
| `app.py` | Chat page: question, sources, "how the agent decided", 👍/👎 feedback |
| `pages/1_Knowledge_Base.py` | Inventory: chunks per source, documents, governance metadata |
| `pages/2_Audit_Log.py` | Every request, who/what/which steps. **network_engineer role only** |
| `pages/3_Evaluation.py` | Plain vs agentic results from the notebook CSV |
| `services/rag_service.py` | Backend logic (ask, KB, audit, feedback, evaluation) |
| `services/client.py` | `LocalClient` or `HttpClient`, chosen by `RAG_API_URL` |
| `services/ui.py` | Sidebar, tables (Markdown fallback if pandas is blocked), result rendering |
| `api.py` | FastAPI: API key, rate limit, input validation, 503 without internals |
| `.streamlit/config.toml` | Theme; usage stats and e-mail prompt off |
| `tests/` | 32 offline tests (service, API, UI) |

## Run it
Prerequisites: `docker start qdrant`, Ollama running, collection ingested.

```powershell
python -m pip install -r requirements-app.txt
python -m streamlit run app.py                    # (A) one command
```
Two-process mode (B):
```powershell
$env:RAG_API_KEY = "a-long-random-value"          # PowerShell; use a .env file for permanence
python -m uvicorn api:app --port 8000             # terminal 1
$env:RAG_API_URL = "http://localhost:8000"
python -m streamlit run app.py                    # terminal 2
```
Docker (all three services): see the header of `docker-compose.yml`.

## Test
```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```
The tests use a scripted fake model and in-memory Qdrant: they prove wiring, access control, safety rules and the UI, not answer quality.

## Security notes (manual, section 6)
- Secrets come from environment variables, `.env` or `.streamlit/secrets.toml`; never from code or GitHub (`.env` and `secrets.toml` stay in `.gitignore`).
- The backend validates every request (role, source, length), rate-limits per client, and returns a generic 503 when Qdrant or Ollama fail.
- Roles are chosen in the sidebar for the prototype. A real deployment takes the identity from SSO and enforces it in the API; the UI is never the authority.
- Streamlit caches only the non-sensitive KB inventory, with a 30 s expiry. Chat results are never cached.
- Feedback (`data/feedback.jsonl`) and the audit log (`data/audit_log.jsonl`) store no answer text.

## Release checklist
- [ ] Qdrant, Ollama and the collection are healthy (sidebar shows "Backend 🟢 ok")
- [ ] `python -m pytest tests -q` passes
- [ ] `RAG_API_KEY` set; `.env` not committed
- [ ] Demo questions checked: normal, cross-source, help request (clarification), password request (refused), guest role
- [ ] Latest notebook comparison saved in `data/` (Evaluation page shows it)
