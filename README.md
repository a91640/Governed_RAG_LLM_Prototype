# Governed RAG LLM Prototype

This is a **read-only public RAG prototype**. It uses Qdrant for vector search and local Ollama models for embeddings and answers. It does not ingest production configurations, credentials, incident tickets, or customer data.

## 1. Create the environment

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Start local services

Open Docker Desktop first. Then start Qdrant (only once):

```powershell
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

On later days, use this instead:

```powershell
docker start qdrant
```

Download the local Ollama models:

```powershell
ollama pull embeddinggemma
ollama pull llama3.2:3b
```

Verify Qdrant is running:

```powershell
docker ps
```

## 3. Included public knowledge base

This ready-to-run package already includes a small, safe public corpus:

- OpenConfig: BGP and interface YANG models
- SONiC: README and BGP router-ID documentation
- Open5GS: 5G Equipment Identity Register guide

The documents are stored in these source folders:

```text
data/raw/openconfig/
data/raw/sonic/
data/raw/open5gs/
data/raw/ietf/
```

To add more approved public files later, use the same folder structure:

- `openconfig-interfaces.yang`, `openconfig-bgp.yang`
- a SONiC README or documentation `.md` file
- an Open5GS documentation `.md` file
- RFC 4271 (BGP) and RFC 6241 (NETCONF) `.txt` files

Allowed formats are `.md`, `.txt`, `.yang`, and `.pdf`.

## 4. Run the notebook

Open `Governed_Network_RAG_Professor_Demo.ipynb` in VS Code. Select the `.venv` Jupyter kernel. Run cells from top to bottom once before the meeting.

The notebook uses the Qdrant collection `network_operations_knowledge`. It safely reuses deterministic chunk IDs, so rerunning ingestion updates existing chunks instead of creating duplicates.

## 5. LangSmith (optional but recommended)

The notebook asks for your API key only when you run the LangSmith cell. Paste the key in the **hidden prompt**, then press Enter. Do not type the key into notebook code or show it during your demonstration.

If your LangSmith workspace is in the EU region, leave this value unchanged:

```python
LANGSMITH_ENDPOINT = "https://eu.api.smith.langchain.com"
```

Open traces at `https://eu.smith.langchain.com` and select project `network-ops-governed-rag`.

## 6. Suggested 8-minute presentation

1. Explain scope: public sources only, read-only assistant.
2. Run / show the pre-flight cell: Qdrant and Ollama are local.
3. Show document validation and a metadata example.
4. Show chunk count and Qdrant collection count.
5. Ask one normal question and point to its citations.
6. Ask an unsafe credential question and show the refusal.
7. Show the small evaluation table and LangSmith trace.
8. Finish with the internal-data roadmap: approval, redaction, authenticated roles, and Qdrant payload filters.

Do not run a large first-time download or a long evaluation live. Complete ingestion before the call, then show the outputs and run only the two short question cells live.
