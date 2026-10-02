"""Governed RAG LLM Prototype — read-only RAG helpers.

The only allowed input is a local public corpus in data/raw/.  The code never
executes device commands and never indexes private credentials intentionally.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"
COLLECTION = "network_operations_knowledge"
EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "llama3.2:3b"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
MIN_SCORE = 0.35
RAW_DIR = Path("data/raw")
ALLOWED_SUFFIXES = {".md", ".txt", ".yang", ".pdf"}

SOURCE_INFO = {
    "openconfig": {
        "product": "OpenConfig",
        "source_url": "https://github.com/openconfig/public",
        "license": "Apache-2.0 (verify per file/repository)",
    },
    "sonic": {
        "product": "SONiC",
        "source_url": "https://github.com/sonic-net/SONiC",
        "license": "Apache-2.0 (verify per file/repository)",
    },
    "open5gs": {
        "product": "Open5GS",
        "source_url": "https://github.com/open5gs/open5gs",
        "license": "AGPL-3.0 (verify per file/repository)",
    },
    "ietf": {
        "product": "IETF RFC",
        "source_url": "https://www.rfc-editor.org/",
        "license": "IETF Trust legal provisions (verify per RFC)",
    },
}

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I),
    re.compile(r"\b(?:password|api[_ -]?key|secret|token)\s*[:=]\s*\S+", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]
UNSAFE_QUESTION = re.compile(
    r"\b(password|credential|private key|api key|secret|token)\b", re.I
)


def product_from_path(path: Path) -> str:
    """Infer product from first folder below data/raw/."""
    parts = [p.lower() for p in path.parts]
    for name in SOURCE_INFO:
        if name in parts:
            return name
    raise ValueError(
        f"Put {path.name} in one of: " + ", ".join(SOURCE_INFO)
    )


def load_one(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    return [Document(page_content=path.read_text(encoding="utf-8", errors="ignore"))]


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def load_and_validate(raw_dir: Path = RAW_DIR) -> tuple[list[Document], list[dict[str, str]]]:
    """Load allowed public files and return rejected-file reasons separately."""
    documents: list[Document] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            rejected.append({"path": str(path), "reason": "unsupported format"})
            continue
        try:
            product_key = product_from_path(path)
            pages = load_one(path)
        except Exception as exc:
            rejected.append({"path": str(path), "reason": f"load error: {exc}"})
            continue
        content = "\n".join(page.page_content for page in pages)
        if contains_secret(content):
            rejected.append({"path": str(path), "reason": "possible secret — quarantine"})
            continue
        info = SOURCE_INFO[product_key]
        for page_number, page in enumerate(pages, start=1):
            clean = re.sub(r"\n{3,}", "\n\n", page.page_content).strip()
            if not clean:
                continue
            page.metadata = {
                **info,
                "source": product_key,
                "source_file": str(path),
                "document_type": path.suffix.lower().lstrip("."),
                "version": "local-public-sandbox",
                "last_reviewed": str(date.today()),
                "classification": "public",
                "allowed_roles": ["network_operator", "network_engineer"],
                "status": "active",
                "page": page.metadata.get("page", page_number),
            }
            page.page_content = clean
            documents.append(page)
    return documents, rejected


def make_chunks(documents: list[Document]) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    ids: list[str] = []
    for index, chunk in enumerate(chunks):
        source = chunk.metadata["source_file"]
        fingerprint = f"{source}|{chunk.metadata.get('page')}|{chunk.page_content}"
        chunk_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = index
        ids.append(chunk_id)
    return chunks, ids


def qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, timeout=30)
    client.get_collections()  # clear early error if Docker/Qdrant is unavailable
    return client


def get_vector_store() -> QdrantVectorStore:
    client = qdrant_client()
    names = {item.name for item in client.get_collections().collections}
    if COLLECTION not in names:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
        for field in ("classification", "source", "product", "status"):
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=OllamaEmbeddings(model=EMBED_MODEL),
    )


def ingest(chunks: list[Document], ids: list[str]) -> int:
    """Upsert public chunks. Stable IDs make reruns safe."""
    if not chunks:
        raise ValueError("No chunks found. Add public documents under data/raw/.")
    store = get_vector_store()
    store.add_documents(chunks, ids=ids, batch_size=32)
    return qdrant_client().count(COLLECTION, exact=True).count


def retrieve(question: str, product: str | None = None) -> list[tuple[Document, float]]:
    if product and product.lower() not in SOURCE_INFO:
        raise ValueError(f"product must be one of {list(SOURCE_INFO)}")
    must = [
        models.FieldCondition(
            key="metadata.classification",
            match=models.MatchValue(value="public"),
        ),
        models.FieldCondition(
            key="metadata.status", match=models.MatchValue(value="active")
        ),
    ]
    if product:
        must.append(
            models.FieldCondition(
                key="metadata.source", match=models.MatchValue(value=product.lower())
            )
        )
    return get_vector_store().similarity_search_with_relevance_scores(
        question, k=TOP_K, filter=models.Filter(must=must)
    )


PROMPT = ChatPromptTemplate.from_template(
    """You are a read-only network-operations knowledge assistant.
Use only the supplied public context. Do not invent commands, configurations, or facts.
If the context is insufficient, reply exactly: I don't know from the approved public corpus.
Add citations like [1] or [2] for every factual statement.

Question: {question}

Context:
{context}
"""
)


def ask(question: str, product: str | None = None) -> dict[str, Any]:
    """Return an answer plus source traceability; never performs external actions."""
    if UNSAFE_QUESTION.search(question):
        return {
            "answer": "I can't help retrieve or disclose credentials, secrets, or private keys.",
            "sources": [],
            "reason": "safety refusal",
        }
    matches = retrieve(question, product)
    if not matches or matches[0][1] < MIN_SCORE:
        return {
            "answer": "I don't know from the approved public corpus.",
            "sources": [],
            "reason": "insufficient evidence",
        }
    context = "\n\n".join(
        f"[{index}] {document.page_content}\nSource: {document.metadata['source_url']}"
        for index, (document, _) in enumerate(matches, start=1)
    )
    answer = ChatOllama(model=CHAT_MODEL, temperature=0).invoke(
        PROMPT.format(question=question, context=context)
    ).content
    sources = [
        {
            "citation": index,
            "product": document.metadata["product"],
            "file": document.metadata["source_file"],
            "url": document.metadata["source_url"],
            "score": round(float(score), 3),
        }
        for index, (document, score) in enumerate(matches, start=1)
    ]
    return {"answer": answer, "sources": sources, "reason": "grounded answer"}


def display_answer(result: dict[str, Any]) -> None:
    print("ANSWER\n" + result["answer"])
    print("\nSOURCE TRACEABILITY")
    if not result["sources"]:
        print("No sources returned — " + result["reason"])
    for source in result["sources"]:
        print(f"[{source['citation']}] {source['product']} | score={source['score']} | {source['url']}")


def corpus_summary(chunks: list[Document], rejected: list[dict[str, str]]) -> None:
    print("Accepted chunks by source:", dict(Counter(c.metadata["source"] for c in chunks)))
    print("Rejected files:", len(rejected))
    for item in rejected:
        print(" -", item["path"], "→", item["reason"])


def enable_langsmith():
    """Enable EU LangSmith tracing for this public corpus, or return None."""
    import getpass

    os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
    if not os.getenv("LANGSMITH_API_KEY"):
        key = getpass.getpass("LangSmith API key (press Enter to skip): ")
        if key:
            os.environ["LANGSMITH_API_KEY"] = key
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LangSmith skipped. Local RAG still works.")
        return None
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "network-ops-governed-rag"
    from langsmith import traceable

    print("LangSmith enabled. Open https://eu.smith.langchain.com")
    return traceable(name="network_ops_rag")(ask)
