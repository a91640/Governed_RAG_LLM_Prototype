from pathlib import Path
from dotenv import load_dotenv
import argparse
import json
import os
import sys
import uuid
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
CHUNKS_FILE = Path("data/processed/network_kb_v1_chunks.jsonl")
COLLECTION_NAME = "network_kb_azure_v1"
EMBEDDING_MODEL = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
VECTOR_SIZE = 1536
BATCH_SIZE = 16

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6336")


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        sys.exit(f"ERROR: Chunk file not found: {CHUNKS_FILE}")

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check configuration without creating a collection or uploading vectors.",
    )
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Chunks loaded: {len(chunks)}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    client = QdrantClient(url=QDRANT_URL)
    embedder = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
)

    test_vector = embedder.embed_query("network telemetry test")
    actual_vector_size = len(test_vector)

    if actual_vector_size != VECTOR_SIZE:
        sys.exit(
            f"ERROR: Expected {VECTOR_SIZE} dimensions, "
            f"but received {actual_vector_size}."
        )

    print(f"Embedding dimension verified: {actual_vector_size}")

    if args.dry_run:
        print("DRY RUN successful. No Qdrant data was changed.")
        return

    if client.collection_exists(COLLECTION_NAME):
        sys.exit(
            f"ERROR: Collection '{COLLECTION_NAME}' already exists. "
            "No data was changed."
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"Created collection: {COLLECTION_NAME}")

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        vectors = embedder.embed_documents(texts)

        points = []
        for item, vector in zip(batch, vectors):
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, item["chunk_id"])
            )

            payload = {
                "chunk_id": item["chunk_id"],
                "source_id": item["source_id"],
                "relative_path": item["relative_path"],
                "document_type": item["document_type"],
                "chunk_number": item["chunk_number"],
                "text": item["text"],
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        completed = min(start + BATCH_SIZE, len(chunks))
        print(f"Uploaded {completed}/{len(chunks)} chunks")

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print("\nIngestion completed successfully.")
    print(f"Vectors stored in {COLLECTION_NAME}: {count}")


if __name__ == "__main__":
    main()