from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"
COLLECTION = "network_kb_azure_v1"


def source_url(source_id: str) -> str:
    value = source_id.lower()

    if "openconfig" in value:
        return "https://github.com/openconfig/public"
    if "sonic" in value:
        return "https://sonic-net.github.io/SONiC/"
    if "open5gs" in value:
        return "https://open5gs.org/open5gs/docs/"
    if "ietf" in value or "rfc" in value:
        return "https://www.rfc-editor.org/"

    return f"Approved public corpus: {source_id}"


client = QdrantClient(url=QDRANT_URL)

source_ids = set()
offset = None

while True:
    points, offset = client.scroll(
        collection_name=COLLECTION,
        limit=1000,
        offset=offset,
        with_payload=["source_id"],
        with_vectors=False,
    )

    for point in points:
        source_id = point.payload.get("source_id")
        if source_id:
            source_ids.add(source_id)

    if offset is None:
        break

for source_id in sorted(source_ids):
    metadata = {
        "classification": "public",
        "status": "active",
        "allowed_roles": ["network_operator", "network_engineer", "admin"],
        "product": source_id,
        "source": source_id,
        "source_url": source_url(source_id),
    }

    client.set_payload(
        collection_name=COLLECTION,
        payload={"metadata": metadata},
        points=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchValue(value=source_id),
                )
            ]
        ),
    )

    print(f"Updated metadata for: {source_id}")

print(f"Completed metadata migration for {len(source_ids)} source(s).")