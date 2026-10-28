from pathlib import Path
import hashlib
import json
import re

import yaml

MANIFEST_PATH = Path("data/manifests/sources.yaml")
RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
OUTPUT_FILE = PROCESSED_ROOT / "network_kb_v1_chunks.jsonl"

SUPPORTED_EXTENSIONS = {".yang", ".md", ".markdown", ".rst", ".txt"}
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.append(current)

        overlap = current[-CHUNK_OVERLAP:] if current else ""
        current = f"{overlap}\n\n{paragraph}".strip()

        while len(current) > CHUNK_SIZE:
            chunks.append(current[:CHUNK_SIZE])
            current = current[CHUNK_SIZE - CHUNK_OVERLAP:].strip()

    if current:
        chunks.append(current)

    return chunks


def main() -> None:
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file)

    enabled_sources = [
        source["id"]
        for source in manifest.get("sources", [])
        if source.get("enabled", False)
    ]

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_chunks = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for source_id in enabled_sources:
            source_folder = RAW_ROOT / source_id

            if not source_folder.exists():
                print(f"SKIPPED: {source_id} folder was not found.")
                continue

            source_files = [
                path for path in source_folder.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            source_chunk_count = 0

            for file_path in source_files:
                try:
                    raw_text = file_path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                except OSError as error:
                    print(f"WARNING: Could not read {file_path}: {error}")
                    continue

                text = clean_text(raw_text)

                if len(text) < 100:
                    continue

                relative_path = file_path.relative_to(source_folder)
                chunks = split_into_chunks(text)

                for index, chunk in enumerate(chunks, start=1):
                    chunk_id_input = f"{source_id}|{relative_path}|{index}|{chunk}"
                    chunk_id = hashlib.sha256(
                        chunk_id_input.encode("utf-8")
                    ).hexdigest()

                    record = {
                        "chunk_id": chunk_id,
                        "source_id": source_id,
                        "relative_path": str(relative_path),
                        "document_type": file_path.suffix.lower().lstrip("."),
                        "chunk_number": index,
                        "text": chunk,
                    }

                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    source_chunk_count += 1

                total_files += 1

            total_chunks += source_chunk_count
            print(
                f"COMPLETED: {source_id} — "
                f"{len(source_files)} file(s), "
                f"{source_chunk_count} chunk(s)"
            )

    print("\nPreparation completed.")
    print(f"Files processed: {total_files}")
    print(f"Chunks created: {total_chunks}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()