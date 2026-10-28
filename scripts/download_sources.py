from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import yaml

MANIFEST_PATH = Path("data/manifests/sources.yaml")
RAW_ROOT = Path("data/raw")


def run_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def download_github_source(source: dict, dry_run: bool) -> None:
    source_id = source["id"]
    repository = source["repository"]
    source_path = source["path"]
    destination = RAW_ROOT / source_id

    if destination.exists():
        print(f"SKIPPED: {source_id} already exists.")
        return

    if dry_run:
        print(f"DRY RUN: Would download {repository} → {destination}")
        return

    print(f"Downloading {source_id}...")

    clone_command = [
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--sparse",
    ]

    if source.get("ref"):
        clone_command.extend(["--branch", source["ref"]])

    clone_command.extend([repository, str(destination)])
    run_command(clone_command)

    run_command([
        "git", "-C", str(destination),
        "sparse-checkout", "set", source_path
    ])

    commit_sha = run_command([
        "git", "-C", str(destination),
        "rev-parse", "HEAD"
    ])

    provenance = {
        "source_id": source_id,
        "repository": repository,
        "selected_path": source_path,
        "commit_sha": commit_sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    shutil.rmtree(destination / ".git", ignore_errors=True)
    print(f"COMPLETED: {source_id}")


def download_rfc(source: dict, dry_run: bool) -> None:
    number = source["number"]
    source_id = source["id"]
    destination = RAW_ROOT / source_id
    file_path = destination / f"rfc{number}.txt"
    url = f"https://www.rfc-editor.org/rfc/rfc{number}.txt"

    if file_path.exists():
        print(f"SKIPPED: RFC {number} already exists.")
        return

    if dry_run:
        print(f"DRY RUN: Would download {url}")
        return

    print(f"Downloading RFC {number}...")
    destination.mkdir(parents=True, exist_ok=True)

    request = Request(url, headers={"User-Agent": "Governed-RAG-Knowledge-Base"})
    with urlopen(request, timeout=30) as response:
        file_path.write_bytes(response.read())

    provenance = {
        "source_id": source_id,
        "source_url": url,
        "rfc_number": number,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    print(f"COMPLETED: RFC {number}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without changing files.",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit(f"ERROR: Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    for source in config.get("sources", []):
        if not source.get("enabled", False):
            continue

        source_type = source["type"]

        if source_type in {"github_yang", "github_docs"}:
            download_github_source(source, args.dry_run)
        elif source_type == "rfc":
            download_rfc(source, args.dry_run)
        else:
            print(f"SKIPPED: Unsupported source type: {source_type}")

    print("\nDownload process completed.")


if __name__ == "__main__":
    main()