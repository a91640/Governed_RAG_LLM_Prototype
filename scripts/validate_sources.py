from pathlib import Path
import sys
import yaml

manifest_path = Path("data/manifests/sources.yaml")

if not manifest_path.exists():
    sys.exit(f"ERROR: File not found: {manifest_path}")

with manifest_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

sources = config.get("sources", [])

if not sources:
    sys.exit("ERROR: No sources were found in sources.yaml")

print(f"Knowledge-base manifest validated: {len(sources)} source(s) found.\n")

for source in sources:
    required_fields = ["id", "type", "topics", "enabled"]
    missing = [field for field in required_fields if field not in source]

    if missing:
        print(f"INVALID: {source.get('id', 'unknown')} — missing: {', '.join(missing)}")
        continue

    state = "ENABLED" if source["enabled"] else "DISABLED"
    print(f"{state}: {source['id']}")
    print(f"  Type: {source['type']}")
    print(f"  Topics: {', '.join(source['topics'])}")

print("\nManifest check completed successfully.")