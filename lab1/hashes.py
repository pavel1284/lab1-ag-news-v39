"""SHA256 file manifest and verification. Raw data are excluded from Git."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RAW_FILES = ("train.csv", "test.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for piece in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def create_manifest(data_dir: Path, manifest_path: Path, source: str) -> dict:
    files = []
    for filename in RAW_FILES:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Data file is missing: {path}")
        files.append({"path": path.as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256(path)})
    result = {
        "schema_version": 1,
        "dataset": "AG News",
        "source": source,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def verify_manifest(manifest_path: Path) -> tuple[bool, list[str]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError) as error:
        return False, [f"Manifest missing or unreadable: {error}"]
    if manifest.get("status") == "pending_download" or not manifest.get("files"):
        return False, ["MISSING RAW FILES: manifest awaits real download and SHA256 calculation"]
    results = []
    valid = True
    for item in manifest["files"]:
        path = Path(item["path"])
        if not path.is_file():
            results.append(f"FAIL missing: {path}")
            valid = False
            continue
        match = path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]
        results.append(f"{'MATCH' if match else 'FAIL SHA256/SIZE'} {path}")
        valid = valid and match
    return valid, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "verify"])
    parser.add_argument("--dir", default="data/raw")
    parser.add_argument("--manifest", default="reports/LAB1/hash_manifest.json")
    parser.add_argument("--optional", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    if args.action == "create":
        created = create_manifest(Path(args.dir), manifest, "https://huggingface.co/datasets/fancyzhx/ag_news")
        print(json.dumps(created, ensure_ascii=False, indent=2))
    else:
        if args.optional and not manifest.exists():
            print("SKIP: raw dataset not downloaded in this environment")
            return
        valid, messages = verify_manifest(manifest)
        print("\n".join(messages))
        print("VERDICT: MATCH" if valid else "VERDICT: MISMATCH")
        if not valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
