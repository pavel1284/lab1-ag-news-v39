"""Download original Hugging Face AG News into stable UTF-8 CSV raw files."""
import argparse
import csv
from pathlib import Path

from lab1.config import read_config
from lab1.hashes import create_manifest


def download(config_path: Path = Path("configs/lab1.yaml")) -> None:
    # Imported lazily: offline smoke tests can run without HF installed.
    from datasets import load_dataset

    config = read_config(config_path)
    if config.variant != 39:
        raise ValueError("This project is explicitly for variant 39")
    config.input_dir.mkdir(parents=True, exist_ok=True)
    for split, expected in (("train", config.expected_train), ("test", config.expected_test)):
        destination = config.input_dir / f"{split}.csv"
        if destination.exists():
            print(f"KEEP {destination}; verify hashes before reusing")
            continue
        data = load_dataset(config.dataset_id, split=split)
        if len(data) != expected:
            raise ValueError(f"{split}: expected {expected} records, got {len(data)}")
        with destination.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(["text", "label"])
            for row in data:
                writer.writerow([row["text"], row["label"]])
        print(f"SAVED {destination}: {expected} records")
    create_manifest(config.input_dir, config.report_dir / "hash_manifest.json",
                    f"https://huggingface.co/datasets/{config.dataset_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lab1.yaml"))
    args = parser.parse_args()
    download(args.config)


if __name__ == "__main__":
    main()
