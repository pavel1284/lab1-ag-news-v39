"""Functional tests; the actual full dataset is deliberately not mocked as real."""
import csv
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from lab1.config import Lab1Config
from lab1.eda import load_split, run_eda, tokenize
from lab1.hashes import create_manifest, verify_manifest


def write_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def test_config_rejects_bad_variant():
    with pytest.raises(ValidationError):
        Lab1Config(variant=88)


def test_text_normalization():
    assert tokenize("Hello, worlds! 25 bucks", 2) == ["hello", "worlds", "bucks"]


def test_bad_label_rejected(tmp_path):
    dest = tmp_path / "train.csv"
    write_csv(dest, [("Text", 5)])
    with pytest.raises(ValueError, match="invalid class"):
        load_split(dest, 1, "train")


def test_wrong_rows_rejected(tmp_path):
    dest = tmp_path / "train.csv"
    write_csv(dest, [("Text", 0)])
    with pytest.raises(ValueError, match="expected 2"):
        load_split(dest, 2, "train")


def test_hash_tamper_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_csv(Path("data/raw/train.csv"), [("hello", 0)])
    write_csv(Path("data/raw/test.csv"), [("world", 1)])
    create_manifest(Path("data/raw"), Path("out/hash_manifest.json"), "fixture")
    assert verify_manifest(Path("out/hash_manifest.json"))[0]
    with Path("data/raw/train.csv").open("a", encoding="utf-8") as f:
        f.write("hacked,2\n")
    assert not verify_manifest(Path("out/hash_manifest.json"))[0]


def test_eda_runs_reproducibly_on_small_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = Path("data/raw")
    write_csv(data / "train.csv", [
        ("A market rally pushes finance shares", 2),
        ("International leaders meet at summit", 0),
        ("Football club wins final match", 1),
        ("Scientists launch new satellite software", 3),
    ])
    write_csv(data / "test.csv", [
        ("Trade and business figures rise", 2),
        ("World leaders discuss security", 0),
        ("Tennis player wins competition", 1),
        ("Researchers discover new device", 3),
    ])
    config = Lab1Config(expected_train=4, expected_test=4, use_mlflow=False)
    create_manifest(data, Path("reports/LAB1/hash_manifest.json"), "fixture")
    result_a = run_eda(config)
    result_b = run_eda(config)
    assert result_a["eda_csv_sha256"] == result_b["eda_csv_sha256"]
    assert result_a["train_count_per_class"] == dict.fromkeys(
        ["World", "Sports", "Business", "Sci/Tech"], 1)
    assert (config.report_dir / "figures/class_balance.png").is_file()
    assert (config.report_dir / "eda_terms.csv").is_file()
