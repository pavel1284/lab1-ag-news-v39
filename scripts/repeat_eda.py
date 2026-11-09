"""Run EDA twice, prove deterministic CSV hash (MLflow disabled for repeat)."""
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab1.config import read_config
from lab1.eda import run_eda
from lab1.hashes import sha256

cfg = read_config()
out = cfg.report_dir
with tempfile.TemporaryDirectory(prefix="lab1_repeat_") as d:
    report_dir = cfg.report_dir
    cfg.report_dir = Path(d)
    manifest = report_dir / "hash_manifest.json"
    (cfg.report_dir / "hash_manifest.json").write_bytes(manifest.read_bytes())
    first = run_eda(cfg, mlflow_enabled=False)
    first_hash = sha256(cfg.report_dir / "eda_stats.csv")
    second = run_eda(cfg, mlflow_enabled=False)
    second_hash = sha256(cfg.report_dir / "eda_stats.csv")
    cfg.report_dir = report_dir
    log = ("PASS: two EDA runs, same inputs and config\n" if first_hash == second_hash
           else "FAIL: inconsistent EDA CSV\n")
    log += f"run1={first_hash}\nrun2={second_hash}\n"
    log += f"n_train={first['n_train']} n_test={second['n_test']}\n"
    (out / "eda_repeat.log").write_text(log, encoding="utf-8")
    print(log)
    if first_hash != second_hash:
        raise SystemExit(1)
