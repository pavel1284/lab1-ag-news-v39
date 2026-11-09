"""Demonstrate three real blocked commits plus SHA256 tamper detection."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def git(folder: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=folder, text=True, capture_output=True)


def blocked_commit_tests(destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="lab1_git_negative_") as temporary:
        folder = Path(temporary)
        (folder / "scripts").mkdir()
        shutil.copy(ROOT / "scripts" / "check_precommit.py", folder / "scripts" / "check_precommit.py")
        git(folder, "init", "-q")
        git(folder, "config", "user.name", "LAB1 Test")
        git(folder, "config", "user.email", "lab1-test@example.invalid")
        (folder / "README.md").write_text("Local test repository\n", encoding="utf-8")
        git(folder, "add", "README.md")
        git(folder, "commit", "-qm", "baseline")
        hook = folder / ".git" / "hooks" / "pre-commit"
        hook.write_text(f"#!/bin/sh\nexec {sys.executable} scripts/check_precommit.py --staged\n", encoding="utf-8")
        os.chmod(hook, 0o755)
        examples = [
            ("01_style", "broken.py", b"def invalid(:\n    pass\n"),
            ("02_secret", "secrets.env", b'API_' + b'KEY = "LAB1_DEMONSTRATION_CREDENTIAL"\n'),
            ("03_large_file", "large_blob.bin", b"X" * (1048576 + 1)),
        ]
        log = ["# Real negative-control git commit tests (temporary repositories)",
               "# Every test attempts `git commit`; return code must be nonzero", ""]
        for case, filename, payload in examples:
            for path in folder.iterdir():
                if path.is_file() and path.name != "README.md":
                    path.unlink()
            git(folder, "reset", "-q", "HEAD")
            (folder / filename).write_bytes(payload)
            git(folder, "add", filename)
            res = git(folder, "commit", "-m", "attempt " + case)
            transcript = (res.stdout + res.stderr).strip()
            ok = res.returncode != 0 and "BLOCK" in transcript
            log += [f"CASE {case}: {'PASS BLOCKED' if ok else 'FAIL NOT BLOCKED'}",
                    f"exit_code={res.returncode}", transcript, ""]
            if not ok:
                raise RuntimeError(f"Negative control {case} was not blocked: {transcript}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(log) + "\n", encoding="utf-8")
        print(f"THREE REAL COMMITS BLOCKED: {destination}")


def hash_tamper_test(destination: Path) -> None:
    from lab1.hashes import create_manifest, verify_manifest
    with tempfile.TemporaryDirectory(prefix="lab1_hash_negative_") as temp:
        root = Path(temp)
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        (raw / "train.csv").write_text("text,label\nalpha,0\n", encoding="utf-8")
        (raw / "test.csv").write_text("text,label\nbeta,1\n", encoding="utf-8")
        manifest = root / "hash_manifest.json"
        # Paths in manifest are relative, so temporarily run with the root as cwd.
        old = Path.cwd()
        try:
            os.chdir(root)
            create_manifest(Path("data/raw"), Path("hash_manifest.json"), "test-fixture")
            first, first_log = verify_manifest(manifest)
            with (raw / "train.csv").open("a", encoding="utf-8") as f:
                f.write("tampered,2\n")
            second, second_log = verify_manifest(manifest)
        finally:
            os.chdir(old)
        if not first or second:
            raise AssertionError("Tamper verification did not detect modified bytes")
        destination.write_text("Before tampering: PASS\n" + "\n".join(first_log)
                               + "\nAfter tampering: MISMATCH EXPECTED\n"
                               + "\n".join(second_log) + "\n", encoding="utf-8")
        print(f"SHA256 CHANGE DETECTED: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="reports/LAB1")
    args = parser.parse_args()
    out = Path(args.report_dir)
    blocked_commit_tests(out / "precommit_block.log")
    hash_tamper_test(out / "hash_negative_test.log")


if __name__ == "__main__":
    main()
