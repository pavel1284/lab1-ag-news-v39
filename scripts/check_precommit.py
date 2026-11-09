#!/usr/bin/env python3
"""Pure-stdlib pre-commit gate: syntax/style, secrets, max staged file size."""
import argparse
import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1048576
SENSITIVE = [
    re.compile(r"(?i)\b(?:api[_-]?key|secret|password|access[_-]?token)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"ghp_[a-zA-Z0-9]{30,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ico", ".gif", ".zip", ".parquet", ".bin"}


def staged_files() -> list[Path]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=True)
    return [ROOT / part.decode("utf-8") for part in result.stdout.split(b"\x00") if part]


def scan(paths: list[Path], checks: set[str]) -> list[str]:
    problems = []
    for path in paths:
        if not path.is_file() or ".git" in path.parts:
            continue
        label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        if "size" in checks and path.stat().st_size > MAX_BYTES:
            problems.append(f"BLOCK large file: {label} ({path.stat().st_size} > {MAX_BYTES} bytes)")
        if path.suffix.lower() in BINARY_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            continue
        if "secrets" in checks:
            for line_number, line in enumerate(content.splitlines(), start=1):
                if any(rule.search(line) for rule in SENSITIVE):
                    problems.append(f"BLOCK secret: {label}:{line_number} (possible credential)")
        if "style" in checks:
            if path.suffix == ".py":
                try:
                    ast.parse(content, filename=str(path))
                except SyntaxError as error:
                    problems.append(f"BLOCK syntax: {label}:{error.lineno}: {error.msg}")
            for line_number, line in enumerate(content.splitlines(), start=1):
                if line.endswith(" ") or line.endswith("\t"):
                    problems.append(f"BLOCK whitespace: {label}:{line_number} (trailing space)")
                if path.suffix == ".py" and line.startswith("\t"):
                    problems.append(f"BLOCK style: {label}:{line_number} (tab indentation)")
                if path.suffix == ".py" and len(line) > 130:
                    problems.append(f"BLOCK style: {label}:{line_number} (line > 130 columns)")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=["style", "secrets", "size", "all"], default="all")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paths", nargs="*")
    group.add_argument("--all", action="store_true")
    group.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    if args.paths is not None:
        paths = [(ROOT / path).resolve() for path in args.paths]
    elif args.all:
        # Restrict to version-controlled files so generated data never get scanned.
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        paths = [(ROOT / part.decode("utf-8")).resolve()
                 for part in result.stdout.split(b"\x00") if part]
    else:
        paths = staged_files()
    checks = {"style", "secrets", "size"} if args.check == "all" else {args.check}
    violations = scan(paths, checks)
    if violations:
        print("\n".join(violations))
        raise SystemExit(1)
    print(f"PASS {args.check}: {len(paths)} files checked")


if __name__ == "__main__":
    main()
