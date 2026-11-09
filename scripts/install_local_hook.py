"""Install an equivalent repository-local Git pre-commit hook without internet."""
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
result = subprocess.run(["git", "rev-parse", "--git-path", "hooks"], cwd=root,
                        capture_output=True, text=True, check=True)
hook_dir = (root / result.stdout.strip()).resolve()
hook_dir.mkdir(parents=True, exist_ok=True)
hook = hook_dir / "pre-commit"
hook.write_text("#!/bin/sh\nexec python3 scripts/check_precommit.py --staged\n", encoding="utf-8")
os.chmod(hook, 0o755)
print(f"Installed {hook}")
