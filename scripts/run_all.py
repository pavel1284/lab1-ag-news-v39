"""One-command complete lab run on a computer with network access."""
import subprocess
import sys

STEPS = [
    [sys.executable, "-m", "lab1.download"],
    [sys.executable, "-m", "lab1.hashes", "verify"],
    [sys.executable, "-m", "lab1.eda"],
    [sys.executable, "scripts/negative_controls.py"],
    [sys.executable, "scripts/repeat_eda.py"],
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "scripts/export_word.py"],
]
for index, command in enumerate(STEPS, start=1):
    print(f"\n[{index}/{len(STEPS)}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
print("LAB1 COMPLETE: datasets, EDA, plots, report, negative controls, tests")
