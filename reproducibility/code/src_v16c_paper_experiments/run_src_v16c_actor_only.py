#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience wrapper: run src_v16c on actor only for 10 baseline-aligned repeats."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
cmd = [
    sys.executable,
    str(ROOT / "run_src_v16c_fill_main_table_10repeats.py"),
    "--only-actor",
    "--max-repeats", "10",
    "--grid", "default",
]
cmd += sys.argv[1:]
print("$ " + " ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd))
