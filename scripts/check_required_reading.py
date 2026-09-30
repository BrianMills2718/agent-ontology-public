#!/usr/bin/env python3
"""Compatibility wrapper for required-read enforcement.

This command delegates to file_context with `--check-reads`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / "meta" / "file_context.py"
if not TARGET.exists():
    TARGET = SCRIPT_DIR / "file_context.py"


if __name__ == "__main__":
    if not TARGET.exists():
        print(f"Missing target script: {TARGET}", file=sys.stderr)
        raise SystemExit(2)

    # Preserve original caller args, force check mode for this helper.
    sys.argv = [str(TARGET), "--check-reads", *sys.argv[1:]]
    runpy.run_path(str(TARGET), run_name="__main__")
