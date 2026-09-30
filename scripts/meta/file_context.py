#!/usr/bin/env python3
"""CLI wrapper for the importable file-context module."""

from __future__ import annotations

from pathlib import Path
import sys


def _detect_repo_root(script_path: Path) -> Path:
    """Resolve repo root for both canonical and installed script layouts."""
    if script_path.parent.name == "meta" and script_path.parent.parent.name == "scripts":
        return script_path.parents[2]
    if script_path.parent.name == "scripts":
        return script_path.parents[1]
    return script_path.parents[1]


REPO_ROOT = _detect_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enforced_planning.file_context import DEFAULT_CONFIG
from enforced_planning.file_context import DEFAULT_READS_FILE
from enforced_planning.file_context import FileContext
from enforced_planning.file_context import ReadCheckResult
from enforced_planning.file_context import check_required_reads
from enforced_planning.file_context import collect_context
from enforced_planning.file_context import load_relationships
from enforced_planning.file_context import load_yaml
from enforced_planning.file_context import main


if __name__ == "__main__":
    raise SystemExit(main())
