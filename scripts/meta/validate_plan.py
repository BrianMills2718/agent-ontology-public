#!/usr/bin/env python3
"""CLI wrapper for the importable plan-validation module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _detect_repo_root(script_path: Path) -> Path:
    """Resolve repo root for both canonical and installed script layouts."""
    if script_path.parent.name == "meta" and script_path.parent.parent.name == "scripts":
        return script_path.parents[2]
    if script_path.parent.name == "scripts":
        return script_path.parents[1]
    return script_path.parents[1]


ROOT = _detect_repo_root(Path(__file__).resolve())
PLANS_DIR = ROOT / "docs" / "plans"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enforced_planning.file_context import collect_context
from enforced_planning.file_context import load_relationships
from enforced_planning.plan_validation import PATH_CLEAN_RE
from enforced_planning.plan_validation import REQUIRED_PLAN_SECTIONS
from enforced_planning.plan_validation import ValidationResult
from enforced_planning.plan_validation import collect_plan_requirements
from enforced_planning.plan_validation import extract_inline_paths
from enforced_planning.plan_validation import extract_paths
from enforced_planning.plan_validation import extract_section
from enforced_planning.plan_validation import find_plan_file
from enforced_planning.plan_validation import get_current_plan_number as _get_current_plan_number
from enforced_planning.plan_validation import get_plan_file as _get_plan_file
from enforced_planning.plan_validation import looks_like_file_path
from enforced_planning.plan_validation import main as _main
from enforced_planning.plan_validation import normalize
from enforced_planning.plan_validation import parse_contracts_used
from enforced_planning.plan_validation import parse_data_flow
from enforced_planning.plan_validation import parse_files_affected
from enforced_planning.plan_validation import parse_mentioned_adrs
from enforced_planning.plan_validation import parse_plan_status
from enforced_planning.plan_validation import parse_references_reviewed
from enforced_planning.plan_validation import parse_tools_used
from enforced_planning.plan_validation import parse_uncertainty_register
from enforced_planning.plan_validation import print_summary
from enforced_planning.plan_validation import read_text
from enforced_planning.plan_validation import split_lines
from enforced_planning.plan_validation import validate_plan


def get_current_plan_number() -> int | None:
    """Infer the active plan number using this repo's checkout root."""
    return _get_current_plan_number(repo_root=ROOT)


def get_plan_file(plan_number: int | None, plans_dir: Path, plan_file: str | None) -> Path:
    """Resolve one plan file using this repo's checkout root."""
    return _get_plan_file(
        plan_number,
        plans_dir,
        plan_file,
        repo_root=ROOT,
    )


def main() -> int:
    """Run plan validation using this repo's local defaults."""
    return _main(repo_root=ROOT, plans_dir=PLANS_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
