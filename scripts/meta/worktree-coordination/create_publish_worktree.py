#!/usr/bin/env python3
"""Create a publish worktree only when the canonical main checkout is clean.

This is a thin wrapper over ``create_worktree.py`` for the specific case where
an operator wants a clean merge/publish lane. Publish lanes are control surfaces
for shared-state actions; they should fail loud when the canonical primary
checkout is already dirty or mid-merge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path


def _load_create_worktree_module():
    """Load the sibling worktree-creation script as a module."""

    module_path = Path(__file__).resolve().with_name("create_worktree.py")
    spec = importlib.util.spec_from_file_location("publish_create_worktree_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load create_worktree module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for publish-worktree creation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repo root or any linked worktree path")
    parser.add_argument("--branch", required=True, help="Publish-lane branch name")
    parser.add_argument(
        "--path",
        help="Explicit worktree path. Defaults to the repo's canonical *_worktrees/<branch> path.",
    )
    parser.add_argument(
        "--start-point",
        default="HEAD",
        help="Commit-ish to branch from when the branch does not already exist",
    )
    parser.add_argument(
        "--keep-failed-worktree",
        action="store_true",
        help="Leave a failed worktree on disk for manual diagnosis.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for publish-worktree creation."""

    args = parse_args(argv)
    module = _load_create_worktree_module()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.path:
        worktree_path = Path(args.path).expanduser().resolve()
    else:
        worktree_path = module.get_default_worktree_dir(repo_root) / args.branch

    result = module.create_worktree(
        repo_root=repo_root,
        worktree_path=worktree_path,
        branch=args.branch,
        start_point=args.start_point,
        split_brain_threshold=5,
        keep_failed_worktree=args.keep_failed_worktree,
        require_clean_main_root=True,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        module._print_human(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
