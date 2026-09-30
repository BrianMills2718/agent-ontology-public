#!/usr/bin/env python3
"""Finish one sanctioned session without worktree cleanup.

For claimed worktree cleanup plus claim release, use ``session_close.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "enforced_planning").is_dir():
            return parent
    raise RuntimeError("Unable to locate repo root containing enforced_planning/")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enforced_planning import session_lifecycle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--worktree-path", required=True)
    parser.add_argument("--note")
    parser.add_argument("--release-claim", action="store_true")
    parser.add_argument("--allow-dirty-handoff", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = session_lifecycle.finish_session(
        agent=args.agent,
        project=args.project,
        scope=args.scope,
        worktree_path=args.worktree_path,
        note=args.note,
        release_claim=args.release_claim,
        allow_dirty_handoff=args.allow_dirty_handoff,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{payload['action']}: clean={payload['clean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
