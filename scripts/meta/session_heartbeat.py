#!/usr/bin/env python3
"""Refresh heartbeat state for one sanctioned session."""

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
    parser.add_argument("--scope")
    parser.add_argument("--branch")
    parser.add_argument("--session-id")
    parser.add_argument("--current-phase")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = session_lifecycle.heartbeat_session(
        agent=args.agent,
        project=args.project,
        session_id=args.session_id,
        scope=args.scope,
        branch=args.branch,
        current_phase=args.current_phase,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"heartbeat: updated {payload['updated_count']} claims "
            f"for session {payload['session_id']} at {payload['heartbeat_at']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
