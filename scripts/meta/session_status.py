#!/usr/bin/env python3
"""Show live session summaries derived from claims plus trackers."""

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
    parser.add_argument("--project")
    parser.add_argument("--agent")
    parser.add_argument("--scope")
    parser.add_argument("--branch")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = session_lifecycle.status_sessions(
        project=args.project,
        agent=args.agent,
        scope=args.scope,
        branch=args.branch,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"Live sessions: {payload['session_count']}")
    for session in payload["sessions"]:
        print(
            f"- {session['project']}:{session['scope']} "
            f"[{session['health_status']}] "
            f"{session['session_name']} :: {session['current_phase']} "
            f"(recovery={session['recovery_action']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
