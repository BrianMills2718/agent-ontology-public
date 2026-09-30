#!/usr/bin/env python3
"""Start or refresh one sanctioned session contract."""

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
    parser.add_argument("--intent", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--worktree-path", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--broader-goal", required=True)
    parser.add_argument("--current-phase", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--allow-unplanned", action="store_true")
    parser.add_argument("--allow-parallel", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--session-name")
    parser.add_argument("--next-phase", action="append", default=[])
    parser.add_argument("--depends-on", action="append", default=[])
    parser.add_argument("--stop-condition", action="append", default=[])
    parser.add_argument("--requires-shared-infra-changes", action="store_true")
    parser.add_argument("--notes")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = session_lifecycle.start_session(
        agent=args.agent,
        project=args.project,
        scope=args.scope,
        intent=args.intent,
        repo_root=args.repo_root,
        worktree_path=args.worktree_path,
        branch=args.branch,
        broader_goal=args.broader_goal,
        current_phase=args.current_phase,
        plan_ref=args.plan,
        allow_unplanned=args.allow_unplanned,
        allow_parallel=args.allow_parallel,
        session_id=args.session_id,
        session_name=args.session_name,
        intended_next_phases=args.next_phase,
        depends_on_repos=args.depends_on,
        requires_shared_infra_changes=args.requires_shared_infra_changes,
        stop_conditions=args.stop_condition,
        notes=args.notes,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"{payload['action']}: {payload['session_name']} "
            f"({payload['broader_goal']}) -> {payload['tracker_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
