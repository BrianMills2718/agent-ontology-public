#!/usr/bin/env python3
"""Check authored AGENTS.md or a legacy CLAUDE-to-AGENTS projection."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()


def _detect_repo_root(script_path: Path) -> Path:
    """Resolve repo root for both canonical and installed script layouts."""

    if script_path.parent.name == "meta" and script_path.parent.parent.name == "scripts":
        return script_path.parents[2]
    if script_path.parent.name == "scripts":
        return script_path.parents[1]
    return script_path.parents[1]


REPO_ROOT = _detect_repo_root(SCRIPT_PATH)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enforced_planning.agents_rendering import build_renderer  # noqa: E402


def _renderer_entrypoint(repo_root: Path) -> Path:
    """Return the truthful render entrypoint path for this repo layout."""

    source_renderer = repo_root / "scripts" / "render_agents_md.py"
    installed_renderer = repo_root / "scripts" / "meta" / "render_agents_md.py"
    candidates = (
        (source_renderer, installed_renderer)
        if SCRIPT_PATH.parent.name == "scripts"
        else (installed_renderer, source_renderer)
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_RENDER_RUNTIME = build_renderer(_renderer_entrypoint(REPO_ROOT))
DEFAULT_TEMPLATE = _RENDER_RUNTIME.default_template


def resolve_inputs(
    repo_root: Path,
    claude_file: str = "CLAUDE.md",
    relationships_file: str = "scripts/relationships.yaml",
    output_file: str = "AGENTS.md",
    template_path: Path = DEFAULT_TEMPLATE,
):
    """Resolve AGENTS canonical inputs for one repo layout."""

    return _RENDER_RUNTIME.resolve_inputs(
        repo_root=repo_root,
        claude_file=claude_file,
        relationships_file=relationships_file,
        output_file=output_file,
        template_path=template_path,
    )


def render_agents_markdown(inputs):
    """Render AGENTS using the truthful local renderer runtime."""

    return _RENDER_RUNTIME.render_agents_markdown(inputs)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sync checker."""
    parser = argparse.ArgumentParser(
        description="Check authored AGENTS.md or a legacy generated projection",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root containing AGENTS.md and optional legacy CLAUDE.md",
    )
    parser.add_argument(
        "--claude-file",
        default="CLAUDE.md",
        help="Repo-relative path to the legacy CLAUDE.md source, when present",
    )
    parser.add_argument(
        "--relationships-file",
        default="scripts/relationships.yaml",
        help="Repo-relative path to canonical relationships.yaml",
    )
    parser.add_argument(
        "--output-file",
        default="AGENTS.md",
        help="Repo-relative path to AGENTS.md",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Path to the AGENTS markdown template",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Accepted for consistency with other repo checkers; check mode is the default",
    )
    return parser.parse_args()


def main() -> int:
    """Check authored AGENTS.md or a legacy generated projection."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    template_path = Path(args.template).resolve()
    source_path = repo_root / args.claude_file
    output_path = repo_root / args.output_file
    if not source_path.exists() and args.claude_file == "CLAUDE.md":
        if output_path.is_symlink() or not output_path.is_file():
            print(f"Authored AGENTS.md is missing or not a regular file: {output_path}")
            return 1
        if "<!-- GENERATED FILE: DO NOT EDIT DIRECTLY -->" in output_path.read_text(encoding="utf-8"):
            print(f"AGENTS.md still declares itself generated without a source: {output_path}")
            return 1
        print(f"AGENTS.md is the authored instruction source: {output_path}")
        return 0
    try:
        inputs = resolve_inputs(
            repo_root=repo_root,
            claude_file=args.claude_file,
            relationships_file=args.relationships_file,
            output_file=args.output_file,
            template_path=template_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    if not inputs.output_path.exists():
        print(f"Generated AGENTS file is missing: {inputs.output_path}")
        print(
            "Run: "
            f"python {_renderer_entrypoint(repo_root)} --repo-root {repo_root}"
        )
        return 1

    expected = render_agents_markdown(inputs)
    actual = inputs.output_path.read_text(encoding="utf-8")
    if actual == expected:
        print(f"AGENTS.md is in sync: {inputs.output_path}")
        return 0

    diff = "\n".join(
        difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=str(inputs.output_path),
            tofile=f"{inputs.output_path} (expected)",
            lineterm="",
        )
    )
    print("AGENTS.md drift detected.")
    print(
        "Regenerate with: "
        f"python {_renderer_entrypoint(repo_root)} --repo-root {repo_root}"
    )
    if diff:
        print(diff)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
