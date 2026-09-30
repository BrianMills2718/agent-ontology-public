#!/usr/bin/env python3
"""Check local markdown link integrity for governance docs.

This checker validates local markdown links (relative and repo-local paths)
and markdown section anchors. External links are ignored.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enforced_planning.worktree_paths import resolve_canonical_target_path

PROJECT_META_ROOT = REPO_ROOT
DEFAULT_TARGETS = ("vision", "STATUS_LEDGER.md")

# Matches inline links and image links: [text](target) / ![alt](target)
INLINE_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")

# Matches reference-style link definitions: [id]: target
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+]\s*:\s*(\S+)")

HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


@dataclass(frozen=True)
class LinkViolation:
    """A broken or missing markdown link with source location and target details."""

    file_path: str
    line_number: int
    target: str
    message: str

    def format(self) -> str:
        """Format as file:line: message (target: ...) for terminal output."""
        return (
            f"{self.file_path}:{self.line_number}: {self.message} "
            f"(target: {self.target})"
        )


def _to_display(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_META_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_targets(raw_targets: list[str], root: Path) -> list[Path]:
    resolved: list[Path] = []
    for item in raw_targets:
        target = Path(item).expanduser()
        if not target.is_absolute():
            target = root / target
        if target.is_file():
            if target.suffix.lower() == ".md":
                resolved.append(target)
            continue
        if target.is_dir():
            resolved.extend(sorted(p for p in target.rglob("*.md") if p.is_file()))
            continue
        raise FileNotFoundError(f"Target does not exist: {item}")
    unique: dict[Path, None] = {}
    for path in resolved:
        unique[path.resolve()] = None
    return sorted(unique.keys())


def _is_external_target(target: str) -> bool:
    lowered = target.lower()
    if lowered.startswith(("#", "http://", "https://", "mailto:", "tel:", "data:")):
        return lowered.startswith(("http://", "https://", "mailto:", "tel:", "data:"))
    if re.match(r"^[a-z][a-z0-9+.-]*:", lowered):
        return True
    return False


def _strip_title_segment(target: str) -> str:
    stripped = target.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return stripped[1:-1].strip()
    for marker in (' "', " '"):
        idx = stripped.find(marker)
        if idx != -1:
            return stripped[:idx].strip()
    return stripped


def _split_link_target(target: str) -> tuple[str, str]:
    if "#" not in target:
        return target, ""
    path_part, fragment = target.split("#", 1)
    return path_part, fragment


def _slugify_heading(value: str) -> str:
    """Generate a GitHub-compatible anchor slug from a heading string.

    Matches GitHub's algorithm: lowercase, strip markdown chars, remove
    non-alphanumeric-except-space-underscore-hyphen, replace spaces with
    hyphens.  GitHub does NOT collapse consecutive hyphens — e.g.
    '## Evidence & Validation' → 'evidence--validation'.
    """
    slug = value.strip().lower()
    slug = re.sub(r"[`*_~]", "", slug)
    slug = re.sub(r"[^a-z0-9 _-]", "", slug)
    slug = slug.replace(" ", "-")
    slug = slug.strip("-")
    return slug


def _extract_markdown_anchors(markdown_path: Path) -> set[str]:
    anchors: set[str] = set()
    duplicate_counts: dict[str, int] = {}

    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        raw_heading = match.group(1).strip().strip("#").strip()
        base_slug = _slugify_heading(raw_heading)
        if not base_slug:
            continue
        if base_slug in duplicate_counts:
            duplicate_counts[base_slug] += 1
            slug = f"{base_slug}-{duplicate_counts[base_slug]}"
        else:
            duplicate_counts[base_slug] = 0
            slug = base_slug
        anchors.add(slug)
    return anchors


def _validate_file(
    markdown_file: Path,
    anchor_cache: dict[Path, set[str]],
    repo_root: Path,
) -> list[LinkViolation]:
    violations: list[LinkViolation] = []
    lines = markdown_file.read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(lines, start=1):
        raw_targets: list[str] = []
        raw_targets.extend(match.group(1) for match in INLINE_LINK_RE.finditer(line))
        reference_match = REFERENCE_LINK_RE.match(line)
        if reference_match:
            raw_targets.append(reference_match.group(1))

        for raw_target in raw_targets:
            cleaned = _strip_title_segment(raw_target)
            if not cleaned:
                continue
            if _is_external_target(cleaned):
                continue

            path_part, fragment = _split_link_target(cleaned)
            decoded_path = unquote(path_part).strip()
            decoded_fragment = unquote(fragment).strip().lower()

            if decoded_path == "":
                target_path = markdown_file
            else:
                candidate = Path(decoded_path).expanduser()
                target_path = candidate if candidate.is_absolute() else (markdown_file.parent / candidate)
                if not target_path.exists():
                    canonical_target = resolve_canonical_target_path(
                        target_path=target_path,
                        repo_root=repo_root,
                    )
                    if canonical_target is not None:
                        target_path = canonical_target

            if not target_path.exists():
                violations.append(
                    LinkViolation(
                        file_path=_to_display(markdown_file),
                        line_number=line_number,
                        target=raw_target,
                        message="MARKDOWN_LINK_MISSING_TARGET",
                    )
                )
                continue

            if decoded_fragment and target_path.is_file() and target_path.suffix.lower() == ".md":
                resolved_target = target_path.resolve()
                if resolved_target not in anchor_cache:
                    anchor_cache[resolved_target] = _extract_markdown_anchors(resolved_target)
                if decoded_fragment not in anchor_cache[resolved_target]:
                    violations.append(
                        LinkViolation(
                            file_path=_to_display(markdown_file),
                            line_number=line_number,
                            target=raw_target,
                            message="MARKDOWN_LINK_MISSING_ANCHOR",
                        )
                    )

    return violations


def check_markdown_links(targets: list[str], root: Path) -> list[LinkViolation]:
    """Validate all local markdown links and anchors across the given targets.

    Resolves target paths/directories, checks that linked files exist and that
    fragment anchors match headings in the destination document.
    """
    markdown_files = _resolve_targets(targets, root)
    anchor_cache: dict[Path, set[str]] = {}
    violations: list[LinkViolation] = []

    for markdown_file in markdown_files:
        violations.extend(_validate_file(markdown_file, anchor_cache, root))

    return violations


def main() -> int:
    """CLI entry point. Parses args and runs the markdown link integrity check."""
    parser = argparse.ArgumentParser(description="Check local markdown link integrity.")
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Markdown files or directories to scan (default: vision STATUS_LEDGER.md)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(PROJECT_META_ROOT),
        help="Repository root for relative target resolution",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    try:
        violations = check_markdown_links(args.targets, repo_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if violations:
        print("Markdown link integrity check failed:")
        for violation in violations:
            print(f"- {violation.format()}")
        return 1

    print("Markdown link integrity OK: no missing local targets or anchors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
