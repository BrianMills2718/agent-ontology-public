#!/usr/bin/env python3
"""Check that documentation is updated when coupled source files change.

Usage:
    python scripts/check_doc_coupling.py [--base BASE_REF] [--suggest]
    python scripts/check_doc_coupling.py --staged  # For pre-commit hook
    python scripts/check_doc_coupling.py --staged --strict --ack-file .doc-coupling-acks

Compares current branch against BASE_REF (default: origin/main) to find
changed files, then checks if coupled docs were also updated.

The --staged option checks only staged files, suitable for pre-commit hooks.
If source files are staged AND their coupled docs are also staged, it passes.

The --ack-file option loads acknowledged gaps from a YAML file. Each entry
requires a ``path`` and a non-empty ``reason``. Acknowledged gaps are
downgraded from strict violations to warnings (printed, non-blocking).

Exit codes:
    0 - All couplings satisfied (or no coupled changes)
    1 - Missing doc updates (strict violations)
"""

import argparse
import fnmatch
import glob
import yaml  # type: ignore[import-untyped]
import subprocess
import sys
from pathlib import Path


def load_yaml(path: Path):
    """Load YAML from path and return dict-like object."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_couplings(config_path: Path) -> list[dict]:
    """Load coupling definitions from legacy or unified config."""
    data = load_yaml(config_path)

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    # New unified format: scripts/relationships.yaml
    couplings = data.get("couplings")
    if isinstance(couplings, list):
        return couplings

    # Legacy format: doc_coupling.yaml
    legacy_couplings = data.get("couplings")
    return legacy_couplings if isinstance(legacy_couplings, list) else []


def resolve_config_path(config_arg: str) -> Path:
    """Resolve default config path with legacy fallback."""
    requested = Path(config_arg)

    if requested.exists():
        return requested

    # Prefer unified file, but fallback to legacy when missing.
    if requested.name == "relationships.yaml":
        legacy = Path("scripts/doc_coupling.yaml")
        if legacy.exists():
            return legacy

    return requested


def get_changed_files(base_ref: str) -> set[str]:
    """Get files changed between base_ref and HEAD."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(result.stdout.strip().split("\n")) - {""}
    except subprocess.CalledProcessError:
        # Fallback: compare against HEAD~1 for local testing
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return set(result.stdout.strip().split("\n")) - {""}
        except subprocess.CalledProcessError:
            return set()


def get_staged_files() -> set[str]:
    """Get files staged for commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(result.stdout.strip().split("\n")) - {""}
    except subprocess.CalledProcessError:
        return set()


def load_couplings(config_path: Path) -> list[dict]:
    """Load coupling definitions from YAML."""
    return resolve_couplings(config_path)


def validate_config(couplings: list[dict]) -> list[str]:
    """Validate that all referenced files in config exist.

    Returns list of warnings for missing files.
    """
    warnings = []
    for coupling in couplings:
        for doc in coupling.get("docs", []):
            if any(ch in doc for ch in "*?[]"):
                if not glob.glob(doc, recursive=True):
                    warnings.append(f"Coupled doc glob doesn't match any files: {doc}")
                continue
            if not Path(doc).exists():
                warnings.append(f"Coupled doc doesn't exist: {doc}")
        # Don't validate source patterns - they're globs
    return warnings


def matches_any_pattern(filepath: str, patterns: list[str]) -> bool:
    """Check if filepath matches any glob pattern."""
    for pattern in patterns:
        if fnmatch.fnmatch(filepath, pattern):
            return True
        # Basename fallback only for patterns with glob wildcards (e.g. "*.md").
        # Plain filenames like "CLAUDE.md" must match the full path to avoid
        # false positives where docs/plans/CLAUDE.md matches root CLAUDE.md.
        if any(c in pattern for c in "*?[") and fnmatch.fnmatch(Path(filepath).name, pattern):
            return True
    return False


def load_ack_file(ack_path: Path) -> dict[str, str]:
    """Load acknowledged doc-coupling gaps from a YAML file.

    Returns a mapping of normalized path -> reason. Entries with empty
    or missing reasons are silently skipped (they will not suppress gaps).
    """
    if not ack_path.exists():
        return {}
    try:
        data = yaml.safe_load(ack_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    acks: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path", "")
        reason = str(entry.get("reason", "")).strip()
        if path and reason:
            acks[str(Path(path))] = reason
    return acks


def filter_violations_with_acks(
    strict_violations: list[dict],
    acks: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Separate strict violations into remaining violations and acknowledged gaps.

    A violation is acknowledged when ALL of its expected docs have an ack entry
    with a non-empty reason. Partially acknowledged violations remain strict.
    """
    remaining = []
    acknowledged = []
    for v in strict_violations:
        expected = v.get("expected_docs", [])
        all_acked = expected and all(
            any(
                fnmatch.fnmatch(ack_path, doc) or fnmatch.fnmatch(doc, ack_path)
                for ack_path in acks
            )
            for doc in expected
        )
        if all_acked:
            acknowledged.append(v)
        else:
            remaining.append(v)
    return remaining, acknowledged


def matches_any_doc(filepath: str, docs: list[str]) -> bool:
    """Check whether a changed file matches any configured documentation target."""

    return matches_any_pattern(filepath, docs)


def run_verify_sync(cmd: str) -> bool:
    """Run a verify_sync command. Returns True if docs are verified in sync."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def check_couplings(
    changed_files: set[str], couplings: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Check which couplings have source changes without doc changes.

    Returns tuple of (strict_violations, soft_warnings).

    Couplings with a ``verify_sync`` command get an automatic check:
    if the command exits 0, the docs are verified as current and the
    co-modification requirement is waived.
    """
    strict_violations = []
    soft_warnings = []

    for coupling in couplings:
        sources = coupling.get("sources", [])
        docs = coupling.get("docs", [])
        description = coupling.get("description", "")
        is_soft = coupling.get("soft", False)
        verify_sync = coupling.get("verify_sync")

        # Find which source patterns matched
        matched_sources = []
        for changed in changed_files:
            if matches_any_pattern(changed, sources):
                matched_sources.append(changed)

        if not matched_sources:
            continue  # No source files changed for this coupling

        # Check if any coupled doc was updated
        docs_updated = any(
            matches_any_doc(changed, docs) for changed in changed_files
        )

        if not docs_updated:
            # If coupling has verify_sync, check if docs are already current
            if verify_sync and run_verify_sync(verify_sync):
                continue  # Docs verified in sync — no co-modification needed

            violation = {
                "description": description,
                "changed_sources": matched_sources,
                "expected_docs": docs,
                "soft": is_soft,
            }
            if is_soft:
                soft_warnings.append(violation)
            else:
                strict_violations.append(violation)

    return strict_violations, soft_warnings


def print_suggestions(changed_files: set[str], couplings: list[dict]) -> None:
    """Print which docs should be updated based on changed files."""
    print("Based on your changes, consider updating:\n")

    suggestions: dict[str, list[str]] = {}  # doc -> [reasons]

    for coupling in couplings:
        sources = coupling.get("sources", [])
        docs = coupling.get("docs", [])
        description = coupling.get("description", "")

        for changed in changed_files:
            if matches_any_pattern(changed, sources):
                for doc in docs:
                    if not matches_any_doc(changed, [doc]):
                        if doc not in suggestions:
                            suggestions[doc] = []
                        suggestions[doc].append(f"{changed} ({description})")

    if not suggestions:
        print("  No documentation updates needed.")
        return

    for doc, reasons in sorted(suggestions.items()):
        print(f"  {doc}")
        for reason in reasons[:3]:  # Limit to 3 reasons
            print(f"    <- {reason}")
        if len(reasons) > 3:
            print(f"    ... and {len(reasons) - 3} more")
        print()


def main() -> int:
    """CLI entry point. Parses args and checks that docs are updated when coupled source files change."""
    parser = argparse.ArgumentParser(description="Check doc-code coupling")
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base ref to compare against (default: origin/main)",
    )
    parser.add_argument(
        "--config",
        default="scripts/relationships.yaml",
        help="Path to coupling config (default: scripts/relationships.yaml)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on strict violations (default: warn only)",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Show which docs to update based on changes",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate that all docs in config exist",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Check staged files only (for pre-commit hook)",
    )
    parser.add_argument(
        "--ack-file",
        default=None,
        help="Path to YAML file with acknowledged gaps (path + reason per entry)",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    couplings = load_couplings(config_path)

    # Validate config if requested
    if args.validate_config:
        warnings = validate_config(couplings)
        if warnings:
            print("Config validation warnings:")
            for w in warnings:
                print(f"  - {w}")
            return 1
        print("Config validation passed.")
        return 0

    # Get changed files based on mode
    if args.staged:
        changed_files = get_staged_files()
        if not changed_files:
            # No staged files = nothing to check
            return 0
    else:
        changed_files = get_changed_files(args.base)
        if not changed_files:
            print("No changed files detected.")
            return 0

    # Suggest mode
    if args.suggest:
        print_suggestions(changed_files, couplings)
        return 0

    strict_violations, soft_warnings = check_couplings(changed_files, couplings)

    # Apply acknowledgments if provided
    acked_warnings: list[dict] = []
    if args.ack_file:
        acks = load_ack_file(Path(args.ack_file))
        if acks:
            strict_violations, acked_warnings = filter_violations_with_acks(
                strict_violations, acks
            )

    if not strict_violations and not soft_warnings and not acked_warnings:
        print("Doc-code coupling check passed.")
        return 0

    # Print violations
    if strict_violations:
        print("=" * 60)
        print("DOC-CODE COUPLING VIOLATIONS (must fix)")
        print("=" * 60)
        print()
        for v in strict_violations:
            print(f"  {v['description']}")
            print(f"    Changed: {', '.join(v['changed_sources'][:3])}")
            if len(v['changed_sources']) > 3:
                print(f"             ... and {len(v['changed_sources']) - 3} more")
            print(f"    Update:  {', '.join(v['expected_docs'])}")
            print()

    if soft_warnings:
        print("=" * 60)
        print("DOC-CODE COUPLING WARNINGS (consider updating)")
        print("=" * 60)
        print()
        for v in soft_warnings:
            print(f"  {v['description']}")
            print(f"    Changed: {', '.join(v['changed_sources'][:3])}")
            if len(v['changed_sources']) > 3:
                print(f"             ... and {len(v['changed_sources']) - 3} more")
            print(f"    Consider: {', '.join(v['expected_docs'])}")
            print()

    if acked_warnings:
        acks = load_ack_file(Path(args.ack_file)) if args.ack_file else {}
        print("=" * 60)
        print("ACKNOWLEDGED GAPS (non-blocking)")
        print("=" * 60)
        print()
        for v in acked_warnings:
            print(f"  {v['description']}")
            print(f"    Changed: {', '.join(v['changed_sources'][:3])}")
            for doc in v["expected_docs"]:
                matching_reason = next(
                    (
                        acks[p]
                        for p in acks
                        if fnmatch.fnmatch(p, doc) or fnmatch.fnmatch(doc, p)
                    ),
                    "acknowledged",
                )
                print(f"    Ack: {doc} — {matching_reason}")
            print()

    if strict_violations:
        print("=" * 60)
        print("If docs are already accurate, update 'Last verified' date.")
        print("=" * 60)

    return 1 if (args.strict and strict_violations) else 0


if __name__ == "__main__":
    sys.exit(main())
