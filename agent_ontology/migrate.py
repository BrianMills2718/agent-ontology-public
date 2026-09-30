#!/usr/bin/env python3
"""
Agent Ontology Spec Migration Tool
=====================================
Handles spec versioning and migration as the ontology evolves.
Transforms spec YAML files from older versions to newer ones by
applying a chain of migration functions.

Usage:
    python3 migrate.py specs/react.yaml --to 2.0            # Migrate one spec
    python3 migrate.py --all specs/ --to 2.0                 # Migrate all specs in a directory
    python3 migrate.py specs/react.yaml --to 2.0 --dry-run   # Preview changes without modifying
    python3 migrate.py --list-versions                        # Show available migrations
"""

import argparse
import copy
import os
import shutil
import sys

import yaml


# ── Color helpers ───────────────────────────────────────────

def _supports_color():
    """Check if stdout supports ANSI color."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


def _c(text, code):
    if _COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text


def green(text):
    return _c(text, "32")


def yellow(text):
    return _c(text, "33")


def red(text):
    return _c(text, "31")


def cyan(text):
    return _c(text, "36")


def bold(text):
    return _c(text, "1")


# ── YAML loading / saving ──────────────────────────────────

def load_yaml(path):
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path, data):
    """Write a dict to a YAML file, preserving readable formatting."""
    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )


# ── Migration functions ────────────────────────────────────
#
# Each migration function takes a spec dict, mutates it in-place,
# and returns a list of human-readable change descriptions.

def migrate_1_0_to_1_1(spec):
    """Migrate from 1.0 to 1.1.

    Changes:
      - Add `retention: session` to any store entity missing it.
      - Add `description` field (empty string) to any process missing it.
    """
    changes = []

    # -- Store entities: ensure retention field --
    for entity in spec.get("entities", []):
        if entity.get("type") == "store" and "retention" not in entity:
            entity["retention"] = "session"
            changes.append(
                f"  + Added retention: session to store '{entity.get('id', '?')}'"
            )

    # -- Processes: ensure description field --
    for process in spec.get("processes", []):
        if "description" not in process:
            process["description"] = ""
            changes.append(
                f"  + Added description: '' to process '{process.get('id', '?')}'"
            )

    spec["version"] = "1.1"
    changes.append("  * Updated version: 1.0 -> 1.1")
    return changes


def migrate_1_1_to_2_0(spec):
    """Migrate from 1.1 to 2.0.

    Changes:
      - Add `tags: []` to every entity.
      - Ensure all schemas have a `description` field.
    """
    changes = []

    # -- Add tags: [] to every entity --
    for entity in spec.get("entities", []):
        if "tags" not in entity:
            entity["tags"] = []
            changes.append(
                f"  + Added tags: [] to entity '{entity.get('id', '?')}'"
            )

    # -- Ensure all schemas have a description field --
    for schema in spec.get("schemas", []):
        if "description" not in schema:
            schema["description"] = ""
            changes.append(
                f"  + Added description: '' to schema '{schema.get('name', '?')}'"
            )

    spec["version"] = "2.0"
    changes.append("  * Updated version: 1.1 -> 2.0")
    return changes


# ── Migration registry ─────────────────────────────────────

MIGRATIONS = {
    ("1.0", "1.1"): migrate_1_0_to_1_1,
    ("1.1", "2.0"): migrate_1_1_to_2_0,
}

# Ordered list of all known versions (used to build migration chains)
ALL_VERSIONS = ["1.0", "1.1", "2.0"]


def get_migration_chain(from_version, to_version):
    """Build a list of (from, to, func) tuples to migrate between versions.

    Returns None if no valid chain exists.
    """
    if from_version == to_version:
        return []

    try:
        start_idx = ALL_VERSIONS.index(from_version)
        end_idx = ALL_VERSIONS.index(to_version)
    except ValueError:
        return None

    if start_idx >= end_idx:
        return None  # Cannot migrate backwards

    chain = []
    for i in range(start_idx, end_idx):
        v_from = ALL_VERSIONS[i]
        v_to = ALL_VERSIONS[i + 1]
        func = MIGRATIONS.get((v_from, v_to))
        if func is None:
            return None  # Missing migration step
        chain.append((v_from, v_to, func))
    return chain


# ── Core migration logic ───────────────────────────────────

def migrate_spec(spec, to_version, dry_run=False):
    """Migrate a spec dict to the target version.

    Args:
        spec: The spec dict (will be mutated unless dry_run).
        to_version: Target version string.
        dry_run: If True, operate on a copy and leave the original unchanged.

    Returns:
        (migrated_spec, all_changes) where all_changes is a list of strings.
        Returns (None, error_message_list) on failure.
    """
    from_version = str(spec.get("version", "1.0"))

    if from_version == to_version:
        return spec, [f"  Already at version {to_version}, no changes needed."]

    chain = get_migration_chain(from_version, to_version)
    if chain is None:
        return None, [
            f"  No migration path from {from_version} to {to_version}."
        ]

    work = copy.deepcopy(spec) if dry_run else spec
    all_changes = []

    for v_from, v_to, func in chain:
        all_changes.append(f"  Migration {v_from} -> {v_to}:")
        step_changes = func(work)
        all_changes.extend(step_changes)

    return work, all_changes


def backup_file(path):
    """Create a .bak backup of a file. Returns the backup path."""
    bak_path = path + ".bak"
    shutil.copy2(path, bak_path)
    return bak_path


# ── Single file migration ──────────────────────────────────

def migrate_file(path, to_version, dry_run=False, create_backup=True):
    """Migrate a single spec file.

    Returns:
        (success: bool, changes: list[str])
    """
    try:
        spec = load_yaml(path)
    except Exception as exc:
        return False, [f"  Failed to load {path}: {exc}"]

    name = spec.get("name", os.path.basename(path))
    from_version = str(spec.get("version", "1.0"))

    result, changes = migrate_spec(spec, to_version, dry_run=dry_run)

    if result is None:
        return False, changes

    if from_version == to_version:
        return True, changes

    if dry_run:
        return True, changes

    # Write the migrated spec
    if create_backup:
        bak = backup_file(path)
        changes.append(f"  Backup saved to {bak}")

    save_yaml(path, result)
    changes.append(f"  Wrote migrated spec to {path}")

    return True, changes


# ── Batch migration ─────────────────────────────────────────

def migrate_directory(dir_path, to_version, dry_run=False, create_backup=True):
    """Migrate all .yaml spec files in a directory.

    Returns:
        (success_count, fail_count, results_by_file)
    """
    results = {}
    success_count = 0
    fail_count = 0

    yaml_files = sorted(
        f for f in os.listdir(dir_path) if f.endswith(".yaml")
    )

    if not yaml_files:
        return 0, 0, {"(none)": [" No YAML files found in directory."]}

    for fname in yaml_files:
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue

        ok, changes = migrate_file(
            fpath, to_version, dry_run=dry_run, create_backup=create_backup
        )
        results[fname] = changes
        if ok:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count, results


# ── Output formatting ───────────────────────────────────────

def format_version_list():
    """Return a formatted string listing all known versions and migrations."""
    lines = []
    lines.append(bold("Known spec versions:"))
    for v in ALL_VERSIONS:
        lines.append(f"  {v}")
    lines.append("")
    lines.append(bold("Available migrations:"))
    for (v_from, v_to), func in MIGRATIONS.items():
        doc = (func.__doc__ or "").strip().split("\n")[0]
        lines.append(f"  {v_from} -> {v_to}  {cyan(doc)}")
    lines.append("")
    lines.append(bold("Migration chains:"))
    for i, v_start in enumerate(ALL_VERSIONS):
        for v_end in ALL_VERSIONS[i + 1 :]:
            chain = get_migration_chain(v_start, v_end)
            if chain:
                path = " -> ".join(
                    [chain[0][0]] + [step[1] for step in chain]
                )
                lines.append(f"  {path}")
    return "\n".join(lines)


def print_file_result(fname, changes, dry_run=False):
    """Print the migration result for a single file."""
    mode = yellow("[DRY RUN] ") if dry_run else ""
    print(f"\n{mode}{bold(fname)}")
    for line in changes:
        # Colorize change indicators
        if line.strip().startswith("+"):
            print(green(line))
        elif line.strip().startswith("~"):
            print(yellow(line))
        elif line.strip().startswith("*"):
            print(cyan(line))
        elif "No migration path" in line or "Failed" in line:
            print(red(line))
        else:
            print(line)


# ── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agent Ontology Spec Migration Tool -- version and migrate agent specs",
        epilog="Examples:\n"
               "  python3 migrate.py specs/react.yaml --to 2.0\n"
               "  python3 migrate.py --all specs/ --to 2.0\n"
               "  python3 migrate.py specs/react.yaml --to 2.0 --dry-run\n"
               "  python3 migrate.py --list-versions\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "spec",
        nargs="?",
        help="Path to a spec YAML file, or directory with --all",
    )
    parser.add_argument(
        "--to",
        dest="to_version",
        help="Target version to migrate to (e.g. 1.1, 2.0)",
    )
    parser.add_argument(
        "--all",
        dest="all_dir",
        metavar="DIR",
        help="Migrate all *.yaml files in the given directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak backup files before modifying",
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="Show all known versions and available migrations",
    )

    args = parser.parse_args()

    # ── List versions mode ──────────────────────────────────
    if args.list_versions:
        print(format_version_list())
        sys.exit(0)

    # ── Validate arguments ──────────────────────────────────
    if args.all_dir:
        # Batch mode
        if not args.to_version:
            print(red("Error: --to is required when migrating"), file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(args.all_dir):
            print(
                red(f"Error: {args.all_dir} is not a directory"),
                file=sys.stderr,
            )
            sys.exit(1)
        if args.to_version not in ALL_VERSIONS:
            print(
                red(f"Error: unknown target version '{args.to_version}'. "
                    f"Known versions: {', '.join(ALL_VERSIONS)}"),
                file=sys.stderr,
            )
            sys.exit(1)

        create_backup = not args.no_backup
        success, fail, results = migrate_directory(
            args.all_dir,
            args.to_version,
            dry_run=args.dry_run,
            create_backup=create_backup,
        )

        for fname, changes in results.items():
            print_file_result(fname, changes, dry_run=args.dry_run)

        # Summary
        print()
        mode = yellow("[DRY RUN] ") if args.dry_run else ""
        print(
            f"{mode}{bold('Summary:')} {green(str(success) + ' succeeded')}, "
            f"{red(str(fail) + ' failed') if fail else '0 failed'} "
            f"(target: {args.to_version})"
        )

        sys.exit(1 if fail else 0)

    elif args.spec:
        # Single file mode
        if not args.to_version:
            print(red("Error: --to is required when migrating"), file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.spec):
            print(
                red(f"Error: {args.spec} not found"),
                file=sys.stderr,
            )
            sys.exit(1)
        if args.to_version not in ALL_VERSIONS:
            print(
                red(f"Error: unknown target version '{args.to_version}'. "
                    f"Known versions: {', '.join(ALL_VERSIONS)}"),
                file=sys.stderr,
            )
            sys.exit(1)

        create_backup = not args.no_backup
        ok, changes = migrate_file(
            args.spec,
            args.to_version,
            dry_run=args.dry_run,
            create_backup=create_backup,
        )

        print_file_result(args.spec, changes, dry_run=args.dry_run)

        if not ok:
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
