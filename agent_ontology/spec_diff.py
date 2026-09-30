#!/usr/bin/env python3
"""
Spec Diff Tool — structured comparison between two YAML agent specs.

Usage:
    python3 spec_diff.py specs/react.yaml specs/autogpt.yaml
    python3 spec_diff.py old_version.yaml new_version.yaml --json
"""

import argparse
import json
import os
import sys
import yaml

# ANSI colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def load_spec(path):
    with open(path) as f:
        return yaml.safe_load(f)


def diff_lists(old_items, new_items, key_fn):
    """Diff two lists by a key function. Returns (added, removed, common_pairs)."""
    old_map = {key_fn(item): item for item in old_items}
    new_map = {key_fn(item): item for item in new_items}

    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    added = [new_map[k] for k in sorted(new_keys - old_keys)]
    removed = [old_map[k] for k in sorted(old_keys - new_keys)]
    common = [(old_map[k], new_map[k]) for k in sorted(old_keys & new_keys)]

    return added, removed, common


def diff_items(old_item, new_item, ignore_keys=None):
    """Find changed fields between two dicts."""
    ignore = set(ignore_keys or [])
    changes = []
    all_keys = set(old_item.keys()) | set(new_item.keys())

    for k in sorted(all_keys):
        if k in ignore:
            continue
        old_val = old_item.get(k)
        new_val = new_item.get(k)
        if old_val != new_val:
            changes.append((k, old_val, new_val))

    return changes


def diff_specs(old_spec, new_spec):
    """Compute structured diff between two specs."""
    result = {
        "metadata": {},
        "entities": {"added": [], "removed": [], "changed": []},
        "processes": {"added": [], "removed": [], "changed": []},
        "edges": {"added": [], "removed": [], "changed": []},
        "schemas": {"added": [], "removed": [], "changed": []},
    }

    # Metadata changes
    for key in ["name", "version", "description", "entry_point"]:
        old_val = old_spec.get(key)
        new_val = new_spec.get(key)
        if old_val != new_val:
            result["metadata"][key] = {"old": old_val, "new": new_val}

    # Entities
    added, removed, common = diff_lists(
        old_spec.get("entities", []),
        new_spec.get("entities", []),
        lambda e: e.get("id", "")
    )
    result["entities"]["added"] = [e.get("id") for e in added]
    result["entities"]["removed"] = [e.get("id") for e in removed]
    for old_e, new_e in common:
        changes = diff_items(old_e, new_e)
        if changes:
            result["entities"]["changed"].append({
                "id": old_e.get("id"),
                "changes": [{"field": k, "old": o, "new": n} for k, o, n in changes]
            })

    # Processes
    added, removed, common = diff_lists(
        old_spec.get("processes", []),
        new_spec.get("processes", []),
        lambda p: p.get("id", "")
    )
    result["processes"]["added"] = [p.get("id") for p in added]
    result["processes"]["removed"] = [p.get("id") for p in removed]
    for old_p, new_p in common:
        changes = diff_items(old_p, new_p)
        if changes:
            result["processes"]["changed"].append({
                "id": old_p.get("id"),
                "changes": [{"field": k, "old": o, "new": n} for k, o, n in changes]
            })

    # Edges (key by from+to+type)
    def edge_key(e):
        return f"{e.get('from', '')}→{e.get('to', '')}:{e.get('type', '')}"

    added, removed, common = diff_lists(
        old_spec.get("edges", []),
        new_spec.get("edges", []),
        edge_key
    )
    result["edges"]["added"] = [edge_key(e) for e in added]
    result["edges"]["removed"] = [edge_key(e) for e in removed]
    for old_e, new_e in common:
        changes = diff_items(old_e, new_e, ignore_keys=["from", "to", "type"])
        if changes:
            result["edges"]["changed"].append({
                "edge": edge_key(old_e),
                "changes": [{"field": k, "old": o, "new": n} for k, o, n in changes]
            })

    # Schemas
    added, removed, common = diff_lists(
        old_spec.get("schemas", []),
        new_spec.get("schemas", []),
        lambda s: s.get("name", "")
    )
    result["schemas"]["added"] = [s.get("name") for s in added]
    result["schemas"]["removed"] = [s.get("name") for s in removed]
    for old_s, new_s in common:
        changes = diff_items(old_s, new_s, ignore_keys=["name"])
        if changes:
            result["schemas"]["changed"].append({
                "name": old_s.get("name"),
                "changes": [{"field": k, "old": o, "new": n} for k, o, n in changes]
            })

    return result


def print_diff(diff, old_name, new_name):
    """Print colored diff to terminal."""
    print(f"\n{BOLD}Spec Diff: {old_name} → {new_name}{RESET}")
    print("=" * 60)

    # Metadata
    if diff["metadata"]:
        print(f"\n{CYAN}Metadata:{RESET}")
        for key, val in diff["metadata"].items():
            print(f"  {YELLOW}{key}:{RESET} {RED}{val['old']}{RESET} → {GREEN}{val['new']}{RESET}")

    # Count totals
    total_added = 0
    total_removed = 0
    total_changed = 0

    for section in ["entities", "processes", "edges", "schemas"]:
        d = diff[section]
        n_add = len(d["added"])
        n_rem = len(d["removed"])
        n_chg = len(d["changed"])
        total_added += n_add
        total_removed += n_rem
        total_changed += n_chg

        if n_add == 0 and n_rem == 0 and n_chg == 0:
            continue

        print(f"\n{CYAN}{section.title()} ({GREEN}+{n_add}{RESET}, {RED}-{n_rem}{RESET}, {YELLOW}~{n_chg}{RESET}{CYAN}):{RESET}")

        for item in d["added"]:
            print(f"  {GREEN}+ {item}{RESET}")
        for item in d["removed"]:
            print(f"  {RED}- {item}{RESET}")
        for item in d["changed"]:
            item_id = item.get("id") or item.get("name") or item.get("edge", "?")
            print(f"  {YELLOW}~ {item_id}:{RESET}")
            for change in item["changes"]:
                field = change["field"]
                old = str(change["old"])[:60] if change["old"] is not None else "∅"
                new = str(change["new"])[:60] if change["new"] is not None else "∅"
                print(f"    {DIM}{field}:{RESET} {RED}{old}{RESET} → {GREEN}{new}{RESET}")

    # Summary
    print(f"\n{'=' * 60}")
    total = total_added + total_removed + total_changed
    if total == 0:
        print(f"  {DIM}No differences found.{RESET}")
    else:
        print(f"  {GREEN}+{total_added}{RESET} added, {RED}-{total_removed}{RESET} removed, {YELLOW}~{total_changed}{RESET} changed")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Spec Diff Tool — structured comparison between two YAML agent specs"
    )
    parser.add_argument("old", help="First (old/base) spec file")
    parser.add_argument("new", help="Second (new/modified) spec file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    old_spec = load_spec(args.old)
    new_spec = load_spec(args.new)

    diff = diff_specs(old_spec, new_spec)

    if args.json:
        print(json.dumps(diff, indent=2, default=str))
    else:
        old_name = os.path.basename(args.old)
        new_name = os.path.basename(args.new)
        print_diff(diff, old_name, new_name)


if __name__ == "__main__":
    main()
