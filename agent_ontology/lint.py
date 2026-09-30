#!/usr/bin/env python3
"""Ontology linting tool — detects common design anti-patterns in agent specs.

Complements validate.py (which checks structural correctness) by flagging
things that are technically valid YAML but likely indicate spec problems.
"""

import argparse
import glob
import json
import os
import sys
from collections import namedtuple, defaultdict

import yaml

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

LintIssue = namedtuple("LintIssue", ["code", "severity", "rule_name", "message"])

SEVERITY_RANK = {"info": 0, "warn": 1, "error": 2}

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _red(t):
    return _c("1;31", t)


def _yellow(t):
    return _c("1;33", t)


def _cyan(t):
    return _c("36", t)


def _dim(t):
    return _c("2", t)


def _bold(t):
    return _c("1", t)


def _severity_color(severity, text):
    if severity == "error":
        return _red(text)
    if severity == "warn":
        return _yellow(text)
    return _cyan(text)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_yaml(path):
    """Load a YAML spec file and return the parsed dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Individual lint rules
# ---------------------------------------------------------------------------


def _check_god_agent(spec):
    """L001: A single agent receives >70% of all invoke edges."""
    issues = []
    edges = spec.get("edges", []) or []
    invoke_edges = [e for e in edges if e.get("type") == "invoke"]
    if not invoke_edges:
        return issues

    target_counts = defaultdict(int)
    for e in invoke_edges:
        target_counts[e.get("to", "")] += 1

    total = len(invoke_edges)
    for target, count in target_counts.items():
        ratio = count / total
        if ratio > 0.70:
            pct = int(ratio * 100)
            issues.append(LintIssue(
                "L001", "warn", "god-agent",
                f"Agent '{target}' receives {pct}% of all invoke edges "
                f"({count}/{total}). Consider decomposing responsibilities."
            ))
    return issues


def _check_linear_chain(spec):
    """L002: Spec is a pure linear chain — no gates, no loops, no fan-out."""
    issues = []
    processes = spec.get("processes", []) or []
    edges = spec.get("edges", []) or []

    # If there are no processes or edges, skip — too trivial to lint.
    if len(processes) < 2 or not edges:
        return issues

    has_gate = any(p.get("type") == "gate" for p in processes)
    has_loop = any(e.get("type") == "loop" for e in edges)

    # Check fan-out: any node with >1 outgoing flow/invoke edge.
    flow_types = {"flow", "invoke"}
    outgoing = defaultdict(int)
    for e in edges:
        if e.get("type") in flow_types:
            outgoing[e.get("from", "")] += 1
    has_fanout = any(c > 1 for c in outgoing.values())

    if not has_gate and not has_loop and not has_fanout:
        issues.append(LintIssue(
            "L002", "info", "linear-chain",
            "Spec is a pure linear chain with no branching or loops"
        ))
    return issues


def _check_dead_store(spec):
    """L003: A store entity has write edges but no read edges, or vice versa."""
    issues = []
    entities = spec.get("entities", []) or []
    edges = spec.get("edges", []) or []

    store_ids = {e["id"] for e in entities if e.get("type") == "store"}
    if not store_ids:
        return issues

    read_targets = set()
    write_targets = set()
    for e in edges:
        etype = e.get("type", "")
        if etype == "read":
            read_targets.add(e.get("to", ""))
        elif etype == "write":
            write_targets.add(e.get("to", ""))

    for sid in sorted(store_ids):
        has_read = sid in read_targets
        has_write = sid in write_targets
        if has_write and not has_read:
            issues.append(LintIssue(
                "L003", "warn", "dead-store",
                f"Store '{sid}' has write edges but no read edges — "
                f"data is written but never consumed"
            ))
        elif has_read and not has_write:
            issues.append(LintIssue(
                "L003", "warn", "dead-store",
                f"Store '{sid}' has read edges but no write edges — "
                f"reading from a never-written store"
            ))
    return issues


def _check_unbounded_loop(spec):
    """L004: A loop edge with no max_iterations and no gate at target."""
    issues = []
    edges = spec.get("edges", []) or []
    processes = spec.get("processes", []) or []

    # Build set of gate process IDs for quick lookup.
    gate_ids = {p["id"] for p in processes if p.get("type") == "gate"}

    loop_edges = [e for e in edges if e.get("type") == "loop"]
    for le in loop_edges:
        if le.get("max_iterations") is not None:
            continue
        target = le.get("to", "")
        # Check if the loop target itself is a gate, or if there is a gate
        # reachable as an immediate successor of the target.
        if target in gate_ids:
            continue

        # Check one hop: does the target have a flow edge to a gate?
        successors = {e.get("to") for e in edges
                      if e.get("from") == target and e.get("type") == "flow"}
        if successors & gate_ids:
            continue

        src = le.get("from", "")
        issues.append(LintIssue(
            "L004", "warn", "unbounded-loop",
            f"Loop from '{src}' to '{target}' has no max_iterations "
            f"and no gate with termination logic"
        ))
    return issues


def _check_schema_mismatch(spec):
    """L005: Invoke edge output schema doesn't match downstream process data_in."""
    issues = []
    edges = spec.get("edges", []) or []
    processes = spec.get("processes", []) or []

    process_map = {p["id"]: p for p in processes}

    invoke_edges = [e for e in edges if e.get("type") == "invoke"]
    for ie in invoke_edges:
        output_schema = ie.get("output")
        if not output_schema:
            continue
        source = ie.get("from", "")
        # Find downstream processes connected via flow from the source.
        downstream_ids = [e.get("to") for e in edges
                          if e.get("from") == source and e.get("type") == "flow"]
        for did in downstream_ids:
            dp = process_map.get(did)
            if not dp:
                continue
            data_in = dp.get("data_in")
            if data_in and data_in != output_schema:
                issues.append(LintIssue(
                    "L005", "warn", "schema-mismatch",
                    f"Invoke edge output schema '{output_schema}' on '{source}' "
                    f"doesn't match downstream process '{did}' data_in '{data_in}'"
                ))
    return issues


def _check_orphan_schema(spec):
    """L006: A schema is defined but never referenced anywhere."""
    issues = []
    schemas = spec.get("schemas", []) or []
    if not schemas:
        return issues

    schema_names = {s.get("name") for s in schemas if s.get("name")}
    if not schema_names:
        return issues

    # Collect all schema references from the spec.
    referenced = set()

    # Edges: input, output fields.
    for e in (spec.get("edges", []) or []):
        for field in ("input", "output", "schema"):
            val = e.get(field)
            if val:
                referenced.add(val)

    # Processes: data_in, data_out fields.
    for p in (spec.get("processes", []) or []):
        for field in ("data_in", "data_out", "schema", "input", "output"):
            val = p.get(field)
            if val:
                referenced.add(val)

    # Entities: schema field.
    for ent in (spec.get("entities", []) or []):
        for field in ("schema", "data_in", "data_out", "input", "output"):
            val = ent.get(field)
            if val:
                referenced.add(val)

    for name in sorted(schema_names - referenced):
        issues.append(LintIssue(
            "L006", "info", "orphan-schema",
            f"Schema '{name}' is defined but never referenced"
        ))
    return issues


def _check_deep_nesting(spec):
    """L007: Longest path from entry node > 15."""
    issues = []
    edges = spec.get("edges", []) or []
    processes = spec.get("processes", []) or []

    if not processes or not edges:
        return issues

    # Build adjacency list from flow-like edges.
    flow_types = {"flow", "invoke", "loop"}
    adj = defaultdict(list)
    all_targets = set()
    all_sources = set()
    for e in edges:
        if e.get("type") in flow_types:
            src = e.get("from", "")
            tgt = e.get("to", "")
            adj[src].append(tgt)
            all_sources.add(src)
            all_targets.add(tgt)

    # Entry nodes: sources that are never targets (in flow edges).
    process_ids = {p["id"] for p in processes}
    entity_ids = {ent["id"] for ent in (spec.get("entities", []) or [])}
    all_ids = process_ids | entity_ids
    entry_nodes = (all_sources - all_targets) & all_ids
    if not entry_nodes:
        # Fall back to first process.
        entry_nodes = {processes[0]["id"]}

    # BFS/DFS to find longest path (with cycle detection).
    max_depth = 0

    def _dfs(node, depth, visited):
        nonlocal max_depth
        if depth > max_depth:
            max_depth = depth
        for nxt in adj.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                _dfs(nxt, depth + 1, visited)
                visited.discard(nxt)

    for entry in entry_nodes:
        _dfs(entry, 0, {entry})

    if max_depth > 15:
        issues.append(LintIssue(
            "L007", "warn", "deep-nesting",
            f"Graph depth is {max_depth} (longest path from entry). "
            f"Depth >15 suggests overly complex flow"
        ))
    return issues


def _check_fan_out_without_aggregation(spec):
    """L008: Fan-out with no downstream join point."""
    issues = []
    edges = spec.get("edges", []) or []

    flow_types = {"flow", "invoke"}
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for e in edges:
        if e.get("type") in flow_types:
            src = e.get("from", "")
            tgt = e.get("to", "")
            outgoing[src].append(tgt)
            incoming[tgt].append(src)

    # Fan-out nodes: nodes with >1 outgoing flow edges.
    fanout_nodes = {n for n, targets in outgoing.items() if len(targets) > 1}

    # Join nodes: nodes with >1 incoming flow edges.
    join_nodes = {n for n, sources in incoming.items() if len(sources) > 1}

    # For each fan-out, check if any of its branches eventually reach a
    # join node (BFS downstream from fan-out targets).
    for fnode in sorted(fanout_nodes):
        targets = outgoing[fnode]
        # BFS from each target to see if any reach a join node.
        has_join = False
        visited = set()
        queue = list(targets)
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in join_nodes:
                has_join = True
                break
            queue.extend(outgoing.get(current, []))

        if not has_join:
            issues.append(LintIssue(
                "L008", "warn", "fan-out-without-aggregation",
                f"Process '{fnode}' fans out to {len(targets)} targets "
                f"but there is no downstream join point — results may be lost"
            ))
    return issues


def _check_gate_without_default(spec):
    """L009: Gate process with branches but no default/else target."""
    issues = []
    processes = spec.get("processes", []) or []

    for p in processes:
        if p.get("type") != "gate":
            continue
        branches = p.get("branches", []) or []
        if not branches:
            continue
        # Look for a default/else branch.
        has_default = False
        for b in branches:
            cond = str(b.get("condition", "")).strip().lower()
            if cond in ("default", "else", "otherwise", "*", "true"):
                has_default = True
                break
        if not has_default:
            issues.append(LintIssue(
                "L009", "info", "gate-without-default",
                f"Gate '{p['id']}' has no default branch — "
                f"execution may hang if no branch matches"
            ))
    return issues


def _check_missing_error_handling(spec):
    """L010: No policy nodes in the spec."""
    issues = []
    entities = spec.get("entities", []) or []
    processes = spec.get("processes", []) or []

    has_policy = any(
        e.get("type") == "policy" for e in entities
    ) or any(
        p.get("type") == "policy" for p in processes
    )

    if not has_policy:
        issues.append(LintIssue(
            "L010", "info", "missing-error-handling",
            "No policy nodes defined — the spec has no error handling, "
            "retry, or guardrail mechanisms"
        ))
    return issues


# ---------------------------------------------------------------------------
# Main lint driver
# ---------------------------------------------------------------------------

ALL_RULES = [
    _check_god_agent,
    _check_linear_chain,
    _check_dead_store,
    _check_unbounded_loop,
    _check_schema_mismatch,
    _check_orphan_schema,
    _check_deep_nesting,
    _check_fan_out_without_aggregation,
    _check_gate_without_default,
    _check_missing_error_handling,
]


def lint_spec(spec):
    """Run all lint rules against a parsed spec dict.

    Returns a list of LintIssue namedtuples.
    """
    issues = []
    for rule_fn in ALL_RULES:
        issues.extend(rule_fn(spec))
    # Sort by severity (error first), then code.
    issues.sort(key=lambda i: (-SEVERITY_RANK.get(i.severity, 0), i.code))
    return issues


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_issue(issue):
    """Format a single issue for terminal output."""
    sev_tag = _severity_color(issue.severity, f"[{issue.severity}]")
    code = _bold(issue.code)
    name = _dim(issue.rule_name + ":")
    return f"  {code} {sev_tag}  {name} {issue.message}"


def _count_summary(issues):
    """Return a human-readable count summary string."""
    counts = defaultdict(int)
    for i in issues:
        counts[i.severity] += 1
    total = len(issues)
    parts = [
        f"{counts.get('error', 0)} errors",
        f"{counts.get('warn', 0)} warnings",
        f"{counts.get('info', 0)} info",
    ]
    noun = "issue" if total == 1 else "issues"
    return f"{total} {noun} ({', '.join(parts)})"


def _short_count(issues):
    """Short count for single-spec output."""
    total = len(issues)
    noun = "issue" if total == 1 else "issues"
    return f"{total} {noun}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _collect_files(paths, recurse_all=False):
    """Resolve CLI path arguments into a list of YAML file paths."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            if recurse_all:
                for root, _dirs, fnames in os.walk(p):
                    for fn in sorted(fnames):
                        if fn.endswith((".yaml", ".yml")):
                            files.append(os.path.join(root, fn))
            else:
                for fn in sorted(os.listdir(p)):
                    if fn.endswith((".yaml", ".yml")):
                        files.append(os.path.join(p, fn))
        else:
            # Could be a glob pattern already expanded by the shell.
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Lint agent ontology specs for design anti-patterns."
    )
    parser.add_argument(
        "specs", nargs="*", default=[],
        help="YAML spec file(s) or directory to lint"
    )
    parser.add_argument(
        "--all", dest="recurse_all", action="store_true",
        help="Recursively lint all YAML files in the given directory"
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--severity", choices=["info", "warn", "error"], default="info",
        help="Minimum severity to show (default: info)"
    )

    args = parser.parse_args()

    if not args.specs:
        parser.error("No spec files provided.")

    files = _collect_files(args.specs, recurse_all=args.recurse_all)
    if not files:
        print("No YAML files found.", file=sys.stderr)
        sys.exit(1)

    min_rank = SEVERITY_RANK.get(args.severity, 0)

    all_results = []  # For JSON output and summary.
    total_issues = 0
    exit_code = 0

    for fpath in files:
        basename = os.path.basename(fpath)
        try:
            spec = load_yaml(fpath)
        except Exception as e:
            msg = f"Failed to load {fpath}: {e}"
            if args.json_output:
                all_results.append({
                    "file": fpath,
                    "error": str(e),
                    "issues": [],
                })
            else:
                print(_red(msg), file=sys.stderr)
            exit_code = 1
            continue

        if spec is None:
            spec = {}

        issues = lint_spec(spec)
        # Filter by severity.
        issues = [i for i in issues if SEVERITY_RANK.get(i.severity, 0) >= min_rank]

        if args.json_output:
            all_results.append({
                "file": fpath,
                "issues": [
                    {
                        "code": i.code,
                        "severity": i.severity,
                        "rule": i.rule_name,
                        "message": i.message,
                    }
                    for i in issues
                ],
            })
        else:
            print(f"Linting: {_bold(basename)}")
            if issues:
                for issue in issues:
                    print(_format_issue(issue))
                print()
                print(f"  {_short_count(issues)}")
            else:
                print(_dim("  No issues found."))
            print()

        total_issues += len(issues)
        if any(i.severity == "error" for i in issues):
            exit_code = 1

    # JSON output.
    if args.json_output:
        print(json.dumps(all_results, indent=2))
    elif len(files) > 1:
        # Multi-file summary.
        counts = defaultdict(int)
        for r in all_results if args.json_output else []:
            pass  # Already printed above.
        # Recount from scratch for the summary line.
        all_flat = []
        for fpath in files:
            try:
                spec = load_yaml(fpath)
                if spec is None:
                    spec = {}
                issues = lint_spec(spec)
                issues = [i for i in issues
                          if SEVERITY_RANK.get(i.severity, 0) >= min_rank]
                all_flat.extend(issues)
            except Exception:
                pass
        error_count = sum(1 for i in all_flat if i.severity == "error")
        warn_count = sum(1 for i in all_flat if i.severity == "warn")
        info_count = sum(1 for i in all_flat if i.severity == "info")
        total = len(all_flat)
        noun = "issue" if total == 1 else "issues"
        spec_noun = "spec" if len(files) == 1 else "specs"
        summary = (
            f"Summary: {len(files)} {spec_noun}, {total} {noun} "
            f"({error_count} errors, {warn_count} warnings, {info_count} info)"
        )
        print(_bold(summary))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
