#!/usr/bin/env python3
"""
Agent Ontology Spec Complexity Scorer
=======================================
Computes structural complexity metrics for agent specs and produces
a normalized 0-100 complexity score.

Usage:
    python3 complexity.py specs/react.yaml              # single spec
    python3 complexity.py --all specs/                   # comparison table
    python3 complexity.py --all specs/ --json            # machine-readable
    python3 complexity.py specs/react.yaml --breakdown   # detailed metric breakdown
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import yaml


# ── YAML loading ────────────────────────────────────────────

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ── Metric computation ──────────────────────────────────────

ENTITY_TYPES = ("agent", "store", "tool", "human", "config", "channel", "team", "conversation")
PROCESS_TYPES = ("step", "gate", "checkpoint", "spawn", "protocol", "policy", "error_handler")
EDGE_TYPES = ("flow", "invoke", "loop", "branch", "read", "write", "modify", "observe", "error", "publish", "subscribe", "handoff")


def _count_by_type(items, key, allowed_types):
    """Count items grouped by their type field."""
    counts = defaultdict(int)
    for item in items:
        t = item.get(key, "unknown")
        counts[t] += 1
    # Ensure all canonical types appear (even if 0)
    for t in allowed_types:
        counts.setdefault(t, 0)
    return dict(counts)


def compute_entity_counts(spec):
    """1. Entity counts by type: agents, tools, stores, humans."""
    entities = spec.get("entities", [])
    return _count_by_type(entities, "type", ENTITY_TYPES)


def compute_process_counts(spec):
    """2. Process counts by type: steps, gates, checkpoints, spawns."""
    processes = spec.get("processes", [])
    return _count_by_type(processes, "type", PROCESS_TYPES)


def compute_edge_counts(spec):
    """3. Edge counts by type: flow, invoke, read, write, loop, branch."""
    edges = spec.get("edges", [])
    return _count_by_type(edges, "type", EDGE_TYPES)


def compute_fan_out(spec):
    """4. Max outgoing flow edges from any single process node."""
    edges = spec.get("edges", [])
    flow_edges = [e for e in edges if e.get("type") == "flow"]
    out_degree = defaultdict(int)
    for e in flow_edges:
        src = e.get("from", "")
        out_degree[src] += 1
    if not out_degree:
        return 0
    return max(out_degree.values())


def compute_loop_depth(spec):
    """5. Number of loop edges (cycles in the graph)."""
    edges = spec.get("edges", [])
    return sum(1 for e in edges if e.get("type") == "loop")


def compute_schema_metrics(spec):
    """6-7. Schema count and average fields per schema."""
    schemas = spec.get("schemas", [])
    count = len(schemas)
    if count == 0:
        return count, 0.0
    total_fields = sum(len(s.get("fields", [])) for s in schemas)
    avg_fields = total_fields / count
    return count, round(avg_fields, 2)


def _build_adjacency(spec):
    """Build a forward-adjacency dict from control-flow edges.

    Includes flow, branch, loop, and invoke edges since all of them
    represent control-flow transitions in the spec graph.  Read/write
    edges are excluded as they represent data movement, not control flow.
    """
    edges = spec.get("edges", [])
    adj = defaultdict(list)
    for e in edges:
        if e.get("type") in ("flow", "branch", "loop", "invoke"):
            adj[e.get("from", "")].append(e.get("to", ""))
    return adj


def _find_all_node_ids(spec):
    """Collect all process ids and entity ids that participate in edges."""
    ids = set()
    for p in spec.get("processes", []):
        ids.add(p.get("id", ""))
    for e in spec.get("entities", []):
        ids.add(e.get("id", ""))
    return ids


def compute_graph_depth(spec):
    """8. Longest path from entry_point to any terminal node (BFS/DFS with memoization)."""
    entry = spec.get("entry_point", "")
    if not entry:
        return 0

    adj = _build_adjacency(spec)

    # Use iterative DFS with memoization to find longest path.
    # Since specs can have loops, we cap to avoid infinite traversal.
    all_nodes = _find_all_node_ids(spec)
    max_path_len = len(all_nodes) + 1  # absolute cap

    memo = {}
    visiting = set()

    def longest(node, depth=0):
        if depth > max_path_len:
            return 0
        if node in memo:
            return memo[node]
        if node in visiting:
            # cycle detected -- do not recurse further
            return 0
        visiting.add(node)
        children = adj.get(node, [])
        if not children:
            result = 0
        else:
            result = max(1 + longest(child, depth + 1) for child in children)
        visiting.discard(node)
        memo[node] = result
        return result

    return longest(entry)


def compute_agent_invocation_density(spec):
    """9. Ratio of invoke edges to total processes."""
    edges = spec.get("edges", [])
    processes = spec.get("processes", [])
    invoke_count = sum(1 for e in edges if e.get("type") == "invoke")
    proc_count = len(processes)
    if proc_count == 0:
        return 0.0
    return round(invoke_count / proc_count, 3)


# ── Overall complexity score ────────────────────────────────

# Weights for the composite score.  Each sub-metric is first mapped through
# a saturation function (log or linear clamp) so that extreme values do not
# dominate, then weighted.  The raw weighted sum is clamped to 0-100.

WEIGHTS = {
    "entity_total":              3.0,
    "process_total":             4.0,
    "edge_total":                3.0,
    "fan_out":                   5.0,
    "loop_depth":                6.0,
    "schema_count":              2.0,
    "schema_complexity":         3.0,
    "graph_depth":               5.0,
    "agent_invocation_density":  4.0,
}


def _saturate(value, cap):
    """Map value into [0, 1] using log saturation.  0 -> 0, cap -> ~1."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(cap))


def compute_overall_score(metrics):
    """10. Weighted combination of sub-metrics, normalized to 0-100."""
    raw = 0.0

    # entity_total: saturation cap at ~15 entities
    raw += WEIGHTS["entity_total"] * _saturate(metrics["entity_total"], 15)

    # process_total: cap at ~20
    raw += WEIGHTS["process_total"] * _saturate(metrics["process_total"], 20)

    # edge_total: cap at ~30
    raw += WEIGHTS["edge_total"] * _saturate(metrics["edge_total"], 30)

    # fan_out: cap at 5
    raw += WEIGHTS["fan_out"] * _saturate(metrics["fan_out"], 5)

    # loop_depth: cap at 4
    raw += WEIGHTS["loop_depth"] * _saturate(metrics["loop_depth"], 4)

    # schema_count: cap at 12
    raw += WEIGHTS["schema_count"] * _saturate(metrics["schema_count"], 12)

    # schema_complexity: cap at 6 fields/schema
    raw += WEIGHTS["schema_complexity"] * _saturate(metrics["schema_complexity"], 6)

    # graph_depth: cap at 12
    raw += WEIGHTS["graph_depth"] * _saturate(metrics["graph_depth"], 12)

    # agent_invocation_density: already 0-N, cap at 1.5
    raw += WEIGHTS["agent_invocation_density"] * _saturate(
        metrics["agent_invocation_density"], 1.5
    )

    max_raw = sum(WEIGHTS.values())
    score = (raw / max_raw) * 100.0
    return round(min(100.0, max(0.0, score)), 1)


# ── Public API ──────────────────────────────────────────────

def analyze_spec(spec):
    """Compute all complexity metrics for a loaded spec dict.

    Returns a dict with all individual metrics plus the overall score.
    """
    entity_counts = compute_entity_counts(spec)
    process_counts = compute_process_counts(spec)
    edge_counts = compute_edge_counts(spec)
    fan_out = compute_fan_out(spec)
    loop_depth = compute_loop_depth(spec)
    schema_count, schema_complexity = compute_schema_metrics(spec)
    graph_depth = compute_graph_depth(spec)
    invocation_density = compute_agent_invocation_density(spec)

    metrics = {
        "name": spec.get("name", "unknown"),
        # Breakdowns
        "entity_counts": entity_counts,
        "process_counts": process_counts,
        "edge_counts": edge_counts,
        # Totals
        "entity_total": sum(entity_counts.values()),
        "process_total": sum(process_counts.values()),
        "edge_total": sum(edge_counts.values()),
        # Derived
        "fan_out": fan_out,
        "loop_depth": loop_depth,
        "schema_count": schema_count,
        "schema_complexity": schema_complexity,
        "graph_depth": graph_depth,
        "agent_invocation_density": invocation_density,
    }
    metrics["overall_score"] = compute_overall_score(metrics)
    return metrics


# ── Formatting helpers ──────────────────────────────────────

def _score_label(score):
    """Human-readable label for the 0-100 score."""
    if score < 20:
        return "trivial"
    elif score < 40:
        return "simple"
    elif score < 60:
        return "moderate"
    elif score < 80:
        return "complex"
    else:
        return "very complex"


def format_breakdown(metrics):
    """Return a multi-line string with the detailed metric breakdown."""
    lines = []
    name = metrics["name"]
    score = metrics["overall_score"]
    lines.append(f"Complexity Report: {name}")
    lines.append("=" * 60)
    lines.append("")

    # Entities
    ec = metrics["entity_counts"]
    lines.append(f"  Entities ({metrics['entity_total']} total)")
    for t in ENTITY_TYPES:
        c = ec.get(t, 0)
        if c:
            lines.append(f"    {t:12s}  {c}")

    # Processes
    pc = metrics["process_counts"]
    lines.append(f"  Processes ({metrics['process_total']} total)")
    for t in PROCESS_TYPES:
        c = pc.get(t, 0)
        if c:
            lines.append(f"    {t:12s}  {c}")

    # Edges
    edc = metrics["edge_counts"]
    lines.append(f"  Edges ({metrics['edge_total']} total)")
    for t in EDGE_TYPES:
        c = edc.get(t, 0)
        if c:
            lines.append(f"    {t:12s}  {c}")

    lines.append("")
    lines.append(f"  Fan-out (max)             {metrics['fan_out']}")
    lines.append(f"  Loop depth                {metrics['loop_depth']}")
    lines.append(f"  Schema count              {metrics['schema_count']}")
    lines.append(f"  Schema complexity (avg)   {metrics['schema_complexity']} fields")
    lines.append(f"  Graph depth               {metrics['graph_depth']}")
    lines.append(f"  Invocation density        {metrics['agent_invocation_density']}")
    lines.append("")
    lines.append(f"  Overall score             {score}/100 ({_score_label(score)})")
    lines.append("")
    return "\n".join(lines)


def format_comparison_table(all_metrics):
    """Return a formatted comparison table string for multiple specs."""
    # Column definitions: (header, key, width, fmt)
    columns = [
        ("Spec",       "name",                       18, "s"),
        ("Ent",        "entity_total",                 4, "d"),
        ("Proc",       "process_total",                5, "d"),
        ("Edge",       "edge_total",                   5, "d"),
        ("Fan",        "fan_out",                      4, "d"),
        ("Loop",       "loop_depth",                   5, "d"),
        ("Sch",        "schema_count",                 4, "d"),
        ("Fld",        "schema_complexity",             5, ".1f"),
        ("Depth",      "graph_depth",                  6, "d"),
        ("Inv.D",      "agent_invocation_density",     6, ".2f"),
        ("Score",      "overall_score",                6, ".1f"),
        ("Level",      None,                           12, "s"),
    ]

    # Header
    header_parts = []
    sep_parts = []
    for hdr, _, w, _ in columns:
        header_parts.append(f"{hdr:>{w}}")
        sep_parts.append("-" * w)
    header = "  ".join(header_parts)
    sep = "  ".join(sep_parts)

    lines = [header, sep]

    # Sort by overall score descending
    sorted_metrics = sorted(all_metrics, key=lambda m: m["overall_score"], reverse=True)

    for m in sorted_metrics:
        parts = []
        for hdr, key, w, fmt in columns:
            if key is None:
                # "Level" column
                val = _score_label(m["overall_score"])
                parts.append(f"{val:>{w}}")
            else:
                val = m[key]
                parts.append(f"{val:>{w}{fmt}}")
        lines.append("  ".join(parts))

    lines.append(sep)
    lines.append(f"  {len(sorted_metrics)} specs analyzed")
    lines.append("")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agent Ontology Spec Complexity Scorer -- structural complexity metrics for agent specs",
        epilog="Examples:\n"
               "  python3 complexity.py specs/react.yaml\n"
               "  python3 complexity.py specs/react.yaml --breakdown\n"
               "  python3 complexity.py --all specs/\n"
               "  python3 complexity.py --all specs/ --json\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spec", help="Path to a spec YAML file, or directory with --all")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all *.yaml files in the given directory")
    parser.add_argument("--breakdown", action="store_true",
                        help="Show detailed metric breakdown (single-spec mode)")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON instead of formatted table")

    args = parser.parse_args()

    if args.all:
        # ── Multi-spec comparison mode ──────────────────────
        spec_dir = args.spec
        if not os.path.isdir(spec_dir):
            print(f"Error: {spec_dir} is not a directory (use --all with a directory)", file=sys.stderr)
            sys.exit(1)

        all_metrics = []
        for fname in sorted(os.listdir(spec_dir)):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(spec_dir, fname)
            try:
                spec = load_yaml(path)
                metrics = analyze_spec(spec)
                metrics["file"] = fname
                all_metrics.append(metrics)
            except Exception as exc:
                print(f"Warning: failed to analyze {fname}: {exc}", file=sys.stderr)

        if not all_metrics:
            print("No spec files found.", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(all_metrics, indent=2))
        else:
            print(format_comparison_table(all_metrics))
    else:
        # ── Single-spec mode ────────────────────────────────
        if not os.path.isfile(args.spec):
            print(f"Error: {args.spec} not found", file=sys.stderr)
            sys.exit(1)

        spec = load_yaml(args.spec)
        metrics = analyze_spec(spec)
        metrics["file"] = os.path.basename(args.spec)

        if args.json:
            print(json.dumps(metrics, indent=2))
        elif args.breakdown:
            print(format_breakdown(metrics))
        else:
            score = metrics["overall_score"]
            name = metrics["name"]
            label = _score_label(score)
            print(f"{name}: {score}/100 ({label})")
            print(f"  entities={metrics['entity_total']}  processes={metrics['process_total']}  "
                  f"edges={metrics['edge_total']}  schemas={metrics['schema_count']}  "
                  f"fan_out={metrics['fan_out']}  loops={metrics['loop_depth']}  "
                  f"depth={metrics['graph_depth']}  inv_density={metrics['agent_invocation_density']}")


if __name__ == "__main__":
    main()
