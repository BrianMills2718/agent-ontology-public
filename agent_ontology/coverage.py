#!/usr/bin/env python3
"""Spec coverage report tool.

Analyzes how well agent specs use the ontology's available types and features.
Computes entity type, process type, edge type, schema, and feature coverage,
then rolls them up into an overall coverage percentage.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# ── canonical type lists ──────────────────────────────────────────────

ENTITY_TYPES = ("agent", "store", "tool", "human", "config", "channel", "team", "conversation")
PROCESS_TYPES = ("step", "gate", "checkpoint", "spawn", "protocol", "policy", "error_handler")
EDGE_TYPES = ("flow", "invoke", "loop", "branch", "read", "write", "modify", "observe", "error", "publish", "subscribe", "handoff")
# Display names live beside the flags deliberately. They were previously kept in
# dashboard.py, and adding channels/teams/handoffs here without adding them
# there crashed `ao-dashboard` with a KeyError for six months.
FEATURE_LABELS = {
    "fan_out": "fan-out",
    "loops": "loops",
    "recursive_spawn": "recursive",
    "human_in_loop": "human-in-loop",
    "stores": "stores",
    "tools": "tools",
    "policies": "policies",
    "channels": "channels",
    "teams": "teams",
    "handoffs": "handoffs",
}
FEATURE_FLAGS = tuple(FEATURE_LABELS)

# ── helpers ───────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    """Load and return a YAML spec file."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _collect_entity_types(spec: dict) -> set:
    """Return the set of entity types present in the spec."""
    used = set()
    for ent in spec.get("entities", []):
        etype = ent.get("type", "").lower()
        if etype in ENTITY_TYPES:
            used.add(etype)
    return used


def _collect_process_types(spec: dict) -> set:
    """Return the set of process types present in the spec."""
    used = set()
    for proc in spec.get("processes", []):
        ptype = proc.get("type", "").lower()
        if ptype in PROCESS_TYPES:
            used.add(ptype)
    return used


def _collect_edge_types(spec: dict) -> set:
    """Return the set of edge types present in the spec."""
    used = set()
    for edge in spec.get("edges", []):
        etype = edge.get("type", "").lower()
        if etype in EDGE_TYPES:
            used.add(etype)
    return used


def _compute_schema_coverage(spec: dict) -> dict:
    """Count schemas and fraction of entity/process nodes referencing them."""
    schemas = spec.get("schemas", [])
    schema_count = len(schemas)

    nodes = list(spec.get("entities", [])) + list(spec.get("processes", []))
    total_nodes = len(nodes)
    if total_nodes == 0:
        return {"schema_count": schema_count, "nodes_total": 0,
                "nodes_with_schema": 0, "schema_pct": 0.0}

    nodes_with_schema = sum(
        1 for n in nodes if n.get("schema") or n.get("schema_ref")
    )
    return {
        "schema_count": schema_count,
        "nodes_total": total_nodes,
        "nodes_with_schema": nodes_with_schema,
        "schema_pct": round(nodes_with_schema / total_nodes * 100, 1),
    }


def _compute_feature_flags(spec: dict) -> dict:
    """Evaluate each feature flag and return a dict of bool values."""
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])

    entity_types_present = {e.get("type", "").lower() for e in entities}
    process_types_present = {p.get("type", "").lower() for p in processes}
    edge_types_present = {e.get("type", "").lower() for e in edges}

    # fan_out: any process with >1 outgoing flow edge
    process_ids = {p.get("id") for p in processes}
    flow_edges = [e for e in edges if e.get("type", "").lower() == "flow"]
    outgoing_flow_counts: dict[str, int] = {}
    for fe in flow_edges:
        src = fe.get("source") or fe.get("from") or fe.get("src")
        if src:
            outgoing_flow_counts[src] = outgoing_flow_counts.get(src, 0) + 1
    fan_out = any(count > 1 for count in outgoing_flow_counts.values())

    # loops: any loop edge
    loops = "loop" in edge_types_present

    # recursive_spawn: any spawn process with recursive:true or template:"self"
    recursive_spawn = False
    for proc in processes:
        if proc.get("type", "").lower() == "spawn":
            if proc.get("recursive") is True:
                recursive_spawn = True
                break
            if str(proc.get("template", "")).lower() == "self":
                recursive_spawn = True
                break

    # human_in_loop: any checkpoint process or human entity
    human_in_loop = (
        "checkpoint" in process_types_present or "human" in entity_types_present
    )

    # stores: any store entity
    stores = "store" in entity_types_present

    # tools: any tool entity
    tools = "tool" in entity_types_present

    # policies: any policy process
    policies = "policy" in process_types_present

    # channels: any channel entity
    channels = "channel" in entity_types_present

    # teams: any team entity
    teams = "team" in entity_types_present

    # handoffs: any handoff edge
    handoffs = "handoff" in edge_types_present

    return {
        "fan_out": fan_out,
        "loops": loops,
        "recursive_spawn": recursive_spawn,
        "human_in_loop": human_in_loop,
        "stores": stores,
        "tools": tools,
        "policies": policies,
        "channels": channels,
        "teams": teams,
        "handoffs": handoffs,
    }


# ── main analysis ─────────────────────────────────────────────────────

def analyze_coverage(spec: dict) -> dict:
    """Analyze a spec and return a coverage metrics dict."""
    name = spec.get("name") or spec.get("id") or "unnamed"

    entity_types_used = _collect_entity_types(spec)
    process_types_used = _collect_process_types(spec)
    edge_types_used = _collect_edge_types(spec)
    schema_info = _compute_schema_coverage(spec)
    features = _compute_feature_flags(spec)

    features_used = sum(1 for v in features.values() if v)
    features_total = len(FEATURE_FLAGS)

    # overall score: average of four ratios
    ratios = [
        len(entity_types_used) / len(ENTITY_TYPES),
        len(process_types_used) / len(PROCESS_TYPES),
        len(edge_types_used) / len(EDGE_TYPES),
        features_used / features_total,
    ]
    overall = sum(ratios) / len(ratios) * 100

    total_features = (
        len(ENTITY_TYPES) + len(PROCESS_TYPES) + len(EDGE_TYPES) + features_total
    )
    used_features = (
        len(entity_types_used) + len(process_types_used)
        + len(edge_types_used) + features_used
    )

    return {
        "name": name,
        "entity_types": {
            "used": sorted(entity_types_used),
            "total": list(ENTITY_TYPES),
            "count": len(entity_types_used),
            "of": len(ENTITY_TYPES),
        },
        "process_types": {
            "used": sorted(process_types_used),
            "total": list(PROCESS_TYPES),
            "count": len(process_types_used),
            "of": len(PROCESS_TYPES),
        },
        "edge_types": {
            "used": sorted(edge_types_used),
            "total": list(EDGE_TYPES),
            "count": len(edge_types_used),
            "of": len(EDGE_TYPES),
        },
        "schema": schema_info,
        "features": {
            "flags": features,
            "count": features_used,
            "of": features_total,
        },
        "overall_pct": round(overall, 1),
        "used_features": used_features,
        "total_features": total_features,
    }


# ── formatters ────────────────────────────────────────────────────────

def format_summary(cov: dict) -> str:
    """One-line summary for a single spec."""
    return (
        f"{cov['name']}: {cov['overall_pct']}% coverage "
        f"({cov['used_features']}/{cov['total_features']} features)\n"
        f"  entity types: {cov['entity_types']['count']}/{cov['entity_types']['of']}  "
        f"process types: {cov['process_types']['count']}/{cov['process_types']['of']}  "
        f"edge types: {cov['edge_types']['count']}/{cov['edge_types']['of']}  "
        f"features: {cov['features']['count']}/{cov['features']['of']}"
    )


def format_breakdown(cov: dict) -> str:
    """Detailed breakdown report for a single spec."""
    lines = []
    lines.append(f"Coverage Report: {cov['name']}")
    lines.append("=" * 60)

    def _section(title, used_list, all_list):
        count = len(used_list)
        total = len(all_list)
        lines.append(f"  {title} ({count}/{total})")
        used_set = set(used_list)
        for item in all_list:
            mark = "v" if item in used_set else "x"
            lines.append(f"    {mark} {item}")
        lines.append("")

    _section("Entity Types", cov["entity_types"]["used"], ENTITY_TYPES)
    _section("Process Types", cov["process_types"]["used"], PROCESS_TYPES)
    _section("Edge Types", cov["edge_types"]["used"], EDGE_TYPES)

    # Features
    flags = cov["features"]["flags"]
    feature_labels = {
        "fan_out": "fan-out",
        "loops": "loops",
        "recursive_spawn": "recursive spawn",
        "human_in_loop": "human-in-the-loop",
        "stores": "stores",
        "tools": "tools",
        "policies": "policies",
        "channels": "channels (pub/sub)",
        "teams": "teams",
        "handoffs": "handoffs",
    }
    count = cov["features"]["count"]
    total = cov["features"]["of"]
    lines.append(f"  Features ({count}/{total})")
    for key in FEATURE_FLAGS:
        mark = "v" if flags[key] else "x"
        label = feature_labels.get(key, key)
        lines.append(f"    {mark} {label}")
    lines.append("")

    # Schema
    si = cov["schema"]
    lines.append(
        f"  Schema coverage: {si['schema_count']} schemas, "
        f"{si['schema_pct']}% of nodes reference schemas"
    )
    lines.append("")
    lines.append(f"  Overall: {cov['overall_pct']}%")

    return "\n".join(lines)


def format_comparison_table(all_coverages: list[dict]) -> str:
    """Render a comparison table sorted by overall coverage descending."""
    sorted_covs = sorted(all_coverages, key=lambda c: c["overall_pct"], reverse=True)

    header = f"{'Spec':<30} {'EntTyp':>6} {'ProcTyp':>7} {'EdgTyp':>6} {'Schema':>8} {'Features':>8} {'Overall%':>8}"
    sep = "-" * len(header)
    lines = [header, sep]

    for cov in sorted_covs:
        name = cov["name"]
        if len(name) > 28:
            name = name[:25] + "..."
        et = f"{cov['entity_types']['count']}/{cov['entity_types']['of']}"
        pt = f"{cov['process_types']['count']}/{cov['process_types']['of']}"
        egt = f"{cov['edge_types']['count']}/{cov['edge_types']['of']}"
        sch = f"{cov['schema']['schema_pct']}%"
        feat = f"{cov['features']['count']}/{cov['features']['of']}"
        ovr = f"{cov['overall_pct']}%"
        lines.append(f"{name:<30} {et:>6} {pt:>7} {egt:>6} {sch:>8} {feat:>8} {ovr:>8}")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────

def find_specs(directory: str) -> list[str]:
    """Find all .yaml / .yml spec files in a directory."""
    specs = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith((".yaml", ".yml")):
            specs.append(os.path.join(directory, fname))
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze spec coverage of the agent ontology."
    )
    parser.add_argument(
        "path",
        help="Path to a spec YAML file, or a directory when used with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_specs",
        help="Analyze all specs in the given directory and print a comparison table.",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Print a detailed breakdown for a single spec.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    if args.all_specs:
        spec_paths = find_specs(args.path)
        if not spec_paths:
            print(f"No YAML specs found in {args.path}", file=sys.stderr)
            sys.exit(1)

        all_coverages = []
        for sp in spec_paths:
            spec = load_yaml(sp)
            if spec is None:
                continue
            all_coverages.append(analyze_coverage(spec))

        if args.json_output:
            print(json.dumps(all_coverages, indent=2))
        else:
            print(format_comparison_table(all_coverages))
    else:
        spec = load_yaml(args.path)
        if spec is None:
            print(f"Failed to load spec: {args.path}", file=sys.stderr)
            sys.exit(1)
        cov = analyze_coverage(spec)

        if args.json_output:
            print(json.dumps(cov, indent=2))
        elif args.breakdown:
            print(format_breakdown(cov))
        else:
            print(format_summary(cov))


if __name__ == "__main__":
    main()
