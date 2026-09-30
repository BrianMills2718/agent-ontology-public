#!/usr/bin/env python3
"""Agent topology classifier.

Analyzes the control-flow graph (flow, branch, loop edges) of an agent spec
and classifies it into one of these topology categories:

    linear       No branches, no loops. A->B->C->D.
    loop         A single loop cycle with no branching.
    branch       Has gates/branches but no loops.
    dag          Directed acyclic graph with fan-out/fan-in, no cycles.
    cyclic       Has exactly one cycle (loop).
    multi-cyclic Has multiple independent cycles.
    tree         Fan-out without fan-in (branching paths don't converge).
    star         One central hub connected to many peripheral nodes.

Usage:
    python3 topology.py specs/react.yaml              # single spec
    python3 topology.py --all specs/                   # comparison table
    python3 topology.py specs/react.yaml --detail      # show graph details
    python3 topology.py --all specs/ --json            # JSON output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Complexity ordering (higher index = more complex)
# ---------------------------------------------------------------------------
TOPOLOGY_ORDER = [
    "linear",
    "loop",
    "branch",
    "tree",
    "star",
    "dag",
    "cyclic",
    "multi-cyclic",
]


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def gather_specs(directory: str | Path) -> list[Path]:
    """Recursively collect all .yaml / .yml files under *directory*."""
    root = Path(directory)
    specs: list[Path] = []
    for ext in ("*.yaml", "*.yml"):
        specs.extend(sorted(root.rglob(ext)))
    return specs


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _collect_nodes(spec: dict) -> set[str]:
    """Return the set of all process-node ids declared in the spec."""
    nodes: set[str] = set()

    # processes / nodes section
    for section_key in ("processes", "nodes", "steps"):
        section = spec.get(section_key, None)
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict) and "id" in item:
                    nodes.add(item["id"])
                elif isinstance(item, str):
                    nodes.add(item)
        elif isinstance(section, dict):
            nodes.update(section.keys())

    return nodes


def _edge_type(edge: dict) -> str:
    """Determine the type label of an edge dict ('flow', 'branch', 'loop')."""
    if isinstance(edge, dict):
        return edge.get("type", edge.get("edge_type", "flow"))
    return "flow"


def _collect_edges(spec: dict) -> list[tuple[str, str, str]]:
    """Return a list of (source, target, edge_type) tuples.

    Looks for edges declared under 'edges', 'flow', 'transitions', or
    'control_flow' keys.  Each edge may be a dict with 'from'/'to' (or
    'source'/'target') and an optional 'type' field.
    """
    edges: list[tuple[str, str, str]] = []

    for section_key in ("edges", "flow", "transitions", "control_flow"):
        section = spec.get(section_key, None)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            src = item.get("from") or item.get("source") or item.get("src", "")
            tgt = item.get("to") or item.get("target") or item.get("dst", "")
            etype = _edge_type(item)
            if src and tgt:
                edges.append((str(src), str(tgt), str(etype)))

    return edges


def _build_graph(spec: dict) -> tuple[set[str], list[tuple[str, str, str]]]:
    """Build the full node-set and edge-list from a spec.

    Nodes that appear only in edges (but not in the nodes section) are added
    automatically so the graph is self-consistent.
    """
    nodes = _collect_nodes(spec)
    edges = _collect_edges(spec)

    # Ensure every node mentioned in an edge is in the node set.
    for src, tgt, _ in edges:
        nodes.add(src)
        nodes.add(tgt)

    return nodes, edges


# ---------------------------------------------------------------------------
# Graph analysis helpers
# ---------------------------------------------------------------------------

def _adjacency(nodes: set[str], edges: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    """Build an adjacency list (forward edges) from nodes and edges."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for src, tgt, _ in edges:
        adj[src].append(tgt)
    return adj


def _reverse_adjacency(nodes: set[str], edges: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    """Build a reverse adjacency list (incoming edges)."""
    rev: dict[str, list[str]] = {n: [] for n in nodes}
    for src, tgt, _ in edges:
        rev[tgt].append(src)
    return rev


# ---------------------------------------------------------------------------
# Cycle detection via DFS back-edge counting
# ---------------------------------------------------------------------------


def _detect_cycles(adj: dict[str, list[str]]) -> tuple[int, list[list[str]]]:
    """Count independent directed cycles and find representative cycle paths.

    Strategy:
    1. Find strongly connected components (SCCs) via Tarjan's algorithm.
       Only SCCs with >= 2 nodes contain directed cycles.
    2. For each non-trivial SCC, compute the number of independent cycles
       using the directed circuit rank within that SCC:
           cycles_in_scc = |edges_in_scc| - |nodes_in_scc| + 1
    3. Sum across all non-trivial SCCs for the total cycle count.
    4. Find representative cycle paths via BFS within each SCC.

    Returns (num_cycles, list_of_cycle_paths).
    """
    sccs = _find_sccs(adj)
    total_cycles = 0
    all_cycle_paths: list[list[str]] = []

    for scc in sccs:
        if len(scc) < 2:
            continue

        scc_set = set(scc)
        # Build sub-adjacency restricted to this SCC
        sub_adj: dict[str, list[str]] = {}
        scc_edges = 0
        for n in scc:
            targets = [t for t in adj.get(n, []) if t in scc_set]
            sub_adj[n] = targets
            scc_edges += len(targets)

        # Independent cycles in this SCC
        scc_cycle_count = scc_edges - len(scc) + 1
        if scc_cycle_count <= 0:
            # Shouldn't happen for an SCC with >= 2 nodes, but guard anyway
            scc_cycle_count = 1
        total_cycles += scc_cycle_count

        # Find representative cycle paths via BFS from each node
        seen_cycle_sets: set[frozenset[str]] = set()
        found_in_scc = 0
        for start_node in sorted(scc):
            if found_in_scc >= scc_cycle_count:
                break
            # BFS to find shortest cycles through start_node
            visited_paths: dict[str, list[str]] = {start_node: [start_node]}
            queue: deque[str] = deque([start_node])
            while queue and found_in_scc < scc_cycle_count:
                curr = queue.popleft()
                for nbr in sub_adj.get(curr, []):
                    if nbr == start_node and len(visited_paths[curr]) > 1:
                        cyc_path = visited_paths[curr] + [start_node]
                        cyc_key = frozenset(cyc_path)
                        if cyc_key not in seen_cycle_sets:
                            seen_cycle_sets.add(cyc_key)
                            all_cycle_paths.append(cyc_path)
                            found_in_scc += 1
                            if found_in_scc >= scc_cycle_count:
                                break
                    elif nbr not in visited_paths:
                        visited_paths[nbr] = visited_paths[curr] + [nbr]
                        queue.append(nbr)

    return total_cycles, all_cycle_paths


def _find_sccs(adj: dict[str, list[str]]) -> list[list[str]]:
    """Find strongly connected components using Tarjan's algorithm."""
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in sorted(adj):
        if v not in index:
            strongconnect(v)

    return sccs


# ---------------------------------------------------------------------------
# Fan-in / fan-out
# ---------------------------------------------------------------------------

def _fan_out(adj: dict[str, list[str]]) -> dict[str, int]:
    return {n: len(targets) for n, targets in adj.items()}


def _fan_in(rev: dict[str, list[str]]) -> dict[str, int]:
    return {n: len(sources) for n, sources in rev.items()}


# ---------------------------------------------------------------------------
# Branch (gate / decision) counting
# ---------------------------------------------------------------------------

def _count_branches(spec: dict, adj: dict[str, list[str]]) -> int:
    """Count gate / decision point nodes.

    A gate is either:
      - a node explicitly typed as 'gate', 'decision', 'branch', 'conditional'
      - OR any node with fan-out > 1 (it routes to multiple successors)
    """
    gate_ids: set[str] = set()

    # Explicit gates from the spec
    for section_key in ("processes", "nodes", "steps"):
        section = spec.get(section_key, None)
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    ntype = str(item.get("type", "")).lower()
                    if ntype in ("gate", "decision", "branch", "conditional",
                                 "router", "switch"):
                        nid = item.get("id", "")
                        if nid:
                            gate_ids.add(nid)
        elif isinstance(section, dict):
            for nid, nval in section.items():
                if isinstance(nval, dict):
                    ntype = str(nval.get("type", "")).lower()
                    if ntype in ("gate", "decision", "branch", "conditional",
                                 "router", "switch"):
                        gate_ids.add(nid)

    # Also count branch-type edges
    for section_key in ("edges", "flow", "transitions", "control_flow"):
        section = spec.get(section_key, None)
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    etype = _edge_type(item)
                    if etype in ("branch", "conditional"):
                        src = item.get("from") or item.get("source") or item.get("src", "")
                        if src:
                            gate_ids.add(str(src))

    # Implicit gates: any node with fan-out > 1
    for node, targets in adj.items():
        if len(targets) > 1:
            gate_ids.add(node)

    return len(gate_ids)


# ---------------------------------------------------------------------------
# Diameter (longest shortest path) via BFS
# ---------------------------------------------------------------------------

def _diameter(adj: dict[str, list[str]]) -> int:
    """Compute the diameter of the directed graph.

    The diameter is the longest shortest-path between any two reachable nodes.
    Uses BFS from every node.
    """
    if not adj:
        return 0

    max_dist = 0
    for start in adj:
        dist: dict[str, int] = {start: 0}
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            for nbr in adj.get(node, []):
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        if dist:
            max_dist = max(max_dist, max(dist.values()))

    return max_dist


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def _find_entry(adj: dict[str, list[str]], rev: dict[str, list[str]],
                spec: dict) -> str | None:
    """Heuristically find the entry node.

    Priority: node explicitly marked 'entry', node with 0 in-degree, or the
    first node alphabetically.
    """
    # Check for explicit entry
    entry = spec.get("entry") or spec.get("entry_point") or spec.get("start")
    if entry and entry in adj:
        return str(entry)

    # Zero in-degree nodes
    zero_in = [n for n in adj if not rev.get(n)]
    if len(zero_in) == 1:
        return zero_in[0]
    if zero_in:
        return sorted(zero_in)[0]

    # Fallback
    if adj:
        return sorted(adj.keys())[0]
    return None


def _is_connected(adj: dict[str, list[str]], entry: str | None) -> bool:
    """Check if all nodes are reachable from *entry* via BFS."""
    if not adj:
        return True
    if entry is None:
        return False

    visited: set[str] = set()
    queue: deque[str] = deque([entry])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for nbr in adj.get(node, []):
            if nbr not in visited:
                queue.append(nbr)

    return visited == set(adj.keys())


def _terminal_nodes(adj: dict[str, list[str]]) -> list[str]:
    """Return nodes with zero out-degree (terminal / sink nodes)."""
    return sorted(n for n, targets in adj.items() if not targets)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(num_cycles: int, max_fo: int, max_fi: int,
              num_branches: int, has_fan_in: bool) -> str:
    """Apply the classification rules and return the topology label.

    For acyclic graphs (num_cycles == 0):
      - linear:  no branching at all (max_fan_out <= 1, no branches)
      - star:    one hub fans out to many (max_fan_out > 3)
      - tree:    fan-out only, paths never reconverge (no fan-in)
      - dag:     fan-out AND fan-in with multiple branch points
      - branch:  simple gate pattern -- fan-out with fan-in, few branches
    """
    if num_cycles == 0:
        if max_fo <= 1 and num_branches == 0:
            return "linear"
        if max_fo > 3:
            return "star"
        if not has_fan_in:
            return "tree"
        # has_fan_in is True from here
        if num_branches > 1 or max_fo > 2:
            return "dag"
        return "branch"
    if num_cycles == 1:
        if num_branches == 0:
            return "loop"
        return "cyclic"
    return "multi-cyclic"


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_topology(spec: dict) -> dict[str, Any]:
    """Analyze a spec and return topology classification plus all metrics.

    Returns a dict with keys:
        name, classification, num_nodes, num_edges, num_cycles, cycles,
        max_fan_out, max_fan_in, num_branches, diameter, is_connected,
        entry, terminals
    """
    name = (spec.get("name")
            or spec.get("metadata", {}).get("name", "")
            or spec.get("id", "unknown"))

    nodes, edges = _build_graph(spec)
    adj = _adjacency(nodes, edges)
    rev = _reverse_adjacency(nodes, edges)

    fo = _fan_out(adj)
    fi = _fan_in(rev)
    max_fo = max(fo.values()) if fo else 0
    max_fi = max(fi.values()) if fi else 0
    has_fan_in = any(v > 1 for v in fi.values())

    num_branches = _count_branches(spec, adj)
    num_cycles, cycle_paths = _detect_cycles(adj)
    diam = _diameter(adj)

    entry = _find_entry(adj, rev, spec)
    connected = _is_connected(adj, entry)
    terminals = _terminal_nodes(adj)

    classification = _classify(num_cycles, max_fo, max_fi,
                               num_branches, has_fan_in)

    return {
        "name": str(name),
        "classification": classification,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_cycles": num_cycles,
        "cycles": cycle_paths,
        "max_fan_out": max_fo,
        "max_fan_in": max_fi,
        "num_branches": num_branches,
        "diameter": diam,
        "is_connected": connected,
        "entry": entry,
        "terminals": terminals,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_single(topo: dict[str, Any]) -> str:
    """One-line summary for a single spec."""
    return (
        f"{topo['name']}: {topo['classification']} "
        f"({topo['num_cycles']} cycle{'s' if topo['num_cycles'] != 1 else ''}, "
        f"{topo['num_nodes']} nodes, {topo['num_edges']} edges)"
    )


def format_detail(topo: dict[str, Any]) -> str:
    """Verbose multi-line report for --detail."""
    lines: list[str] = []
    lines.append(f"Topology Report: {topo['name']}")
    lines.append("=" * 60)
    lines.append(f"  Classification: {topo['classification']}")
    lines.append("")
    lines.append("  Graph metrics:")
    lines.append(f"    Nodes:       {topo['num_nodes']}")
    lines.append(f"    Edges:       {topo['num_edges']}")
    lines.append(f"    Cycles:      {topo['num_cycles']}")
    lines.append(f"    Max fan-out: {topo['max_fan_out']}")
    lines.append(f"    Max fan-in:  {topo['max_fan_in']}")
    lines.append(f"    Branches:    {topo['num_branches']}")
    lines.append(f"    Diameter:    {topo['diameter']}")
    lines.append(f"    Connected:   {'yes' if topo['is_connected'] else 'no'}")

    if topo["cycles"]:
        lines.append("")
        lines.append("  Cycle details:")
        for i, cycle in enumerate(topo["cycles"], 1):
            path_str = " -> ".join(cycle)
            lines.append(f"    Cycle {i}: {path_str}")

    lines.append("")
    lines.append(f"  Entry: {topo['entry'] or '(none)'}")
    terminals_str = ", ".join(topo["terminals"]) if topo["terminals"] else "(none)"
    lines.append(f"  Terminal nodes: {terminals_str}")

    return "\n".join(lines)


def _topology_sort_key(topo: dict[str, Any]) -> tuple[int, str]:
    """Sort key: complexity descending, then name ascending."""
    idx = TOPOLOGY_ORDER.index(topo["classification"]) if topo["classification"] in TOPOLOGY_ORDER else -1
    # Negate index so higher complexity sorts first
    return (-idx, topo["name"].lower())


def _trunc(text: str, width: int) -> str:
    """Truncate *text* to *width*, adding ellipsis if needed."""
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def format_comparison_table(all_topos: list[dict[str, Any]]) -> str:
    """Pretty-print a comparison table sorted by complexity descending."""
    all_topos = sorted(all_topos, key=_topology_sort_key)

    # Column definitions: (header, key, width, align)
    columns = [
        ("Spec",      "name",           20, ">"),
        ("Topology",  "classification", 12, "<"),
        ("Nodes",     "num_nodes",       6, ">"),
        ("Edges",     "num_edges",       6, ">"),
        ("Cycles",    "num_cycles",      7, ">"),
        ("Fan",       "max_fan_out",     4, ">"),
        ("Branches",  "num_branches",    9, ">"),
        ("Diameter",  "diameter",        9, ">"),
    ]

    # Header
    header_parts: list[str] = []
    for hdr, _, width, align in columns:
        fmt = f"{{:{align}{width}}}"
        header_parts.append(fmt.format(hdr))
    header = "  ".join(header_parts)

    sep = "  " + "-" * (len(header) + 2)

    lines: list[str] = [header, sep]

    for topo in all_topos:
        row_parts: list[str] = []
        for _, key, width, align in columns:
            val = topo[key]
            if isinstance(val, str):
                val = _trunc(val, width)
            else:
                val = str(val)
            fmt = f"{{:{align}{width}}}"
            row_parts.append(fmt.format(val))
        lines.append("  ".join(row_parts))

    lines.append(sep)
    lines.append(f"  {len(all_topos)} specs analyzed")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent topology classifier: categorize agent specs by control-flow graph structure.",
    )
    parser.add_argument(
        "path",
        help="Path to a single YAML spec file, or a directory (with --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_specs",
        help="Analyze all YAML specs in the given directory and print a comparison table.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print a detailed topology report for a single spec.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    target = Path(args.path)

    if args.all_specs:
        if not target.is_dir():
            print(f"Error: {target} is not a directory. --all requires a directory.", file=sys.stderr)
            sys.exit(1)

        spec_files = gather_specs(target)
        if not spec_files:
            print(f"No YAML specs found in {target}", file=sys.stderr)
            sys.exit(1)

        all_topos: list[dict[str, Any]] = []
        for sf in spec_files:
            try:
                spec = load_yaml(sf)
                topo = analyze_topology(spec)
                topo["file"] = str(sf)
                all_topos.append(topo)
            except Exception as exc:
                print(f"Warning: skipping {sf}: {exc}", file=sys.stderr)

        if args.json_output:
            # Strip non-serialisable bits (cycle paths are lists of strings, fine)
            print(json.dumps(all_topos, indent=2, default=str))
        else:
            print(format_comparison_table(all_topos))

    else:
        if not target.is_file():
            print(f"Error: {target} is not a file.", file=sys.stderr)
            sys.exit(1)

        spec = load_yaml(target)
        topo = analyze_topology(spec)
        topo["file"] = str(target)

        if args.json_output:
            print(json.dumps(topo, indent=2, default=str))
        elif args.detail:
            print(format_detail(topo))
        else:
            print(format_single(topo))


if __name__ == "__main__":
    main()
