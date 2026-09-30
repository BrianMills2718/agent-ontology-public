#!/usr/bin/env python3
"""
Agent Ontology Comparative Spec Analysis Report
==================================================
Loads all specs from specs/, computes topology, complexity, coverage,
similarity, and feature data, then produces a comprehensive formatted
text report.

Usage:
    python3 comparative_report.py
    python3 comparative_report.py --specs-dir path/to/specs
    python3 comparative_report.py --json

Dependencies: pyyaml (+ standard library only)
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# ======================================================================
# YAML loading
# ======================================================================

def load_yaml(path):
    """Load and return a YAML spec file."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def find_specs(directory):
    """Find all .yaml spec files in a directory (non-recursive)."""
    specs = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith((".yaml", ".yml")):
            specs.append(os.path.join(directory, fname))
    return specs


# ======================================================================
# Canonical ontology types (from ONTOLOGY.yaml / coverage.py)
# ======================================================================

ENTITY_TYPES = ("agent", "store", "tool", "human", "config", "channel", "team", "conversation")
PROCESS_TYPES = ("step", "gate", "checkpoint", "spawn", "protocol", "policy", "error_handler")
EDGE_TYPES = ("flow", "invoke", "loop", "branch", "read", "write", "modify", "observe", "error", "publish", "subscribe", "handoff")
FEATURE_FLAGS = ("fan_out", "loops", "recursive_spawn", "human_in_loop", "stores", "tools", "policies", "channels", "teams", "handoffs")

FEATURE_LABELS = {
    "fan_out": "Fan-out",
    "loops": "Loops",
    "recursive_spawn": "Recursive spawn",
    "human_in_loop": "Human-in-the-loop",
    "stores": "Stores",
    "tools": "Tools",
    "policies": "Policies",
    "channels": "Channels (pub/sub)",
    "teams": "Teams",
    "handoffs": "Handoffs",
}

# ======================================================================
# Topology classification (inlined from topology.py)
# ======================================================================

TOPOLOGY_ORDER = [
    "linear", "loop", "branch", "tree", "star", "dag", "cyclic", "multi-cyclic",
]


def _topo_collect_nodes(spec):
    nodes = set()
    for section_key in ("processes", "nodes", "steps"):
        section = spec.get(section_key, None)
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict) and "id" in item:
                    nodes.add(item["id"])
        elif isinstance(section, dict):
            nodes.update(section.keys())
    return nodes


def _topo_collect_edges(spec):
    edges = []
    for section_key in ("edges", "flow", "transitions", "control_flow"):
        section = spec.get(section_key, None)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            src = item.get("from") or item.get("source") or item.get("src", "")
            tgt = item.get("to") or item.get("target") or item.get("dst", "")
            etype = item.get("type", item.get("edge_type", "flow"))
            if src and tgt:
                edges.append((str(src), str(tgt), str(etype)))
    return edges


def _topo_adjacency(nodes, edges):
    adj = {n: [] for n in nodes}
    for src, tgt, _ in edges:
        if src not in adj:
            adj[src] = []
        adj[src].append(tgt)
        if tgt not in adj:
            adj[tgt] = []
    return adj


def _topo_reverse(nodes, edges):
    rev = {n: [] for n in nodes}
    for src, tgt, _ in edges:
        if tgt not in rev:
            rev[tgt] = []
        rev[tgt].append(src)
        if src not in rev:
            rev[src] = []
    return rev


def _find_sccs(adj):
    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    sccs = []

    def strongconnect(v):
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
            scc = []
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


def _detect_cycles(adj):
    sccs = _find_sccs(adj)
    total_cycles = 0
    for scc in sccs:
        if len(scc) < 2:
            continue
        scc_set = set(scc)
        scc_edges = 0
        for n in scc:
            targets = [t for t in adj.get(n, []) if t in scc_set]
            scc_edges += len(targets)
        scc_cycle_count = max(1, scc_edges - len(scc) + 1)
        total_cycles += scc_cycle_count
    return total_cycles


def classify_topology(spec):
    """Classify the topology of a spec. Returns the classification string."""
    nodes = _topo_collect_nodes(spec)
    edges = _topo_collect_edges(spec)
    for src, tgt, _ in edges:
        nodes.add(src)
        nodes.add(tgt)

    adj = _topo_adjacency(nodes, edges)
    rev = _topo_reverse(nodes, edges)

    fo = {n: len(targets) for n, targets in adj.items()}
    fi = {n: len(sources) for n, sources in rev.items()}
    max_fo = max(fo.values()) if fo else 0
    max_fi = max(fi.values()) if fi else 0
    has_fan_in = any(v > 1 for v in fi.values())

    # Count branches
    gate_ids = set()
    for section_key in ("processes", "nodes", "steps"):
        section = spec.get(section_key, None)
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    ntype = str(item.get("type", "")).lower()
                    if ntype in ("gate", "decision", "branch", "conditional", "router", "switch"):
                        nid = item.get("id", "")
                        if nid:
                            gate_ids.add(nid)
    for section_key in ("edges", "flow", "transitions", "control_flow"):
        section = spec.get(section_key, None)
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    etype = item.get("type", item.get("edge_type", "flow"))
                    if etype in ("branch", "conditional"):
                        src = item.get("from") or item.get("source") or item.get("src", "")
                        if src:
                            gate_ids.add(str(src))
    for node, targets in adj.items():
        if len(targets) > 1:
            gate_ids.add(node)
    num_branches = len(gate_ids)

    num_cycles = _detect_cycles(adj)

    # Classify
    if num_cycles == 0:
        if max_fo <= 1 and num_branches == 0:
            return "linear"
        if max_fo > 3:
            return "star"
        if not has_fan_in:
            return "tree"
        if num_branches > 1 or max_fo > 2:
            return "dag"
        return "branch"
    if num_cycles == 1:
        if num_branches == 0:
            return "loop"
        return "cyclic"
    return "multi-cyclic"


# ======================================================================
# Complexity scoring (inlined from complexity.py)
# ======================================================================

def _saturate(value, cap):
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(cap))


COMPLEXITY_WEIGHTS = {
    "entity_total": 3.0,
    "process_total": 4.0,
    "edge_total": 3.0,
    "fan_out": 5.0,
    "loop_depth": 6.0,
    "schema_count": 2.0,
    "schema_complexity": 3.0,
    "graph_depth": 5.0,
    "agent_invocation_density": 4.0,
}


def compute_complexity(spec):
    """Compute complexity metrics and overall score for a spec."""
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])
    schemas = spec.get("schemas", [])

    entity_total = len(entities)
    process_total = len(processes)
    edge_total = len(edges)

    # Fan-out
    flow_edges = [e for e in edges if e.get("type") == "flow"]
    out_degree = defaultdict(int)
    for e in flow_edges:
        src = e.get("from", "")
        out_degree[src] += 1
    fan_out = max(out_degree.values()) if out_degree else 0

    # Loop depth
    loop_depth = sum(1 for e in edges if e.get("type") == "loop")

    # Schema metrics
    schema_count = len(schemas)
    if schema_count > 0:
        total_fields = sum(len(s.get("fields", [])) for s in schemas)
        schema_complexity = total_fields / schema_count
    else:
        schema_complexity = 0.0

    # Graph depth
    entry = spec.get("entry_point", "")
    if entry:
        adj = defaultdict(list)
        for e in edges:
            if e.get("type") in ("flow", "branch", "loop", "invoke"):
                adj[e.get("from", "")].append(e.get("to", ""))
        all_ids = set()
        for p in processes:
            all_ids.add(p.get("id", ""))
        for ent in entities:
            all_ids.add(ent.get("id", ""))
        max_path = len(all_ids) + 1

        memo = {}
        visiting = set()

        def longest(node, depth=0):
            if depth > max_path:
                return 0
            if node in memo:
                return memo[node]
            if node in visiting:
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

        graph_depth = longest(entry)
    else:
        graph_depth = 0

    # Invocation density
    invoke_count = sum(1 for e in edges if e.get("type") == "invoke")
    agent_invocation_density = round(invoke_count / process_total, 3) if process_total > 0 else 0.0

    metrics = {
        "entity_total": entity_total,
        "process_total": process_total,
        "edge_total": edge_total,
        "fan_out": fan_out,
        "loop_depth": loop_depth,
        "schema_count": schema_count,
        "schema_complexity": round(schema_complexity, 2),
        "graph_depth": graph_depth,
        "agent_invocation_density": agent_invocation_density,
    }

    # Overall score
    raw = 0.0
    raw += COMPLEXITY_WEIGHTS["entity_total"] * _saturate(entity_total, 15)
    raw += COMPLEXITY_WEIGHTS["process_total"] * _saturate(process_total, 20)
    raw += COMPLEXITY_WEIGHTS["edge_total"] * _saturate(edge_total, 30)
    raw += COMPLEXITY_WEIGHTS["fan_out"] * _saturate(fan_out, 5)
    raw += COMPLEXITY_WEIGHTS["loop_depth"] * _saturate(loop_depth, 4)
    raw += COMPLEXITY_WEIGHTS["schema_count"] * _saturate(schema_count, 12)
    raw += COMPLEXITY_WEIGHTS["schema_complexity"] * _saturate(schema_complexity, 6)
    raw += COMPLEXITY_WEIGHTS["graph_depth"] * _saturate(graph_depth, 12)
    raw += COMPLEXITY_WEIGHTS["agent_invocation_density"] * _saturate(agent_invocation_density, 1.5)
    max_raw = sum(COMPLEXITY_WEIGHTS.values())
    score = (raw / max_raw) * 100.0
    metrics["overall_score"] = round(min(100.0, max(0.0, score)), 1)

    return metrics


def complexity_label(score):
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


# ======================================================================
# Feature flags (inlined from coverage.py)
# ======================================================================

def compute_feature_flags(spec):
    """Compute feature flags as a dict of {flag_name: bool}."""
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])

    entity_types = {e.get("type", "").lower() for e in entities}
    process_types = {p.get("type", "").lower() for p in processes}
    edge_types = {e.get("type", "").lower() for e in edges}

    # fan_out
    flow_out = {}
    for e in edges:
        if e.get("type", "").lower() == "flow":
            src = e.get("from") or e.get("source") or e.get("src", "")
            if src:
                flow_out[src] = flow_out.get(src, 0) + 1
    fan_out = any(c > 1 for c in flow_out.values())

    # loops
    loops = "loop" in edge_types

    # recursive_spawn
    recursive_spawn = False
    for proc in processes:
        if proc.get("type", "").lower() == "spawn":
            if proc.get("recursive") is True or str(proc.get("template", "")).lower() == "self":
                recursive_spawn = True
                break

    # human_in_loop
    human_in_loop = "checkpoint" in process_types or "human" in entity_types

    # stores
    stores = "store" in entity_types

    # tools
    tools = "tool" in entity_types

    # policies
    policies = "policy" in process_types

    # channels (pub/sub)
    channels = "channel" in entity_types

    # teams
    teams = "team" in entity_types

    # handoffs
    handoffs = "handoff" in edge_types

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


# ======================================================================
# Coverage analysis
# ======================================================================

def compute_coverage(spec):
    """Compute which ontology types/features a spec uses."""
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])

    entity_types_used = set()
    for ent in entities:
        t = ent.get("type", "").lower()
        if t in ENTITY_TYPES:
            entity_types_used.add(t)

    process_types_used = set()
    for proc in processes:
        t = proc.get("type", "").lower()
        if t in PROCESS_TYPES:
            process_types_used.add(t)

    edge_types_used = set()
    for edge in edges:
        t = edge.get("type", "").lower()
        if t in EDGE_TYPES:
            edge_types_used.add(t)

    return {
        "entity_types": entity_types_used,
        "process_types": process_types_used,
        "edge_types": edge_types_used,
    }


# ======================================================================
# Similarity (inlined from similarity.py)
# ======================================================================

def _jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _numeric_similarity(nums_a, nums_b):
    keys = sorted(set(nums_a.keys()) | set(nums_b.keys()))
    if not keys:
        return 1.0
    diffs_sq = []
    for k in keys:
        a = nums_a.get(k, 0)
        b = nums_b.get(k, 0)
        max_val = max(abs(a), abs(b), 1)
        norm_a = a / max_val
        norm_b = b / max_val
        diffs_sq.append((norm_a - norm_b) ** 2)
    dist = math.sqrt(sum(diffs_sq) / len(diffs_sq))
    return max(0.0, 1.0 - dist)


def compute_similarity_features(spec, topology, complexity_score):
    """Extract similarity features from a spec."""
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])
    schemas = spec.get("schemas", [])

    entity_types = {e.get("type", "").lower() for e in entities if e.get("type")} & set(ENTITY_TYPES)
    process_types = {p.get("type", "").lower() for p in processes if p.get("type")} & set(PROCESS_TYPES)
    edge_types = {e.get("type", "").lower() for e in edges if e.get("type")} & set(EDGE_TYPES)
    feature_flags = set()
    ff = compute_feature_flags(spec)
    for k, v in ff.items():
        if v:
            feature_flags.add(k)

    numerics = {
        "entity_count": len(entities),
        "process_count": len(processes),
        "edge_count": len(edges),
        "schema_count": len(schemas),
        "complexity_score": complexity_score,
    }

    return {
        "entity_types": entity_types,
        "process_types": process_types,
        "edge_types": edge_types,
        "feature_flags": feature_flags,
        "topology": topology,
        "numerics": numerics,
    }


def compute_pairwise_similarity(feat_a, feat_b):
    """Compute weighted similarity between two feature dicts."""
    entity_sim = _jaccard(feat_a["entity_types"], feat_b["entity_types"])
    process_sim = _jaccard(feat_a["process_types"], feat_b["process_types"])
    edge_sim = _jaccard(feat_a["edge_types"], feat_b["edge_types"])
    feature_sim = _jaccard(feat_a["feature_flags"], feat_b["feature_flags"])
    topology_sim = 1.0 if feat_a["topology"] == feat_b["topology"] else 0.0
    numeric_sim = _numeric_similarity(feat_a["numerics"], feat_b["numerics"])

    similarity = (
        0.30 * entity_sim
        + 0.20 * process_sim
        + 0.20 * edge_sim
        + 0.15 * feature_sim
        + 0.05 * topology_sim
        + 0.10 * numeric_sim
    )
    return round(similarity, 4)


# ======================================================================
# Architecture pattern taxonomy
# ======================================================================

TAXONOMY_RULES = [
    {
        "pattern": "reactive",
        "description": "Stimulus-response loop with tool use (observe-think-act cycle)",
        "match": lambda s, t, f: t in ("cyclic", "multi-cyclic") and f.get("tools") and f.get("loops"),
    },
    {
        "pattern": "planning",
        "description": "Decompose goal into steps, then execute sequentially",
        "match": lambda s, t, f: "plan" in s.get("name", "").lower() or "plan" in s.get("description", "").lower(),
    },
    {
        "pattern": "search",
        "description": "Explore a space of solutions (tree/graph search, MCTS)",
        "match": lambda s, t, f: (
            any(kw in s.get("name", "").lower() for kw in ("tree", "search", "lats"))
            or any(kw in s.get("description", "").lower() for kw in ("tree search", "mcts", "monte carlo"))
        ),
    },
    {
        "pattern": "multi-agent",
        "description": "Multiple cooperating agents with distinct roles",
        "match": lambda s, t, f: (
            sum(1 for e in s.get("entities", []) if e.get("type") == "agent") >= 3
            and t not in ("linear",)
        ),
    },
    {
        "pattern": "refinement",
        "description": "Iterative self-improvement loop (generate then critique)",
        "match": lambda s, t, f: (
            any(kw in s.get("name", "").lower() for kw in ("refine", "self_refine", "self-refine"))
            or any(kw in s.get("description", "").lower() for kw in ("refine", "critique", "self-improve"))
        ),
    },
    {
        "pattern": "map-reduce",
        "description": "Parallel fan-out processing then aggregation",
        "match": lambda s, t, f: (
            any(kw in s.get("name", "").lower() for kw in ("map", "reduce", "map_reduce"))
            or (f.get("fan_out") and any(p.get("type") == "spawn" for p in s.get("processes", [])))
        ),
    },
    {
        "pattern": "debate",
        "description": "Adversarial or dialectic multi-agent deliberation",
        "match": lambda s, t, f: (
            any(kw in s.get("name", "").lower() for kw in ("debate", "socratic", "dialectic"))
        ),
    },
    {
        "pattern": "pipeline",
        "description": "Linear chain of transformations",
        "match": lambda s, t, f: t in ("linear",) and not f.get("loops"),
    },
    {
        "pattern": "retrieval-augmented",
        "description": "Retrieve context from external sources before generation",
        "match": lambda s, t, f: (
            any(kw in s.get("name", "").lower() for kw in ("rag", "retrieval"))
            or any(e.get("store_type") == "vector" for e in s.get("entities", []) if e.get("type") == "store")
        ),
    },
    {
        "pattern": "autonomous",
        "description": "Open-ended self-directed agent with persistent learning",
        "match": lambda s, t, f: (
            any(kw in s.get("name", "").lower() for kw in ("autogpt", "auto-gpt", "voyager", "babyagi"))
            or (f.get("stores") and f.get("loops") and f.get("tools"))
        ),
    },
]


def classify_architecture(spec, topology, features):
    """Return a list of matching architecture pattern names."""
    patterns = []
    for rule in TAXONOMY_RULES:
        try:
            if rule["match"](spec, topology, features):
                patterns.append(rule["pattern"])
        except Exception:
            pass
    if not patterns:
        patterns.append("uncategorized")
    return patterns


# ======================================================================
# Main analysis: analyze a single spec
# ======================================================================

def analyze_spec(spec, filepath):
    """Analyze a single spec and return a comprehensive metrics dict."""
    name = spec.get("name") or spec.get("id") or os.path.basename(filepath).replace(".yaml", "")
    filename = os.path.basename(filepath)

    # Topology
    topology = classify_topology(spec)

    # Complexity
    cx = compute_complexity(spec)

    # Feature flags
    ff = compute_feature_flags(spec)

    # Coverage
    cov = compute_coverage(spec)

    # Counts
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])
    schemas = spec.get("schemas", [])

    # Architecture pattern
    arch_patterns = classify_architecture(spec, topology, ff)

    # Similarity features (for later pairwise comparison)
    sim_features = compute_similarity_features(spec, topology, cx["overall_score"])

    return {
        "name": name,
        "filename": filename,
        "topology": topology,
        "complexity_score": cx["overall_score"],
        "complexity_label": complexity_label(cx["overall_score"]),
        "entity_count": len(entities),
        "process_count": len(processes),
        "edge_count": len(edges),
        "schema_count": len(schemas),
        "fan_out": cx["fan_out"],
        "graph_depth": cx["graph_depth"],
        "loop_depth": cx["loop_depth"],
        "invocation_density": cx["agent_invocation_density"],
        "features": ff,
        "coverage": cov,
        "architecture_patterns": arch_patterns,
        "sim_features": sim_features,
    }


# ======================================================================
# Report generation
# ======================================================================

def trunc(text, width):
    """Truncate text to fit in a column."""
    if len(text) <= width:
        return text
    return text[:width - 2] + ".."


def generate_report(all_data):
    """Generate the full comparative analysis report as a string."""
    lines = []

    def heading(title, char="="):
        lines.append("")
        lines.append(char * 72)
        lines.append(title)
        lines.append(char * 72)
        lines.append("")

    def subheading(title):
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
        lines.append("")

    # ── Title ────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  AGENT ONTOLOGY COMPARATIVE SPEC ANALYSIS REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Specs analyzed: {len(all_data)}")
    lines.append(f"  Directory:      specs/")
    lines.append("")

    # ── 1. Summary Table ─────────────────────────────────────

    heading("1. SUMMARY TABLE")

    # Sorted by complexity descending
    sorted_data = sorted(all_data, key=lambda d: d["complexity_score"], reverse=True)

    col_defs = [
        ("Spec",      "name",             22, "s"),
        ("Topology",  "topology",         13, "s"),
        ("Ent",       "entity_count",      4, "d"),
        ("Proc",      "process_count",     5, "d"),
        ("Edge",      "edge_count",        5, "d"),
        ("Sch",       "schema_count",      4, "d"),
        ("Fan",       "fan_out",           4, "d"),
        ("Loop",      "loop_depth",        5, "d"),
        ("Depth",     "graph_depth",       6, "d"),
        ("Score",     "complexity_score",   6, ".1f"),
        ("Level",     "complexity_label",  12, "s"),
    ]

    # Build header
    header_parts = []
    sep_parts = []
    for hdr, _, w, _ in col_defs:
        header_parts.append(f"{hdr:>{w}}")
        sep_parts.append("-" * w)
    lines.append("  ".join(header_parts))
    lines.append("  ".join(sep_parts))

    for d in sorted_data:
        parts = []
        for hdr, key, w, fmt in col_defs:
            val = d[key]
            if isinstance(val, str):
                val = trunc(val, w)
                parts.append(f"{val:>{w}}")
            elif fmt.startswith("."):
                parts.append(f"{val:>{w}{fmt}}")
            else:
                parts.append(f"{val:>{w}{fmt}}")
        lines.append("  ".join(parts))

    lines.append("  ".join(sep_parts))
    lines.append(f"  {len(sorted_data)} specs analyzed")

    # ── 2. Taxonomy Grouping ─────────────────────────────────

    heading("2. TAXONOMY GROUPING (Architecture Patterns)")

    # Collect specs by pattern
    pattern_groups = defaultdict(list)
    for d in all_data:
        for pat in d["architecture_patterns"]:
            pattern_groups[pat].append(d["name"])

    # Get descriptions
    pattern_desc = {}
    for rule in TAXONOMY_RULES:
        pattern_desc[rule["pattern"]] = rule["description"]
    pattern_desc["uncategorized"] = "Does not match any defined pattern"

    # Sort patterns by number of members descending
    sorted_patterns = sorted(pattern_groups.items(), key=lambda kv: -len(kv[1]))

    for pat, members in sorted_patterns:
        desc = pattern_desc.get(pat, "")
        lines.append(f"  {pat.upper()} ({len(members)} spec{'s' if len(members) != 1 else ''})")
        lines.append(f"    {desc}")
        for m in sorted(members):
            lines.append(f"      - {m}")
        lines.append("")

    # Also show each spec's primary classification
    subheading("  Per-spec classification:")
    name_width = max(len(d["name"]) for d in all_data)
    for d in sorted(all_data, key=lambda x: x["name"].lower()):
        pats = ", ".join(d["architecture_patterns"])
        lines.append(f"    {d['name']:<{name_width}}  ->  {pats}")

    # ── 3. Feature Matrix ────────────────────────────────────

    heading("3. FEATURE MATRIX")

    # Column: each feature flag
    name_w = max(len(d["name"]) for d in all_data) + 2
    name_w = max(name_w, 20)
    flag_w = 6

    # Header
    hdr = f"  {'Spec':<{name_w}}"
    for flag in FEATURE_FLAGS:
        label = FEATURE_LABELS.get(flag, flag)
        short = trunc(label, flag_w)
        hdr += f"  {short:>{flag_w}}"
    hdr += f"  {'Total':>5}"
    lines.append(hdr)
    sep = "  " + "-" * (name_w + (flag_w + 2) * len(FEATURE_FLAGS) + 7)
    lines.append(sep)

    for d in sorted(all_data, key=lambda x: x["name"].lower()):
        row = f"  {d['name']:<{name_w}}"
        count = 0
        for flag in FEATURE_FLAGS:
            val = d["features"].get(flag, False)
            mark = "YES" if val else " - "
            if val:
                count += 1
            row += f"  {mark:>{flag_w}}"
        row += f"  {count:>5}"
        lines.append(row)

    lines.append(sep)

    # Column totals
    totals_row = f"  {'TOTAL':<{name_w}}"
    for flag in FEATURE_FLAGS:
        total = sum(1 for d in all_data if d["features"].get(flag, False))
        totals_row += f"  {total:>{flag_w}}"
    totals_row += f"  {'':<5}"
    lines.append(totals_row)
    lines.append("")

    # Feature adoption rates
    subheading("  Feature adoption rates:")
    for flag in FEATURE_FLAGS:
        total = sum(1 for d in all_data if d["features"].get(flag, False))
        pct = total / len(all_data) * 100
        bar_len = int(pct / 5)
        bar = "#" * bar_len + "." * (20 - bar_len)
        label = FEATURE_LABELS.get(flag, flag)
        lines.append(f"    {label:<20}  {total:>2}/{len(all_data):<2}  ({pct:>5.1f}%)  [{bar}]")

    # ── 4. Complexity Ranking ────────────────────────────────

    heading("4. COMPLEXITY RANKING (Simplest to Most Complex)")

    ranked = sorted(all_data, key=lambda d: d["complexity_score"])
    max_score = max(d["complexity_score"] for d in all_data) if all_data else 100

    for rank, d in enumerate(ranked, 1):
        bar_len = int(d["complexity_score"] / max_score * 30) if max_score > 0 else 0
        bar = "#" * bar_len + "." * (30 - bar_len)
        lines.append(
            f"  {rank:>2}. {d['name']:<24} {d['complexity_score']:>5.1f}/100 "
            f"({d['complexity_label']:<12}) [{bar}]"
        )
    lines.append("")

    # Score distribution
    subheading("  Score distribution:")
    buckets = {"trivial (0-19)": 0, "simple (20-39)": 0, "moderate (40-59)": 0,
               "complex (60-79)": 0, "very complex (80-100)": 0}
    for d in all_data:
        s = d["complexity_score"]
        if s < 20:
            buckets["trivial (0-19)"] += 1
        elif s < 40:
            buckets["simple (20-39)"] += 1
        elif s < 60:
            buckets["moderate (40-59)"] += 1
        elif s < 80:
            buckets["complex (60-79)"] += 1
        else:
            buckets["very complex (80-100)"] += 1

    for label, count in buckets.items():
        bar = "#" * (count * 3)
        lines.append(f"    {label:<22}  {count:>2}  {bar}")

    # Statistics
    scores = [d["complexity_score"] for d in all_data]
    avg_score = sum(scores) / len(scores)
    median_idx = len(scores) // 2
    sorted_scores = sorted(scores)
    median_score = sorted_scores[median_idx]
    std_dev = math.sqrt(sum((s - avg_score) ** 2 for s in scores) / len(scores))
    lines.append("")
    lines.append(f"    Mean:   {avg_score:.1f}")
    lines.append(f"    Median: {median_score:.1f}")
    lines.append(f"    StdDev: {std_dev:.1f}")
    lines.append(f"    Range:  {min(scores):.1f} - {max(scores):.1f}")

    # ── 5. Similarity Clusters ───────────────────────────────

    heading("5. SIMILARITY ANALYSIS (Top 5 Most Similar Pairs)")

    # Compute pairwise similarities
    n = len(all_data)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_pairwise_similarity(
                all_data[i]["sim_features"],
                all_data[j]["sim_features"]
            )
            pairs.append((sim, all_data[i]["name"], all_data[j]["name"]))

    pairs.sort(key=lambda x: -x[0])
    top_5 = pairs[:5]

    max_pair_w = max(len(f"{a} <-> {b}") for _, a, b in top_5) if top_5 else 30

    for rank, (sim, a, b) in enumerate(top_5, 1):
        pair_str = f"{a} <-> {b}"
        bar_len = int(sim * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        lines.append(f"  {rank}. {pair_str:<{max_pair_w}}  {sim:.4f}  [{bar}]")

    lines.append("")

    # Bottom 5 most different
    subheading("  Bottom 5 Most Different Pairs:")
    bottom_5 = pairs[-5:]
    bottom_5.reverse()  # least similar first

    for rank, (sim, a, b) in enumerate(bottom_5, 1):
        pair_str = f"{a} <-> {b}"
        bar_len = int(sim * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        lines.append(f"  {rank}. {pair_str:<{max_pair_w}}  {sim:.4f}  [{bar}]")

    # ── 6. Coverage Analysis ─────────────────────────────────

    heading("6. COVERAGE ANALYSIS (Ontology Feature Usage)")

    # Entity types coverage
    subheading("  Entity types:")
    for et in ENTITY_TYPES:
        specs_using = [d["name"] for d in all_data if et in d["coverage"]["entity_types"]]
        count = len(specs_using)
        pct = count / len(all_data) * 100
        mark = "[USED]  " if count > 0 else "[UNUSED]"
        lines.append(f"    {mark} {et:<12}  {count:>2}/{len(all_data)} specs ({pct:>5.1f}%)")
        if count > 0 and count <= 5:
            lines.append(f"             Used by: {', '.join(specs_using)}")

    # Process types coverage
    subheading("  Process types:")
    for pt in PROCESS_TYPES:
        specs_using = [d["name"] for d in all_data if pt in d["coverage"]["process_types"]]
        count = len(specs_using)
        pct = count / len(all_data) * 100
        mark = "[USED]  " if count > 0 else "[UNUSED]"
        lines.append(f"    {mark} {pt:<12}  {count:>2}/{len(all_data)} specs ({pct:>5.1f}%)")
        if count > 0 and count <= 5:
            lines.append(f"             Used by: {', '.join(specs_using)}")

    # Edge types coverage
    subheading("  Edge types:")
    for et in EDGE_TYPES:
        specs_using = [d["name"] for d in all_data if et in d["coverage"]["edge_types"]]
        count = len(specs_using)
        pct = count / len(all_data) * 100
        mark = "[USED]  " if count > 0 else "[UNUSED]"
        lines.append(f"    {mark} {et:<12}  {count:>2}/{len(all_data)} specs ({pct:>5.1f}%)")
        if count > 0 and count <= 5:
            lines.append(f"             Used by: {', '.join(specs_using)}")

    # Summary of unused features
    unused_entity = [et for et in ENTITY_TYPES
                     if not any(et in d["coverage"]["entity_types"] for d in all_data)]
    unused_process = [pt for pt in PROCESS_TYPES
                      if not any(pt in d["coverage"]["process_types"] for d in all_data)]
    unused_edge = [et for et in EDGE_TYPES
                   if not any(et in d["coverage"]["edge_types"] for d in all_data)]
    unused_features = [f for f in FEATURE_FLAGS
                       if not any(d["features"].get(f) for d in all_data)]

    subheading("  UNUSED ontology features (not used by ANY spec):")
    total_unused = len(unused_entity) + len(unused_process) + len(unused_edge) + len(unused_features)
    total_all = len(ENTITY_TYPES) + len(PROCESS_TYPES) + len(EDGE_TYPES) + len(FEATURE_FLAGS)

    if unused_entity:
        lines.append(f"    Entity types:  {', '.join(unused_entity)}")
    else:
        lines.append(f"    Entity types:  (all covered)")
    if unused_process:
        lines.append(f"    Process types: {', '.join(unused_process)}")
    else:
        lines.append(f"    Process types: (all covered)")
    if unused_edge:
        lines.append(f"    Edge types:    {', '.join(unused_edge)}")
    else:
        lines.append(f"    Edge types:    (all covered)")
    if unused_features:
        lines.append(f"    Feature flags: {', '.join(unused_features)}")
    else:
        lines.append(f"    Feature flags: (all covered)")

    used_count = total_all - total_unused
    pct = used_count / total_all * 100
    lines.append("")
    lines.append(f"    Overall ontology coverage: {used_count}/{total_all} features used ({pct:.1f}%)")

    # ── 7. Recommendations ───────────────────────────────────

    heading("7. RECOMMENDATIONS FOR NEW SPECS")

    lines.append("  To improve ontology coverage, consider adding specs that exercise")
    lines.append("  the following unused or underused features:")
    lines.append("")

    rec_number = 1

    # Unused features
    if unused_entity:
        for et in unused_entity:
            lines.append(f"  {rec_number}. Entity type '{et}' is unused.")
            if et == "config":
                lines.append("     Suggestion: Add a spec with a configuration entity, e.g., a")
                lines.append("     parameterized agent that reads from a config object.")
            lines.append("")
            rec_number += 1

    if unused_process:
        for pt in unused_process:
            lines.append(f"  {rec_number}. Process type '{pt}' is unused.")
            if pt == "protocol":
                lines.append("     Suggestion: Add a multi-party negotiation or consensus spec")
                lines.append("     (e.g., auction, voting, contract-net protocol).")
            elif pt == "policy":
                lines.append("     Suggestion: Add a spec with guardrails, rate limits, or safety")
                lines.append("     filters (e.g., a content-moderated chatbot).")
            elif pt == "spawn":
                lines.append("     Suggestion: Add a spec with dynamic agent spawning (e.g.,")
                lines.append("     orchestrator-workers, MapReduce with dynamic fan-out).")
            elif pt == "checkpoint":
                lines.append("     Suggestion: Add a spec with human-in-the-loop approval gates")
                lines.append("     (e.g., a document approval workflow).")
            lines.append("")
            rec_number += 1

    if unused_edge:
        for et in unused_edge:
            lines.append(f"  {rec_number}. Edge type '{et}' is unused.")
            if et == "modify":
                lines.append("     Suggestion: Add a spec where a policy or config dynamically")
                lines.append("     modifies agent behavior at runtime.")
            elif et == "observe":
                lines.append("     Suggestion: Add a spec with a meta-cognitive observer or")
                lines.append("     logging/tracing node that monitors execution.")
            elif et == "error":
                lines.append("     Suggestion: Add a spec with explicit error handling edges")
                lines.append("     that route failures to recovery processes.")
            lines.append("")
            rec_number += 1

    if unused_features:
        for ff_name in unused_features:
            label = FEATURE_LABELS.get(ff_name, ff_name)
            lines.append(f"  {rec_number}. Feature flag '{label}' is never triggered.")
            if ff_name == "recursive_spawn":
                lines.append("     Suggestion: Add a spec with self-referential spawning (e.g.,")
                lines.append("     a recursive web crawler or fractal decomposition solver).")
            elif ff_name == "policies":
                lines.append("     Suggestion: Add a spec with policy processes for guardrails.")
            lines.append("")
            rec_number += 1

    # Underused features (used by <=2 specs)
    subheading("  Underused features (used by 2 or fewer specs):")
    for flag in FEATURE_FLAGS:
        count = sum(1 for d in all_data if d["features"].get(flag))
        if 0 < count <= 2:
            label = FEATURE_LABELS.get(flag, flag)
            users = [d["name"] for d in all_data if d["features"].get(flag)]
            lines.append(f"    - {label}: used by {count} spec(s) ({', '.join(users)})")

    for et in ENTITY_TYPES:
        count = sum(1 for d in all_data if et in d["coverage"]["entity_types"])
        if 0 < count <= 2:
            users = [d["name"] for d in all_data if et in d["coverage"]["entity_types"]]
            lines.append(f"    - Entity '{et}': used by {count} spec(s) ({', '.join(users)})")

    for pt in PROCESS_TYPES:
        count = sum(1 for d in all_data if pt in d["coverage"]["process_types"])
        if 0 < count <= 2:
            users = [d["name"] for d in all_data if pt in d["coverage"]["process_types"]]
            lines.append(f"    - Process '{pt}': used by {count} spec(s) ({', '.join(users)})")

    for et in EDGE_TYPES:
        count = sum(1 for d in all_data if et in d["coverage"]["edge_types"])
        if 0 < count <= 2:
            users = [d["name"] for d in all_data if et in d["coverage"]["edge_types"]]
            lines.append(f"    - Edge '{et}': used by {count} spec(s) ({', '.join(users)})")

    # Topology distribution
    subheading("  Topology distribution:")
    topo_counts = defaultdict(int)
    for d in all_data:
        topo_counts[d["topology"]] += 1

    for topo in TOPOLOGY_ORDER:
        count = topo_counts.get(topo, 0)
        bar = "#" * (count * 2)
        lines.append(f"    {topo:<14}  {count:>2}  {bar}")

    missing_topos = [t for t in TOPOLOGY_ORDER if t not in topo_counts]
    if missing_topos:
        lines.append("")
        lines.append(f"  Missing topologies: {', '.join(missing_topos)}")
        lines.append("  Consider adding specs that produce these graph structures.")

    # ── Footer ───────────────────────────────────────────────

    lines.append("")
    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agent Ontology Comparative Spec Analysis Report",
    )
    parser.add_argument(
        "--specs-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs"),
        help="Directory containing spec YAML files (default: ./specs)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw analysis data as JSON instead of formatted report",
    )
    # Every sibling CLI (ao-lint, ao-coverage, ao-complexity, ao-mermaid) takes
    # `--all <dir>`, and the README documents ao-report the same way. This tool
    # always reports on a whole directory, so --all is accepted for symmetry and
    # the directory may be given positionally.
    parser.add_argument(
        "--all",
        action="store_true",
        help="Report on every spec in the directory (the only mode; accepted for symmetry with the other ao-* tools)",
    )
    parser.add_argument(
        "specs_dir_positional",
        nargs="?",
        metavar="SPECS_DIR",
        help="Directory containing spec YAML files (overrides --specs-dir)",
    )
    args = parser.parse_args()

    specs_dir = args.specs_dir_positional or args.specs_dir
    if not os.path.isdir(specs_dir):
        print(f"Error: {specs_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    spec_paths = find_specs(specs_dir)
    if not spec_paths:
        print(f"Error: no YAML spec files found in {specs_dir}", file=sys.stderr)
        sys.exit(1)

    # Analyze all specs
    all_data = []
    for path in spec_paths:
        try:
            spec = load_yaml(path)
            data = analyze_spec(spec, path)
            all_data.append(data)
        except Exception as exc:
            print(f"Warning: failed to analyze {path}: {exc}", file=sys.stderr)

    if not all_data:
        print("Error: no specs could be analyzed.", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        # Serialize: convert sets to sorted lists for JSON
        json_data = []
        for d in all_data:
            jd = dict(d)
            jd["coverage"] = {
                k: sorted(v) for k, v in d["coverage"].items()
            }
            # Remove sim_features (internal)
            sf = jd.pop("sim_features", {})
            jd["sim_entity_types"] = sorted(sf.get("entity_types", set()))
            jd["sim_process_types"] = sorted(sf.get("process_types", set()))
            jd["sim_edge_types"] = sorted(sf.get("edge_types", set()))
            jd["sim_feature_flags"] = sorted(sf.get("feature_flags", set()))
            json_data.append(jd)
        print(json.dumps(json_data, indent=2))
    else:
        report = generate_report(all_data)
        print(report)


if __name__ == "__main__":
    main()
