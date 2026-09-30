#!/usr/bin/env python3
"""
Agent Ontology Spec Similarity Analyzer
=========================================
Computes pairwise similarity between agent specs using Jaccard similarity
on categorical/set features and cosine similarity on numerical features.
Supports similarity matrix output, top-N most similar pairs, and
agglomerative clustering.

Usage:
    python3 similarity.py --all specs/                 # Print similarity matrix
    python3 similarity.py --all specs/ --top 5         # Show top 5 most similar pairs
    python3 similarity.py --all specs/ --clusters 3    # Cluster into 3 groups
    python3 similarity.py --all specs/ --json          # JSON output
    python3 similarity.py specs/react.yaml specs/autogpt.yaml  # Compare two specs
"""

import argparse
import json
import math
import os
import sys

import yaml


# ── ANSI colors ────────────────────────────────────────────────────

ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"

# Thresholds for color coding similarity values
HIGH_SIMILARITY = 0.70
MEDIUM_SIMILARITY = 0.40


def _use_color():
    """Return True if stdout supports ANSI color."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _color_val(val, is_diagonal=False):
    """Color a similarity value for terminal output."""
    if not _use_color():
        return f"{val:.2f}"
    if is_diagonal:
        return f"{ANSI_DIM}{val:.2f}{ANSI_RESET}"
    if val >= HIGH_SIMILARITY:
        return f"{ANSI_GREEN}{val:.2f}{ANSI_RESET}"
    if val >= MEDIUM_SIMILARITY:
        return f"{ANSI_YELLOW}{val:.2f}{ANSI_RESET}"
    return f"{val:.2f}"


# ── YAML loading ──────────────────────────────────────────────────

def load_yaml(path):
    """Load and return a YAML spec file."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def find_specs(directory):
    """Find all .yaml files in a directory."""
    specs = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith((".yaml", ".yml")):
            specs.append(os.path.join(directory, fname))
    return specs


# ── Canonical type sets ───────────────────────────────────────────

ENTITY_TYPES = ("agent", "store", "tool", "human", "config", "channel", "team", "conversation")
PROCESS_TYPES = ("step", "gate", "checkpoint", "spawn", "protocol", "policy", "error_handler")
EDGE_TYPES = ("flow", "invoke", "loop", "branch", "read", "write", "modify", "observe", "error", "publish", "subscribe", "handoff")
FEATURE_FLAGS = ("fan_out", "loops", "recursive_spawn", "human_in_loop", "stores", "tools", "policies", "channels", "teams", "handoffs")


# ── Feature extraction ───────────────────────────────────────────

def _collect_types(items, key):
    """Return the set of type values from a list of dicts."""
    return {item.get(key, "").lower() for item in items if item.get(key)}


def _compute_feature_flags(spec):
    """Evaluate feature flags from spec structure, returning a set of active flags."""
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])

    entity_types = _collect_types(entities, "type")
    process_types = _collect_types(processes, "type")
    edge_types = _collect_types(edges, "type")

    active = set()

    # fan_out: any process node with >1 outgoing flow edge
    flow_out = {}
    for e in edges:
        if e.get("type", "").lower() == "flow":
            src = e.get("from") or e.get("source") or e.get("src", "")
            if src:
                flow_out[src] = flow_out.get(src, 0) + 1
    if any(c > 1 for c in flow_out.values()):
        active.add("fan_out")

    # loops
    if "loop" in edge_types:
        active.add("loops")

    # recursive_spawn
    for proc in processes:
        if proc.get("type", "").lower() == "spawn":
            if proc.get("recursive") is True or str(proc.get("template", "")).lower() == "self":
                active.add("recursive_spawn")
                break

    # human_in_loop
    if "checkpoint" in process_types or "human" in entity_types:
        active.add("human_in_loop")

    # stores
    if "store" in entity_types:
        active.add("stores")

    # tools
    if "tool" in entity_types:
        active.add("tools")

    # policies
    if "policy" in process_types:
        active.add("policies")

    # channels (pub/sub)
    if "channel" in entity_types:
        active.add("channels")

    # teams
    if "team" in entity_types:
        active.add("teams")

    # handoffs
    if "handoff" in edge_types:
        active.add("handoffs")

    return active


def _get_topology(spec):
    """Classify the topology of a spec.

    Uses a simplified inline classifier to avoid requiring topology.py as
    a hard dependency, but produces compatible classifications.
    """
    try:
        from topology import analyze_topology
        result = analyze_topology(spec)
        return result.get("classification", "unknown")
    except ImportError:
        pass

    # Inline fallback: simple classification based on edge types
    edges = spec.get("edges", [])
    edge_types = {e.get("type", "").lower() for e in edges}
    has_loops = "loop" in edge_types
    has_branches = "branch" in edge_types

    # Build adjacency to check fan-out
    adj = {}
    for e in edges:
        src = e.get("from") or e.get("source") or ""
        if src:
            adj.setdefault(src, []).append(e.get("to") or e.get("target") or "")
    max_fan_out = max((len(v) for v in adj.values()), default=0)

    if has_loops and has_branches:
        return "multi-cyclic"
    if has_loops:
        return "cyclic"
    if max_fan_out > 3:
        return "star"
    if has_branches or max_fan_out > 1:
        return "dag"
    return "linear"


def extract_features(spec):
    """Extract all feature components from a spec for similarity comparison.

    Returns a dict with:
        name: str
        entity_types: set of entity type strings
        process_types: set of process type strings
        edge_types: set of edge type strings
        feature_flags: set of active feature flag strings
        topology: str classification
        numerics: dict of numeric feature values
    """
    name = spec.get("name") or spec.get("id") or "unknown"
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])
    schemas = spec.get("schemas", [])

    entity_types = _collect_types(entities, "type") & set(ENTITY_TYPES)
    process_types = _collect_types(processes, "type") & set(PROCESS_TYPES)
    edge_types = _collect_types(edges, "type") & set(EDGE_TYPES)
    feature_flags = _compute_feature_flags(spec)
    topology = _get_topology(spec)

    # Compute complexity score inline to avoid hard dependency
    try:
        from complexity import analyze_spec
        metrics = analyze_spec(spec)
        complexity_score = metrics.get("overall_score", 0)
    except ImportError:
        complexity_score = 0

    numerics = {
        "entity_count": len(entities),
        "process_count": len(processes),
        "edge_count": len(edges),
        "schema_count": len(schemas),
        "complexity_score": complexity_score,
    }

    return {
        "name": name,
        "entity_types": entity_types,
        "process_types": process_types,
        "edge_types": edge_types,
        "feature_flags": feature_flags,
        "topology": topology,
        "numerics": numerics,
    }


# ── Similarity computation ───────────────────────────────────────

def _jaccard(set_a, set_b):
    """Compute Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _numeric_similarity(nums_a, nums_b):
    """Compute similarity from numeric features using normalized euclidean distance.

    Each feature is min-max normalized across the pair, then euclidean distance
    is computed and converted to a similarity in [0, 1].
    """
    keys = sorted(set(nums_a.keys()) | set(nums_b.keys()))
    if not keys:
        return 1.0

    # Normalize each feature to [0, 1] relative to the max across both values
    diffs_sq = []
    for k in keys:
        a = nums_a.get(k, 0)
        b = nums_b.get(k, 0)
        max_val = max(abs(a), abs(b), 1)  # avoid division by zero
        norm_a = a / max_val
        norm_b = b / max_val
        diffs_sq.append((norm_a - norm_b) ** 2)

    # Euclidean distance, normalized by sqrt(num_features) so max distance = 1
    dist = math.sqrt(sum(diffs_sq) / len(diffs_sq))
    return max(0.0, 1.0 - dist)


def compute_similarity(feat_a, feat_b):
    """Compute weighted similarity between two feature dicts.

    Weights:
        0.30 entity_types (Jaccard)
        0.20 process_types (Jaccard)
        0.20 edge_types (Jaccard)
        0.15 feature_flags (Jaccard)
        0.05 topology (exact match)
        0.10 numerics (euclidean-based similarity)
    """
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


def compute_similarity_matrix(features_list):
    """Compute pairwise similarity matrix for a list of feature dicts.

    Returns:
        names: list of spec names
        matrix: list of lists (NxN) of similarity values
    """
    n = len(features_list)
    names = [f["name"] for f in features_list]
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = compute_similarity(features_list[i], features_list[j])
            matrix[i][j] = sim
            matrix[j][i] = sim

    return names, matrix


# ── Agglomerative clustering ────────────────────────────────────

def agglomerative_cluster(names, matrix, k):
    """Simple agglomerative clustering using average linkage.

    Args:
        names: list of spec names
        matrix: NxN similarity matrix (higher = more similar)
        k: target number of clusters

    Returns:
        list of lists, each inner list contains spec names in one cluster
    """
    n = len(names)
    if k >= n:
        return [[name] for name in names]
    if k <= 0:
        return [list(names)]

    # Initialize: each item is its own cluster
    # clusters maps cluster_id -> set of original indices
    clusters = {i: {i} for i in range(n)}
    active_ids = set(range(n))

    # Precompute pairwise similarities between cluster pairs
    # We use average linkage: sim(C1, C2) = avg of sim(i, j) for i in C1, j in C2
    def cluster_similarity(c1_indices, c2_indices):
        total = 0.0
        count = 0
        for i in c1_indices:
            for j in c2_indices:
                total += matrix[i][j]
                count += 1
        return total / count if count > 0 else 0.0

    next_id = n

    while len(active_ids) > k:
        # Find the two most similar clusters
        best_sim = -1.0
        best_pair = None
        active_list = sorted(active_ids)

        for idx_a in range(len(active_list)):
            for idx_b in range(idx_a + 1, len(active_list)):
                ca = active_list[idx_a]
                cb = active_list[idx_b]
                sim = cluster_similarity(clusters[ca], clusters[cb])
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (ca, cb)

        if best_pair is None:
            break

        # Merge the two most similar clusters
        ca, cb = best_pair
        merged = clusters[ca] | clusters[cb]
        clusters[next_id] = merged
        active_ids.discard(ca)
        active_ids.discard(cb)
        active_ids.add(next_id)
        next_id += 1

    # Build result: map indices back to names
    result = []
    for cid in sorted(active_ids):
        members = sorted(clusters[cid])
        result.append([names[i] for i in members])

    # Sort clusters by size descending, then by first member name
    result.sort(key=lambda c: (-len(c), c[0].lower()))
    return result


# ── Formatting helpers ───────────────────────────────────────────

def _short_name(name, width=14):
    """Truncate a name to fit in a column."""
    if len(name) <= width:
        return name
    return name[:width - 2] + ".."


def format_similarity_matrix(names, matrix):
    """Format the similarity matrix as an ASCII table with ANSI colors."""
    n = len(names)
    use_color = _use_color()

    # Determine column width based on name lengths
    col_width = max(6, min(14, max((len(n) for n in names), default=6)))
    label_width = max(col_width, max((len(n) for n in names), default=10))
    label_width = min(label_width, 20)

    lines = []

    if use_color:
        lines.append(f"\n{ANSI_BOLD}  SIMILARITY MATRIX{ANSI_RESET}")
    else:
        lines.append("\n  SIMILARITY MATRIX")
    lines.append("")

    # Header row
    header = " " * (label_width + 2)
    for name in names:
        header += f"  {_short_name(name, col_width):>{col_width}}"
    lines.append(header)

    # Separator
    sep = " " * (label_width + 2) + "  " + "-" * ((col_width + 2) * n - 2)
    lines.append(sep)

    # Data rows
    for i in range(n):
        row_name = _short_name(names[i], label_width)
        row = f"  {row_name:>{label_width}}"
        for j in range(n):
            val = matrix[i][j]
            colored = _color_val(val, is_diagonal=(i == j))
            # Pad manually since ANSI codes mess up alignment
            if use_color and i != j and (val >= HIGH_SIMILARITY or val >= MEDIUM_SIMILARITY):
                # colored string has ANSI codes, pad the raw portion
                row += f"  {colored:>{col_width + (len(colored) - 4)}}"
            else:
                row += f"  {colored:>{col_width}}"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def format_top_pairs(names, matrix, top_n):
    """Format the top-N most similar pairs."""
    n = len(names)
    use_color = _use_color()

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((matrix[i][j], names[i], names[j]))

    pairs.sort(key=lambda x: -x[0])
    top = pairs[:top_n]

    lines = []
    if use_color:
        lines.append(f"\n{ANSI_BOLD}  TOP {top_n} MOST SIMILAR PAIRS{ANSI_RESET}")
    else:
        lines.append(f"\n  TOP {top_n} MOST SIMILAR PAIRS")
    lines.append("")

    max_pair_width = max(
        (len(f"{a} <-> {b}") for _, a, b in top),
        default=20,
    )

    for rank, (sim, a, b) in enumerate(top, 1):
        pair_str = f"{a} <-> {b}"
        val_str = _color_val(sim)
        lines.append(f"  {rank:>3}. {pair_str:<{max_pair_width}}  {val_str}")

    lines.append("")
    return "\n".join(lines)


def format_clusters(clusters, k):
    """Format cluster membership."""
    use_color = _use_color()

    lines = []
    if use_color:
        lines.append(f"\n{ANSI_BOLD}  CLUSTERS (k={k}){ANSI_RESET}")
    else:
        lines.append(f"\n  CLUSTERS (k={k})")
    lines.append("")

    for i, cluster in enumerate(clusters, 1):
        members = ", ".join(cluster)
        lines.append(f"  Cluster {i}: {members}")

    lines.append("")
    return "\n".join(lines)


def format_pair_comparison(feat_a, feat_b, similarity):
    """Format a detailed comparison between two specs."""
    use_color = _use_color()

    lines = []
    name_a = feat_a["name"]
    name_b = feat_b["name"]

    if use_color:
        lines.append(f"\n{ANSI_BOLD}  SPEC COMPARISON{ANSI_RESET}")
    else:
        lines.append("\n  SPEC COMPARISON")
    lines.append(f"  {name_a}  vs  {name_b}")
    lines.append("  " + "=" * 56)

    # Component similarities
    entity_sim = _jaccard(feat_a["entity_types"], feat_b["entity_types"])
    process_sim = _jaccard(feat_a["process_types"], feat_b["process_types"])
    edge_sim = _jaccard(feat_a["edge_types"], feat_b["edge_types"])
    feature_sim = _jaccard(feat_a["feature_flags"], feat_b["feature_flags"])
    topology_sim = 1.0 if feat_a["topology"] == feat_b["topology"] else 0.0
    numeric_sim = _numeric_similarity(feat_a["numerics"], feat_b["numerics"])

    lines.append("")
    lines.append("  Component Similarities:")
    lines.append(f"    Entity types  (0.30):  {_color_val(entity_sim)}")
    lines.append(f"    Process types (0.20):  {_color_val(process_sim)}")
    lines.append(f"    Edge types    (0.20):  {_color_val(edge_sim)}")
    lines.append(f"    Feature flags (0.15):  {_color_val(feature_sim)}")
    lines.append(f"    Topology      (0.05):  {_color_val(topology_sim)}")
    lines.append(f"    Numeric       (0.10):  {_color_val(numeric_sim)}")
    lines.append("")

    if use_color:
        lines.append(f"  {ANSI_BOLD}Overall similarity: {_color_val(similarity)}{ANSI_RESET}")
    else:
        lines.append(f"  Overall similarity: {similarity:.2f}")
    lines.append("")

    # Detail: set differences
    lines.append("  Feature Details:")
    label_w = 18

    def _set_detail(label, set_a, set_b):
        shared = sorted(set_a & set_b)
        only_a = sorted(set_a - set_b)
        only_b = sorted(set_b - set_a)
        lines.append(f"    {label}")
        if shared:
            lines.append(f"      shared:     {', '.join(shared)}")
        if only_a:
            lines.append(f"      only {_short_name(name_a, 10)}: {', '.join(only_a)}")
        if only_b:
            lines.append(f"      only {_short_name(name_b, 10)}: {', '.join(only_b)}")

    _set_detail("Entity types:", feat_a["entity_types"], feat_b["entity_types"])
    _set_detail("Process types:", feat_a["process_types"], feat_b["process_types"])
    _set_detail("Edge types:", feat_a["edge_types"], feat_b["edge_types"])
    _set_detail("Feature flags:", feat_a["feature_flags"], feat_b["feature_flags"])

    lines.append(f"    Topology: {feat_a['topology']} vs {feat_b['topology']}")

    lines.append("")
    lines.append("  Numeric Features:")
    for key in sorted(feat_a["numerics"].keys()):
        va = feat_a["numerics"][key]
        vb = feat_b["numerics"][key]
        lines.append(f"    {key:>20s}:  {va:>6.1f}  vs  {vb:<6.1f}")

    lines.append("")
    return "\n".join(lines)


def build_json_output(names, matrix, features_list, clusters=None):
    """Build a JSON-serializable dict of results."""
    n = len(names)

    # Similarity matrix
    sim_matrix = {}
    for i in range(n):
        sim_matrix[names[i]] = {}
        for j in range(n):
            sim_matrix[names[i]][names[j]] = round(matrix[i][j], 4)

    # Top pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "spec_a": names[i],
                "spec_b": names[j],
                "similarity": round(matrix[i][j], 4),
            })
    pairs.sort(key=lambda x: -x["similarity"])

    # Features
    features_out = []
    for feat in features_list:
        features_out.append({
            "name": feat["name"],
            "entity_types": sorted(feat["entity_types"]),
            "process_types": sorted(feat["process_types"]),
            "edge_types": sorted(feat["edge_types"]),
            "feature_flags": sorted(feat["feature_flags"]),
            "topology": feat["topology"],
            "numerics": feat["numerics"],
        })

    result = {
        "similarity_matrix": sim_matrix,
        "pairs": pairs,
        "features": features_out,
    }

    if clusters is not None:
        result["clusters"] = clusters

    return result


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agent Ontology Spec Similarity Analyzer -- pairwise similarity and clustering for agent specs",
        epilog="Examples:\n"
               "  python3 similarity.py --all specs/\n"
               "  python3 similarity.py --all specs/ --top 5\n"
               "  python3 similarity.py --all specs/ --clusters 3\n"
               "  python3 similarity.py --all specs/ --json\n"
               "  python3 similarity.py specs/react.yaml specs/autogpt.yaml\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "specs",
        nargs="+",
        help="Path(s) to spec YAML files, or a directory with --all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_specs",
        help="Analyze all *.yaml files in the given directory",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="Show top N most similar pairs",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=0,
        metavar="K",
        help="Cluster specs into K groups using agglomerative clustering",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # ── Load specs ──────────────────────────────────────────
    spec_paths = []

    if args.all_specs:
        # Treat first positional arg as directory
        spec_dir = args.specs[0]
        if not os.path.isdir(spec_dir):
            print(f"Error: {spec_dir} is not a directory (use --all with a directory)",
                  file=sys.stderr)
            sys.exit(1)
        spec_paths = find_specs(spec_dir)
        if not spec_paths:
            print(f"No YAML specs found in {spec_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        # Individual spec files
        for path in args.specs:
            if not os.path.isfile(path):
                print(f"Error: {path} not found", file=sys.stderr)
                sys.exit(1)
            spec_paths.append(path)

    # ── Extract features ────────────────────────────────────
    features_list = []
    for path in spec_paths:
        try:
            spec = load_yaml(path)
            feat = extract_features(spec)
            feat["file"] = os.path.basename(path)
            features_list.append(feat)
        except Exception as exc:
            print(f"Warning: failed to analyze {path}: {exc}", file=sys.stderr)

    if len(features_list) < 2:
        print("Error: need at least 2 specs to compare", file=sys.stderr)
        sys.exit(1)

    # ── Two-spec comparison mode ────────────────────────────
    if not args.all_specs and len(features_list) == 2:
        feat_a, feat_b = features_list
        sim = compute_similarity(feat_a, feat_b)

        if args.json_output:
            result = {
                "spec_a": feat_a["name"],
                "spec_b": feat_b["name"],
                "similarity": round(sim, 4),
                "features_a": {
                    "entity_types": sorted(feat_a["entity_types"]),
                    "process_types": sorted(feat_a["process_types"]),
                    "edge_types": sorted(feat_a["edge_types"]),
                    "feature_flags": sorted(feat_a["feature_flags"]),
                    "topology": feat_a["topology"],
                    "numerics": feat_a["numerics"],
                },
                "features_b": {
                    "entity_types": sorted(feat_b["entity_types"]),
                    "process_types": sorted(feat_b["process_types"]),
                    "edge_types": sorted(feat_b["edge_types"]),
                    "feature_flags": sorted(feat_b["feature_flags"]),
                    "topology": feat_b["topology"],
                    "numerics": feat_b["numerics"],
                },
            }
            print(json.dumps(result, indent=2))
        else:
            print(format_pair_comparison(feat_a, feat_b, sim))
        return

    # ── Multi-spec mode ─────────────────────────────────────
    names, matrix = compute_similarity_matrix(features_list)

    # Compute clusters if requested
    clusters = None
    if args.clusters > 0:
        clusters = agglomerative_cluster(names, matrix, args.clusters)

    if args.json_output:
        result = build_json_output(names, matrix, features_list, clusters)
        print(json.dumps(result, indent=2))
        return

    # Default: print similarity matrix
    print(format_similarity_matrix(names, matrix))

    # Top pairs
    if args.top > 0:
        print(format_top_pairs(names, matrix, args.top))

    # Clusters
    if clusters is not None:
        print(format_clusters(clusters, args.clusters))


if __name__ == "__main__":
    main()
