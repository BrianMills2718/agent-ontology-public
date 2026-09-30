#!/usr/bin/env python3
"""
Pattern Library — Reusable architectural building blocks for agent specs.

Extracts 7 canonical patterns from reference specs and provides a library
for composing, mutating, and evolving agent architectures at the pattern level.

Usage:
    python3 patterns.py                          # List all patterns
    python3 patterns.py --info reasoning_loop    # Show pattern details
    python3 patterns.py --compat reasoning_loop critique_cycle  # Check compatibility
"""

import argparse
import copy
import os
import sys
from collections import deque

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPECS_DIR = os.path.join(SCRIPT_DIR, "specs")


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _load_spec(name):
    """Load a spec YAML from the specs directory."""
    path = os.path.join(SPECS_DIR, f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _process_by_id(spec, pid):
    for p in spec.get("processes", []):
        if p.get("id") == pid:
            return p
    return None


def _entity_by_id(spec, eid):
    for e in spec.get("entities", []):
        if e.get("id") == eid:
            return e
    return None


def _schema_by_name(spec, name):
    for s in spec.get("schemas", []):
        if s.get("name") == name:
            return s
    return None


# ════════════════════════════════════════════════════════════════════
# Pattern extraction
# ════════════════════════════════════════════════════════════════════

def _extract_pattern(spec, entry, exits, name, description, config=None,
                     inputs=None, outputs=None):
    """Extract a subgraph pattern from a spec via BFS from entry to exits.

    Args:
        spec: Full spec dict
        entry: Entry process ID for the pattern
        exits: List of exit process IDs (included in pattern)
        name: Pattern name
        description: Human-readable description
        config: Dict of tunable params with defaults
        inputs: List of required input state field names
        outputs: List of produced output state field names

    Returns:
        Pattern dict with processes, edges, entities, schemas, interface
    """
    # BFS from entry, collecting all reachable processes
    process_ids = set()
    queue = deque([entry])
    while queue:
        pid = queue.popleft()
        if pid in process_ids:
            continue
        proc = _process_by_id(spec, pid)
        if not proc:
            continue
        process_ids.add(pid)
        # Don't expand beyond exit points
        if pid in exits:
            continue
        # Follow flow, loop, branch edges from this process
        for edge in spec.get("edges", []):
            if edge.get("from") != pid:
                continue
            if edge.get("type") in ("flow", "loop", "branch"):
                target = edge.get("to")
                if target and _process_by_id(spec, target):
                    queue.append(target)
        # Follow gate branch targets
        if proc.get("type") == "gate":
            for branch in proc.get("branches", []):
                target = branch.get("target")
                if target:
                    queue.append(target)

    # Collect processes
    processes = []
    for p in spec.get("processes", []):
        if p.get("id") in process_ids:
            processes.append(copy.deepcopy(p))

    # Collect edges: internal edges between pattern processes + invoke/read/write
    entity_ids = set()
    schema_names = set()
    edges = []
    for edge in spec.get("edges", []):
        efrom = edge.get("from")
        eto = edge.get("to")
        etype = edge.get("type")

        # Include flow/loop/branch edges between pattern processes
        if etype in ("flow", "loop", "branch"):
            if efrom in process_ids and eto in process_ids:
                edges.append(copy.deepcopy(edge))
        # Include invoke/read/write edges from pattern processes
        elif etype in ("invoke", "read", "write", "observe"):
            if efrom in process_ids:
                edges.append(copy.deepcopy(edge))
                entity_ids.add(eto)
                # Collect schema refs from edge
                for field in ("data", "input", "output"):
                    ref = edge.get(field)
                    if ref:
                        schema_names.add(ref)

    # Collect entities referenced by pattern edges
    entities = []
    for e in spec.get("entities", []):
        if e.get("id") in entity_ids:
            entities.append(copy.deepcopy(e))
            # Collect schema refs from entity
            for field in ("input_schema", "output_schema", "schema"):
                ref = e.get(field)
                if ref:
                    schema_names.add(ref)

    # Collect schema refs from processes
    for p in processes:
        for field in ("data_in", "data_out", "schema"):
            ref = p.get(field)
            if ref:
                schema_names.add(ref)

    # Recursively resolve schema references (schemas that reference other schemas)
    resolved = set()
    to_resolve = set(schema_names)
    while to_resolve:
        sname = to_resolve.pop()
        if sname in resolved:
            continue
        resolved.add(sname)
        schema = _schema_by_name(spec, sname)
        if schema:
            for field in schema.get("fields", []):
                ftype = field.get("type", "")
                # Check for references like "list<SubProblem>" or just "SubProblem"
                inner = ftype.replace("list<", "").replace(">", "").strip()
                if inner and _schema_by_name(spec, inner):
                    to_resolve.add(inner)

    # Collect schemas
    schemas = []
    for s in spec.get("schemas", []):
        if s.get("name") in resolved:
            schemas.append(copy.deepcopy(s))

    return {
        "name": name,
        "description": description,
        "source_spec": spec.get("name", "unknown"),
        "processes": processes,
        "edges": edges,
        "entities": entities,
        "schemas": schemas,
        "config": config or {},
        "interface": {
            "entry": entry,
            "exits": exits,
            "inputs": inputs or [],
            "outputs": outputs or [],
        },
    }


# ════════════════════════════════════════════════════════════════════
# Pattern definitions — extracted from reference specs
# ════════════════════════════════════════════════════════════════════

def _build_pattern_library():
    """Build the pattern library by extracting from reference specs."""
    library = {}

    # 1. reasoning_loop from react.yaml
    try:
        spec = _load_spec("react")
        library["reasoning_loop"] = _extract_pattern(
            spec,
            entry="think_or_act",
            exits=["emit_answer"],
            name="reasoning_loop",
            description="ReAct-style reasoning loop: think, act with tools, observe, repeat until answer",
            config={"max_steps": 10},
            inputs=["query"],
            outputs=["answer", "trajectory"],
        )
    except Exception as e:
        print(f"Warning: Could not extract reasoning_loop: {e}", file=sys.stderr)

    # 2. critique_cycle from self_refine.yaml
    try:
        spec = _load_spec("self_refine")
        library["critique_cycle"] = _extract_pattern(
            spec,
            entry="generate",
            exits=["finalize"],
            name="critique_cycle",
            description="Generate-critique-refine loop: produce output, evaluate quality, refine if below threshold",
            config={"max_rounds": 3, "quality_threshold": 7},
            inputs=["task"],
            outputs=["final_output", "final_score", "total_rounds"],
        )
    except Exception as e:
        print(f"Warning: Could not extract critique_cycle: {e}", file=sys.stderr)

    # 3. debate from debate.yaml
    try:
        spec = _load_spec("debate")
        library["debate"] = _extract_pattern(
            spec,
            entry="initialize_debate",
            exits=["end_debate"],
            name="debate",
            description="Multi-agent debate: pro argues, con rebuts, judge evaluates, repeats for N rounds",
            config={"max_rounds": 3},
            inputs=["proposition", "pro_position", "con_position"],
            outputs=["winner", "summary", "pro_score", "con_score"],
        )
    except Exception as e:
        print(f"Warning: Could not extract debate: {e}", file=sys.stderr)

    # 4. retrieval from rag.yaml
    try:
        spec = _load_spec("rag")
        library["retrieval"] = _extract_pattern(
            spec,
            entry="rewrite_query",
            exits=["emit_answer", "no_answer"],
            name="retrieval",
            description="RAG pipeline: rewrite query, retrieve from vector store, filter by relevance, generate answer",
            config={"top_k": 5},
            inputs=["query"],
            outputs=["answer", "citations"],
        )
    except Exception as e:
        print(f"Warning: Could not extract retrieval: {e}", file=sys.stderr)

    # 5. decomposition from plan_and_solve.yaml
    try:
        spec = _load_spec("plan_and_solve")
        library["decomposition"] = _extract_pattern(
            spec,
            entry="decompose",
            exits=["synthesize"],
            name="decomposition",
            description="Plan-and-solve: decompose problem into sub-problems, solve each with verification, synthesize final answer",
            config={"max_retries": 2},
            inputs=["problem"],
            outputs=["final_answer", "synthesis_notes"],
        )
    except Exception as e:
        print(f"Warning: Could not extract decomposition: {e}", file=sys.stderr)

    # 6. fan_out_aggregate from map_reduce.yaml
    try:
        spec = _load_spec("map_reduce")
        library["fan_out_aggregate"] = _extract_pattern(
            spec,
            entry="chunk_document",
            exits=["finalize_output"],
            name="fan_out_aggregate",
            description="MapReduce: split input into chunks, process each in parallel, shuffle, reduce, quality-check",
            config={"chunk_size": 1000},
            inputs=["full_text", "task_description"],
            outputs=["final_output"],
        )
    except Exception as e:
        print(f"Warning: Could not extract fan_out_aggregate: {e}", file=sys.stderr)

    # 7. reflection from reflexion.yaml
    try:
        spec = _load_spec("reflexion")
        library["reflection"] = _extract_pattern(
            spec,
            entry="attempt_task",
            exits=["finalize_success", "finalize_failure"],
            name="reflection",
            description="Reflexion: attempt task, evaluate, self-reflect on failure, retry with accumulated episodic memory",
            config={"max_trials": 5},
            inputs=["task", "success_criteria"],
            outputs=["final_answer", "total_trials", "final_score", "outcome"],
        )
    except Exception as e:
        print(f"Warning: Could not extract reflection: {e}", file=sys.stderr)

    return library


# Build library at import time
PATTERN_LIBRARY = _build_pattern_library()


# ════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════

def list_patterns():
    """Return list of pattern names."""
    return sorted(PATTERN_LIBRARY.keys())


def get_pattern(name):
    """Return a deep copy of a pattern by name."""
    if name not in PATTERN_LIBRARY:
        raise ValueError(f"Unknown pattern '{name}'. Available: {list_patterns()}")
    return copy.deepcopy(PATTERN_LIBRARY[name])


def pattern_info(name):
    """Return a formatted string with pattern details."""
    p = PATTERN_LIBRARY.get(name)
    if not p:
        return f"Unknown pattern: {name}"
    lines = [
        f"Pattern: {p['name']}",
        f"  Source: {p['source_spec']}",
        f"  Description: {p['description']}",
        f"  Processes: {len(p['processes'])} ({', '.join(pr['id'] for pr in p['processes'])})",
        f"  Entities: {len(p['entities'])} ({', '.join(e['id'] for e in p['entities'])})",
        f"  Edges: {len(p['edges'])}",
        f"  Schemas: {len(p['schemas'])} ({', '.join(s['name'] for s in p['schemas'])})",
        f"  Config: {p['config']}",
        f"  Interface:",
        f"    Entry: {p['interface']['entry']}",
        f"    Exits: {p['interface']['exits']}",
        f"    Inputs: {p['interface']['inputs']}",
        f"    Outputs: {p['interface']['outputs']}",
    ]
    return "\n".join(lines)


def compatible_patterns(pattern_a_name, pattern_b_name):
    """Check if pattern A's outputs can feed pattern B's inputs.

    Returns True if at least one output field of A matches an input field of B,
    or if B has no required inputs (accepts anything).
    """
    a = PATTERN_LIBRARY.get(pattern_a_name)
    b = PATTERN_LIBRARY.get(pattern_b_name)
    if not a or not b:
        return False
    a_outputs = set(a["interface"]["outputs"])
    b_inputs = set(b["interface"]["inputs"])
    if not b_inputs:
        return True
    return bool(a_outputs & b_inputs)


def detect_patterns(spec, spec_path=None):
    """Detect which patterns are present in a spec.

    Uses OWL DL structural classification when available, falling back to
    ID-based matching. Returns a list of (pattern_name, matched_process_ids, prefix) tuples.
    """
    # Try OWL structural detection first (label-independent, topology-based)
    try:
        from .owl_bridge import build_bridge_ontology_in_world, spec_to_owl
        from .ontology_owl import classify_structural
        from owlready2 import World

        world = World()
        onto = build_bridge_ontology_in_world(world)
        spec_inst = spec_to_owl(onto, spec if spec_path is None else spec_path)
        results = classify_structural(onto, [spec_inst])

        owl_detected = []
        for _spec_name, owl_patterns in results.items():
            for owl_pat in owl_patterns:
                canon = _OWL_TO_PATTERN.get(owl_pat, owl_pat.lower())
                owl_detected.append((canon, set(), "owl"))
        if owl_detected:
            # Merge with ID-based for process ID info
            id_results = _detect_patterns_by_id(spec)
            id_map = {name: (pids, prefix) for name, pids, prefix in id_results}
            merged = []
            seen = set()
            for name, _, _ in owl_detected:
                if name in id_map:
                    merged.append((name, id_map[name][0], id_map[name][1]))
                else:
                    merged.append((name, set(), "owl"))
                seen.add(name)
            # Add any ID-only detections not found by OWL
            for name, pids, prefix in id_results:
                if name not in seen:
                    merged.append((name, pids, prefix))
            return merged
    except Exception:
        pass

    # Fallback to ID-based detection
    return _detect_patterns_by_id(spec)


def _detect_patterns_by_id(spec):
    """Detect patterns by matching process IDs (original heuristic method).

    Returns a list of (pattern_name, matched_process_ids, prefix) tuples.
    """
    spec_process_ids = {p.get("id") for p in spec.get("processes", []) if p.get("id")}
    detected = []

    for pname, pattern in PATTERN_LIBRARY.items():
        pattern_pids = {p["id"] for p in pattern["processes"]}
        if not pattern_pids:
            continue

        # Try exact match (no prefix)
        overlap = pattern_pids & spec_process_ids
        if len(overlap) >= len(pattern_pids) * 0.6:
            detected.append((pname, overlap, ""))
            continue

        # Try with prefix detection
        prefixes = set()
        for spid in spec_process_ids:
            for ppid in pattern_pids:
                if spid.endswith(f"_{ppid}") or spid.endswith(ppid):
                    prefix = spid[: len(spid) - len(ppid)].rstrip("_")
                    if prefix:
                        prefixes.add(prefix)

        for prefix in prefixes:
            matched = set()
            for ppid in pattern_pids:
                candidate = f"{prefix}_{ppid}"
                if candidate in spec_process_ids:
                    matched.add(candidate)
            if len(matched) >= len(pattern_pids) * 0.6:
                detected.append((pname, matched, prefix))
                break

    return detected


# ════════════════════════════════════════════════════════════════════
# OWL / DL Structural Pattern Detection
# ════════════════════════════════════════════════════════════════════

# Mapping from OWL DL pattern class names → patterns.py canonical names
_OWL_TO_PATTERN = {
    "ReasoningLoop": "reasoning_loop",
    "CritiqueCycle": "critique_cycle",
    "MultiAgentDebate": "debate",
    "RetrievalAugmented": "retrieval",
    "FanOut": "fan_out_aggregate",
    "MemoryBacked": "reflection",
    # OWL-only patterns (no patterns.py library equivalent)
    "HumanInLoop": "human_in_loop",
    "PubSub": "pub_sub",
    "Handoff": "handoff",
}


def detect_patterns_structural(spec, spec_path=None):
    """Detect patterns using OWL structural classification (graph topology).

    Uses classify_structural() from ontology_owl.py which analyzes the actual
    graph structure (loops, invocations, reads/writes, fan-out) rather than
    matching process ID strings.

    Returns a list of (pattern_name, method) tuples where method is 'owl-structural'.
    Falls back to ID-based detect_patterns() if OWL modules aren't available.
    """
    try:
        from .owl_bridge import build_bridge_ontology_in_world, spec_to_owl
        from .ontology_owl import classify_structural
        from owlready2 import World

        world = World()
        onto = build_bridge_ontology_in_world(world)
        spec_inst = spec_to_owl(onto, spec if spec_path is None else spec_path)
        results = classify_structural(onto, [spec_inst])

        detected = []
        for spec_name, owl_patterns in results.items():
            for owl_pat in owl_patterns:
                canon = _OWL_TO_PATTERN.get(owl_pat, owl_pat.lower())
                detected.append((canon, "owl-structural"))
        return detected

    except Exception:
        # Fall back to ID-based detection
        id_results = detect_patterns(spec)
        return [(name, "id-match") for name, _, _ in id_results]


def detect_patterns_combined(spec, spec_path=None):
    """Detect patterns using both OWL structural and ID-based methods.

    Returns a list of dicts with:
      - name: canonical pattern name
      - method: 'owl-structural', 'id-match', or 'both'
      - matched_ids: set of matched process IDs (for id-match)
      - prefix: namespace prefix (for id-match)
    """
    # Run ID-based detection
    id_results = detect_patterns(spec)
    id_map = {name: (pids, prefix) for name, pids, prefix in id_results}

    # Run OWL structural detection
    owl_results = set()
    try:
        from .owl_bridge import build_bridge_ontology_in_world, spec_to_owl
        from .ontology_owl import classify_structural
        from owlready2 import World

        world = World()
        onto = build_bridge_ontology_in_world(world)
        spec_inst = spec_to_owl(onto, spec if spec_path is None else spec_path)
        results = classify_structural(onto, [spec_inst])
        for spec_name, owl_patterns in results.items():
            for owl_pat in owl_patterns:
                canon = _OWL_TO_PATTERN.get(owl_pat, owl_pat.lower())
                owl_results.add(canon)
    except Exception:
        pass

    # Merge results
    all_names = set(id_map.keys()) | owl_results
    combined = []
    for name in sorted(all_names):
        in_id = name in id_map
        in_owl = name in owl_results
        method = "both" if (in_id and in_owl) else ("owl-structural" if in_owl else "id-match")
        pids, prefix = id_map.get(name, (set(), ""))
        combined.append({
            "name": name,
            "method": method,
            "matched_ids": pids,
            "prefix": prefix,
        })
    return combined


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pattern Library — reusable architectural building blocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 patterns.py                          # List all patterns\n"
            "  python3 patterns.py --info reasoning_loop    # Show pattern details\n"
            "  python3 patterns.py --compat reasoning_loop critique_cycle\n"
        ),
    )
    parser.add_argument("--info", "-i", metavar="PATTERN", help="Show details for a pattern")
    parser.add_argument("--compat", nargs=2, metavar=("A", "B"),
                        help="Check if pattern A's outputs feed pattern B's inputs")
    parser.add_argument("--detect", metavar="SPEC",
                        help="Detect patterns in a spec file (ID-based)")
    parser.add_argument("--detect-structural", metavar="SPEC",
                        help="Detect patterns using OWL structural classification")
    parser.add_argument("--detect-all", metavar="SPEC",
                        help="Detect patterns using both ID-based and OWL structural methods")

    args = parser.parse_args()

    if args.info:
        print(pattern_info(args.info))
    elif args.compat:
        a, b = args.compat
        ok = compatible_patterns(a, b)
        print(f"Compatible ({a} -> {b}): {ok}")
        if ok:
            pa = PATTERN_LIBRARY.get(a, {})
            pb = PATTERN_LIBRARY.get(b, {})
            overlap = set(pa.get("interface", {}).get("outputs", [])) & \
                      set(pb.get("interface", {}).get("inputs", []))
            print(f"  Shared fields: {sorted(overlap)}")
    elif args.detect:
        spec = yaml.safe_load(open(args.detect))
        results = detect_patterns(spec)
        if results:
            for pname, pids, prefix in results:
                prefix_str = f" (prefix: {prefix})" if prefix else ""
                print(f"  {pname}{prefix_str}: {len(pids)} processes matched")
        else:
            print("  No patterns detected")
    elif args.detect_structural:
        spec = yaml.safe_load(open(args.detect_structural))
        results = detect_patterns_structural(spec, args.detect_structural)
        if results:
            for pname, method in results:
                print(f"  {pname} [{method}]")
        else:
            print("  No patterns detected")
    elif args.detect_all:
        spec = yaml.safe_load(open(args.detect_all))
        results = detect_patterns_combined(spec, args.detect_all)
        if results:
            for r in results:
                ids_str = f" ({len(r['matched_ids'])} IDs)" if r['matched_ids'] else ""
                prefix_str = f" prefix={r['prefix']}" if r['prefix'] else ""
                print(f"  {r['name']:25s} [{r['method']}]{ids_str}{prefix_str}")
        else:
            print("  No patterns detected")
    else:
        # Default: list all patterns
        print(f"Pattern Library ({len(PATTERN_LIBRARY)} patterns):\n")
        for name in list_patterns():
            p = PATTERN_LIBRARY[name]
            proc_count = len(p["processes"])
            ent_count = len(p["entities"])
            print(f"  {name:25s}  {proc_count} procs, {ent_count} entities  "
                  f"[{p['source_spec']}]")
            print(f"  {'':25s}  {p['description'][:80]}")
            print()


if __name__ == "__main__":
    main()
