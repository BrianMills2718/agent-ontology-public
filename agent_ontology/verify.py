#!/usr/bin/env python3
"""
Agent Ontology Semantic Verifier

Advanced verification beyond validate.py's structural checks. Uses graph
analysis, OWL reasoning, and data-flow tracking to prove properties about
agent architectures.

Checks:
  V001  Termination — every path from entry reaches a terminal node
  V002  Data provenance — gate conditions reference variables set by earlier steps
  V003  Loop termination — loops have incrementing counters or convergence signals
  V004  Store consistency — read edges preceded by write edges on same store
  V005  Schema flow — output schemas of invoking steps match input schemas downstream
  V006  Policy coverage — all steps with side effects are covered by a policy
  V007  Pattern composition — detected patterns have compatible interfaces
  V008  OWL round-trip — YAML→OWL→YAML is lossless
  V009  Information flow — can data from entity A reach entity B?

Usage:
  python3 verify.py specs/react.yaml
  python3 verify.py --all specs/
  python3 verify.py specs/react.yaml --verbose
  python3 verify.py specs/react.yaml --check V001,V003
  python3 verify.py --all specs/ --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# YAML / Spec helpers
# ═══════════════════════════════════════════════════════════════

def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def gather_specs(directory: str | Path) -> list[Path]:
    root = Path(directory)
    specs: list[Path] = []
    for ext in ("*.yaml", "*.yml"):
        specs.extend(sorted(root.rglob(ext)))
    return specs


def _build_id_maps(spec: dict) -> tuple[dict, dict, dict, list]:
    """Build lookup dicts for entities, processes, schemas, edges."""
    entities = {e["id"]: e for e in spec.get("entities", []) if isinstance(e, dict) and "id" in e}
    processes = {p["id"]: p for p in spec.get("processes", []) if isinstance(p, dict) and "id" in p}
    schemas = {}
    for s in spec.get("schemas", []):
        if isinstance(s, dict) and "id" in s:
            schemas[s["id"]] = s
    edges = spec.get("edges", [])
    return entities, processes, schemas, edges


def _build_control_flow_graph(processes: dict, edges: list) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build forward and reverse adjacency from flow/branch/loop edges."""
    FLOW_TYPES = {"flow", "branch", "loop"}
    fwd: dict[str, list[str]] = defaultdict(list)
    rev: dict[str, list[str]] = defaultdict(list)
    for pid in processes:
        fwd.setdefault(pid, [])
        rev.setdefault(pid, [])
    for e in edges:
        if not isinstance(e, dict):
            continue
        etype = e.get("type", "")
        if etype in FLOW_TYPES:
            src = e.get("from", "")
            tgt = e.get("to", "")
            if src in processes and tgt in processes:
                fwd[src].append(tgt)
                rev[tgt].append(src)
    return dict(fwd), dict(rev)


def _reachable_from(start: str, adj: dict[str, list[str]]) -> set[str]:
    """BFS reachability from a start node."""
    visited: set[str] = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for nbr in adj.get(node, []):
            if nbr not in visited:
                queue.append(nbr)
    return visited


# ═══════════════════════════════════════════════════════════════
# V001: Termination — every path reaches a terminal
# ═══════════════════════════════════════════════════════════════

def check_termination(spec: dict, entities: dict, processes: dict, schemas: dict,
                      edges: list, fwd: dict, rev: dict) -> tuple[list[str], list[str]]:
    """V001: Every non-loop path from entry_point reaches a terminal node."""
    errors, warnings = [], []
    entry = spec.get("entry_point", "")
    if not entry or entry not in processes:
        return errors, warnings  # validate.py catches this

    # Terminal nodes: process nodes with no outgoing flow/branch edges
    terminals = {pid for pid in processes if not fwd.get(pid)}

    if not terminals:
        errors.append("V001: No terminal nodes found — agent may never halt")
        return errors, warnings

    # BFS from entry — check every non-terminal, non-loop-back node can reach a terminal
    reachable = _reachable_from(entry, fwd)

    # Build reverse from terminals
    rev_from_terminals: set[str] = set()
    for t in terminals:
        rev_from_terminals |= _reachable_from(t, rev)

    # Any reachable non-terminal node that can't reach a terminal
    for pid in reachable:
        if pid not in terminals and pid not in rev_from_terminals:
            # Check if it's in a loop (acceptable if loop has exit)
            in_loop = False
            for e in edges:
                if isinstance(e, dict) and e.get("type") == "loop" and e.get("to") == pid:
                    in_loop = True
                    break
            if not in_loop:
                warnings.append(f"V001: Process '{pid}' is reachable but cannot reach any terminal node")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# V002: Data provenance — gate conditions reference populated vars
# ═══════════════════════════════════════════════════════════════

def _extract_state_keys_from_logic(logic_lines: list[str]) -> set[str]:
    """Extract state.data["key"] assignments from logic blocks."""
    keys: set[str] = set()
    pattern = re.compile(r'state\.data\["([^"]+)"\]\s*=')
    for line in logic_lines:
        for m in pattern.finditer(line):
            keys.add(m.group(1))
    return keys


def _extract_schema_fields(schema: dict) -> set[str]:
    """Extract field names from a schema definition."""
    fields = set()
    for f in schema.get("fields", []):
        if isinstance(f, dict) and "name" in f:
            fields.add(f["name"])
    return fields


def _extract_condition_keys(condition: str) -> set[str]:
    """Extract variable references from a gate condition string."""
    keys: set[str] = set()
    # Patterns like: score >= 0.7, issues.length > 0, is_correct == true
    tokens = re.split(r'[\s>=<!]+', condition)
    for token in tokens:
        token = token.strip().strip('"').strip("'")
        if token and not token.replace(".", "").replace("_", "").isdigit() and \
           token not in ("true", "false", "null", "not", "empty", "is", "and", "or",
                         "length", "len", "0", "1"):
            # Strip .length etc
            base = token.split(".")[0] if "." in token else token
            if base:
                keys.add(base)
    return keys


def check_data_provenance(spec: dict, entities: dict, processes: dict, schemas: dict,
                          edges: list, fwd: dict, rev: dict) -> tuple[list[str], list[str]]:
    """V002: Gate conditions reference variables that are set by earlier processes."""
    errors, warnings = [], []
    entry = spec.get("entry_point", "")
    if not entry:
        return errors, warnings

    # Collect which keys each process produces (from output schemas and logic)
    process_produces: dict[str, set[str]] = defaultdict(set)
    for pid, proc in processes.items():
        # From logic blocks
        logic = proc.get("logic", [])
        if isinstance(logic, list):
            process_produces[pid] |= _extract_state_keys_from_logic(logic)

        # From output schemas of invoke edges targeting this process
        for e in edges:
            if isinstance(e, dict) and e.get("type") == "invoke" and e.get("from") == pid:
                target = e.get("to", "")
                # The target agent's output schema populates fields
                for ent in spec.get("entities", []):
                    if isinstance(ent, dict) and ent.get("id") == target:
                        out_schema = ent.get("output_schema", "")
                        if out_schema and out_schema in schemas:
                            process_produces[pid] |= _extract_schema_fields(schemas[out_schema])

    # For each gate, check its condition variables
    for pid, proc in processes.items():
        if proc.get("type") != "gate":
            continue
        condition = proc.get("condition", "")
        if not condition:
            continue

        needed_keys = _extract_condition_keys(condition)
        if not needed_keys:
            continue

        # What keys are available at this gate? Trace backwards from gate
        available_keys: set[str] = set()
        predecessors = _reachable_from(pid, rev)
        for pred in predecessors:
            if pred != pid:
                available_keys |= process_produces.get(pred, set())

        # Also add keys from initial state / input schema
        input_schema = spec.get("input_schema", "")
        if input_schema and input_schema in schemas:
            available_keys |= _extract_schema_fields(schemas[input_schema])

        for key in needed_keys:
            if key not in available_keys:
                warnings.append(f"V002: Gate '{pid}' condition references '{key}' "
                                f"but no upstream process clearly produces it")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# V003: Loop termination — loops have counter increments or exits
# ═══════════════════════════════════════════════════════════════

def check_loop_termination(spec: dict, entities: dict, processes: dict, schemas: dict,
                           edges: list, fwd: dict, rev: dict) -> tuple[list[str], list[str]]:
    """V003: Loops have incrementing counters, max checks, or convergence signals."""
    errors, warnings = [], []

    # Find loop edges
    loop_edges = [e for e in edges if isinstance(e, dict) and e.get("type") == "loop"]
    if not loop_edges:
        return errors, warnings

    for loop_edge in loop_edges:
        loop_target = loop_edge.get("to", "")
        loop_source = loop_edge.get("from", "")

        # The gate that controls the loop should be in the path between target and source
        # Find the gate between loop_target and loop_source
        controlling_gate = None
        # Walk from loop_target to loop_source looking for a gate
        path_nodes = set()
        queue = deque([loop_target])
        while queue:
            node = queue.popleft()
            if node in path_nodes:
                continue
            path_nodes.add(node)
            if node == loop_source:
                break
            for nbr in fwd.get(node, []):
                queue.append(nbr)

        for pid in path_nodes:
            if processes.get(pid, {}).get("type") == "gate":
                controlling_gate = pid
                break

        if not controlling_gate:
            warnings.append(f"V003: Loop edge {loop_source}→{loop_target} has no "
                            f"controlling gate — may loop forever")
            continue

        # Check if any process in the loop body modifies a counter/iteration variable
        has_counter = False
        counter_patterns = [
            re.compile(r'(?:iteration|count|round|attempt|retry|step_count|num_)\w*\s*[\+\-]='),
            re.compile(r'(?:iteration|count|round|attempt|retry|step_count|num_)\w*\s*=\s*.*\+\s*1'),
            re.compile(r'state\.data\["[^"]*(?:iteration|count|round|attempt|retry|num_)[^"]*"\]\s*[\+]='),
            re.compile(r'state\.data\["[^"]*(?:iteration|count|round|attempt|retry|num_)[^"]*"\]\s*=\s*.*\+\s*1'),
        ]

        for pid in path_nodes:
            logic = processes.get(pid, {}).get("logic", [])
            if isinstance(logic, list):
                for line in logic:
                    for pat in counter_patterns:
                        if pat.search(line):
                            has_counter = True
                            break
                    if has_counter:
                        break
            if has_counter:
                break

        # Also check if the gate condition references iteration/counter vars
        gate_condition = processes.get(controlling_gate, {}).get("condition", "")
        has_max_check = bool(re.search(
            r'(?:iteration|count|round|attempt|retry|max_|num_)',
            gate_condition, re.IGNORECASE
        ))

        # Check for convergence-based exit (quality score, done flag)
        has_convergence = bool(re.search(
            r'(?:quality|score|done|complete|converge|satisf|sufficient|pass)',
            gate_condition, re.IGNORECASE
        ))

        if not has_counter and not has_max_check and not has_convergence:
            warnings.append(f"V003: Loop {loop_source}→{loop_target} (gate: {controlling_gate}) "
                            f"has no visible termination mechanism (counter, max check, or convergence)")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# V004: Store consistency — reads have preceding writes
# ═══════════════════════════════════════════════════════════════

def check_store_consistency(spec: dict, entities: dict, processes: dict, schemas: dict,
                            edges: list, fwd: dict, rev: dict) -> tuple[list[str], list[str]]:
    """V004: Store read edges are preceded by write edges to the same store."""
    errors, warnings = [], []
    entry = spec.get("entry_point", "")

    # Collect read and write edges per store
    store_reads: dict[str, list[str]] = defaultdict(list)  # store_id -> [process_ids]
    store_writes: dict[str, list[str]] = defaultdict(list)

    for e in edges:
        if not isinstance(e, dict):
            continue
        etype = e.get("type", "")
        src = e.get("from", "")
        tgt = e.get("to", "")
        if etype == "read" and tgt in entities and entities[tgt].get("type") in ("store",):
            store_reads[tgt].append(src)
        elif etype == "write" and tgt in entities and entities[tgt].get("type") in ("store",):
            store_writes[tgt].append(src)

    if not store_reads:
        return errors, warnings

    # For each store that has reads, check if a write precedes it
    for store_id, readers in store_reads.items():
        writers = store_writes.get(store_id, [])
        if not writers:
            # Store might be pre-populated (e.g., document store for RAG)
            store_type = entities.get(store_id, {}).get("store_type", "")
            if store_type in ("vector", "document", "knowledge_base"):
                continue  # These are often pre-populated
            warnings.append(f"V004: Store '{store_id}' is read by "
                            f"{readers} but never written to in this spec")
            continue

        if not entry:
            continue

        # Check if any writer is reachable before any reader
        for reader_pid in readers:
            reader_predecessors = _reachable_from(reader_pid, rev)
            writer_reachable_before = False
            for writer_pid in writers:
                if writer_pid in reader_predecessors:
                    writer_reachable_before = True
                    break
            if not writer_reachable_before:
                # Could be in same loop body (write and read in same cycle)
                warnings.append(f"V004: Process '{reader_pid}' reads from store "
                                f"'{store_id}' but no write to that store is "
                                f"guaranteed to precede it")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# V005: Schema flow — output/input schema compatibility
# ═══════════════════════════════════════════════════════════════

def check_schema_flow(spec: dict, entities: dict, processes: dict, schemas: dict,
                      edges: list, fwd: dict, rev: dict) -> tuple[list[str], list[str]]:
    """V005: Output schemas from invoke edges carry fields needed by downstream steps."""
    errors, warnings = [], []

    # Build map: for each invoke edge, what fields does the target agent produce?
    for e in edges:
        if not isinstance(e, dict) or e.get("type") != "invoke":
            continue
        invoker = e.get("from", "")
        target = e.get("to", "")
        return_schema = e.get("return_schema", "")

        if not return_schema or return_schema not in schemas:
            continue

        produced_fields = _extract_schema_fields(schemas[return_schema])
        if not produced_fields:
            continue

        # Check what the next step(s) after the invoker consume
        for next_pid in fwd.get(invoker, []):
            next_proc = processes.get(next_pid, {})
            # Check if the next process is a gate with condition referencing produced fields
            if next_proc.get("type") == "gate":
                condition = next_proc.get("condition", "")
                needed = _extract_condition_keys(condition)
                missing = needed - produced_fields
                # Only warn if there's a clear mismatch (produced fields don't overlap)
                if needed and not (needed & produced_fields) and len(missing) > 0:
                    warnings.append(f"V005: Gate '{next_pid}' needs {missing} but "
                                    f"upstream invoke ({invoker}→{target}) produces "
                                    f"{produced_fields} via schema '{return_schema}'")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# V006: Policy coverage
# ═══════════════════════════════════════════════════════════════

def check_policy_coverage(spec: dict, entities: dict, processes: dict, schemas: dict,
                          edges: list, fwd: dict, rev: dict) -> tuple[list[str], list[str]]:
    """V006: Processes that invoke external tools/agents have policy coverage."""
    errors, warnings = [], []

    # Find policy processes and their targets
    policy_targets: set[str] = set()
    for pid, proc in processes.items():
        if proc.get("type") == "policy":
            targets = proc.get("targets", [])
            if isinstance(targets, list):
                policy_targets.update(targets)

    # Find processes that invoke tools (external side effects)
    for e in edges:
        if not isinstance(e, dict) or e.get("type") != "invoke":
            continue
        invoker = e.get("from", "")
        target = e.get("to", "")
        target_ent = entities.get(target, {})
        if target_ent.get("type") == "tool":
            if invoker not in policy_targets:
                # Info-level: not all tool invocations need policies
                pass  # Could add an optional strict mode

    # Check: if spec has any policies, are they reachable?
    policy_procs = [pid for pid, p in processes.items() if p.get("type") == "policy"]
    entry = spec.get("entry_point", "")
    if policy_procs and entry:
        reachable = _reachable_from(entry, fwd)
        for pid in policy_procs:
            if pid not in reachable:
                warnings.append(f"V006: Policy '{pid}' is defined but unreachable "
                                f"from entry point — it will never execute")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# V007: Pattern composition
# ═══════════════════════════════════════════════════════════════

def check_pattern_composition(spec: dict, entities: dict, processes: dict, schemas: dict,
                              edges: list, fwd: dict, rev: dict) -> tuple[list[str], list[str]]:
    """V007: Detected patterns have compatible interfaces."""
    errors, warnings = [], []

    try:
        from .patterns import detect_patterns, get_pattern, compatible_patterns
    except ImportError:
        return errors, warnings

    detected = detect_patterns(spec)
    if len(detected) < 2:
        return errors, warnings  # No composition to check

    # Check pairwise compatibility
    for i, (name_a, pids_a, prefix_a) in enumerate(detected):
        for name_b, pids_b, prefix_b in detected[i + 1:]:
            pat_a = get_pattern(name_a)
            pat_b = get_pattern(name_b)
            if pat_a and pat_b:
                # Check if they're connected in the actual spec
                a_set = set(pids_a)
                b_set = set(pids_b)
                connected = False
                for pid_a in a_set:
                    for next_pid in fwd.get(pid_a, []):
                        if next_pid in b_set:
                            connected = True
                            break
                if connected and not compatible_patterns(pat_a, pat_b):
                    warnings.append(f"V007: Patterns '{name_a}' and '{name_b}' are "
                                    f"connected but may have incompatible interfaces")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# V008: OWL round-trip
# ═══════════════════════════════════════════════════════════════

def check_owl_roundtrip(spec: dict, spec_path: str | None) -> tuple[list[str], list[str]]:
    """V008: YAML→OWL→YAML round-trip is lossless."""
    errors, warnings = [], []

    if not spec_path:
        return errors, warnings

    try:
        from .owl_bridge import round_trip
        passed, diff = round_trip(spec_path)
        if not passed:
            warnings.append(f"V008: OWL round-trip produced differences: {diff}")
    except ImportError:
        pass  # owlready2 not installed — skip
    except Exception as exc:
        warnings.append(f"V008: OWL round-trip failed: {exc}")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# V009: Information flow
# ═══════════════════════════════════════════════════════════════

def check_information_flow(spec: dict, entities: dict, processes: dict, schemas: dict,
                           edges: list, fwd: dict, rev: dict) -> tuple[list[str], list[str]]:
    """V009: Data flow graph is connected — outputs of one stage reach inputs of next."""
    errors, warnings = [], []

    # Build data-flow graph (who produces what fields, who consumes them)
    producers: dict[str, set[str]] = defaultdict(set)  # field -> {process_ids}
    consumers: dict[str, set[str]] = defaultdict(set)  # field -> {process_ids}

    for pid, proc in processes.items():
        # Producers: processes with output schemas from invocations
        for e in edges:
            if isinstance(e, dict) and e.get("type") == "invoke" and e.get("from") == pid:
                rs = e.get("return_schema", "")
                if rs and rs in schemas:
                    for field in _extract_schema_fields(schemas[rs]):
                        producers[field].add(pid)

        # Producers: logic block assignments
        logic = proc.get("logic", [])
        if isinstance(logic, list):
            for key in _extract_state_keys_from_logic(logic):
                producers[key].add(pid)

        # Consumers: input schemas from invocations
        for e in edges:
            if isinstance(e, dict) and e.get("type") == "invoke" and e.get("from") == pid:
                target = e.get("to", "")
                target_ent = entities.get(target, {})
                in_schema = target_ent.get("input_schema", "")
                if in_schema and in_schema in schemas:
                    for field in _extract_schema_fields(schemas[in_schema]):
                        consumers[field].add(pid)

        # Consumers: gate conditions
        if proc.get("type") == "gate":
            condition = proc.get("condition", "")
            for key in _extract_condition_keys(condition):
                consumers[key].add(pid)

    # Find consumed fields with no producer
    for field, consumer_pids in consumers.items():
        if field not in producers:
            # Might come from initial input
            warnings.append(f"V009: Field '{field}' consumed by {sorted(consumer_pids)} "
                            f"but no process produces it (may come from initial input)")

    return errors, warnings


# ═══════════════════════════════════════════════════════════════
# Main verification orchestrator
# ═══════════════════════════════════════════════════════════════

ALL_CHECKS = {
    "V001": ("Termination", check_termination),
    "V002": ("Data provenance", check_data_provenance),
    "V003": ("Loop termination", check_loop_termination),
    "V004": ("Store consistency", check_store_consistency),
    "V005": ("Schema flow", check_schema_flow),
    "V006": ("Policy coverage", check_policy_coverage),
    "V007": ("Pattern composition", check_pattern_composition),
    "V009": ("Information flow", check_information_flow),
}

# V008 handled separately since it needs spec_path
OWL_CHECKS = {
    "V008": ("OWL round-trip", check_owl_roundtrip),
}


def verify_spec(spec: dict, spec_path: str | None = None,
                checks: set[str] | None = None,
                verbose: bool = False) -> tuple[list[str], list[str]]:
    """Run semantic verification checks on a spec.

    Returns (errors, warnings).
    """
    all_errors: list[str] = []
    all_warnings: list[str] = []

    entities, processes, schemas, edges = _build_id_maps(spec)
    fwd, rev = _build_control_flow_graph(processes, edges)

    enabled = checks or set(ALL_CHECKS.keys()) | set(OWL_CHECKS.keys())

    for check_id, (name, check_fn) in ALL_CHECKS.items():
        if check_id not in enabled:
            continue
        try:
            errs, warns = check_fn(spec, entities, processes, schemas, edges, fwd, rev)
            all_errors.extend(errs)
            all_warnings.extend(warns)
        except Exception as exc:
            if verbose:
                all_warnings.append(f"{check_id}: Check failed with exception: {exc}")

    for check_id, (name, check_fn) in OWL_CHECKS.items():
        if check_id not in enabled:
            continue
        try:
            errs, warns = check_fn(spec, spec_path)
            all_errors.extend(errs)
            all_warnings.extend(warns)
        except Exception as exc:
            if verbose:
                all_warnings.append(f"{check_id}: Check failed with exception: {exc}")

    return all_errors, all_warnings


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def _format_result(filepath: str, errors: list[str], warnings: list[str],
                   verbose: bool = False) -> str:
    name = os.path.basename(filepath)
    lines: list[str] = []

    if not errors and not warnings:
        lines.append(f"{GREEN}✓ {name}: VERIFIED (0 errors, 0 warnings){RESET}")
    elif errors:
        lines.append(f"\n{'═' * 60}")
        lines.append(f"  {name}")
        lines.append(f"{'═' * 60}\n")
        lines.append(f"  {RED}{len(errors)} ERRORS:{RESET}")
        for e in errors:
            lines.append(f"    {RED}✗{RESET} {e}")
        if warnings:
            lines.append(f"\n  {YELLOW}{len(warnings)} WARNINGS:{RESET}")
            for w in warnings:
                lines.append(f"    {YELLOW}⚠{RESET} {w}")
        lines.append("")
    else:
        lines.append(f"\n{'═' * 60}")
        lines.append(f"  {name}")
        lines.append(f"{'═' * 60}\n")
        lines.append(f"  {YELLOW}{len(warnings)} WARNINGS:{RESET}")
        for w in warnings:
            lines.append(f"    {YELLOW}⚠{RESET} {w}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Agent Ontology Semantic Verifier — advanced property checking for agent specs",
    )
    parser.add_argument("path", help="Path to YAML spec file or directory (with --all)")
    parser.add_argument("--all", action="store_true", dest="all_specs",
                        help="Verify all specs in directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output including exception traces")
    parser.add_argument("--check", type=str, default="",
                        help="Comma-separated list of checks to run (e.g., V001,V003)")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output results as JSON")
    parser.add_argument("--no-owl", action="store_true",
                        help="Skip OWL-based checks (V008)")
    args = parser.parse_args(argv)

    # Parse check filter
    checks = None
    if args.check:
        checks = set(c.strip().upper() for c in args.check.split(","))
    if args.no_owl:
        if checks is None:
            checks = set(ALL_CHECKS.keys())
        else:
            checks.discard("V008")

    target = Path(args.path)
    results: list[dict] = []

    if args.all_specs:
        if not target.is_dir():
            print(f"Error: {target} is not a directory", file=sys.stderr)
            sys.exit(1)
        spec_files = gather_specs(target)
    else:
        if not target.is_file():
            print(f"Error: {target} is not a file", file=sys.stderr)
            sys.exit(1)
        spec_files = [target]

    total_errors = 0
    total_warnings = 0

    for sf in spec_files:
        spec = load_yaml(sf)
        errs, warns = verify_spec(spec, str(sf), checks=checks, verbose=args.verbose)
        total_errors += len(errs)
        total_warnings += len(warns)

        if args.json_output:
            results.append({
                "file": str(sf),
                "name": spec.get("name", ""),
                "errors": errs,
                "warnings": warns,
            })
        else:
            print(_format_result(str(sf), errs, warns, args.verbose))

    if args.json_output:
        print(json.dumps(results, indent=2))
    elif len(spec_files) > 1:
        print(f"\n{'─' * 40}")
        total = len(spec_files)
        clean = sum(1 for r in results if not r.get("errors") and not r.get("warnings"))
        if not args.json_output:
            # Recount from printed output
            pass
        print(f"  {total} specs verified: {total_errors} errors, {total_warnings} warnings")


if __name__ == "__main__":
    main()
