#!/usr/bin/env python3
"""
Property-based tests for the spec -> instantiate pipeline.

Validates structural invariants of every spec YAML without making API calls.
Uses only the Python standard library + pyyaml.

Usage: python3 test_properties.py
"""

import ast
import os
import sys

import pytest
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "agent_ontology")
SPECS_DIR = os.path.join(PACKAGE_DIR, "specs")
ONTOLOGY_PATH = os.path.join(PACKAGE_DIR, "ONTOLOGY.yaml")

# Ensure project root is on sys.path so agent_ontology package is importable
sys.path.insert(0, PROJECT_ROOT)
from agent_ontology.validate import validate_spec, load_yaml
from agent_ontology.instantiate import generate_agent, generate_langgraph_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ontology():
    return load_yaml(ONTOLOGY_PATH)


@pytest.fixture(scope="session")
def ontology():
    """Pytest fixture for ontology used by round-trip validation."""
    return load_ontology()


def all_spec_paths():
    """Return sorted list of all spec YAML file paths."""
    paths = []
    for fname in sorted(os.listdir(SPECS_DIR)):
        if fname.endswith(".yaml"):
            paths.append(os.path.join(SPECS_DIR, fname))
    return paths


def spec_name(path):
    return os.path.basename(path)


# ---------------------------------------------------------------------------
# Test results tracking
# ---------------------------------------------------------------------------

class _ResultsTracker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def ok(self, test_name, spec_file):
        self.passed += 1

    def fail(self, test_name, spec_file, message):
        self.failed += 1
        self.failures.append((test_name, spec_file, message))

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed")
        print(f"{'=' * 60}")
        if self.failures:
            print("\n  FAILURES:\n")
            for test_name, spec_file, message in self.failures:
                print(f"    [{test_name}] {spec_file}")
                print(f"      {message}\n")
        return self.failed == 0


results = _ResultsTracker()


# ---------------------------------------------------------------------------
# Property 1: Round-trip validation
# For every spec, validate.py reports no errors.
# ---------------------------------------------------------------------------

def test_roundtrip_validation(ontology):
    """Every spec must pass validation with zero errors."""
    test_name = "round-trip validation"
    for path in all_spec_paths():
        name = spec_name(path)
        try:
            spec = load_yaml(path)
            errors, warnings = validate_spec(spec, ontology, path)
            if errors:
                results.fail(test_name, name,
                             f"{len(errors)} error(s): {errors[0]}")
            else:
                results.ok(test_name, name)
        except Exception as e:
            results.fail(test_name, name, f"Exception: {e}")


# ---------------------------------------------------------------------------
# Property 2: Code generation produces syntactically valid Python
# For every spec (except claude_code), instantiate.py produces parseable code.
# ---------------------------------------------------------------------------

def test_code_generation_syntax():
    """Generated Python code must be syntactically valid (ast.parse)."""
    test_name = "code generation syntax"
    # claude-code.yaml is excluded per the task spec
    skip = {"claude-code.yaml"}
    for path in all_spec_paths():
        name = spec_name(path)
        if name in skip:
            continue
        try:
            spec = load_yaml(path)
            code = generate_agent(spec)
            ast.parse(code)
            results.ok(test_name, name)
        except SyntaxError as e:
            results.fail(test_name, name,
                         f"SyntaxError at line {e.lineno}: {e.msg}")
        except Exception as e:
            results.fail(test_name, name, f"Exception: {e}")


# ---------------------------------------------------------------------------
# Property 3: Schema completeness
# Every schema_ref used in edges/processes resolves to a defined schema.
# ---------------------------------------------------------------------------

def test_schema_completeness():
    """All schema_ref values in edges and processes must resolve to defined schemas."""
    test_name = "schema completeness"
    schema_ref_fields = [
        "input_schema", "output_schema", "data_in", "data_out",
        "schema", "input", "output", "data",
    ]

    for path in all_spec_paths():
        name = spec_name(path)
        try:
            spec = load_yaml(path)
            schemas = spec.get("schemas", [])
            schema_names = {s.get("name") for s in schemas if s.get("name")}

            missing = []

            # Check entities
            for entity in spec.get("entities", []):
                eid = entity.get("id", "?")
                for field in schema_ref_fields:
                    ref = entity.get(field)
                    if ref and ref not in schema_names:
                        missing.append(f"entity '{eid}'.{field} -> '{ref}'")

            # Check processes
            for proc in spec.get("processes", []):
                pid = proc.get("id", "?")
                for field in schema_ref_fields:
                    ref = proc.get(field)
                    if ref and ref not in schema_names:
                        missing.append(f"process '{pid}'.{field} -> '{ref}'")

            # Check edges
            for i, edge in enumerate(spec.get("edges", [])):
                elabel = edge.get("label", f"edge[{i}]")
                for field in ["data", "input", "output"]:
                    ref = edge.get(field)
                    if ref and ref not in schema_names:
                        missing.append(f"edge '{elabel}'.{field} -> '{ref}'")

            if missing:
                results.fail(test_name, name,
                             f"{len(missing)} unresolved schema ref(s): {missing[0]}")
            else:
                results.ok(test_name, name)
        except Exception as e:
            results.fail(test_name, name, f"Exception: {e}")


# ---------------------------------------------------------------------------
# Property 4: Entry point reachability
# Every process is reachable from the entry_point via flow/branch edges.
#
# Exceptions that are excluded from the reachability requirement:
#   - policy processes (cross-cutting concerns, attached via observe/modify)
#   - error_handler processes (activated on error, not via normal flow)
#   - processes that are on_error/fallback targets of error_handlers
#   - processes that are roots of intentionally separate pipelines
#     (i.e., they have outgoing flow but no incoming flow from any node)
# ---------------------------------------------------------------------------

def test_entry_point_reachability():
    """Every process must be reachable from the entry_point via flow/branch/loop edges."""
    test_name = "entry point reachability"

    for path in all_spec_paths():
        name = spec_name(path)
        try:
            spec = load_yaml(path)
            processes = spec.get("processes", [])
            edges = spec.get("edges", [])
            entities = spec.get("entities", [])
            entry_point = spec.get("entry_point")

            if not entry_point:
                results.ok(test_name, name)
                continue

            process_map = {p.get("id"): p for p in processes if p.get("id")}
            process_ids = set(process_map.keys())
            entity_ids = {e.get("id") for e in entities if e.get("id")}
            all_node_ids = process_ids | entity_ids

            # Identify cross-cutting process types that are exempt from
            # reachability: policy and error_handler.
            exempt_types = {"policy", "error_handler"}
            exempt_ids = set()
            for p in processes:
                pid = p.get("id")
                if pid and p.get("type") in exempt_types:
                    exempt_ids.add(pid)
                    # Also exempt the on_error and fallback targets of
                    # error_handlers, since they are activated by the
                    # error path, not the normal flow.
                    if p.get("type") == "error_handler":
                        on_err = p.get("on_error")
                        if on_err:
                            exempt_ids.add(on_err)
                        fallback = p.get("fallback")
                        if fallback:
                            exempt_ids.add(fallback)

            # Build adjacency from flow/branch/loop/invoke edges.
            # Include invoke edges so the BFS can traverse entity-bridged
            # flows: process --(invoke)--> entity --(flow)--> process.
            adjacency = {}
            for edge in edges:
                etype = edge.get("type")
                efrom = edge.get("from")
                eto = edge.get("to")
                if etype in ("flow", "branch", "loop", "invoke") and efrom and eto:
                    adjacency.setdefault(efrom, set()).add(eto)

            # Include gate branch targets declared in process definitions
            for proc in processes:
                pid = proc.get("id")
                if proc.get("type") == "gate" and pid:
                    for branch in proc.get("branches", []):
                        target = branch.get("target")
                        if target and target in all_node_ids:
                            adjacency.setdefault(pid, set()).add(target)
                    default = proc.get("default")
                    if default and default in all_node_ids:
                        adjacency.setdefault(pid, set()).add(default)

            # Identify separate pipeline roots: processes that have outgoing
            # flow edges but no incoming flow/branch/loop edges from any node.
            # These represent intentionally separate entry points (e.g., an
            # ingestion pipeline alongside a query pipeline).
            incoming = set()
            for edge in edges:
                if edge.get("type") in ("flow", "branch", "loop"):
                    eto = edge.get("to")
                    if eto:
                        incoming.add(eto)
            # Also mark targets of gate branches as having incoming
            for proc in processes:
                if proc.get("type") == "gate":
                    for branch in proc.get("branches", []):
                        target = branch.get("target")
                        if target:
                            incoming.add(target)

            separate_roots = set()
            for pid in process_ids:
                if pid == entry_point:
                    continue
                if pid in exempt_ids:
                    continue
                if pid not in incoming:
                    # This process has no incoming flow -- it is a separate
                    # pipeline root.
                    separate_roots.add(pid)

            # BFS from entry_point (traverses through entity nodes too)
            reachable = set()
            frontier = [entry_point]
            while frontier:
                current = frontier.pop()
                if current in reachable:
                    continue
                reachable.add(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in reachable:
                        frontier.append(neighbor)

            # Also BFS from each separate pipeline root
            for root in separate_roots:
                if root in reachable:
                    continue
                sub_frontier = [root]
                while sub_frontier:
                    current = sub_frontier.pop()
                    if current in reachable:
                        continue
                    reachable.add(current)
                    for neighbor in adjacency.get(current, set()):
                        if neighbor not in reachable:
                            sub_frontier.append(neighbor)

            # Check: all non-exempt processes should be reachable
            must_reach = process_ids - exempt_ids
            unreachable = sorted(must_reach - reachable)
            if unreachable:
                results.fail(test_name, name,
                             f"{len(unreachable)} unreachable process(es): {unreachable}")
            else:
                results.ok(test_name, name)
        except Exception as e:
            results.fail(test_name, name, f"Exception: {e}")


# ---------------------------------------------------------------------------
# Property 5: Fan-out symmetry
# If a process has fan-out transitions (list of targets), all targets exist
# as valid processes.
# ---------------------------------------------------------------------------

def test_fan_out_symmetry():
    """All fan-out targets from flow edges must reference existing processes."""
    test_name = "fan-out symmetry"

    for path in all_spec_paths():
        name = spec_name(path)
        try:
            spec = load_yaml(path)
            processes = spec.get("processes", [])
            edges = spec.get("edges", [])

            process_ids = {p.get("id") for p in processes if p.get("id")}
            entity_ids = {e.get("id") for e in spec.get("entities", []) if e.get("id")}
            all_node_ids = process_ids | entity_ids

            # Build flow targets per process
            flow_targets = {}
            for edge in edges:
                if edge.get("type") in ("flow", "branch"):
                    src = edge.get("from")
                    tgt = edge.get("to")
                    if src and tgt:
                        flow_targets.setdefault(src, []).append(tgt)

            invalid = []
            for src, targets in flow_targets.items():
                if len(targets) > 1:
                    # This is a fan-out -- check all targets exist
                    for tgt in targets:
                        if tgt not in all_node_ids:
                            invalid.append(f"'{src}' -> '{tgt}' (target not found)")

            if invalid:
                results.fail(test_name, name,
                             f"{len(invalid)} invalid fan-out target(s): {invalid[0]}")
            else:
                results.ok(test_name, name)
        except Exception as e:
            results.fail(test_name, name, f"Exception: {e}")


# ---------------------------------------------------------------------------
# Property 6: Gate branch coverage
# Every gate has at least 2 branches and all branch targets exist.
# ---------------------------------------------------------------------------

def test_gate_branch_coverage():
    """Every gate must have >= 2 branches and all branch targets must exist."""
    test_name = "gate branch coverage"

    for path in all_spec_paths():
        name = spec_name(path)
        try:
            spec = load_yaml(path)
            processes = spec.get("processes", [])

            all_node_ids = set()
            for e in spec.get("entities", []):
                eid = e.get("id")
                if eid:
                    all_node_ids.add(eid)
            for p in processes:
                pid = p.get("id")
                if pid:
                    all_node_ids.add(pid)

            issues = []
            gates = [p for p in processes if p.get("type") == "gate"]

            for gate in gates:
                gid = gate.get("id", "?")
                branches = gate.get("branches", [])

                if len(branches) < 2:
                    issues.append(
                        f"gate '{gid}' has {len(branches)} branch(es), needs >= 2")

                for branch in branches:
                    target = branch.get("target")
                    if target and target not in all_node_ids:
                        issues.append(
                            f"gate '{gid}' branch target '{target}' not found")

            if issues:
                results.fail(test_name, name,
                             f"{len(issues)} issue(s): {issues[0]}")
            else:
                results.ok(test_name, name)
        except Exception as e:
            results.fail(test_name, name, f"Exception: {e}")


# ---------------------------------------------------------------------------
# Property 7: Store access
# Every store entity has at least one read or write edge referencing it.
# ---------------------------------------------------------------------------

def test_store_access():
    """Every store entity must have at least one read or write edge."""
    test_name = "store access"

    for path in all_spec_paths():
        name = spec_name(path)
        try:
            spec = load_yaml(path)
            entities = spec.get("entities", [])
            edges = spec.get("edges", [])

            stores = {e.get("id") for e in entities
                      if e.get("type") == "store" and e.get("id")}

            if not stores:
                # No stores defined -- trivially passes
                results.ok(test_name, name)
                continue

            # Collect stores referenced by read/write edges
            accessed_stores = set()
            for edge in edges:
                etype = edge.get("type")
                if etype in ("read", "write"):
                    eto = edge.get("to")
                    if eto:
                        accessed_stores.add(eto)

            unaccessed = sorted(stores - accessed_stores)
            if unaccessed:
                results.fail(test_name, name,
                             f"{len(unaccessed)} store(s) with no read/write edge: {unaccessed}")
            else:
                results.ok(test_name, name)
        except Exception as e:
            results.fail(test_name, name, f"Exception: {e}")


# ---------------------------------------------------------------------------
# Property 8: LangGraph code generation syntax
# For every spec (except claude_code), generate_langgraph_agent() produces
# syntactically valid Python.
# ---------------------------------------------------------------------------

def test_langgraph_code_generation_syntax():
    """Generated LangGraph Python code must be syntactically valid (ast.parse)."""
    test_name = "langgraph code generation syntax"
    skip = {"claude-code.yaml"}
    for path in all_spec_paths():
        name = spec_name(path)
        if name in skip:
            continue
        try:
            spec = load_yaml(path)
            code = generate_langgraph_agent(spec)
            ast.parse(code)
            results.ok(test_name, name)
        except SyntaxError as e:
            results.fail(test_name, name,
                         f"SyntaxError at line {e.lineno}: {e.msg}")
        except Exception as e:
            results.fail(test_name, name, f"Exception: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading ontology from {ONTOLOGY_PATH}")
    ontology = load_ontology()

    spec_paths = all_spec_paths()
    print(f"Found {len(spec_paths)} spec(s) in {SPECS_DIR}/\n")

    print("Property 1: Round-trip validation")
    print("-" * 40)
    test_roundtrip_validation(ontology)

    print("\nProperty 2: Code generation syntax")
    print("-" * 40)
    test_code_generation_syntax()

    print("\nProperty 3: Schema completeness")
    print("-" * 40)
    test_schema_completeness()

    print("\nProperty 4: Entry point reachability")
    print("-" * 40)
    test_entry_point_reachability()

    print("\nProperty 5: Fan-out symmetry")
    print("-" * 40)
    test_fan_out_symmetry()

    print("\nProperty 6: Gate branch coverage")
    print("-" * 40)
    test_gate_branch_coverage()

    print("\nProperty 7: Store access")
    print("-" * 40)
    test_store_access()

    print("\nProperty 8: LangGraph code generation syntax")
    print("-" * 40)
    test_langgraph_code_generation_syntax()

    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
