#!/usr/bin/env python3
"""
Agent Ontology OWL Bridge — Phase B: Dual Representation

Bidirectional YAML <-> OWL conversion with lossless round-trip.

Architecture:
  - Structural model (OWL classes + object properties) -> enables reasoning
  - Raw data (JSON-encoded YAML dicts in data properties) -> enables round-trip

The bridge extends ontology_owl.py's structural model with data properties
that store original YAML values. Reasoning tools use the structural model;
reconstruction uses the stored data.

Usage:
  python3 owl_bridge.py --round-trip                     # Test all 22 specs
  python3 owl_bridge.py --round-trip specs/react.yaml    # Test one spec
  python3 owl_bridge.py --classify                       # Pattern classification
  python3 owl_bridge.py --export specs/react.yaml        # Print reconstructed YAML
"""

import json
import os
import sys
import yaml
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from owlready2 import *


# ═══════════════════════════════════════════════════════════════
# 1. BUILD EXTENDED ONTOLOGY
# ═══════════════════════════════════════════════════════════════

def build_bridge_ontology():
    """Build OWL ontology with structural model + round-trip data properties.

    Extends ontology_owl.py's class hierarchy with:
    - Schema class for data schema modeling
    - Data properties for storing raw YAML values
    - Order-preserving properties for lossless reconstruction
    """
    from .ontology_owl import build_ontology
    onto = build_ontology()

    with onto:
        # ── Schema modeling ──────────────────────────────────
        class Schema(Thing):
            """A data schema definition."""
            pass

        class has_schema(onto.Spec >> Schema):
            """Spec has schemas."""
            pass

        class has_schema_name_val(Schema >> str):
            """Schema name."""
            pass

        class has_schema_fields_json(Schema >> str):
            """JSON array of field dicts."""
            pass

        class has_schema_description_val(Schema >> str):
            """Schema description."""
            pass

        # ── Round-trip data properties ───────────────────────
        class has_raw_yaml(Thing >> str):
            """JSON-encoded original YAML dict for this instance."""
            pass

        class has_spec_meta_json(onto.Spec >> str):
            """JSON-encoded spec-level metadata."""
            pass

        class has_edges_json(onto.Spec >> str):
            """JSON-encoded array of all edge dicts (preserves order)."""
            pass

        class has_schemas_raw_json(onto.Spec >> str):
            """JSON-encoded array of all schema dicts (preserves order)."""
            pass

        class has_entity_order_json(onto.Spec >> str):
            """JSON array of entity IDs in spec order."""
            pass

        class has_process_order_json(onto.Spec >> str):
            """JSON array of process IDs in spec order."""
            pass

        class has_key_order_json(onto.Spec >> str):
            """JSON array of top-level YAML keys in order."""
            pass

        # ── Additional structural properties ─────────────────
        class has_data_in_ref(onto.Process >> str):
            """Input schema name reference."""
            pass

        class has_data_out_ref(onto.Process >> str):
            """Output schema name reference."""
            pass

        class has_description_val(Thing >> str):
            """Description text."""
            pass

        class has_system_prompt_val(onto.Agent >> str):
            """Agent system prompt."""
            pass

    return onto


# ═══════════════════════════════════════════════════════════════
# 2. YAML -> OWL
# ═══════════════════════════════════════════════════════════════

def spec_to_owl(onto, spec_path):
    """Load a YAML spec into OWL with full structural model + raw data.

    Creates:
    - Spec instance with metadata
    - Entity instances with type-correct OWL classes + raw data
    - Process instances with structural properties + raw data
    - Schema instances with field data
    - OWL object properties for edges (structural model)
    - Stored arrays for lossless round-trip
    """
    if isinstance(spec_path, dict):
        spec = spec_path
        file_id = spec.get("name", "unknown").replace(" ", "_").lower()
    else:
        with open(spec_path) as f:
            spec = yaml.safe_load(f)
        file_id = os.path.basename(spec_path).replace(".yaml", "")

    if not spec:
        return None

    safe_name = file_id.replace(" ", "_").replace("-", "_").replace(".", "_")

    with onto:
        # ── Spec instance ────────────────────────────────────
        spec_inst = onto.Spec(f"spec_{safe_name}")

        # Store ALL spec-level keys except list sections
        meta = {}
        list_keys = {"entities", "processes", "edges", "schemas"}
        for key, val in spec.items():
            if key not in list_keys:
                meta[key] = val
        spec_inst.has_spec_meta_json = [json.dumps(meta)]

        # Preserve top-level key order
        spec_inst.has_key_order_json = [json.dumps(list(spec.keys()))]

        # ── Schemas ──────────────────────────────────────────
        for s in spec.get("schemas", []):
            schema_safe = f"{safe_name}_schema_{s['name']}"
            schema_inst = onto.Schema(schema_safe)
            schema_inst.has_schema_name_val = [s["name"]]
            schema_inst.has_schema_fields_json = [json.dumps(s.get("fields", []))]
            if "description" in s:
                schema_inst.has_schema_description_val = [s["description"]]
            spec_inst.has_schema.append(schema_inst)

        # Store raw schemas array for round-trip
        spec_inst.has_schemas_raw_json = [json.dumps(spec.get("schemas", []))]

        # ── Entities ─────────────────────────────────────────
        entity_instances = {}
        entity_order = []

        cls_map = {
            "agent": onto.Agent, "store": onto.Store, "tool": onto.Tool,
            "human": onto.Human, "channel": onto.Channel, "team": onto.Team,
            "conversation": onto.Conversation, "config": onto.Config,
        }

        for e in spec.get("entities", []):
            eid = e["id"]
            etype = e["type"]
            safe_eid = f"{safe_name}_{eid}"
            entity_order.append(eid)

            cls = cls_map.get(etype, onto.Entity)
            inst = cls(safe_eid)

            # Structural properties (for reasoning)
            inst.has_label = [e.get("label", eid)]
            if etype == "agent" and "model" in e:
                inst.has_model = [e["model"]]
            if etype == "agent" and "system_prompt" in e:
                inst.has_system_prompt_val = [e["system_prompt"]]
            if etype == "store" and "store_type" in e:
                inst.has_store_type = [e["store_type"]]
            if etype == "tool" and "tool_type" in e:
                inst.has_tool_type = [e["tool_type"]]
            if etype == "team" and "strategy" in e:
                inst.has_strategy = [e["strategy"]]
            if "description" in e:
                inst.has_description_val = [e["description"]]

            # Raw YAML for lossless round-trip
            inst.has_raw_yaml = [json.dumps(e)]

            spec_inst.has_entity.append(inst)
            entity_instances[eid] = inst

        spec_inst.has_entity_order_json = [json.dumps(entity_order)]

        # ── Processes ────────────────────────────────────────
        process_instances = {}
        process_order = []
        proc_order_map = {}

        proc_cls_map = {
            "step": onto.Step, "gate": onto.Gate, "checkpoint": onto.Checkpoint,
            "spawn": onto.Spawn, "protocol": onto.Protocol, "policy": onto.Policy,
            "error_handler": onto.ErrorHandler,
        }

        for idx, p in enumerate(spec.get("processes", [])):
            pid = p["id"]
            ptype = p["type"]
            safe_pid = f"{safe_name}_{pid}"
            process_order.append(pid)
            proc_order_map[pid] = idx

            cls = proc_cls_map.get(ptype, onto.Process)
            inst = cls(safe_pid)

            # Structural properties (for reasoning)
            inst.has_label = [p.get("label", pid)]
            if ptype == "gate" and "condition" in p:
                inst.has_condition = [p["condition"]]
            if ptype == "step" and "logic" in p:
                inst.has_logic = [p["logic"]]  # Full logic, not truncated
            if "data_in" in p:
                inst.has_data_in_ref = [p["data_in"]]
            if "data_out" in p:
                inst.has_data_out_ref = [p["data_out"]]
            if "description" in p:
                inst.has_description_val = [p["description"]]

            # Raw YAML for lossless round-trip
            inst.has_raw_yaml = [json.dumps(p)]

            spec_inst.has_process.append(inst)
            process_instances[pid] = inst

        spec_inst.has_process_order_json = [json.dumps(process_order)]

        # ── Edges (structural OWL model) ─────────────────────
        for e in spec.get("edges", []):
            etype = e["type"]
            src = e.get("from", "")
            tgt = e.get("to", "")
            src_inst = process_instances.get(src) or entity_instances.get(src)
            tgt_inst = process_instances.get(tgt) or entity_instances.get(tgt)

            if not src_inst or not tgt_inst:
                continue

            if etype == "flow" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Process):
                if tgt_inst not in src_inst.flows_to:
                    src_inst.flows_to.append(tgt_inst)
            elif etype == "invoke" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Entity):
                if tgt_inst not in src_inst.invokes:
                    src_inst.invokes.append(tgt_inst)
            elif etype == "loop" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Process):
                if tgt_inst not in src_inst.loops_to:
                    src_inst.loops_to.append(tgt_inst)
            elif etype == "branch" and isinstance(src_inst, onto.Gate) and isinstance(tgt_inst, onto.Process):
                if tgt_inst not in src_inst.branches_to:
                    src_inst.branches_to.append(tgt_inst)
            elif etype == "read" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Store):
                if tgt_inst not in src_inst.reads_from:
                    src_inst.reads_from.append(tgt_inst)
            elif etype == "write" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Store):
                if tgt_inst not in src_inst.writes_to:
                    src_inst.writes_to.append(tgt_inst)
            elif etype == "publish" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Channel):
                if tgt_inst not in src_inst.publishes_to:
                    src_inst.publishes_to.append(tgt_inst)
            elif etype == "subscribe" and isinstance(tgt_inst, onto.Process) and isinstance(src_inst, onto.Channel):
                if src_inst not in tgt_inst.subscribes_from:
                    tgt_inst.subscribes_from.append(src_inst)
            elif etype == "handoff" and isinstance(src_inst, onto.Agent) and isinstance(tgt_inst, onto.Agent):
                if tgt_inst not in src_inst.hands_off_to:
                    src_inst.hands_off_to.append(tgt_inst)

        # Handle gate branches from process definitions (for reasoning)
        for p in spec.get("processes", []):
            if p["type"] == "gate":
                pid = p["id"]
                gate_inst = process_instances.get(pid)
                if gate_inst:
                    gate_order = proc_order_map.get(pid, 999)
                    for branch in p.get("branches", []):
                        target_id = branch.get("target", "")
                        target_inst = process_instances.get(target_id)
                        if target_inst:
                            if target_inst not in gate_inst.branches_to:
                                gate_inst.branches_to.append(target_inst)
                            target_order = proc_order_map.get(target_id, 999)
                            if isinstance(target_inst, onto.Step) and target_order < gate_order:
                                if target_inst not in gate_inst.loops_to:
                                    gate_inst.loops_to.append(target_inst)

        # Detect backward flow edges (for reasoning)
        for edge in spec.get("edges", []):
            if edge["type"] == "flow":
                src_id = edge.get("from", "")
                tgt_id = edge.get("to", "")
                src_order = proc_order_map.get(src_id, 999)
                tgt_order = proc_order_map.get(tgt_id, 999)
                if tgt_order < src_order:
                    src_inst = process_instances.get(src_id)
                    tgt_inst = process_instances.get(tgt_id)
                    if src_inst and tgt_inst and isinstance(tgt_inst, onto.Step):
                        if tgt_inst not in src_inst.loops_to:
                            src_inst.loops_to.append(tgt_inst)

        # Store full edge array for round-trip
        spec_inst.has_edges_json = [json.dumps(spec.get("edges", []))]

    return spec_inst


# ═══════════════════════════════════════════════════════════════
# 3. OWL -> YAML
# ═══════════════════════════════════════════════════════════════

def owl_to_spec(onto, spec_inst):
    """Reconstruct a YAML-compatible spec dict from an OWL spec instance.

    Uses stored raw YAML data for lossless reconstruction.
    Preserves original key order and list element order.
    """
    result = {}

    # ── Spec-level metadata ──────────────────────────────────
    meta_json = spec_inst.has_spec_meta_json
    if meta_json:
        meta = json.loads(meta_json[0])
        result.update(meta)

    # ── Key ordering ─────────────────────────────────────────
    key_order_json = spec_inst.has_key_order_json
    if key_order_json:
        key_order = json.loads(key_order_json[0])
    else:
        key_order = list(result.keys()) + ["entities", "processes", "edges", "schemas"]

    # ── Entities (in original order) ─────────────────────────
    entity_order_json = spec_inst.has_entity_order_json
    entity_order = json.loads(entity_order_json[0]) if entity_order_json else []

    entity_map = {}
    for e_inst in spec_inst.has_entity:
        raw_json = e_inst.has_raw_yaml
        if raw_json:
            entity_dict = json.loads(raw_json[0])
            entity_map[entity_dict["id"]] = entity_dict

    entities = []
    for eid in entity_order:
        if eid in entity_map:
            entities.append(entity_map[eid])
    if entities:
        result["entities"] = entities

    # ── Processes (in original order) ────────────────────────
    process_order_json = spec_inst.has_process_order_json
    process_order = json.loads(process_order_json[0]) if process_order_json else []

    proc_map = {}
    for p_inst in spec_inst.has_process:
        raw_json = p_inst.has_raw_yaml
        if raw_json:
            proc_dict = json.loads(raw_json[0])
            proc_map[proc_dict["id"]] = proc_dict

    processes = []
    for pid in process_order:
        if pid in proc_map:
            processes.append(proc_map[pid])
    if processes:
        result["processes"] = processes

    # ── Edges ────────────────────────────────────────────────
    edges_json = spec_inst.has_edges_json
    if edges_json:
        edges = json.loads(edges_json[0])
        if edges:
            result["edges"] = edges

    # ── Schemas ──────────────────────────────────────────────
    schemas_json = spec_inst.has_schemas_raw_json
    if schemas_json:
        schemas = json.loads(schemas_json[0])
        if schemas:
            result["schemas"] = schemas

    # ── Reorder keys to match original ───────────────────────
    ordered = {}
    for key in key_order:
        if key in result:
            ordered[key] = result[key]
    # Add any remaining keys not in original order
    for key in result:
        if key not in ordered:
            ordered[key] = result[key]

    return ordered


# ═══════════════════════════════════════════════════════════════
# 4. ROUND-TRIP TESTING
# ═══════════════════════════════════════════════════════════════

def _deep_diff(a, b, path=""):
    """Deep comparison of two values, returning list of differences."""
    diffs = []

    if type(a) != type(b):
        diffs.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return diffs

    if isinstance(a, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        for k in sorted(a_keys - b_keys):
            diffs.append(f"{path}.{k}: missing in reconstructed")
        for k in sorted(b_keys - a_keys):
            diffs.append(f"{path}.{k}: extra in reconstructed")
        for k in sorted(a_keys & b_keys):
            diffs.extend(_deep_diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        for i in range(min(len(a), len(b))):
            diffs.extend(_deep_diff(a[i], b[i], f"{path}[{i}]"))
    elif a != b:
        a_str = repr(a)[:80]
        b_str = repr(b)[:80]
        diffs.append(f"{path}: {a_str} != {b_str}")

    return diffs


def round_trip(spec_path, onto=None):
    """Test YAML -> OWL -> YAML round-trip for one spec.

    Returns (passed: bool, diffs: list[str]).
    """
    # Load original
    with open(spec_path) as f:
        original = yaml.safe_load(f)

    if not original:
        return False, ["Empty spec"]

    # Build ontology if needed (use isolated world to avoid state leakage)
    if onto is None:
        world = World()
        onto = build_bridge_ontology_in_world(world)

    # YAML -> OWL
    spec_inst = spec_to_owl(onto, spec_path)
    if not spec_inst:
        return False, ["Failed to load into OWL"]

    # OWL -> YAML
    reconstructed = owl_to_spec(onto, spec_inst)

    # Compare
    diffs = _deep_diff(original, reconstructed)
    return len(diffs) == 0, diffs


def build_bridge_ontology_in_world(world):
    """Build bridge ontology in an isolated owlready2 World.

    Each world has its own triple store, preventing state leakage
    between round-trip tests.
    """
    onto = world.get_ontology("http://agent-ontology.org/ontology#")

    with onto:
        # ── Core classes (from ontology_owl.py) ──────────────
        class Spec(Thing): pass
        class Node(Thing): pass
        class Entity(Node): pass
        class Process(Node): pass
        class Edge(Thing): pass

        class Agent(Entity): pass
        class Store(Entity): pass
        class Tool(Entity): pass
        class Human(Entity): pass
        class Channel(Entity): pass
        class Team(Entity): pass
        class Conversation(Entity): pass
        class Config(Entity): pass

        class Step(Process): pass
        class Gate(Process): pass
        class Checkpoint(Process): pass
        class Spawn(Process): pass
        class Protocol(Process): pass
        class Policy(Process): pass
        class ErrorHandler(Process): pass

        # ── Object properties ────────────────────────────────
        class has_entity(Spec >> Entity): pass
        class has_process(Spec >> Process): pass
        class flows_to(Process >> Process): pass
        class loops_to(Process >> Process): pass
        class branches_to(Gate >> Process): pass
        class invokes(Process >> Entity): pass
        class reads_from(Process >> Store): pass
        class writes_to(Process >> Store): pass
        class publishes_to(Process >> Channel): pass
        class subscribes_from(Process >> Channel): pass
        class hands_off_to(Agent >> Agent): pass

        # ── Core data properties ─────────────────────────────
        class has_label(Node >> str): pass
        class has_model(Agent >> str): pass
        class has_condition(Gate >> str): pass
        class has_logic(Step >> str): pass
        class has_store_type(Store >> str): pass
        class has_tool_type(Tool >> str): pass
        class has_strategy(Team >> str): pass

        # ── Pattern definitions (DL concepts) ────────────────
        class ReasoningLoopSpec(Spec):
            equivalent_to = [
                Spec & has_process.some(
                    Step & invokes.some(Agent)
                    & flows_to.some(Gate & loops_to.some(Step))
                )
            ]

        class CritiqueCycleSpec(Spec):
            equivalent_to = [
                Spec
                & has_process.some(Step & invokes.some(Agent) & flows_to.some(Step & invokes.some(Agent)))
                & has_process.some(Gate & loops_to.some(Step))
            ]

        class RetrievalAugmentedSpec(Spec):
            equivalent_to = [
                Spec
                & has_process.some(Step & reads_from.some(Store))
                & has_process.some(Step & invokes.some(Agent))
            ]

        class FanOutSpec(Spec):
            equivalent_to = [
                Spec & has_process.some(Step & invokes.min(2, Agent))
            ]

        class HumanInLoopSpec(Spec):
            equivalent_to = [Spec & has_process.some(Checkpoint)]

        class MultiAgentDebateSpec(Spec):
            equivalent_to = [
                Spec
                & has_entity.min(3, Agent)
                & has_process.some(Gate & loops_to.some(Step))
            ]

        class PubSubSpec(Spec):
            equivalent_to = [
                Spec
                & has_entity.some(Channel)
                & has_process.some(Step & publishes_to.some(Channel))
            ]

        class HandoffSpec(Spec):
            equivalent_to = [
                Spec & has_entity.some(Agent & hands_off_to.some(Agent))
            ]

        class MemoryBackedSpec(Spec):
            equivalent_to = [
                Spec
                & has_process.some(Step & reads_from.some(Store))
                & has_process.some(Step & writes_to.some(Store))
            ]

        # ── Schema modeling ──────────────────────────────────
        class Schema(Thing): pass
        class has_schema(Spec >> Schema): pass
        class has_schema_name_val(Schema >> str): pass
        class has_schema_fields_json(Schema >> str): pass
        class has_schema_description_val(Schema >> str): pass

        # ── Round-trip data properties ───────────────────────
        class has_raw_yaml(Thing >> str): pass
        class has_spec_meta_json(Spec >> str): pass
        class has_edges_json(Spec >> str): pass
        class has_schemas_raw_json(Spec >> str): pass
        class has_entity_order_json(Spec >> str): pass
        class has_process_order_json(Spec >> str): pass
        class has_key_order_json(Spec >> str): pass

        # ── Additional structural properties ─────────────────
        class has_data_in_ref(Process >> str): pass
        class has_data_out_ref(Process >> str): pass
        class has_description_val(Thing >> str): pass
        class has_system_prompt_val(Agent >> str): pass

    return onto


def round_trip_all(specs_dir=None):
    """Run round-trip test on all specs. Returns {name: (passed, diffs)}."""
    if specs_dir is None:
        specs_dir = os.path.join(SCRIPT_DIR, "specs")

    spec_files = sorted(glob.glob(os.path.join(specs_dir, "*.yaml")))
    results = {}

    # Use a single world + ontology for all specs (faster)
    world = World()
    onto = build_bridge_ontology_in_world(world)

    for sf in spec_files:
        name = os.path.basename(sf).replace(".yaml", "")
        try:
            passed, diffs = round_trip(sf, onto=onto)
            results[name] = (passed, diffs)
        except Exception as ex:
            results[name] = (False, [f"Exception: {ex}"])

    return results


# ═══════════════════════════════════════════════════════════════
# 5. PATTERN CLASSIFICATION (delegates to ontology_owl.py)
# ═══════════════════════════════════════════════════════════════

def classify_all(specs_dir=None):
    """Classify all specs using structural pattern matching.

    Uses the OWL structural model for pattern detection.
    Returns {spec_name: [pattern_names]}.
    """
    from .ontology_owl import classify_structural

    if specs_dir is None:
        specs_dir = os.path.join(SCRIPT_DIR, "specs")

    spec_files = sorted(glob.glob(os.path.join(specs_dir, "*.yaml")))

    world = World()
    onto = build_bridge_ontology_in_world(world)

    spec_instances = []
    for sf in spec_files:
        try:
            inst = spec_to_owl(onto, sf)
            if inst:
                spec_instances.append(inst)
        except Exception:
            pass

    return classify_structural(onto, spec_instances)


# ═══════════════════════════════════════════════════════════════
# 6. CONVENIENCE: EXPORT SPEC AS OWL/RDF
# ═══════════════════════════════════════════════════════════════

def export_owl(spec_path, format="ntriples"):
    """Export a spec to OWL serialization (RDF/XML, N-Triples, etc.).

    Returns the serialized string.
    """
    world = World()
    onto = build_bridge_ontology_in_world(world)
    spec_to_owl(onto, spec_path)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".owl", delete=False) as f:
        tmp = f.name

    try:
        onto.save(file=tmp, format=format)
        with open(tmp) as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    if "--round-trip" in sys.argv:
        # Round-trip test
        spec_args = [a for a in sys.argv[1:] if not a.startswith("--")]
        if spec_args:
            # Test specific specs
            world = World()
            onto = build_bridge_ontology_in_world(world)
            total = 0
            passed_count = 0
            for sf in spec_args:
                name = os.path.basename(sf).replace(".yaml", "")
                total += 1
                passed, diffs = round_trip(sf, onto=onto)
                status = "PASS" if passed else "FAIL"
                if passed:
                    passed_count += 1
                print(f"  {name:<40} {status}")
                if diffs:
                    for d in diffs[:10]:
                        print(f"    {d}")
            print(f"\n  {passed_count}/{total} passed")
        else:
            # Test all specs
            print("Running YAML -> OWL -> YAML round-trip on all specs...")
            print()
            results = round_trip_all()
            passed_count = 0
            total = len(results)
            for name, (passed, diffs) in sorted(results.items()):
                status = "PASS" if passed else "FAIL"
                if passed:
                    passed_count += 1
                print(f"  {name:<40} {status}")
                if diffs:
                    for d in diffs[:5]:
                        print(f"    {d}")
            print()
            print(f"  {passed_count}/{total} passed")

    elif "--classify" in sys.argv:
        # Pattern classification
        print("Classifying specs using OWL structural model...")
        print()
        results = classify_all()

        name_width = max(len(n) for n in results) + 2
        print(f"  {'Spec':<{name_width}} Patterns")
        print(f"  {'─' * name_width} {'─' * 50}")
        for name, patterns in sorted(results.items()):
            pstr = ", ".join(patterns) if patterns else "(none)"
            print(f"  {name:<{name_width}} {pstr}")

        all_patterns = set()
        for ps in results.values():
            all_patterns.update(ps)
        print(f"\n  {len(results)} specs, {len(all_patterns)} distinct patterns")

    elif "--export" in sys.argv:
        # Export spec as reconstructed YAML
        spec_args = [a for a in sys.argv[1:] if not a.startswith("--")]
        if not spec_args:
            print("Usage: owl_bridge.py --export <spec.yaml>", file=sys.stderr)
            sys.exit(1)
        world = World()
        onto = build_bridge_ontology_in_world(world)
        for sf in spec_args:
            inst = spec_to_owl(onto, sf)
            if inst:
                reconstructed = owl_to_spec(onto, inst)
                print(yaml.dump(reconstructed, default_flow_style=False, sort_keys=False))

    else:
        print(__doc__.strip())
        print()
        print("Commands:")
        print("  --round-trip [spec.yaml ...]  Test YAML->OWL->YAML round-trip")
        print("  --classify                    Pattern classification via OWL")
        print("  --export <spec.yaml>          Export reconstructed YAML")


if __name__ == "__main__":
    main()
