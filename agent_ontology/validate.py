#!/usr/bin/env python3
"""
Agent Ontology Spec Validator
Validates agent specs against ONTOLOGY.yaml.

Usage: python3 validate.py specs/claude-code.yaml
       python3 validate.py specs/*.yaml
"""

import glob
import sys
import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(SCRIPT_DIR, "ONTOLOGY.yaml")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def validate_spec(spec, ontology, filepath):
    errors = []
    warnings = []

    def err(msg):
        errors.append(msg)

    def warn(msg):
        warnings.append(msg)

    # ── Extract ontology type sets ──
    valid_entity_types = set(ontology.get("entity_types", {}).keys())
    valid_process_types = set(ontology.get("process_types", {}).keys())
    valid_edge_types = set(ontology.get("edge_types", {}).keys())

    entity_type_defs = ontology.get("entity_types", {})
    process_type_defs = ontology.get("process_types", {})
    edge_type_defs = ontology.get("edge_types", {})

    # ── Extract spec contents ──
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])
    schemas = spec.get("schemas", [])
    entry_point = spec.get("entry_point")

    # Build ID maps
    all_nodes = {}  # id -> {type, category, data}
    for e in entities:
        eid = e.get("id")
        if not eid:
            err("Entity missing 'id'")
            continue
        if eid in all_nodes:
            err(f"Duplicate node id: '{eid}'")
        all_nodes[eid] = {"type": e.get("type"), "category": "entity", "data": e}

    for p in processes:
        pid = p.get("id")
        if not pid:
            err("Process missing 'id'")
            continue
        if pid in all_nodes:
            err(f"Duplicate node id: '{pid}'")
        all_nodes[pid] = {"type": p.get("type"), "category": "process", "data": p}

    schema_names = {s.get("name") for s in schemas if s.get("name")}

    # ════════════════════════════════════════════════════
    # 1. Spec-level required fields
    # ════════════════════════════════════════════════════
    if not spec.get("name"):
        err("Spec missing required field: 'name'")
    if not spec.get("version"):
        err("Spec missing required field: 'version'")
    if not entities:
        err("Spec has no entities")
    if not processes:
        err("Spec has no processes")

    # ════════════════════════════════════════════════════
    # 2. Entity type validation
    # ════════════════════════════════════════════════════
    for e in entities:
        eid = e.get("id", "?")
        etype = e.get("type")
        if not etype:
            err(f"Entity '{eid}' missing 'type'")
        elif etype not in valid_entity_types:
            err(f"Entity '{eid}' has invalid type '{etype}'. Valid: {sorted(valid_entity_types)}")
        else:
            # Check required properties
            typedef = entity_type_defs[etype]
            for prop in typedef.get("required", {}):
                if prop not in e:
                    err(f"Entity '{eid}' (type={etype}) missing required property: '{prop}'")

    # ════════════════════════════════════════════════════
    # 3. Process type validation
    # ════════════════════════════════════════════════════
    for p in processes:
        pid = p.get("id", "?")
        ptype = p.get("type")
        if not ptype:
            err(f"Process '{pid}' missing 'type'")
        elif ptype not in valid_process_types:
            err(f"Process '{pid}' has invalid type '{ptype}'. Valid: {sorted(valid_process_types)}")
        else:
            typedef = process_type_defs[ptype]
            for prop in typedef.get("required", {}):
                if prop not in p:
                    err(f"Process '{pid}' (type={ptype}) missing required property: '{prop}'")

    # ════════════════════════════════════════════════════
    # 4. Edge type + reference validation
    # ════════════════════════════════════════════════════
    for i, e in enumerate(edges):
        etype = e.get("type")
        efrom = e.get("from")
        eto = e.get("to")
        elabel = e.get("label", f"edge[{i}]")

        if not etype:
            err(f"Edge '{elabel}' missing 'type'")
        elif etype not in valid_edge_types:
            err(f"Edge '{elabel}' has invalid type '{etype}'. Valid: {sorted(valid_edge_types)}")

        if not efrom:
            err(f"Edge '{elabel}' missing 'from'")
        elif efrom not in all_nodes:
            err(f"Edge '{elabel}': 'from' references unknown node '{efrom}'")

        if not eto:
            err(f"Edge '{elabel}' missing 'to'")
        elif eto not in all_nodes:
            err(f"Edge '{elabel}': 'to' references unknown node '{eto}'")

        # Type constraint checks (from ontology edge_type definitions)
        if etype and efrom in all_nodes and eto in all_nodes:
            from_node = all_nodes[efrom]
            to_node = all_nodes[eto]
            _check_edge_constraints(etype, efrom, eto, from_node, to_node, elabel, edge_type_defs, err, warn)

        # Schema reference checks on edges
        for field in ("data", "input", "output"):
            ref = e.get(field)
            if ref and ref not in schema_names:
                err(f"Edge '{elabel}': {field} references unknown schema '{ref}'")

    # ════════════════════════════════════════════════════
    # 5. Ontology validation rules
    # ════════════════════════════════════════════════════

    # Rule: at least one agent entity
    agent_entities = [e for e in entities if e.get("type") == "agent"]
    if not agent_entities:
        err("Spec must have at least one agent entity")

    # Rule: entry_point exists
    if entry_point:
        if entry_point not in all_nodes:
            err(f"entry_point '{entry_point}' references unknown node")
    else:
        # Check for exactly one node with no incoming flow/invoke
        incoming = set()
        for e in edges:
            if e.get("type") in ("flow", "invoke"):
                incoming.add(e.get("to"))
        process_ids = {p.get("id") for p in processes}
        roots = process_ids - incoming
        if len(roots) == 0:
            warn("No entry_point defined and no root process found (all processes have incoming edges)")
        elif len(roots) > 1:
            warn(f"No entry_point defined and multiple root processes found: {sorted(roots)}")

    # Rule: every gate has at least 2 branches
    for p in processes:
        if p.get("type") == "gate":
            branches = p.get("branches", [])
            if len(branches) < 2:
                err(f"Gate '{p.get('id')}' must have at least 2 branches, has {len(branches)}")
            # Also check branch targets resolve
            for b in branches:
                target = b.get("target")
                if target and target not in all_nodes:
                    err(f"Gate '{p.get('id')}' branch target '{target}' references unknown node")

    # Rule: every loop targets an earlier node in the flow
    # (We approximate "earlier" by checking the loop target exists — full
    # topological ordering would require building the flow graph)
    for e in edges:
        if e.get("type") == "loop":
            if e.get("to") not in all_nodes:
                err(f"Loop edge targets unknown node '{e.get('to')}'")

    # Rule: every schema_ref resolves
    _check_schema_refs(entities, processes, schema_names, err)

    # Rule: no orphan nodes
    connected = set()
    for e in edges:
        connected.add(e.get("from"))
        connected.add(e.get("to"))
    for nid in all_nodes:
        if nid not in connected:
            warn(f"Orphan node: '{nid}' has no edges")

    # Rule: agent tools[] should be defined as tool entities
    tool_entity_ids = {e.get("id") for e in entities if e.get("type") == "tool"}
    for e in entities:
        if e.get("type") == "agent" and e.get("tools"):
            for tool_ref in e["tools"]:
                if tool_ref not in tool_entity_ids:
                    warn(f"Agent '{e.get('id')}' references tool '{tool_ref}' not defined as a tool entity")

    # Rule: spawn templates valid
    for p in processes:
        if p.get("type") == "spawn":
            template = p.get("template")
            if not template:
                err(f"Spawn '{p.get('id')}' missing required 'template'")
            elif template == "self":
                pass  # valid recursive reference
            elif template not in all_nodes:
                # Could be a spec_ref (external file) — warn, don't error
                warn(f"Spawn '{p.get('id')}' template '{template}' is not a node in this spec (may be external spec_ref)")

    # Rule: recursive spawns should document bounds
    for p in processes:
        if p.get("type") == "spawn" and (p.get("recursive") or p.get("template") == "self"):
            if not p.get("max_depth"):
                warn(f"Recursive spawn '{p.get('id')}' should specify max_depth or document natural bounds")

    # Rule: team members must reference valid agent entities
    agent_entity_ids = {e.get("id") for e in entities if e.get("type") == "agent"}
    for e in entities:
        if e.get("type") == "team":
            members = e.get("members", [])
            for member in members:
                if member not in agent_entity_ids:
                    err(f"Team '{e.get('id')}' member '{member}' is not a valid agent entity")
            manager = e.get("manager")
            if manager and manager not in members:
                warn(f"Team '{e.get('id')}' manager '{manager}' is not in the members list")

    # Rule: channel message_schema must resolve
    for e in entities:
        if e.get("type") == "channel":
            msg_schema = e.get("message_schema")
            if msg_schema and msg_schema not in schema_names:
                err(f"Channel '{e.get('id')}' message_schema references unknown schema '{msg_schema}'")

    # Rule: conversation participants must reference valid entities
    for e in entities:
        if e.get("type") == "conversation":
            participants = e.get("participants", [])
            for p_ref in participants:
                if p_ref not in all_nodes:
                    warn(f"Conversation '{e.get('id')}' participant '{p_ref}' references unknown node")

    # Rule: protocol participants must reference valid entities
    for p in processes:
        if p.get("type") == "protocol":
            participants = p.get("participants", [])
            if len(participants) < 2:
                err(f"Protocol '{p.get('id')}' must have at least 2 participants")
            for part in participants:
                entity_ref = part.get("entity")
                if entity_ref and entity_ref not in all_nodes:
                    err(f"Protocol '{p.get('id')}' participant references unknown entity '{entity_ref}'")

    # ════════════════════════════════════════════════════
    # 6. Advanced structural checks
    # ════════════════════════════════════════════════════

    # Build adjacency structures for graph analysis
    # outgoing_flow[pid] = set of process ids reachable via flow/branch/loop edges from pid
    outgoing_flow = {}   # node_id -> set of target node_ids (flow/branch/loop edges)
    incoming_flow = {}   # node_id -> set of source node_ids (flow/branch/loop edges)
    outgoing_flow_only = {}  # node_id -> set of target node_ids (flow edges only)
    invoke_from = {}     # node_id -> set of target entity_ids (invoke edges)
    read_from = {}       # node_id -> set of target store_ids (read edges)
    write_from = {}      # node_id -> set of target store_ids (write edges)

    for e in edges:
        etype = e.get("type")
        efrom = e.get("from")
        eto = e.get("to")
        if not efrom or not eto:
            continue
        if etype in ("flow", "branch", "loop"):
            outgoing_flow.setdefault(efrom, set()).add(eto)
            incoming_flow.setdefault(eto, set()).add(efrom)
        if etype == "flow":
            outgoing_flow_only.setdefault(efrom, set()).add(eto)
        if etype == "invoke":
            invoke_from.setdefault(efrom, set()).add(eto)
        if etype == "read":
            read_from.setdefault(efrom, set()).add(eto)
        if etype == "write":
            write_from.setdefault(efrom, set()).add(eto)

    # Also include gate branch targets declared in process definitions
    # (some specs declare branches in the gate's branches[] array rather
    # than as explicit branch-type edges in the edges list)
    for p in processes:
        pid = p.get("id")
        if not pid:
            continue
        if p.get("type") == "gate":
            for b in p.get("branches", []):
                target = b.get("target")
                if target and target in all_nodes:
                    outgoing_flow.setdefault(pid, set()).add(target)
                    incoming_flow.setdefault(target, set()).add(pid)
        # Also include gate default targets
        default_target = p.get("default")
        if p.get("type") == "gate" and default_target and default_target in all_nodes:
            outgoing_flow.setdefault(pid, set()).add(default_target)
            incoming_flow.setdefault(default_target, set()).add(pid)

    process_ids = {p.get("id") for p in processes if p.get("id")}

    # ── Rule 6a: Unreachable processes ──
    # Starting from entry_point, follow flow/branch/loop edges to find all
    # reachable processes. Any process not reachable produces a WARNING.
    effective_entry = entry_point
    if not effective_entry:
        # Fall back to auto-detected root if no entry_point declared
        all_incoming = set()
        for e in edges:
            if e.get("type") in ("flow", "invoke"):
                all_incoming.add(e.get("to"))
        auto_roots = process_ids - all_incoming
        if len(auto_roots) == 1:
            effective_entry = auto_roots.pop()

    if effective_entry and effective_entry in all_nodes:
        reachable = set()
        frontier = [effective_entry]
        while frontier:
            current = frontier.pop()
            if current in reachable:
                continue
            reachable.add(current)
            for target in outgoing_flow.get(current, set()):
                if target not in reachable:
                    frontier.append(target)
        for pid in sorted(process_ids):
            if pid not in reachable:
                warn(f"Process '{pid}' is unreachable from entry_point '{effective_entry}'")

    # ── Rule 6b: Fan-out without join ──
    # When a process has multiple outgoing flow edges, check that all
    # targets eventually converge to a common downstream process.
    def _find_all_downstream(start_id, adjacency, visited=None):
        """Return all nodes reachable from start_id via adjacency."""
        if visited is None:
            visited = set()
        frontier = [start_id]
        result = set()
        while frontier:
            node = frontier.pop()
            if node in visited:
                continue
            visited.add(node)
            result.add(node)
            for nxt in adjacency.get(node, set()):
                if nxt not in visited:
                    frontier.append(nxt)
        return result

    for pid in process_ids:
        flow_targets = outgoing_flow_only.get(pid, set())
        if len(flow_targets) > 1:
            # This process fans out — check if the branches converge
            downstream_sets = []
            for target in flow_targets:
                ds = _find_all_downstream(target, outgoing_flow)
                downstream_sets.append(ds)
            # Check intersection: is there a common node all branches reach?
            if downstream_sets:
                common = downstream_sets[0]
                for ds in downstream_sets[1:]:
                    common = common & ds
                if not common:
                    warn(f"Process '{pid}' has fan-out flow edges to {sorted(flow_targets)} that never converge to a common downstream process")

    # ── Rule 6c: Empty processes ──
    # If a process has type "step" but has no logic field AND no invoke
    # edges from it AND no read/write edges from it, it's an empty shell.
    for p in processes:
        pid = p.get("id")
        if not pid:
            continue
        if p.get("type") == "step":
            has_logic = bool(p.get("logic"))
            has_invoke = pid in invoke_from
            has_read = pid in read_from
            has_write = pid in write_from
            if not has_logic and not has_invoke and not has_read and not has_write:
                warn(f"Process '{pid}' has no logic, invocations, or store access (empty shell)")

    # ── Rule 6d: Schema field collisions in fan-out ──
    # When a process has multiple outgoing flow edges and the target
    # processes invoke agents with the same output schema field names,
    # warn about potential field collisions.
    schema_map = {s.get("name"): s for s in schemas if s.get("name")}

    def _get_output_field_names(process_id):
        """Get the set of output schema field names for agents invoked by a process."""
        field_names = set()
        for inv_edge in edges:
            if inv_edge.get("type") != "invoke":
                continue
            if inv_edge.get("from") != process_id:
                continue
            output_schema_name = inv_edge.get("output")
            if output_schema_name and output_schema_name in schema_map:
                schema_def = schema_map[output_schema_name]
                for field in schema_def.get("fields", []):
                    fname = field.get("name")
                    if fname:
                        field_names.add((fname, output_schema_name))
        return field_names

    for pid in process_ids:
        flow_targets = outgoing_flow_only.get(pid, set())
        if len(flow_targets) > 1:
            # Collect output fields from each fan-out target's invoke edges
            target_fields = {}  # target_id -> set of (field_name, schema_name)
            for target in flow_targets:
                fields = _get_output_field_names(target)
                if fields:
                    target_fields[target] = fields

            if len(target_fields) >= 2:
                # Check for field name collisions across targets
                all_field_names = {}  # field_name -> list of (target, schema_name)
                for target, fields in target_fields.items():
                    for fname, sname in fields:
                        all_field_names.setdefault(fname, []).append((target, sname))

                for fname, sources in all_field_names.items():
                    if len(sources) >= 2:
                        involved = ", ".join(f"'{t}' (schema '{s}')" for t, s in sources)
                        warn(f"Fan-out from '{pid}': output field '{fname}' produced by multiple targets: {involved} (potential field collision)")

    # ── Rule 6e: Disconnected flow chain ──
    # If process A flows to process B, but B has no outgoing flow/branch/loop
    # edges and is not a terminal node (no _done in logic), produce WARNING.
    for p in processes:
        pid = p.get("id")
        if not pid:
            continue
        # Only check process nodes (not entities)
        if all_nodes.get(pid, {}).get("category") != "process":
            continue
        has_incoming_flow = pid in incoming_flow
        has_outgoing_flow = pid in outgoing_flow
        if has_incoming_flow and not has_outgoing_flow:
            # This process is a dead-end — check if it's intentionally terminal
            logic_text = p.get("logic", "") or ""
            is_terminal = "_done" in logic_text
            if not is_terminal:
                warn(f"Process '{pid}' has incoming flow but no outgoing flow/branch/loop edges and no '_done' in logic (disconnected chain end)")

    return errors, warnings


def _check_edge_constraints(etype, efrom, eto, from_node, to_node, elabel, edge_type_defs, err, warn):
    """Check that edge from/to types match ontology constraints."""
    typedef = edge_type_defs.get(etype, {})
    required = typedef.get("required", {})

    from_constraint = required.get("from", {}).get("type", "")
    to_constraint = required.get("to", {}).get("type", "")

    if from_constraint and from_constraint != "node_ref":
        allowed_from = _parse_type_constraint(from_constraint)
        actual_from = from_node["type"]
        if allowed_from and actual_from not in allowed_from:
            warn(f"Edge '{elabel}' ({etype}): 'from' node '{efrom}' is type '{actual_from}', "
                 f"but {etype} expects from to be {from_constraint}")

    if to_constraint and to_constraint != "node_ref":
        allowed_to = _parse_type_constraint(to_constraint)
        actual_to = to_node["type"]
        if allowed_to and actual_to not in allowed_to:
            warn(f"Edge '{elabel}' ({etype}): 'to' node '{eto}' is type '{actual_to}', "
                 f"but {etype} expects to to be {to_constraint}")


def _parse_type_constraint(constraint):
    """Parse type constraints like 'step | gate | spawn' or 'agent | tool' or 'store'."""
    constraint = constraint.strip('"').strip("'")
    if "|" in constraint:
        return {t.strip() for t in constraint.split("|")}
    return {constraint}


def _check_schema_refs(entities, processes, schema_names, err):
    """Check that all schema references in entities and processes resolve."""
    schema_ref_fields = ["input_schema", "output_schema", "data_in", "data_out", "schema"]
    for e in entities:
        eid = e.get("id", "?")
        for field in schema_ref_fields:
            ref = e.get(field)
            if ref and ref not in schema_names:
                err(f"Entity '{eid}': {field} references unknown schema '{ref}'")
    for p in processes:
        pid = p.get("id", "?")
        for field in schema_ref_fields:
            ref = p.get(field)
            if ref and ref not in schema_names:
                err(f"Process '{pid}': {field} references unknown schema '{ref}'")


def print_results(filepath, errors, warnings):
    name = os.path.basename(filepath)
    total = len(errors) + len(warnings)

    if total == 0:
        print(f"\033[32m✓ {name}: VALID (0 errors, 0 warnings)\033[0m")
        return True

    print(f"\n{'═' * 60}")
    print(f"  {name}")
    print(f"{'═' * 60}")

    if errors:
        print(f"\n  \033[31m{len(errors)} ERROR{'S' if len(errors) != 1 else ''}:\033[0m")
        for e in errors:
            print(f"    \033[31m✗\033[0m {e}")

    if warnings:
        print(f"\n  \033[33m{len(warnings)} WARNING{'S' if len(warnings) != 1 else ''}:\033[0m")
        for w in warnings:
            print(f"    \033[33m⚠\033[0m {w}")

    print()
    return len(errors) == 0


def main():
    # `--all <dir>` matches ao-lint / ao-coverage / ao-complexity / ao-mermaid,
    # and is how .github/workflows/validate.yml has always invoked this module.
    # It was never implemented; the CI job would have failed on its first line
    # had the workflow file been parseable.
    argv = sys.argv[1:]
    if "--all" in argv:
        argv = [a for a in argv if a != "--all"]
        expanded = []
        for target in argv:
            if os.path.isdir(target):
                expanded.extend(sorted(glob.glob(os.path.join(target, "*.yaml"))))
            else:
                expanded.append(target)
        argv = expanded
        if not argv:
            print("No spec files found.")
            sys.exit(1)

    if not argv:
        print(f"Usage: {sys.argv[0]} <spec.yaml> [spec2.yaml ...]")
        print(f"       {sys.argv[0]} --all <specs-dir>")
        sys.exit(1)

    # Load ontology
    try:
        ontology = load_yaml(ONTOLOGY_PATH)
    except Exception as e:
        print(f"\033[31mFailed to load ontology: {e}\033[0m")
        sys.exit(1)

    all_valid = True
    for filepath in argv:
        try:
            spec = load_yaml(filepath)
        except Exception as e:
            print(f"\033[31m✗ {filepath}: Failed to parse YAML: {e}\033[0m")
            all_valid = False
            continue

        errors, warnings = validate_spec(spec, ontology, filepath)
        valid = print_results(filepath, errors, warnings)
        if not valid:
            all_valid = False

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
