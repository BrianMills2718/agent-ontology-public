#!/usr/bin/env python3
"""
Compose Operator — Build agent specs by combining patterns.

Takes a declarative compose spec (YAML) that lists patterns and wiring,
and produces a valid full agent spec.

Usage:
    python3 compose.py compose_specs/react_refine.yaml -o specs/react_refine.yaml
    python3 compose.py compose_specs/react_refine.yaml --dry-run
"""

import argparse
import copy
import os
import re
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from .patterns import get_pattern, PATTERN_LIBRARY, compatible_patterns


# ════════════════════════════════════════════════════════════════════
# YAML formatting helpers (shared with mutate.py)
# ════════════════════════════════════════════════════════════════════

class LiteralStr(str):
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralStr, _literal_representer)


def _preserve_multiline(obj):
    if isinstance(obj, dict):
        return {k: _preserve_multiline(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_preserve_multiline(v) for v in obj]
    elif isinstance(obj, str) and "\n" in obj:
        return LiteralStr(obj)
    return obj


def dump_spec(spec):
    tagged = _preserve_multiline(spec)
    return yaml.dump(tagged, default_flow_style=False, sort_keys=False,
                     width=120, allow_unicode=True)


# ════════════════════════════════════════════════════════════════════
# Namespacing
# ════════════════════════════════════════════════════════════════════

def _namespace_id(prefix, original_id):
    """Namespace an ID: 'think_or_act' -> 'rl_think_or_act'."""
    return f"{prefix}_{original_id}"


def _namespace_schema(prefix, schema_name):
    """Namespace a schema name: 'ReActStep' -> 'RlReActStep'."""
    cap_prefix = prefix.capitalize()
    return f"{cap_prefix}{schema_name}"


def _namespace_pattern(pattern, prefix):
    """Create a namespaced copy of a pattern. All IDs get prefixed.

    Returns a new pattern dict with all internal references updated.
    """
    p = copy.deepcopy(pattern)

    # Build ID mappings
    proc_map = {}  # old_id -> new_id
    for proc in p["processes"]:
        old_id = proc["id"]
        new_id = _namespace_id(prefix, old_id)
        proc_map[old_id] = new_id

    entity_map = {}
    for ent in p["entities"]:
        old_id = ent["id"]
        new_id = _namespace_id(prefix, old_id)
        entity_map[old_id] = new_id

    schema_map = {}
    for schema in p["schemas"]:
        old_name = schema["name"]
        new_name = _namespace_schema(prefix, old_name)
        schema_map[old_name] = new_name

    id_map = {**proc_map, **entity_map}

    # Apply to processes
    for proc in p["processes"]:
        proc["id"] = proc_map[proc["id"]]
        if proc.get("label"):
            proc["label"] = f"[{prefix}] {proc['label']}"
        # Update schema refs
        for field in ("data_in", "data_out", "schema"):
            if proc.get(field) and proc[field] in schema_map:
                proc[field] = schema_map[proc[field]]
        # Update gate branch targets
        if proc.get("branches"):
            for branch in proc["branches"]:
                old_target = branch.get("target")
                if old_target and old_target in proc_map:
                    branch["target"] = proc_map[old_target]

    # Apply to entities
    for ent in p["entities"]:
        ent["id"] = entity_map[ent["id"]]
        if ent.get("label"):
            ent["label"] = f"[{prefix}] {ent['label']}"
        for field in ("input_schema", "output_schema", "schema"):
            if ent.get(field) and ent[field] in schema_map:
                ent[field] = schema_map[ent[field]]

    # Apply to edges
    for edge in p["edges"]:
        if edge.get("from") in id_map:
            edge["from"] = id_map[edge["from"]]
        if edge.get("to") in id_map:
            edge["to"] = id_map[edge["to"]]
        for field in ("data", "input", "output"):
            if edge.get(field) and edge[field] in schema_map:
                edge[field] = schema_map[edge[field]]

    # Apply to schemas
    for schema in p["schemas"]:
        schema["name"] = schema_map[schema["name"]]
        # Update field types that reference other schemas
        for field in schema.get("fields", []):
            ftype = field.get("type", "")
            for old_sname, new_sname in schema_map.items():
                if old_sname in ftype:
                    field["type"] = ftype.replace(old_sname, new_sname)

    # Update interface
    iface = p["interface"]
    if iface["entry"] in proc_map:
        iface["entry"] = proc_map[iface["entry"]]
    iface["exits"] = [proc_map.get(e, e) for e in iface["exits"]]

    return p


# ════════════════════════════════════════════════════════════════════
# Compose
# ════════════════════════════════════════════════════════════════════

def compose(compose_spec):
    """Compose a full agent spec from a compose spec.

    The compose spec format:
        name: "Agent Name"
        description: "..."
        version: "1.0"
        model: "gemini-3-flash-preview"
        compose:
          - pattern: reasoning_loop
            as: rl
            config: {max_steps: 10}
          - pattern: critique_cycle
            as: cc
            graft_after: rl  # Connect after rl's exit(s)
            config: {max_rounds: 3}

    Returns a full agent spec dict.
    """
    name = compose_spec.get("name", "Composed Agent")
    description = compose_spec.get("description", "")
    version = compose_spec.get("version", "1.0")
    model = compose_spec.get("model", "gemini-3-flash-preview")
    blocks = compose_spec.get("compose", [])

    if not blocks:
        raise ValueError("Compose spec has no 'compose' blocks")

    # Phase 1: Load and namespace all patterns
    namespaced = []
    for block in blocks:
        pattern_name = block["pattern"]
        prefix = block.get("as", pattern_name)
        config = block.get("config", {})

        pattern = get_pattern(pattern_name)

        # Apply config overrides to process logic
        _apply_config(pattern, config)

        # Namespace the pattern
        ns_pattern = _namespace_pattern(pattern, prefix)
        namespaced.append({
            "block": block,
            "pattern": ns_pattern,
            "prefix": prefix,
            "pattern_name": pattern_name,
        })

    # Phase 2: Build the full spec
    all_processes = []
    all_edges = []
    all_entities = []
    all_schemas = []
    seen_schema_names = set()
    seen_entity_ids = set()

    # Create boilerplate entry step
    first_pattern = namespaced[0]["pattern"]
    first_inputs = PATTERN_LIBRARY[namespaced[0]["pattern_name"]]["interface"]["inputs"]

    receive_step = {
        "id": "receive_input",
        "type": "step",
        "label": "Receive Input",
        "description": "Accept input and initialize state",
        "logic": _make_receive_logic(first_inputs),
    }
    all_processes.append(receive_step)

    # Add user entity
    user_entity = {"id": "user", "type": "human", "label": "User"}
    all_entities.append(user_entity)
    seen_entity_ids.add("user")

    # Add user->receive flow
    all_edges.append({
        "type": "flow",
        "from": "user",
        "to": "receive_input",
        "label": "User input",
    })

    # Connect receive_input to first pattern's entry
    all_edges.append({
        "type": "flow",
        "from": "receive_input",
        "to": first_pattern["interface"]["entry"],
        "label": "Start processing",
    })

    # Collect all pattern components
    for i, ns in enumerate(namespaced):
        pat = ns["pattern"]
        all_processes.extend(pat["processes"])

        for ent in pat["entities"]:
            eid = ent["id"]
            if eid not in seen_entity_ids:
                # Apply model override
                if ent.get("type") == "agent" and model:
                    ent["model"] = model
                all_entities.append(ent)
                seen_entity_ids.add(eid)

        for schema in pat["schemas"]:
            sname = schema["name"]
            if sname not in seen_schema_names:
                all_schemas.append(schema)
                seen_schema_names.add(sname)

        all_edges.extend(pat["edges"])

    # Phase 3: Wire patterns together via graft_after
    for i, ns in enumerate(namespaced):
        block = ns["block"]
        graft_after = block.get("graft_after")
        if not graft_after:
            continue

        # Resolve the graft target: can be "prefix" (connect after that pattern's exits)
        # or "prefix.process_id" (connect after a specific process)
        target_prefix = graft_after.split(".")[0] if "." in graft_after else graft_after

        # Find the target namespaced pattern
        target_ns = None
        for prev_ns in namespaced[:i]:
            if prev_ns["prefix"] == target_prefix:
                target_ns = prev_ns
                break

        if not target_ns:
            raise ValueError(f"graft_after '{graft_after}' references unknown prefix '{target_prefix}'")

        target_pat = target_ns["pattern"]
        current_pat = ns["pattern"]

        if "." in graft_after:
            # Specific process: "prefix.process_id"
            specific_proc = graft_after.split(".", 1)[1]
            exit_processes = [_namespace_id(target_prefix, specific_proc)]
        else:
            # All exits of the target pattern
            exit_processes = target_pat["interface"]["exits"]

        # Wire each exit to the current pattern's entry
        for exit_pid in exit_processes:
            # Check if exit process sets _done — if so, remove that from logic
            _remove_done_flag(all_processes, exit_pid)

            all_edges.append({
                "type": "flow",
                "from": exit_pid,
                "to": current_pat["interface"]["entry"],
                "label": f"Graft: {target_prefix} -> {ns['prefix']}",
            })

    # Phase 4: Create finalize step connected to last pattern's exits
    last_pat = namespaced[-1]["pattern"]
    last_pattern_name = namespaced[-1]["pattern_name"]
    last_outputs = PATTERN_LIBRARY[last_pattern_name]["interface"]["outputs"]

    finalize_step = {
        "id": "finalize",
        "type": "step",
        "label": "Finalize",
        "description": "Produce final output",
        "logic": _make_finalize_logic(last_outputs),
    }
    all_processes.append(finalize_step)

    for exit_pid in last_pat["interface"]["exits"]:
        _remove_done_flag(all_processes, exit_pid)
        all_edges.append({
            "type": "flow",
            "from": exit_pid,
            "to": "finalize",
            "label": "Finalize output",
        })

    # Build final spec
    spec = {
        "name": name,
        "version": version,
        "description": description,
        "entry_point": "receive_input",
        "entities": all_entities,
        "processes": all_processes,
        "edges": all_edges,
        "schemas": all_schemas,
    }

    return spec


def _apply_config(pattern, config):
    """Apply config overrides to pattern process logic blocks."""
    for key, value in config.items():
        for proc in pattern["processes"]:
            logic = proc.get("logic", "")
            if not logic:
                continue
            # Replace default assignments like: state.data["max_steps"] = 10
            # with the override value
            old_pattern = re.compile(
                rf'(state\.data\["{key}"\]\s*=\s*(?:state\.data\.get\("{key}",\s*)?)(\d+|True|False)',
            )
            match = old_pattern.search(logic)
            if match:
                proc["logic"] = old_pattern.sub(
                    rf'\g<1>{value}', logic
                )


def _remove_done_flag(processes, pid):
    """Remove state.data['_done'] = True from a process's logic."""
    for proc in processes:
        if proc.get("id") != pid:
            continue
        logic = proc.get("logic", "")
        if not logic:
            return
        lines = logic.split("\n")
        filtered = [l for l in lines if '_done' not in l]
        proc["logic"] = "\n".join(filtered)


def _make_receive_logic(input_fields):
    """Generate logic for the boilerplate receive_input step."""
    lines = ['print(f"    Starting composed agent...")']
    for field in input_fields:
        lines.append(f'print(f"    {field}: {{state.data.get(\'{field}\', \'\')[:100]}}")')
    return "\n".join(lines) + "\n"


def _make_finalize_logic(output_fields):
    """Generate logic for the boilerplate finalize step."""
    lines = ['print(f"    Finalizing output...")']
    for field in output_fields:
        lines.append(
            f'print(f"    {field}: {{str(state.data.get(\'{field}\', \'\'))[:200]}}")'
        )
    lines.append('state.data["_done"] = True')
    return "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compose agent specs from patterns",
        epilog=(
            "Example:\n"
            "  python3 compose.py compose_specs/react_refine.yaml -o specs/react_refine.yaml\n"
            "  python3 compose.py compose_specs/react_refine.yaml --dry-run\n"
        ),
    )
    parser.add_argument("compose_spec", help="Path to the compose spec YAML")
    parser.add_argument("-o", "--output", help="Output file path for the composed spec")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print composed spec to stdout without writing")
    parser.add_argument("--validate", action="store_true",
                        help="Validate the composed spec after generation")

    args = parser.parse_args()

    with open(args.compose_spec) as f:
        cspec = yaml.safe_load(f)

    spec = compose(cspec)
    output = dump_spec(spec)

    if args.dry_run or not args.output:
        print(output)
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote composed spec to {args.output}")

        proc_count = len(spec.get("processes", []))
        ent_count = len(spec.get("entities", []))
        edge_count = len(spec.get("edges", []))
        schema_count = len(spec.get("schemas", []))
        print(f"  {proc_count} processes, {ent_count} entities, "
              f"{edge_count} edges, {schema_count} schemas")

    if args.validate:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(output)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "validate.py"), tmp_path],
                capture_output=True, text=True, cwd=SCRIPT_DIR,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
