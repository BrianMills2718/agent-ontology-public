#!/usr/bin/env python3
"""
Google ADK Importer — Parse Google Agent Development Kit Python files into Agent Ontology YAML specs.

Supports google.adk patterns: Agent/LlmAgent, SequentialAgent, ParallelAgent,
LoopAgent, AgentTool, function tools, google_search, exit_loop, sub_agents composition.

Mapping:
  Agent/LlmAgent       → agent entity + step process
  SequentialAgent       → flow edges between sub_agent steps
  ParallelAgent         → fan-out step invoking all sub_agents
  LoopAgent             → gate + loop edge around sub_agent steps
  tools=[func]          → tool entity + invoke edge
  AgentTool(agent=...)  → invoke edge from step to sub-agent
  google_search         → tool entity (type: search)
  exit_loop             → tool entity (type: function)
  output_key            → step logic (state data flow)
  output_schema         → schema
  Runner(agent=...)     → entry point detection
  sub_agents=[...]      → recursive composition tree

Usage:
  python3 import_google_adk.py agent_file.py -o specs/imported.yaml
  python3 import_google_adk.py agent_file.py --validate
  python3 import_google_adk.py agent_file.py --llm-augment
"""

from __future__ import annotations

import ast
import argparse
import os
import re
import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentInfo:
    name: str
    var_name: str
    agent_type: str  # "llm", "sequential", "parallel", "loop"
    id: str = ""
    model: str = "unknown"
    instruction: str = ""
    description: str = ""
    tool_refs: list[str] = field(default_factory=list)
    sub_agent_refs: list[str] = field(default_factory=list)
    output_key: str = ""
    output_schema: str = ""
    input_schema: str = ""
    max_iterations: int | None = None
    before_agent_callback: str = ""
    after_agent_callback: str = ""


@dataclass
class ToolInfo:
    name: str
    id: str = ""
    description: str = ""
    tool_type: str = "function"
    is_builtin: bool = False


@dataclass
class AgentToolRef:
    """AgentTool(agent=...) reference."""
    var_name: str = ""
    agent_var: str = ""


# ═══════════════════════════════════════════════════════════════
# AST Helpers
# ═══════════════════════════════════════════════════════════════

def _get_str(node) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _get_keyword(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _get_str_keyword(call: ast.Call, name: str) -> str:
    val = _get_keyword(call, name)
    if val is None:
        return ""
    return _get_str(val)


def _get_int_keyword(call: ast.Call, name: str) -> int | None:
    val = _get_keyword(call, name)
    if isinstance(val, ast.Constant) and isinstance(val.value, int):
        return val.value
    return None


def _get_list_names(node) -> list[str]:
    names = []
    if isinstance(node, ast.List):
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Call):
                # AgentTool(agent=...) or FunctionTool(...)
                func = elt.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name:
                    names.append(f"__call__{name}")
    return names


def _sanitize_id(name: str) -> str:
    return re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')


def _snake_to_label(name: str) -> str:
    return name.replace("_", " ").title()


def _get_call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _get_docstring(node) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value.strip()
    return ""


def _annotation_to_type(ann) -> str:
    if isinstance(ann, ast.Name):
        type_map = {
            "str": "string", "int": "integer", "float": "float",
            "bool": "boolean", "list": "list", "dict": "object",
        }
        return type_map.get(ann.id, "string")
    if isinstance(ann, ast.Subscript):
        if isinstance(ann.value, ast.Name):
            if ann.value.id == "list":
                return "list<string>"
            if ann.value.id in ("Optional", "Union"):
                return _annotation_to_type(ann.slice)
        return "string"
    return "string"


# ═══════════════════════════════════════════════════════════════
# Extraction Functions
# ═══════════════════════════════════════════════════════════════

ADK_AGENT_CLASSES = {"Agent", "LlmAgent"}
ADK_WORKFLOW_CLASSES = {
    "SequentialAgent": "sequential",
    "ParallelAgent": "parallel",
    "LoopAgent": "loop",
}
ADK_BUILTIN_TOOLS = {
    "google_search": ("search", "Google web search"),
    "exit_loop": ("function", "Exit the current loop"),
    "transfer_to_agent": ("function", "Transfer to another agent"),
    "google_maps_grounding": ("function", "Google Maps grounding"),
    "url_context": ("function", "URL context retrieval"),
    "load_memory": ("function", "Load memory from session"),
    "preload_memory": ("function", "Preload memory into session"),
    "load_artifacts": ("function", "Load artifacts"),
    "get_user_choice": ("function", "Get user choice input"),
}


def _extract_agents(tree: ast.Module) -> dict[str, AgentInfo]:
    """Extract Agent/LlmAgent/SequentialAgent/ParallelAgent/LoopAgent definitions."""
    agents: dict[str, AgentInfo] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        call_name = _get_call_name(call)

        if call_name in ADK_AGENT_CLASSES:
            agent_type = "llm"
        elif call_name in ADK_WORKFLOW_CLASSES:
            agent_type = ADK_WORKFLOW_CLASSES[call_name]
        else:
            continue

        var_name = node.targets[0].id
        agent_name = _get_str_keyword(call, "name") or var_name
        agent_id = _sanitize_id(agent_name)

        ag = AgentInfo(
            name=agent_name,
            var_name=var_name,
            agent_type=agent_type,
            id=agent_id,
        )

        # model
        model_node = _get_keyword(call, "model")
        if model_node:
            model_str = _get_str(model_node)
            if model_str:
                ag.model = model_str

        # instruction
        inst_node = _get_keyword(call, "instruction")
        if inst_node:
            ag.instruction = _get_str(inst_node)

        # description
        desc = _get_str_keyword(call, "description")
        if desc:
            ag.description = desc

        # tools
        tools_node = _get_keyword(call, "tools")
        if tools_node:
            ag.tool_refs = _get_list_names(tools_node)

        # sub_agents
        sa_node = _get_keyword(call, "sub_agents")
        if sa_node:
            ag.sub_agent_refs = _get_list_names(sa_node)

        # output_key
        ag.output_key = _get_str_keyword(call, "output_key")

        # output_schema
        os_node = _get_keyword(call, "output_schema")
        if isinstance(os_node, ast.Name):
            ag.output_schema = os_node.id

        # input_schema
        is_node = _get_keyword(call, "input_schema")
        if isinstance(is_node, ast.Name):
            ag.input_schema = is_node.id

        # max_iterations (LoopAgent)
        ag.max_iterations = _get_int_keyword(call, "max_iterations")

        # callbacks
        cb = _get_keyword(call, "before_agent_callback")
        if isinstance(cb, ast.Name):
            ag.before_agent_callback = cb.id
        cb = _get_keyword(call, "after_agent_callback")
        if isinstance(cb, ast.Name):
            ag.after_agent_callback = cb.id

        agents[var_name] = ag

    return agents


def _extract_function_tools(tree: ast.Module, agent_tool_refs: set[str]) -> dict[str, ToolInfo]:
    """Extract function definitions that are referenced as tools."""
    functions: dict[str, ToolInfo] = {}

    # Collect all function definitions
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_name = node.name
        # Skip if it's a known callback pattern
        if func_name.startswith("before_") or func_name.startswith("after_") or func_name.startswith("on_"):
            continue

        # Only include if it's referenced in agent_tool_refs
        if func_name not in agent_tool_refs:
            continue

        functions[func_name] = ToolInfo(
            name=func_name,
            id=_sanitize_id(func_name),
            description=_get_docstring(node),
            tool_type="function",
        )

    return functions


def _extract_agent_tools(tree: ast.Module) -> dict[str, AgentToolRef]:
    """Extract AgentTool(agent=...) variable assignments."""
    agent_tools: dict[str, AgentToolRef] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call_name = _get_call_name(node.value)
        if call_name != "AgentTool":
            continue

        var_name = node.targets[0].id
        agent_var = ""
        ag_node = _get_keyword(node.value, "agent")
        if isinstance(ag_node, ast.Name):
            agent_var = ag_node.id
        elif node.value.args and isinstance(node.value.args[0], ast.Name):
            agent_var = node.value.args[0].id

        if agent_var:
            agent_tools[var_name] = AgentToolRef(var_name=var_name, agent_var=agent_var)

    return agent_tools


def _extract_output_types(tree: ast.Module) -> dict[str, list[dict]]:
    """Extract Pydantic BaseModel class definitions."""
    schemas: dict[str, list[dict]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        is_schema = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ("BaseModel", "TypedDict"):
                is_schema = True
            elif isinstance(base, ast.Attribute) and base.attr in ("BaseModel", "TypedDict"):
                is_schema = True

        if not is_schema:
            continue

        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fname = item.target.id
                ftype = _annotation_to_type(item.annotation)
                fields.append({"name": fname, "type": ftype})

        if fields:
            schemas[node.name] = fields

    return schemas


def _find_runner_agent(tree: ast.Module) -> str | None:
    """Find the agent passed to Runner(agent=...)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _get_call_name(node)
        if call_name != "Runner":
            continue
        ag_node = _get_keyword(node, "agent")
        if isinstance(ag_node, ast.Name):
            return ag_node.id
    return None


# ═══════════════════════════════════════════════════════════════
# Recursive Agent Tree → Spec
# ═══════════════════════════════════════════════════════════════

def _flatten_agent_tree(
    var_name: str,
    agents: dict[str, AgentInfo],
    all_tools: dict[str, ToolInfo],
    agent_tool_map: dict[str, AgentToolRef],
    entities: list[dict],
    processes: list[dict],
    edges: list[dict],
    entity_ids: set[str],
    process_ids: set[str],
    prefix: str = "",
) -> tuple[str | None, str | None]:
    """Recursively flatten an agent composition tree into spec entities/processes/edges.
    Returns (entry_step_id, exit_step_id) for this subtree.
    For loops, exit_step_id is the gate ID (so sequential can wire the 'done' branch)."""

    if var_name not in agents:
        return None, None

    ag = agents[var_name]
    step_prefix = f"{prefix}{ag.id}" if prefix else ag.id

    if ag.agent_type == "llm":
        # Create agent entity
        if ag.id not in entity_ids:
            entity: dict[str, Any] = {
                "id": ag.id,
                "type": "agent",
                "label": _snake_to_label(ag.name),
                "model": ag.model,
            }
            if ag.instruction:
                entity["system_prompt"] = ag.instruction
            if ag.description:
                entity["description"] = ag.description
            if ag.output_schema:
                entity["output_schema"] = ag.output_schema
            entities.append(entity)
            entity_ids.add(ag.id)

        # Create step process
        step_id = f"run_{step_prefix}"
        if step_id not in process_ids:
            logic_lines = []
            if ag.output_key:
                logic_lines.append(f'state.data["{ag.output_key}"] = state.data.get("output_text", "")')
            logic = "\n".join(logic_lines) + "\n" if logic_lines else ""

            processes.append({
                "id": step_id,
                "type": "step",
                "label": f"Run {_snake_to_label(ag.name)}",
                **({"logic": logic} if logic else {}),
            })
            process_ids.add(step_id)

            # Invoke agent
            edges.append({
                "type": "invoke",
                "from": step_id,
                "to": ag.id,
                "label": f"Call {_snake_to_label(ag.name)}",
            })

            # Wire tools
            for tref in ag.tool_refs:
                if tref.startswith("__call__AgentTool"):
                    continue

                if tref in agent_tool_map:
                    at = agent_tool_map[tref]
                    target_id = agents[at.agent_var].id if at.agent_var in agents else _sanitize_id(at.agent_var)
                    edges.append({
                        "type": "invoke",
                        "from": step_id,
                        "to": target_id,
                        "label": f"Delegate to {_snake_to_label(at.agent_var)}",
                    })
                elif tref in all_tools:
                    edges.append({
                        "type": "invoke",
                        "from": step_id,
                        "to": all_tools[tref].id,
                        "label": f"Use {_snake_to_label(tref)}",
                    })

            # Wire sub_agents as handoff targets (for LLM agents with sub_agents)
            for sa_ref in ag.sub_agent_refs:
                if sa_ref in agents:
                    sa = agents[sa_ref]
                    if sa.id not in entity_ids:
                        _flatten_agent_tree(
                            sa_ref, agents, all_tools, agent_tool_map,
                            entities, processes, edges, entity_ids, process_ids, prefix
                        )
                    edges.append({
                        "type": "handoff",
                        "from": ag.id,
                        "to": sa.id,
                        "label": f"Delegate to {_snake_to_label(sa.name)}",
                    })

        return step_id, step_id  # entry == exit for simple steps

    elif ag.agent_type == "sequential":
        entries_exits: list[tuple[str | None, str | None]] = []
        for sa_ref in ag.sub_agent_refs:
            entry, exit_ = _flatten_agent_tree(
                sa_ref, agents, all_tools, agent_tool_map,
                entities, processes, edges, entity_ids, process_ids, prefix
            )
            if entry:
                entries_exits.append((entry, exit_))

        # Wire flow from exit of each step to entry of next
        for i in range(len(entries_exits) - 1):
            _, prev_exit = entries_exits[i]
            next_entry, _ = entries_exits[i + 1]
            if prev_exit and next_entry:
                # If prev_exit is a gate (loop exit), update its "done" branch target
                is_gate = any(p["id"] == prev_exit and p["type"] == "gate" for p in processes)
                if is_gate:
                    # Rewire the gate's "done" branch and its branch edge
                    for p in processes:
                        if p["id"] == prev_exit and p["type"] == "gate":
                            for br in p.get("branches", []):
                                if br.get("condition") == "done":
                                    br["target"] = next_entry
                            if p.get("default") == "_done_placeholder":
                                p["default"] = next_entry
                    for e in edges:
                        if (e.get("type") == "branch" and e.get("from") == prev_exit
                                and e.get("to") == "_done_placeholder"):
                            e["to"] = next_entry
                else:
                    edges.append({
                        "type": "flow",
                        "from": prev_exit,
                        "to": next_entry,
                    })

        first_entry = entries_exits[0][0] if entries_exits else None
        last_exit = entries_exits[-1][1] if entries_exits else None
        return first_entry, last_exit

    elif ag.agent_type == "parallel":
        # Fan-out step invokes all sub_agents directly (no individual step processes)
        fan_out_id = f"fanout_{step_prefix}"
        if fan_out_id not in process_ids:
            processes.append({
                "id": fan_out_id,
                "type": "step",
                "label": f"Parallel: {_snake_to_label(ag.name)}",
            })
            process_ids.add(fan_out_id)

        for sa_ref in ag.sub_agent_refs:
            if sa_ref in agents:
                sa = agents[sa_ref]
                # Create agent entity if needed
                if sa.id not in entity_ids:
                    entity = {
                        "id": sa.id,
                        "type": "agent",
                        "label": _snake_to_label(sa.name),
                        "model": sa.model,
                    }
                    if sa.instruction:
                        entity["system_prompt"] = sa.instruction
                    if sa.description:
                        entity["description"] = sa.description
                    if sa.output_schema:
                        entity["output_schema"] = sa.output_schema
                    entities.append(entity)
                    entity_ids.add(sa.id)

                edges.append({
                    "type": "invoke",
                    "from": fan_out_id,
                    "to": sa.id,
                    "label": f"Call {_snake_to_label(sa.name)}",
                })

                # Wire tools for sub-agents
                for tref in sa.tool_refs:
                    if tref in all_tools:
                        edges.append({
                            "type": "invoke",
                            "from": fan_out_id,
                            "to": all_tools[tref].id,
                            "label": f"Use {_snake_to_label(tref)}",
                        })

        return fan_out_id, fan_out_id

    elif ag.agent_type == "loop":
        sub_entries_exits: list[tuple[str | None, str | None]] = []
        for sa_ref in ag.sub_agent_refs:
            entry, exit_ = _flatten_agent_tree(
                sa_ref, agents, all_tools, agent_tool_map,
                entities, processes, edges, entity_ids, process_ids, prefix
            )
            if entry:
                sub_entries_exits.append((entry, exit_))

        if not sub_entries_exits:
            return None, None

        # Wire flow between loop steps
        for i in range(len(sub_entries_exits) - 1):
            _, prev_exit = sub_entries_exits[i]
            next_entry, _ = sub_entries_exits[i + 1]
            if prev_exit and next_entry:
                edges.append({
                    "type": "flow",
                    "from": prev_exit,
                    "to": next_entry,
                })

        first_entry = sub_entries_exits[0][0]
        last_exit = sub_entries_exits[-1][1]

        # Gate at end of loop
        gate_id = f"check_{step_prefix}"
        max_iter = ag.max_iterations or 10
        condition = f"iteration < {max_iter}"

        processes.append({
            "id": gate_id,
            "type": "gate",
            "label": f"Check {_snake_to_label(ag.name)}",
            "condition": condition,
            "branches": [
                {"condition": "continue", "target": first_entry},
                {"condition": "done", "target": "_done_placeholder"},
            ],
            "default": "_done_placeholder",
        })
        process_ids.add(gate_id)

        # Flow from last step to gate
        edges.append({
            "type": "flow",
            "from": last_exit,
            "to": gate_id,
        })

        # Loop edge
        edges.append({
            "type": "loop",
            "from": gate_id,
            "to": first_entry,
            "condition": "continue",
        })

        # Branch edge (placeholder, will be rewired by sequential parent)
        edges.append({
            "type": "branch",
            "from": gate_id,
            "to": "_done_placeholder",
            "condition": "done",
        })

        return first_entry, gate_id  # exit is the gate (sequential parent rewires "done" branch)

    return None, None


# ═══════════════════════════════════════════════════════════════
# Spec Assembly
# ═══════════════════════════════════════════════════════════════

def import_google_adk(source_path: Path) -> dict:
    """Parse a Google ADK Python file and produce an Agent Ontology YAML spec."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    agents = _extract_agents(tree)
    agent_tool_refs_map = _extract_agent_tools(tree)
    output_schemas = _extract_output_types(tree)

    # Collect all tool references from all agents
    all_tool_refs: set[str] = set()
    for ag in agents.values():
        for tref in ag.tool_refs:
            if not tref.startswith("__call__"):
                all_tool_refs.add(tref)

    # Identify built-in tools vs function tools
    builtin_tools: dict[str, ToolInfo] = {}
    for tref in all_tool_refs:
        if tref in ADK_BUILTIN_TOOLS:
            ttype, desc = ADK_BUILTIN_TOOLS[tref]
            builtin_tools[tref] = ToolInfo(
                name=tref, id=_sanitize_id(tref),
                description=desc, tool_type=ttype, is_builtin=True,
            )

    # Extract function tools (only functions referenced in tools=[])
    func_tool_refs = all_tool_refs - set(builtin_tools.keys()) - set(agent_tool_refs_map.keys())
    func_tools = _extract_function_tools(tree, func_tool_refs)

    # Merge all tools
    all_tools: dict[str, ToolInfo] = {}
    all_tools.update(builtin_tools)
    all_tools.update(func_tools)

    # Find root agent
    runner_agent_var = _find_runner_agent(tree)
    root_var = None

    # Priority: Runner(agent=...) > variable named root_agent > first agent defined
    if runner_agent_var and runner_agent_var in agents:
        root_var = runner_agent_var
    elif "root_agent" in agents:
        root_var = "root_agent"
    else:
        # Find agent that is not a sub_agent of any other agent
        all_sub_refs = set()
        for ag in agents.values():
            for sa_ref in ag.sub_agent_refs:
                all_sub_refs.add(sa_ref)
        top_level = [v for v in agents if v not in all_sub_refs]
        if top_level:
            root_var = top_level[0]
        elif agents:
            root_var = list(agents.keys())[0]

    # Build spec
    spec_name = source_path.stem.replace("_", " ").title()
    spec: dict[str, Any] = {
        "name": spec_name,
        "version": "1.0",
        "description": f"Imported from Google ADK file: {source_path.name}",
        "entities": [],
        "processes": [],
        "edges": [],
        "schemas": [],
    }

    entities = spec["entities"]
    processes = spec["processes"]
    edges = spec["edges"]
    schemas = spec["schemas"]

    # Add tool entities
    for tname, tool in all_tools.items():
        entities.append({
            "id": tool.id,
            "type": "tool",
            "label": _snake_to_label(tname),
            "tool_type": tool.tool_type,
            **({"description": tool.description} if tool.description else {}),
        })

    # Flatten agent tree
    entity_ids: set[str] = {e["id"] for e in entities}
    process_ids: set[str] = set()
    entry_step = None
    if root_var:
        entry_exit = _flatten_agent_tree(
            root_var, agents, all_tools, agent_tool_refs_map,
            entities, processes, edges, entity_ids, process_ids,
        )
        if entry_exit:
            entry_step = entry_exit[0]

    # Add output schemas
    for class_name, fields in output_schemas.items():
        schemas.append({"name": class_name, "fields": fields})

    # Terminal step
    if processes:
        done_id = "_done"
        processes.append({
            "id": done_id,
            "type": "step",
            "label": "Done",
            "logic": 'state.data["_done"] = True\n',
        })

        # Replace _done_placeholder references in gates and edges
        for p in processes:
            if p.get("type") == "gate":
                for br in p.get("branches", []):
                    if br.get("target") == "_done_placeholder":
                        br["target"] = done_id
                if p.get("default") == "_done_placeholder":
                    p["default"] = done_id

        for e in edges:
            if e.get("to") == "_done_placeholder":
                e["to"] = done_id

        # Connect last step to done (if not a loop)
        non_gate_steps = [p["id"] for p in processes
                          if p["type"] == "step" and p["id"] != done_id]
        if non_gate_steps:
            # Find steps with no outgoing flow/branch/loop edges
            has_outgoing = set()
            for e in edges:
                if e["type"] in ("flow", "branch", "loop"):
                    has_outgoing.add(e["from"])
            terminal_steps = [s for s in non_gate_steps if s not in has_outgoing]
            for ts in terminal_steps:
                edges.append({
                    "type": "flow",
                    "from": ts,
                    "to": done_id,
                    "label": "Done",
                })

    # Entry point
    if entry_step:
        spec["entry_point"] = entry_step
    elif processes:
        spec["entry_point"] = processes[0]["id"]

    # Default schema
    if not schemas:
        schemas.append({
            "name": "AgentOutput",
            "fields": [
                {"name": "output", "type": "string", "description": "Agent output text"},
            ],
        })

    # Deduplicate edges
    seen: set[tuple] = set()
    unique_edges = []
    for e in edges:
        key = (e["type"], e["from"], e["to"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)
    spec["edges"] = unique_edges

    return spec


# ═══════════════════════════════════════════════════════════════
# LLM Augmentation
# ═══════════════════════════════════════════════════════════════

def llm_augment_spec(source: str, skeleton_spec: dict, model: str = "gemini-2.0-flash") -> dict:
    """Use LLM to enrich the AST-extracted skeleton."""
    try:
        from .import_langgraph import llm_augment_spec as _augment
        return _augment(source, skeleton_spec, model=model)
    except ImportError:
        return skeleton_spec


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Import a Google ADK Python file into an Agent Ontology YAML spec"
    )
    parser.add_argument("input", help="Path to Google ADK Python file")
    parser.add_argument("-o", "--output", help="Output YAML file path (default: stdout)")
    parser.add_argument("--validate", action="store_true", help="Validate the output spec")
    parser.add_argument("--quiet", action="store_true", help="Suppress info messages")
    parser.add_argument("--llm-augment", action="store_true",
                        help="Use LLM to enrich the AST skeleton")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Model for LLM augmentation")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        spec = import_google_adk(input_path)
    except SyntaxError as exc:
        print(f"Error: {input_path} is not valid Python: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.llm_augment:
        if not args.quiet:
            print("LLM-augmenting spec...", file=sys.stderr)
        source = input_path.read_text(encoding="utf-8")
        spec = llm_augment_spec(source, spec, model=args.model)

    n_ent = len(spec.get("entities", []))
    n_proc = len(spec.get("processes", []))
    n_edge = len(spec.get("edges", []))
    n_schema = len(spec.get("schemas", []))

    if not args.quiet:
        print(f"Imported: {n_ent} entities, {n_proc} processes, {n_edge} edges, {n_schema} schemas",
              file=sys.stderr)

    yaml_str = yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    if args.output:
        Path(args.output).write_text(yaml_str, encoding="utf-8")
        if not args.quiet:
            print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(yaml_str)

    if args.validate:
        try:
            from .validate import validate_spec, load_yaml
            ontology_path = os.path.join(SCRIPT_DIR, "ONTOLOGY.yaml")
            ontology = load_yaml(ontology_path)
            errors, warnings = validate_spec(spec, ontology, str(input_path))
            if errors:
                print(f"\n  {len(errors)} validation error(s):", file=sys.stderr)
                for e in errors:
                    print(f"    \u2717 {e}", file=sys.stderr)
            if warnings:
                print(f"\n  {len(warnings)} warning(s):", file=sys.stderr)
                for w in warnings:
                    print(f"    \u26a0 {w}", file=sys.stderr)
            if not errors:
                print(f"  \u2713 Valid spec", file=sys.stderr)
        except ImportError:
            print("Warning: validate.py not found, skipping validation", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
