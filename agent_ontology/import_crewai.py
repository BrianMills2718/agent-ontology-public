#!/usr/bin/env python3
"""
CrewAI Importer — Parse CrewAI Python files into Agent Ontology YAML specs.

Extracts Agent, Task, Crew, and optionally Flow definitions from CrewAI code
using AST analysis, producing valid Agent Ontology YAML specs.

Mapping:
  CrewAI Agent   → agent entity (role→label, goal→description, tools→tool entities)
  CrewAI Task    → step process (description→logic prompt, expected_output→output schema)
  CrewAI Crew    → team entity + flow/invoke edges
  Process.sequential → flow edges between tasks in order
  Process.hierarchical → team entity with strategy: hierarchical + manager
  Task.context   → data flow edges (invoke with return_schema)
  Task.tools     → tool entities + invoke edges
  Flow @start/@listen/@router → step/gate processes + flow/branch edges

Usage:
  python3 import_crewai.py crew_file.py -o specs/imported_crew.yaml
  python3 import_crewai.py crew_file.py --validate
  python3 import_crewai.py crew_file.py -o specs/my_crew.yaml --validate
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# AST Extraction Helpers
# ═══════════════════════════════════════════════════════════════

def _get_str(node: ast.expr) -> str:
    """Extract a string from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        # f-string — extract literal parts
        parts = []
        for val in node.values:
            if isinstance(val, ast.Constant):
                parts.append(str(val.value))
            else:
                parts.append("{...}")
        return "".join(parts)
    return ""


def _get_keyword(call: ast.Call, name: str) -> ast.expr | None:
    """Get a keyword argument value from a Call node."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _get_str_keyword(call: ast.Call, name: str) -> str:
    """Get a string keyword argument from a Call node."""
    val = _get_keyword(call, name)
    if val:
        return _get_str(val)
    return ""


def _get_bool_keyword(call: ast.Call, name: str) -> bool | None:
    """Get a boolean keyword argument from a Call node."""
    val = _get_keyword(call, name)
    if isinstance(val, ast.Constant) and isinstance(val.value, bool):
        return val.value
    return None


def _get_list_names(node: ast.expr) -> list[str]:
    """Extract a list of variable names/strings from a List node."""
    names: list[str] = []
    if isinstance(node, ast.List):
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.append(elt.value)
    return names


def _to_snake(name: str) -> str:
    """Convert CamelCase or mixed to snake_case."""
    s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s2 = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s1)
    return re.sub(r'[\s\-]+', '_', s2).lower()


# ═══════════════════════════════════════════════════════════════
# CrewAI Component Extraction
# ═══════════════════════════════════════════════════════════════

class CrewAgent:
    def __init__(self, var_name: str, role: str = "", goal: str = "",
                 backstory: str = "", tools: list[str] | None = None,
                 llm: str = "", verbose: bool = True,
                 allow_delegation: bool = False):
        self.var_name = var_name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.llm = llm
        self.verbose = verbose
        self.allow_delegation = allow_delegation


class CrewTask:
    def __init__(self, var_name: str, description: str = "",
                 expected_output: str = "", agent_var: str = "",
                 tools: list[str] | None = None,
                 context_vars: list[str] | None = None,
                 async_execution: bool = False,
                 human_input: bool = False,
                 output_pydantic: str = ""):
        self.var_name = var_name
        self.description = description
        self.expected_output = expected_output
        self.agent_var = agent_var
        self.tools = tools or []
        self.context_vars = context_vars or []
        self.async_execution = async_execution
        self.human_input = human_input
        self.output_pydantic = output_pydantic


class CrewDef:
    def __init__(self, var_name: str = "", agent_vars: list[str] | None = None,
                 task_vars: list[str] | None = None,
                 process: str = "sequential",
                 manager_llm: str = "", manager_agent_var: str = "",
                 verbose: bool = True, name: str = ""):
        self.var_name = var_name
        self.agent_vars = agent_vars or []
        self.task_vars = task_vars or []
        self.process = process
        self.manager_llm = manager_llm
        self.manager_agent_var = manager_agent_var
        self.verbose = verbose
        self.name = name


def _extract_agent(node: ast.Assign | ast.AnnAssign, call: ast.Call) -> CrewAgent | None:
    """Extract a CrewAI Agent from an AST assignment."""
    if isinstance(node, ast.Assign) and node.targets:
        var_name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        var_name = node.target.id
    else:
        return None

    agent = CrewAgent(var_name=var_name)
    agent.role = _get_str_keyword(call, "role")
    agent.goal = _get_str_keyword(call, "goal")
    agent.backstory = _get_str_keyword(call, "backstory")
    agent.llm = _get_str_keyword(call, "llm")
    verbose = _get_bool_keyword(call, "verbose")
    agent.verbose = verbose if verbose is not None else True
    delegation = _get_bool_keyword(call, "allow_delegation")
    agent.allow_delegation = delegation if delegation is not None else False

    tools_node = _get_keyword(call, "tools")
    if tools_node:
        agent.tools = _get_list_names(tools_node)

    return agent


def _extract_task(node: ast.Assign | ast.AnnAssign, call: ast.Call) -> CrewTask | None:
    """Extract a CrewAI Task from an AST assignment."""
    if isinstance(node, ast.Assign) and node.targets:
        var_name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        var_name = node.target.id
    else:
        return None

    task = CrewTask(var_name=var_name)
    task.description = _get_str_keyword(call, "description")
    task.expected_output = _get_str_keyword(call, "expected_output")
    task.async_execution = _get_bool_keyword(call, "async_execution") or False
    task.human_input = _get_bool_keyword(call, "human_input") or False
    task.output_pydantic = _get_str_keyword(call, "output_pydantic")

    agent_node = _get_keyword(call, "agent")
    if isinstance(agent_node, ast.Name):
        task.agent_var = agent_node.id

    tools_node = _get_keyword(call, "tools")
    if tools_node:
        task.tools = _get_list_names(tools_node)

    context_node = _get_keyword(call, "context")
    if context_node:
        task.context_vars = _get_list_names(context_node)

    return task


def _extract_crew(node: ast.Assign | ast.AnnAssign, call: ast.Call) -> CrewDef | None:
    """Extract a CrewAI Crew from an AST assignment."""
    if isinstance(node, ast.Assign) and node.targets:
        var_name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        var_name = node.target.id
    else:
        return None

    crew = CrewDef(var_name=var_name)

    agents_node = _get_keyword(call, "agents")
    if agents_node:
        crew.agent_vars = _get_list_names(agents_node)

    tasks_node = _get_keyword(call, "tasks")
    if tasks_node:
        crew.task_vars = _get_list_names(tasks_node)

    process_node = _get_keyword(call, "process")
    if isinstance(process_node, ast.Attribute):
        crew.process = process_node.attr  # e.g., Process.sequential -> "sequential"
    elif isinstance(process_node, ast.Constant):
        crew.process = str(process_node.value)

    crew.manager_llm = _get_str_keyword(call, "manager_llm")

    manager_node = _get_keyword(call, "manager_agent")
    if isinstance(manager_node, ast.Name):
        crew.manager_agent_var = manager_node.id

    return crew


def _is_crewai_call(call: ast.Call, class_name: str) -> bool:
    """Check if a Call node is calling a specific class (Agent, Task, Crew)."""
    if isinstance(call.func, ast.Name):
        return call.func.id == class_name
    if isinstance(call.func, ast.Attribute):
        return call.func.attr == class_name
    return False


# ═══════════════════════════════════════════════════════════════
# Pydantic Model Extraction (for output schemas)
# ═══════════════════════════════════════════════════════════════

def _extract_pydantic_models(tree: ast.Module) -> dict[str, list[dict]]:
    """Extract Pydantic BaseModel class definitions as schema field lists."""
    models: dict[str, list[dict]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Check if it inherits from BaseModel
        is_pydantic = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseModel":
                is_pydantic = True
            elif isinstance(base, ast.Attribute) and base.attr == "BaseModel":
                is_pydantic = True
        if not is_pydantic:
            continue

        fields: list[dict] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_name = item.target.id
                field_type = "string"  # default
                if isinstance(item.annotation, ast.Name):
                    type_map = {"str": "string", "int": "integer", "float": "float",
                                "bool": "boolean", "list": "list", "dict": "object",
                                "List": "list", "Dict": "object"}
                    field_type = type_map.get(item.annotation.id, "string")
                elif isinstance(item.annotation, ast.Subscript):
                    if isinstance(item.annotation.value, ast.Name):
                        if item.annotation.value.id in ("List", "list"):
                            field_type = "list"
                        elif item.annotation.value.id in ("Dict", "dict"):
                            field_type = "object"
                        elif item.annotation.value.id == "Optional":
                            field_type = "string"  # simplification
                fields.append({"name": field_name, "type": field_type})

        if fields:
            models[node.name] = fields

    return models


# ═══════════════════════════════════════════════════════════════
# Main Parser
# ═══════════════════════════════════════════════════════════════

def parse_crewai(source: str, filename: str = "<string>") -> dict:
    """Parse a CrewAI Python file and produce an Agent Ontology spec dict."""
    tree = ast.parse(source, filename=filename)

    agents: list[CrewAgent] = []
    tasks: list[CrewTask] = []
    crew: CrewDef | None = None
    pydantic_models = _extract_pydantic_models(tree)

    # Walk all assignments looking for Agent(), Task(), Crew()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        # Get the value (RHS of assignment)
        value = node.value
        if value is None:
            continue

        if not isinstance(value, ast.Call):
            continue

        if _is_crewai_call(value, "Agent"):
            agent = _extract_agent(node, value)
            if agent:
                agents.append(agent)
        elif _is_crewai_call(value, "Task"):
            task = _extract_task(node, value)
            if task:
                tasks.append(task)
        elif _is_crewai_call(value, "Crew"):
            crew_def = _extract_crew(node, value)
            if crew_def:
                crew = crew_def

    # Also check for class-based patterns (methods returning Agent/Task)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Check return statements for Agent(), Task(), Crew() calls
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                if _is_crewai_call(child.value, "Agent"):
                    agent = CrewAgent(var_name=node.name)
                    agent.role = _get_str_keyword(child.value, "role")
                    agent.goal = _get_str_keyword(child.value, "goal")
                    agent.backstory = _get_str_keyword(child.value, "backstory")
                    agent.llm = _get_str_keyword(child.value, "llm")
                    tools_node = _get_keyword(child.value, "tools")
                    if tools_node:
                        agent.tools = _get_list_names(tools_node)
                    agents.append(agent)
                elif _is_crewai_call(child.value, "Task"):
                    task = CrewTask(var_name=node.name)
                    task.description = _get_str_keyword(child.value, "description")
                    task.expected_output = _get_str_keyword(child.value, "expected_output")
                    task.async_execution = _get_bool_keyword(child.value, "async_execution") or False
                    task.human_input = _get_bool_keyword(child.value, "human_input") or False
                    agent_node = _get_keyword(child.value, "agent")
                    if isinstance(agent_node, ast.Attribute):
                        task.agent_var = agent_node.attr
                    elif isinstance(agent_node, ast.Name):
                        task.agent_var = agent_node.id
                    context_node = _get_keyword(child.value, "context")
                    if context_node:
                        task.context_vars = _get_list_names(context_node)
                    tasks.append(task)

    if not agents and not tasks:
        print(f"Warning: No CrewAI Agent or Task definitions found in {filename}", file=sys.stderr)

    return _build_spec(agents, tasks, crew, pydantic_models, filename)


# ═══════════════════════════════════════════════════════════════
# Spec Construction
# ═══════════════════════════════════════════════════════════════

def _build_spec(agents: list[CrewAgent], tasks: list[CrewTask],
                crew: CrewDef | None, pydantic_models: dict[str, list[dict]],
                filename: str) -> dict:
    """Build an Agent Ontology spec from extracted CrewAI components."""

    # Derive spec name
    name = crew.name if crew and crew.name else Path(filename).stem
    name = name.replace("_", " ").title()

    spec: dict[str, Any] = {
        "name": name,
        "version": "1.0",
        "description": f"Imported from CrewAI: {filename}",
    }

    spec_entities: list[dict] = []
    spec_processes: list[dict] = []
    spec_edges: list[dict] = []
    spec_schemas: list[dict] = []

    # Variable name → agent entity ID
    agent_var_to_id: dict[str, str] = {}
    # Variable name → task step ID
    task_var_to_id: dict[str, str] = {}
    # Track tool entities to deduplicate
    tool_ids: set[str] = set()
    # Track step_id → checkpoint_id for human_input tasks
    checkpoint_map: dict[str, str] = {}

    # ── Create agent entities ──
    for agent in agents:
        agent_id = _to_snake(agent.var_name or agent.role or "agent")
        agent_var_to_id[agent.var_name] = agent_id

        entity: dict[str, Any] = {
            "id": agent_id,
            "type": "agent",
            "label": agent.role or agent.var_name,
            "description": agent.goal,
            "model": agent.llm or "unknown",
        }
        if agent.backstory:
            entity["system_prompt"] = agent.backstory

        spec_entities.append(entity)

        # Create tool entities for agent's tools
        for tool_name in agent.tools:
            tool_id = _to_snake(tool_name)
            if tool_id not in tool_ids:
                tool_ids.add(tool_id)
                spec_entities.append({
                    "id": tool_id,
                    "type": "tool",
                    "label": tool_name,
                    "tool_type": "function",
                })

    # ── Build agent→tools mapping for later step→tool edges ──
    agent_tool_map: dict[str, list[str]] = {}
    for agent in agents:
        agent_id = agent_var_to_id.get(agent.var_name, "")
        if agent.tools:
            agent_tool_map[agent_id] = [_to_snake(t) for t in agent.tools]

    # ── Create step processes from tasks ──
    for i, task in enumerate(tasks):
        step_id = _to_snake(task.var_name or f"task_{i}")
        task_var_to_id[task.var_name] = step_id

        process: dict[str, Any] = {
            "id": step_id,
            "type": "step",
            "label": task.description[:80] if task.description else step_id,
            "description": task.description,
        }
        spec_processes.append(process)

        # Create invoke edge from step to assigned agent
        if task.agent_var and task.agent_var in agent_var_to_id:
            agent_id = agent_var_to_id[task.agent_var]
            # Create input/output schemas
            input_schema_id = f"{step_id}_input"
            output_schema_id = f"{step_id}_output"

            spec_schemas.append({
                "name": input_schema_id,
                "fields": [
                    {"name": "task_description", "type": "string"},
                    {"name": "context", "type": "string"},
                ],
            })

            # Output schema from pydantic model or expected_output
            output_fields = [{"name": "result", "type": "string"}]
            if task.output_pydantic and task.output_pydantic in pydantic_models:
                output_fields = pydantic_models[task.output_pydantic]
            spec_schemas.append({
                "name": output_schema_id,
                "fields": output_fields,
            })

            spec_edges.append({
                "type": "invoke",
                "from": step_id,
                "to": agent_id,
                "input_schema": input_schema_id,
                "return_schema": output_schema_id,
            })

            # Create invoke edges from step to agent's tools
            for tool_id in agent_tool_map.get(agent_id, []):
                spec_edges.append({
                    "type": "invoke",
                    "from": step_id,
                    "to": tool_id,
                    "label": f"use {tool_id}",
                })

        # Create invoke edges for task-specific tools
        for tool_name in task.tools:
            tool_id = _to_snake(tool_name)
            if tool_id not in tool_ids:
                tool_ids.add(tool_id)
                spec_entities.append({
                    "id": tool_id,
                    "type": "tool",
                    "label": tool_name,
                    "tool_type": "function",
                })
            spec_edges.append({
                "type": "invoke",
                "from": step_id,
                "to": tool_id,
                "label": f"use {tool_id}",
            })

        # Create checkpoint for human input
        if task.human_input:
            checkpoint_id = f"{step_id}_review"
            spec_processes.append({
                "id": checkpoint_id,
                "type": "checkpoint",
                "label": f"Human review for {step_id}",
            })
            # Wire step → checkpoint; flow edges will use checkpoint as the outgoing node
            spec_edges.append({"type": "flow", "from": step_id, "to": checkpoint_id})
            checkpoint_map[step_id] = checkpoint_id

    # ── Create flow edges based on process type ──
    process_type = crew.process if crew else "sequential"

    if process_type == "sequential":
        # Sequential: flow edges between tasks in order, respecting async_execution
        ordered_tasks = [t for t in tasks if t.var_name in task_var_to_id]
        task_ids = [task_var_to_id[t.var_name] for t in ordered_tasks]

        # Group consecutive async tasks into fan-out batches
        i = 0
        while i < len(ordered_tasks):
            if ordered_tasks[i].async_execution:
                # Collect consecutive async tasks
                batch_start = i
                while i < len(ordered_tasks) and ordered_tasks[i].async_execution:
                    i += 1
                batch_ids = task_ids[batch_start:i]
                # Find predecessor (task before the batch)
                if batch_start > 0:
                    pred = task_ids[batch_start - 1]
                    for bid in batch_ids:
                        spec_edges.append({"type": "flow", "from": pred, "to": bid})
                else:
                    # Async batch starts at position 0 — create a _start node
                    start_id = "_start"
                    spec_processes.insert(0, {
                        "id": start_id,
                        "type": "step",
                        "label": "Start",
                        "description": "Fan-out to parallel tasks",
                    })
                    for bid in batch_ids:
                        spec_edges.append({"type": "flow", "from": start_id, "to": bid})
                    spec["entry_point"] = start_id
                # Find successor (first non-async task after batch)
                if i < len(ordered_tasks):
                    succ = task_ids[i]
                    for bid in batch_ids:
                        spec_edges.append({"type": "flow", "from": bid, "to": succ})
            else:
                if i + 1 < len(ordered_tasks):
                    spec_edges.append({
                        "type": "flow",
                        "from": task_ids[i],
                        "to": task_ids[i + 1],
                    })
                i += 1

        # Set entry point to first task (if not already set by async batch)
        if task_ids and "entry_point" not in spec:
            spec["entry_point"] = task_ids[0]

    elif process_type == "hierarchical":
        # Hierarchical: team entity with manager
        team_id = _to_snake(crew.var_name if crew else "crew") + "_team"
        team_entity: dict[str, Any] = {
            "id": team_id,
            "type": "team",
            "label": name,
            "strategy": "hierarchical",
            "members": list(agent_var_to_id.values()),
        }
        if crew and crew.manager_agent_var and crew.manager_agent_var in agent_var_to_id:
            team_entity["manager"] = agent_var_to_id[crew.manager_agent_var]

        spec_entities.append(team_entity)

        # Create a coordination step
        coord_id = "coordinate"
        spec_processes.append({
            "id": coord_id,
            "type": "step",
            "label": "Manager coordinates task delegation",
        })
        spec["entry_point"] = coord_id

        # Flow from coordination to each task
        for task in tasks:
            if task.var_name in task_var_to_id:
                spec_edges.append({
                    "type": "flow",
                    "from": coord_id,
                    "to": task_var_to_id[task.var_name],
                })

    # ── Rewire flow edges through checkpoints ──
    # If a step has a checkpoint, outgoing flow edges should start from the checkpoint
    if checkpoint_map:
        for edge in spec_edges:
            if edge["type"] == "flow" and edge["from"] in checkpoint_map:
                edge["from"] = checkpoint_map[edge["from"]]

    # ── Add terminal logic to last task(s) ──
    if process_type == "sequential" and spec_processes:
        # Find tasks with no outgoing flow edge
        process_ids = {p["id"] for p in spec_processes}
        has_outgoing = {e["from"] for e in spec_edges if e["type"] == "flow" and e["from"] in process_ids}
        for p in spec_processes:
            if p["type"] == "step" and p["id"] not in has_outgoing:
                p["logic"] = 'state.data["_done"] = True\n'

    # ── Context dependencies (data flow) ──
    for task in tasks:
        if not task.context_vars:
            continue
        for ctx_var in task.context_vars:
            if ctx_var in task_var_to_id and task.var_name in task_var_to_id:
                src_step = task_var_to_id[ctx_var]
                tgt_step = task_var_to_id[task.var_name]
                # Check if flow edge already exists
                existing = any(
                    e.get("type") == "flow" and e.get("from") == src_step and e.get("to") == tgt_step
                    for e in spec_edges
                )
                if not existing:
                    spec_edges.append({
                        "type": "flow",
                        "from": src_step,
                        "to": tgt_step,
                        "label": "context dependency",
                    })

    # ── Assemble spec ──
    spec["entities"] = spec_entities
    spec["processes"] = spec_processes
    spec["edges"] = spec_edges
    spec["schemas"] = spec_schemas

    return spec


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Import a CrewAI Python file into an Agent Ontology YAML spec",
    )
    parser.add_argument("input", help="Path to CrewAI Python file")
    parser.add_argument("-o", "--output", help="Output YAML file path")
    parser.add_argument("--validate", action="store_true",
                        help="Validate generated spec against ontology")
    parser.add_argument("--print", action="store_true", dest="print_yaml",
                        help="Print YAML to stdout (default if no -o)")
    parser.add_argument("--llm-augment", action="store_true",
                        help="Use LLM to enrich the AST-extracted skeleton with semantic information")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Model for LLM augmentation (default: gemini-2.0-flash)")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    source = input_path.read_text(encoding="utf-8")
    spec = parse_crewai(source, str(input_path))

    if args.llm_augment:
        from .import_langgraph import llm_augment_spec
        print("LLM-augmenting spec...", file=sys.stderr)
        spec = llm_augment_spec(source, spec, model=args.model)

    yaml_output = yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if args.output:
        Path(args.output).write_text(yaml_output, encoding="utf-8")
        print(f"  Wrote {args.output}")
        if args.print_yaml:
            print(yaml_output)
    else:
        print(yaml_output)

    if args.validate:
        from .validate import validate_spec, load_yaml as load_ontology_yaml
        ontology_path = os.path.join(SCRIPT_DIR, "ONTOLOGY.yaml")
        ontology = load_ontology_yaml(ontology_path)
        errors, warnings = validate_spec(spec, ontology, str(input_path))
        if errors:
            print(f"\n  {len(errors)} validation errors:")
            for e in errors:
                print(f"    ✗ {e}")
        if warnings:
            print(f"\n  {len(warnings)} warnings:")
            for w in warnings:
                print(f"    ⚠ {w}")
        if not errors:
            print(f"  ✓ Validation passed ({len(warnings)} warnings)")


if __name__ == "__main__":
    main()
