#!/usr/bin/env python3
"""
OpenAI Agents SDK Importer — Parse OpenAI Agents SDK Python files into Agent Ontology YAML specs.

Supports the `agents` package (pip install openai-agents) patterns.

Mapping:
  Agent()             → agent entity
  Agent.instructions  → agent.system_prompt
  Agent.model         → agent.model
  Agent.tools         → tool entities + invoke edges
  @function_tool      → tool entity (type: function)
  WebSearchTool       → tool entity (type: search)
  FileSearchTool      → tool entity (type: file_search)
  CodeInterpreterTool → tool entity (type: code_interpreter)
  Agent.handoffs      → handoff edges between agents
  agent.as_tool()     → invoke edge from orchestrator step to agent
  @input_guardrail    → policy entity (effect: block)
  @output_guardrail   → policy entity (effect: block)
  Runner.run()        → step process + flow edges
  asyncio.gather()    → fan-out step processes
  Agent.output_type   → schema (output)

Usage:
  python3 import_openai_agents.py agent_file.py -o specs/imported.yaml
  python3 import_openai_agents.py agent_file.py --validate
  python3 import_openai_agents.py agent_file.py --llm-augment
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
    id: str = ""
    var_name: str = ""
    instructions: str = ""
    model: str = "unknown"
    tool_refs: list[str] = field(default_factory=list)  # variable/function names
    handoff_refs: list[str] = field(default_factory=list)  # variable names
    as_tool_refs: list[str] = field(default_factory=list)  # agent vars used via as_tool()
    output_type: str = ""
    input_guardrails: list[str] = field(default_factory=list)
    output_guardrails: list[str] = field(default_factory=list)
    handoff_description: str = ""


@dataclass
class ToolInfo:
    name: str
    id: str = ""
    description: str = ""
    tool_type: str = "function"
    var_name: str = ""  # variable name if assigned to a var


@dataclass
class GuardrailInfo:
    name: str
    id: str = ""
    guardrail_type: str = "input"  # input or output
    description: str = ""


@dataclass
class RunCallInfo:
    """Represents a Runner.run() or Runner.run_sync() call."""
    agent_var: str = ""
    input_expr: str = ""
    result_var: str = ""
    is_async: bool = True
    line_no: int = 0


@dataclass
class AsToolInfo:
    """Represents agent.as_tool() call in another agent's tools list."""
    source_agent_var: str = ""  # the agent being converted to a tool
    tool_name: str = ""
    tool_description: str = ""


# ═══════════════════════════════════════════════════════════════
# AST Helpers
# ═══════════════════════════════════════════════════════════════

def _get_str(node) -> str:
    """Extract string value from AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "<f-string>"
    return ""


def _get_keyword(call: ast.Call, name: str):
    """Get a keyword argument node from a Call."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _get_str_keyword(call: ast.Call, name: str) -> str:
    """Get a string keyword argument value."""
    val = _get_keyword(call, name)
    if val is None:
        return ""
    return _get_str(val)


def _get_int_keyword(call: ast.Call, name: str) -> int | None:
    """Get an integer keyword argument value."""
    val = _get_keyword(call, name)
    if isinstance(val, ast.Constant) and isinstance(val.value, int):
        return val.value
    return None


def _get_list_names(node) -> list[str]:
    """Extract variable names from a list literal."""
    names = []
    if isinstance(node, ast.List):
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Call):
                # handoff(agent=...) or agent.as_tool(...)
                func = elt.func
                if isinstance(func, ast.Name):
                    names.append(f"__call__{func.id}")
                elif isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        names.append(f"__attr__{func.value.id}.{func.attr}")
    return names


def _sanitize_id(name: str) -> str:
    """Convert a variable name to a valid spec ID."""
    return re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')


def _snake_to_label(name: str) -> str:
    """Convert snake_case to Title Case label."""
    return name.replace("_", " ").title()


def _get_call_name(node: ast.Call) -> str:
    """Get the function/class name from a Call node."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    # Agent[TypeParam](...) → Subscript
    if isinstance(func, ast.Subscript):
        if isinstance(func.value, ast.Name):
            return func.value.id
    return ""


def _get_docstring(node) -> str:
    """Extract docstring from a function definition."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value.strip()
    return ""


# ═══════════════════════════════════════════════════════════════
# Extraction Functions
# ═══════════════════════════════════════════════════════════════

def _build_var_map(tree: ast.Module) -> dict[str, ast.AST]:
    """Build a map of variable name → AST node for simple assignments."""
    var_map: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                var_map[target.id] = node.value
    return var_map


def _extract_agents(tree: ast.Module, var_map: dict) -> dict[str, AgentInfo]:
    """Extract Agent() definitions."""
    agents: dict[str, AgentInfo] = {}  # var_name -> AgentInfo

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        name = _get_call_name(call)
        if name != "Agent":
            continue

        var_name = node.targets[0].id
        agent_name = _get_str_keyword(call, "name") or var_name
        agent_id = _sanitize_id(agent_name)

        ag = AgentInfo(name=agent_name, id=agent_id, var_name=var_name)

        # instructions
        instructions_node = _get_keyword(call, "instructions")
        if instructions_node:
            ag.instructions = _get_str(instructions_node)

        # model
        model_node = _get_keyword(call, "model")
        if model_node:
            model_str = _get_str(model_node)
            if model_str:
                ag.model = model_str

        # tools
        tools_node = _get_keyword(call, "tools")
        if tools_node:
            ag.tool_refs = _get_list_names(tools_node)

        # handoffs
        handoffs_node = _get_keyword(call, "handoffs")
        if handoffs_node:
            ag.handoff_refs = _get_list_names(handoffs_node)

        # output_type
        output_node = _get_keyword(call, "output_type")
        if isinstance(output_node, ast.Name):
            ag.output_type = output_node.id

        # input_guardrails
        ig_node = _get_keyword(call, "input_guardrails")
        if ig_node:
            ag.input_guardrails = _get_list_names(ig_node)

        # output_guardrails
        og_node = _get_keyword(call, "output_guardrails")
        if og_node:
            ag.output_guardrails = _get_list_names(og_node)

        # handoff_description
        hd = _get_str_keyword(call, "handoff_description")
        if hd:
            ag.handoff_description = hd

        agents[var_name] = ag

    return agents


def _extract_function_tools(tree: ast.Module) -> dict[str, ToolInfo]:
    """Extract @function_tool decorated functions."""
    tools: dict[str, ToolInfo] = {}  # func_name -> ToolInfo

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        is_function_tool = False
        tool_name_override = ""
        tool_desc_override = ""

        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "function_tool":
                is_function_tool = True
            elif isinstance(dec, ast.Call):
                dec_name = _get_call_name(dec)
                if dec_name == "function_tool":
                    is_function_tool = True
                    tool_name_override = _get_str_keyword(dec, "name_override")
                    tool_desc_override = _get_str_keyword(dec, "description_override")

        if not is_function_tool:
            continue

        func_name = node.name
        tool_name = tool_name_override or func_name
        tool_id = _sanitize_id(tool_name)

        tool = ToolInfo(
            name=tool_name,
            id=tool_id,
            description=tool_desc_override or _get_docstring(node),
            tool_type="function",
        )
        tools[func_name] = tool

    return tools


def _extract_hosted_tools(tree: ast.Module) -> dict[str, ToolInfo]:
    """Extract hosted tool instances (WebSearchTool, FileSearchTool, etc.)."""
    hosted_tools: dict[str, ToolInfo] = {}  # var_name -> ToolInfo

    HOSTED_TOOL_MAP = {
        "WebSearchTool": ("search", "Web search tool"),
        "FileSearchTool": ("file_search", "File search tool"),
        "CodeInterpreterTool": ("code_interpreter", "Code interpreter tool"),
        "ImageGenerationTool": ("function", "Image generation tool"),
        "ComputerTool": ("function", "Computer use tool"),
        "HostedMCPTool": ("mcp", "Hosted MCP tool"),
        "ShellTool": ("shell", "Shell execution tool"),
        "ApplyPatchTool": ("function", "Apply patch tool"),
        "LocalShellTool": ("shell", "Local shell tool"),
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call_name = _get_call_name(node.value)
        if call_name not in HOSTED_TOOL_MAP:
            continue

        var_name = node.targets[0].id
        tool_type, description = HOSTED_TOOL_MAP[call_name]
        tool_id = _sanitize_id(var_name)

        hosted_tools[var_name] = ToolInfo(
            name=var_name,
            id=tool_id,
            description=description,
            tool_type=tool_type,
            var_name=var_name,
        )

    return hosted_tools


def _extract_as_tool_calls(tree: ast.Module) -> list[AsToolInfo]:
    """Extract agent.as_tool() calls found in tools lists."""
    as_tools: list[AsToolInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "as_tool":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue

        source_var = node.func.value.id
        tool_name = _get_str_keyword(node, "tool_name") or f"{source_var}_tool"
        tool_desc = _get_str_keyword(node, "tool_description") or ""

        as_tools.append(AsToolInfo(
            source_agent_var=source_var,
            tool_name=tool_name,
            tool_description=tool_desc,
        ))

    return as_tools


def _extract_guardrails(tree: ast.Module) -> dict[str, GuardrailInfo]:
    """Extract @input_guardrail and @output_guardrail decorated functions."""
    guardrails: dict[str, GuardrailInfo] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        guardrail_type = ""
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                if dec.id == "input_guardrail":
                    guardrail_type = "input"
                elif dec.id == "output_guardrail":
                    guardrail_type = "output"
            elif isinstance(dec, ast.Call):
                dec_name = _get_call_name(dec)
                if dec_name == "input_guardrail":
                    guardrail_type = "input"
                elif dec_name == "output_guardrail":
                    guardrail_type = "output"

        if not guardrail_type:
            continue

        func_name = node.name
        gid = _sanitize_id(func_name)
        guardrails[func_name] = GuardrailInfo(
            name=func_name,
            id=gid,
            guardrail_type=guardrail_type,
            description=_get_docstring(node),
        )

    return guardrails


def _extract_run_calls(tree: ast.Module) -> list[RunCallInfo]:
    """Extract Runner.run() and Runner.run_sync() calls in execution order."""
    run_calls: list[RunCallInfo] = []

    for node in ast.walk(tree):
        call = None
        result_var = ""

        # Pattern: result = await Runner.run(...) or result = Runner.run_sync(...)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                result_var = target.id

            value = node.value
            # Unwrap Await
            if isinstance(value, ast.Await):
                value = value.value
            if isinstance(value, ast.Call):
                call = value

        # Pattern: bare await Runner.run(...)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
            if isinstance(node.value.value, ast.Call):
                call = node.value.value

        if call is None:
            continue
        if not isinstance(call.func, ast.Attribute):
            continue

        method = call.func.attr
        if method not in ("run", "run_sync"):
            continue

        # Check it's Runner.run() not some_other.run()
        caller = call.func.value
        if not (isinstance(caller, ast.Name) and caller.id == "Runner"):
            continue

        rc = RunCallInfo(
            result_var=result_var,
            is_async=(method == "run"),
            line_no=getattr(node, "lineno", 0),
        )

        # First positional arg = agent
        if call.args:
            arg0 = call.args[0]
            if isinstance(arg0, ast.Name):
                rc.agent_var = arg0.id
            elif isinstance(arg0, ast.Attribute) and isinstance(arg0.value, ast.Name):
                rc.agent_var = f"{arg0.value.id}"
        # keyword: starting_agent=
        if not rc.agent_var:
            sa_node = _get_keyword(call, "starting_agent")
            if isinstance(sa_node, ast.Name):
                rc.agent_var = sa_node.id

        # Second positional arg = input
        if len(call.args) > 1:
            arg1 = call.args[1]
            if isinstance(arg1, ast.Constant):
                rc.input_expr = str(arg1.value)[:100]
            elif isinstance(arg1, ast.Name):
                rc.input_expr = arg1.id
        # keyword: input=
        if not rc.input_expr:
            inp_node = _get_keyword(call, "input")
            if isinstance(inp_node, ast.Name):
                rc.input_expr = inp_node.id
            elif isinstance(inp_node, ast.Constant):
                rc.input_expr = str(inp_node.value)[:100]

        run_calls.append(rc)

    return run_calls


def _extract_gather_calls(tree: ast.Module) -> list[list[str]]:
    """Extract asyncio.gather(Runner.run(...), ...) patterns → lists of agent vars."""
    gather_groups: list[list[str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # asyncio.gather(...)
        is_gather = False
        if isinstance(func, ast.Attribute) and func.attr == "gather":
            if isinstance(func.value, ast.Name) and func.value.id == "asyncio":
                is_gather = True
        if not is_gather:
            continue

        agent_vars = []
        for arg in node.args:
            # Each arg should be Runner.run(agent, ...)
            inner = arg
            if isinstance(inner, ast.Await):
                inner = inner.value
            if not isinstance(inner, ast.Call):
                continue
            if not isinstance(inner.func, ast.Attribute):
                continue
            if inner.func.attr not in ("run", "run_sync"):
                continue
            if inner.args and isinstance(inner.args[0], ast.Name):
                agent_vars.append(inner.args[0].id)

        if agent_vars:
            gather_groups.append(agent_vars)

    return gather_groups


def _extract_handoff_appends(tree: ast.Module) -> list[tuple[str, str]]:
    """Extract agent.handoffs.append(other_agent) patterns."""
    appends: list[tuple[str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "append":
            continue

        # agent.handoffs.append(other)
        mid = call.func.value
        if not isinstance(mid, ast.Attribute):
            continue
        if mid.attr != "handoffs":
            continue
        if not isinstance(mid.value, ast.Name):
            continue

        agent_var = mid.value.id
        if call.args and isinstance(call.args[0], ast.Name):
            target_var = call.args[0].id
            appends.append((agent_var, target_var))

    return appends


def _extract_output_types(tree: ast.Module) -> dict[str, list[dict]]:
    """Extract Pydantic BaseModel / dataclass / TypedDict definitions for output schemas."""
    schemas: dict[str, list[dict]] = {}  # class_name -> fields

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if it inherits from BaseModel, TypedDict, or is @dataclass
        is_schema = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ("BaseModel", "TypedDict"):
                is_schema = True
            elif isinstance(base, ast.Attribute) and base.attr in ("BaseModel", "TypedDict"):
                is_schema = True

        if not is_schema:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id == "dataclass":
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


def _annotation_to_type(ann) -> str:
    """Convert a Python type annotation AST node to ontology type string."""
    if isinstance(ann, ast.Name):
        type_map = {
            "str": "string", "int": "integer", "float": "float",
            "bool": "boolean", "list": "list", "dict": "object",
        }
        return type_map.get(ann.id, "string")
    if isinstance(ann, ast.Attribute):
        return "string"
    if isinstance(ann, ast.Subscript):
        if isinstance(ann.value, ast.Name):
            if ann.value.id == "list":
                return "list<string>"
            if ann.value.id in ("Optional", "Union"):
                return _annotation_to_type(ann.slice)
            if ann.value.id == "Annotated":
                if isinstance(ann.slice, ast.Tuple) and ann.slice.elts:
                    return _annotation_to_type(ann.slice.elts[0])
        return "string"
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        return "string"
    return "string"


# ═══════════════════════════════════════════════════════════════
# Spec Assembly
# ═══════════════════════════════════════════════════════════════

def import_openai_agents(source_path: Path) -> dict:
    """Parse an OpenAI Agents SDK Python file and produce an Agent Ontology YAML spec."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    var_map = _build_var_map(tree)

    agents = _extract_agents(tree, var_map)
    func_tools = _extract_function_tools(tree)
    hosted_tools = _extract_hosted_tools(tree)
    as_tool_calls = _extract_as_tool_calls(tree)
    guardrails = _extract_guardrails(tree)
    run_calls = _extract_run_calls(tree)
    gather_groups = _extract_gather_calls(tree)
    handoff_appends = _extract_handoff_appends(tree)
    output_schemas = _extract_output_types(tree)

    # Merge all tools
    all_tools: dict[str, ToolInfo] = {}
    all_tools.update(func_tools)
    all_tools.update(hosted_tools)

    # Build tool var → tool name mapping
    tool_var_map: dict[str, str] = {}
    for fname, tool in func_tools.items():
        tool_var_map[fname] = fname
    for vname, tool in hosted_tools.items():
        tool_var_map[vname] = vname

    # Build spec
    spec_name = source_path.stem.replace("_", " ").title()
    spec: dict[str, Any] = {
        "name": spec_name,
        "version": "1.0",
        "description": f"Imported from OpenAI Agents SDK file: {source_path.name}",
        "entities": [],
        "processes": [],
        "edges": [],
        "schemas": [],
    }

    entities = spec["entities"]
    processes = spec["processes"]
    edges = spec["edges"]
    schemas = spec["schemas"]

    var_to_id: dict[str, str] = {}  # var_name -> entity_id
    agent_tool_ids: dict[str, list[str]] = {}  # agent_var -> [tool_id, ...]

    # ── Agent entities ──
    for var_name, ag in agents.items():
        var_to_id[var_name] = ag.id

        entity: dict[str, Any] = {
            "id": ag.id,
            "type": "agent",
            "label": _snake_to_label(ag.name),
            "model": ag.model,
        }
        if ag.instructions:
            entity["system_prompt"] = ag.instructions
        if ag.handoff_description:
            entity["description"] = ag.handoff_description
        if ag.output_type:
            entity["output_schema"] = ag.output_type

        entities.append(entity)

        # Resolve tool references
        resolved_tools: list[str] = []
        for tref in ag.tool_refs:
            # Direct function_tool reference
            if tref in tool_var_map:
                resolved_tools.append(tool_var_map[tref])
            # __attr__agent_var.as_tool pattern
            elif tref.startswith("__attr__") and ".as_tool" in tref:
                # handled separately via as_tool_calls
                pass
            else:
                # Might be a variable pointing to a tool
                if tref in var_map:
                    # Try to resolve
                    pass
                resolved_tools.append(tref)

        agent_tool_ids[var_name] = [
            all_tools[t].id for t in resolved_tools if t in all_tools
        ]

    # ── Tool entities ──
    for tool_name, tool in all_tools.items():
        entities.append({
            "id": tool.id,
            "type": "tool",
            "label": _snake_to_label(tool.name),
            "tool_type": tool.tool_type,
            **({"description": tool.description} if tool.description else {}),
        })

    # ── Guardrail/Policy processes ──
    # Collect targets first, then create policy processes
    guardrail_targets: dict[str, list[str]] = {}  # guardrail_func_name -> [agent_id, ...]
    for var_name, ag in agents.items():
        for gref in ag.input_guardrails:
            guardrail_targets.setdefault(gref, []).append(ag.id)
        for gref in ag.output_guardrails:
            guardrail_targets.setdefault(gref, []).append(ag.id)

    for gname, gr in guardrails.items():
        targets = guardrail_targets.get(gname, [])
        processes.append({
            "id": gr.id,
            "type": "policy",
            "label": _snake_to_label(gname),
            "targets": targets,
            "effect": "block",
            **({"description": gr.description} if gr.description else {}),
        })

    # ── Output type schemas ──
    for class_name, fields in output_schemas.items():
        schemas.append({
            "name": class_name,
            "fields": fields,
        })

    # ── Handoff edges from Agent.handoffs ──
    for var_name, ag in agents.items():
        for href in ag.handoff_refs:
            if href.startswith("__call__handoff"):
                # handoff(agent=...) — skip for now, complex
                continue
            target_id = var_to_id.get(href, _sanitize_id(href))
            edges.append({
                "type": "handoff",
                "from": ag.id,
                "to": target_id,
                "label": f"Handoff to {_snake_to_label(href)}",
            })

    # ── Handoff edges from .handoffs.append() ──
    for agent_var, target_var in handoff_appends:
        src_id = var_to_id.get(agent_var, _sanitize_id(agent_var))
        tgt_id = var_to_id.get(target_var, _sanitize_id(target_var))
        edges.append({
            "type": "handoff",
            "from": src_id,
            "to": tgt_id,
            "label": f"Handoff to {_snake_to_label(target_var)}",
        })

    # (Guardrail targets already set above during policy process creation)

    # ── Process generation from Runner.run() calls ──
    entry_point = None
    prev_step_id = None

    # If we have run_calls, create step processes for each
    if run_calls:
        for i, rc in enumerate(run_calls):
            agent_id = var_to_id.get(rc.agent_var, _sanitize_id(rc.agent_var))
            step_id = f"run_{rc.agent_var}" if rc.agent_var else f"run_step_{i}"

            processes.append({
                "id": step_id,
                "type": "step",
                "label": f"Run {_snake_to_label(rc.agent_var or f'step_{i}')}",
            })

            # Invoke the agent
            edges.append({
                "type": "invoke",
                "from": step_id,
                "to": agent_id,
                "label": f"Call {_snake_to_label(rc.agent_var)}",
            })

            # Wire tool invokes from the step
            for tool_id in agent_tool_ids.get(rc.agent_var, []):
                edges.append({
                    "type": "invoke",
                    "from": step_id,
                    "to": tool_id,
                    "label": f"Use {_snake_to_label(tool_id)}",
                })

            # Also wire tools from handoff-target agents (reachable via handoff)
            if rc.agent_var in agents:
                for href in agents[rc.agent_var].handoff_refs:
                    target_var = href
                    for tool_id in agent_tool_ids.get(target_var, []):
                        edges.append({
                            "type": "invoke",
                            "from": step_id,
                            "to": tool_id,
                            "label": f"Use {_snake_to_label(tool_id)}",
                        })

            if entry_point is None:
                entry_point = step_id

            # Flow from previous step
            if prev_step_id:
                edges.append({
                    "type": "flow",
                    "from": prev_step_id,
                    "to": step_id,
                })

            prev_step_id = step_id

    # ── Fan-out processes from asyncio.gather ──
    for group_idx, agent_vars in enumerate(gather_groups):
        fan_out_step = f"fan_out_{group_idx}"
        processes.append({
            "id": fan_out_step,
            "type": "step",
            "label": f"Parallel Execution {group_idx + 1}",
        })

        if entry_point is None:
            entry_point = fan_out_step

        if prev_step_id:
            edges.append({
                "type": "flow",
                "from": prev_step_id,
                "to": fan_out_step,
            })

        for avar in agent_vars:
            agent_id = var_to_id.get(avar, _sanitize_id(avar))
            edges.append({
                "type": "invoke",
                "from": fan_out_step,
                "to": agent_id,
                "label": f"Call {_snake_to_label(avar)}",
            })
            for tool_id in agent_tool_ids.get(avar, []):
                edges.append({
                    "type": "invoke",
                    "from": fan_out_step,
                    "to": tool_id,
                    "label": f"Use {_snake_to_label(tool_id)}",
                })

        prev_step_id = fan_out_step

    # ── If no run_calls or gather, create process structure from agent relationships ──
    if not run_calls and not gather_groups and agents:
        # Find the "entry" agent — one with handoffs to others but not a handoff target
        handoff_targets = set()
        for ag in agents.values():
            for href in ag.handoff_refs:
                target_var = href
                if target_var in var_to_id:
                    handoff_targets.add(var_to_id[target_var])
        for agent_var, target_var in handoff_appends:
            if target_var in var_to_id:
                handoff_targets.add(var_to_id[target_var])

        entry_agents = [v for k, v in agents.items()
                        if v.id not in handoff_targets and v.handoff_refs]
        if not entry_agents:
            entry_agents = list(agents.values())[:1]

        if entry_agents:
            main_agent = entry_agents[0]
            step_id = f"run_{main_agent.var_name}"
            processes.append({
                "id": step_id,
                "type": "step",
                "label": f"Run {_snake_to_label(main_agent.name)}",
            })
            edges.append({
                "type": "invoke",
                "from": step_id,
                "to": main_agent.id,
                "label": f"Call {_snake_to_label(main_agent.name)}",
            })
            for tool_id in agent_tool_ids.get(main_agent.var_name, []):
                edges.append({
                    "type": "invoke",
                    "from": step_id,
                    "to": tool_id,
                    "label": f"Use {_snake_to_label(tool_id)}",
                })
            entry_point = step_id
            prev_step_id = step_id

    # ── as_tool invoke edges ──
    for at in as_tool_calls:
        src_id = var_to_id.get(at.source_agent_var, _sanitize_id(at.source_agent_var))
        # Find which agent's tools list contains this as_tool call
        # Wire from any step that invokes the orchestrator agent
        for step in processes:
            step_invoke_targets = [e["to"] for e in edges
                                   if e["type"] == "invoke" and e["from"] == step["id"]]
            # If this step invokes an agent that uses the as_tool agent
            for var_name, ag in agents.items():
                for tref in ag.tool_refs:
                    if (tref.startswith("__attr__") and at.source_agent_var in tref
                            and ag.id in step_invoke_targets):
                        edges.append({
                            "type": "invoke",
                            "from": step["id"],
                            "to": src_id,
                            "label": at.tool_description or f"Delegate to {_snake_to_label(at.source_agent_var)}",
                        })

    # ── Terminal step ──
    if processes:
        done_id = "_done"
        processes.append({
            "id": done_id,
            "type": "step",
            "label": "Done",
            "logic": 'state.data["_done"] = True\n',
        })
        if prev_step_id:
            edges.append({
                "type": "flow",
                "from": prev_step_id,
                "to": done_id,
                "label": "Done",
            })

    # ── Entry point ──
    if entry_point:
        spec["entry_point"] = entry_point
    elif processes:
        spec["entry_point"] = processes[0]["id"]

    # ── Default schema ──
    if not schemas:
        schemas.append({
            "name": "AgentOutput",
            "fields": [
                {"name": "output", "type": "string", "description": "Agent output text"},
            ],
        })

    # ── Deduplicate edges ──
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
# LLM Augmentation (reuse shared function)
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
        description="Import an OpenAI Agents SDK Python file into an Agent Ontology YAML spec"
    )
    parser.add_argument("input", help="Path to OpenAI Agents SDK Python file")
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
        spec = import_openai_agents(input_path)
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
