#!/usr/bin/env python3
"""
AutoGen Importer — Parse AutoGen Python files into Agent Ontology YAML specs.

Supports both pyautogen v0.2 (ConversableAgent, GroupChat, GroupChatManager)
and autogen-agentchat v0.4 (Teams, Handoff, FunctionTool) patterns.

Mapping:
  AssistantAgent      → agent entity
  UserProxyAgent      → agent (NEVER) or human+agent (ALWAYS/TERMINATE)
  GroupChat           → team entity + loop/gate processes
  GroupChatManager    → coordinator agent
  register_for_llm   → tool entity + invoke edge
  code_execution_cfg → tool entity (shell) + invoke edge
  initiate_chat      → entry step + flow edges
  initiate_chats     → sequential steps + flow edges
  register_nested_chats → sub-process composition
  Handoff (v0.4)     → handoff edge
  RoundRobinGroupChat → team (round_robin)
  SelectorGroupChat  → team (dynamic)
  Swarm              → team (dynamic)

Usage:
  python3 import_autogen.py agent_file.py -o specs/imported.yaml
  python3 import_autogen.py agent_file.py --validate
  python3 import_autogen.py agent_file.py --llm-augment
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
        parts = []
        for val in node.values:
            if isinstance(val, ast.Constant):
                parts.append(str(val.value))
            else:
                parts.append("{...}")
        return "".join(parts)
    return ""


def _get_int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _get_bool(node: ast.expr) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _get_keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _get_str_keyword(call: ast.Call, name: str) -> str:
    val = _get_keyword(call, name)
    return _get_str(val) if val else ""


def _get_int_keyword(call: ast.Call, name: str) -> int | None:
    val = _get_keyword(call, name)
    return _get_int(val) if val else None


def _get_bool_keyword(call: ast.Call, name: str) -> bool | None:
    val = _get_keyword(call, name)
    return _get_bool(val) if val else None


def _call_func_name(call: ast.Call) -> str:
    """Get the function/class name from a Call node."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _call_full_name(call: ast.Call) -> str:
    """Get full dotted name like autogen.AssistantAgent."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        obj = call.func.value
        if isinstance(obj, ast.Name):
            return f"{obj.id}.{call.func.attr}"
        if isinstance(obj, ast.Attribute):
            return f"{_call_full_name_expr(obj)}.{call.func.attr}"
    return ""


def _call_full_name_expr(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_full_name_expr(node.value)}.{node.attr}"
    return ""


def _snake_to_label(s: str) -> str:
    return s.replace("_", " ").title()


def _sanitize_id(name: str) -> str:
    """Convert an agent name to a valid YAML ID."""
    return re.sub(r'[^a-z0-9_]', '_', name.lower().strip())


# ═══════════════════════════════════════════════════════════════
# Known AutoGen Classes
# ═══════════════════════════════════════════════════════════════

AGENT_CLASSES = {
    "ConversableAgent", "AssistantAgent", "UserProxyAgent",
    "RetrieveAssistantAgent", "RetrieveUserProxyAgent",
    "CompressibleAgent", "GPTAssistantAgent",
}

GROUP_CLASSES = {
    "GroupChat",
    "RoundRobinGroupChat", "SelectorGroupChat", "Swarm",
    "MagenticOneGroupChat",
}

MANAGER_CLASSES = {"GroupChatManager"}

TOOL_CLASSES = {"FunctionTool"}

TERMINATION_CLASSES = {
    "TextMentionTermination", "MaxMessageTermination",
    "HandoffTermination", "TokenUsageTermination",
    "TimeoutTermination", "SourceMatchTermination",
}


# ═══════════════════════════════════════════════════════════════
# Extracted Data Structures
# ═══════════════════════════════════════════════════════════════

class AgentInfo:
    def __init__(self, var_name: str, name: str, class_type: str):
        self.var_name = var_name
        self.name = name
        self.class_type = class_type
        self.system_message = ""
        self.model = "unknown"
        self.human_input_mode = "NEVER"
        self.code_execution = False
        self.max_auto_reply: int | None = None
        self.termination_msg = ""
        self.description = ""
        self.tools: list[str] = []
        self.handoffs: list[str] = []

    @property
    def id(self) -> str:
        return _sanitize_id(self.name or self.var_name)

    @property
    def is_human(self) -> bool:
        return self.human_input_mode in ("ALWAYS", "TERMINATE")


class GroupChatInfo:
    def __init__(self, var_name: str):
        self.var_name = var_name
        self.agent_vars: list[str] = []
        self.max_round: int | None = None
        self.speaker_method = "auto"
        self.speaker_transitions: dict[str, list[str]] = {}
        self.allow_repeat = False


class ManagerInfo:
    def __init__(self, var_name: str):
        self.var_name = var_name
        self.groupchat_var = ""
        self.model = "unknown"
        self.termination_msg = ""


class ToolInfo:
    def __init__(self, func_name: str):
        self.func_name = func_name
        self.description = ""
        self.caller_var = ""
        self.executor_var = ""

    @property
    def id(self) -> str:
        return _sanitize_id(self.func_name)


class ChatInfo:
    def __init__(self, caller_var: str, recipient_var: str):
        self.caller_var = caller_var
        self.recipient_var = recipient_var
        self.message = ""
        self.max_turns: int | None = None
        self.summary_method = ""


class NestedChatInfo:
    def __init__(self, agent_var: str):
        self.agent_var = agent_var
        self.trigger_var = ""
        self.chat_queue: list[dict] = []


# ═══════════════════════════════════════════════════════════════
# Variable Resolution
# ═══════════════════════════════════════════════════════════════

def _build_var_map(tree: ast.Module) -> dict[str, ast.expr]:
    """Build a map from variable names to their assigned values."""
    var_map: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_map[target.id] = node.value
    return var_map


def _extract_model_from_llm_config(node: ast.expr, var_map: dict) -> str:
    """Extract model name from an llm_config dict or variable reference."""
    if isinstance(node, ast.Name) and node.id in var_map:
        node = var_map[node.id]

    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is None:
                continue
            key_str = _get_str(k) if isinstance(k, ast.Constant) else ""
            if key_str == "model":
                return _get_str(v) or "unknown"
            if key_str == "config_list":
                return _extract_model_from_config_list(v, var_map)
    return "unknown"


def _extract_model_from_config_list(node: ast.expr, var_map: dict) -> str:
    """Extract model from a config_list (list of dicts or variable)."""
    if isinstance(node, ast.Name) and node.id in var_map:
        node = var_map[node.id]

    if isinstance(node, ast.List) and node.elts:
        first = node.elts[0]
        if isinstance(first, ast.Dict):
            for k, v in zip(first.keys, first.values):
                if k is None:
                    continue
                if _get_str(k) == "model":
                    return _get_str(v) or "unknown"
    return "unknown"


def _resolve_list_vars(node: ast.expr) -> list[str]:
    """Extract variable names from a list expression."""
    names = []
    if isinstance(node, ast.List):
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.append(elt.value)
    return names


# ═══════════════════════════════════════════════════════════════
# AST Extraction: Agents
# ═══════════════════════════════════════════════════════════════

def _extract_agents(tree: ast.Module, var_map: dict) -> dict[str, AgentInfo]:
    """Extract all AutoGen agent definitions."""
    agents: dict[str, AgentInfo] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        func_name = _call_func_name(call)

        if func_name not in AGENT_CLASSES:
            continue

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            var_name = target.id

            name = _get_str_keyword(call, "name") or var_name
            agent = AgentInfo(var_name, name, func_name)

            agent.system_message = _get_str_keyword(call, "system_message")
            agent.description = _get_str_keyword(call, "description")

            # Extract model from llm_config
            llm_cfg = _get_keyword(call, "llm_config")
            if llm_cfg:
                agent.model = _extract_model_from_llm_config(llm_cfg, var_map)

            # v0.4: model_client
            model_client = _get_keyword(call, "model_client")
            if model_client and isinstance(model_client, ast.Call):
                m = _get_str_keyword(model_client, "model")
                if m:
                    agent.model = m

            # Human input mode
            him = _get_str_keyword(call, "human_input_mode")
            if him:
                agent.human_input_mode = him

            # Code execution
            cec = _get_keyword(call, "code_execution_config")
            if cec:
                if isinstance(cec, ast.Dict):
                    agent.code_execution = True
                elif isinstance(cec, ast.Constant) and cec.value not in (False, None):
                    agent.code_execution = True

            # Max auto reply
            agent.max_auto_reply = _get_int_keyword(call, "max_consecutive_auto_reply")

            # Termination message
            term = _get_keyword(call, "is_termination_msg")
            if term:
                agent.termination_msg = ast.dump(term)

            # v0.4: tools list
            tools_node = _get_keyword(call, "tools")
            if tools_node:
                agent.tools = _resolve_list_vars(tools_node)

            # v0.4: handoffs
            handoffs_node = _get_keyword(call, "handoffs")
            if handoffs_node:
                if isinstance(handoffs_node, ast.List):
                    for elt in handoffs_node.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            agent.handoffs.append(elt.value)
                        elif isinstance(elt, ast.Call) and _call_func_name(elt) == "Handoff":
                            target = _get_str_keyword(elt, "target")
                            if target:
                                agent.handoffs.append(target)

            agents[var_name] = agent

    return agents


# ═══════════════════════════════════════════════════════════════
# AST Extraction: GroupChat + Manager
# ═══════════════════════════════════════════════════════════════

def _extract_group_chats(tree: ast.Module, var_map: dict) -> dict[str, GroupChatInfo]:
    """Extract GroupChat definitions."""
    groups: dict[str, GroupChatInfo] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        func_name = _call_func_name(call)

        if func_name not in GROUP_CLASSES:
            continue

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            var_name = target.id

            gc = GroupChatInfo(var_name)

            # agents list
            agents_node = _get_keyword(call, "agents") or _get_keyword(call, "participants")
            if agents_node:
                gc.agent_vars = _resolve_list_vars(agents_node)

            # max_round / max_turns
            gc.max_round = _get_int_keyword(call, "max_round") or _get_int_keyword(call, "max_turns")

            # speaker selection
            sm = _get_str_keyword(call, "speaker_selection_method")
            if sm:
                gc.speaker_method = sm

            # Map class name to speaker method for v0.4
            if func_name == "RoundRobinGroupChat":
                gc.speaker_method = "round_robin"
            elif func_name in ("SelectorGroupChat", "Swarm", "MagenticOneGroupChat"):
                gc.speaker_method = "auto"

            # Speaker transitions
            trans = _get_keyword(call, "allowed_or_disallowed_speaker_transitions")
            if isinstance(trans, ast.Dict):
                for k, v in zip(trans.keys, trans.values):
                    if k is None:
                        continue
                    src = k.id if isinstance(k, ast.Name) else ""
                    if src and isinstance(v, ast.List):
                        gc.speaker_transitions[src] = [
                            elt.id for elt in v.elts if isinstance(elt, ast.Name)
                        ]

            gc.allow_repeat = _get_bool_keyword(call, "allow_repeat_speaker") or False

            groups[var_name] = gc

    return groups


def _extract_managers(tree: ast.Module, var_map: dict) -> dict[str, ManagerInfo]:
    """Extract GroupChatManager definitions."""
    managers: dict[str, ManagerInfo] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        func_name = _call_func_name(call)

        if func_name not in MANAGER_CLASSES:
            continue

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            var_name = target.id

            mgr = ManagerInfo(var_name)

            gc_node = _get_keyword(call, "groupchat")
            if isinstance(gc_node, ast.Name):
                mgr.groupchat_var = gc_node.id

            llm_cfg = _get_keyword(call, "llm_config")
            if llm_cfg:
                mgr.model = _extract_model_from_llm_config(llm_cfg, var_map)

            managers[var_name] = mgr

    return managers


# ═══════════════════════════════════════════════════════════════
# AST Extraction: Tools
# ═══════════════════════════════════════════════════════════════

def _extract_tools(tree: ast.Module) -> dict[str, ToolInfo]:
    """Extract tool registrations from decorators and method calls."""
    tools: dict[str, ToolInfo] = {}

    # Pass 1: Decorator-based registration
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            if not isinstance(dec.func, ast.Attribute):
                continue

            attr_name = dec.func.attr
            obj = dec.func.value

            if attr_name == "register_for_llm" and isinstance(obj, ast.Name):
                tool = tools.setdefault(node.name, ToolInfo(node.name))
                tool.caller_var = obj.id
                desc = _get_str_keyword(dec, "description")
                if desc:
                    tool.description = desc
                elif node.body and isinstance(node.body[0], ast.Expr):
                    ds = node.body[0].value
                    if isinstance(ds, ast.Constant) and isinstance(ds.value, str):
                        tool.description = ds.value

            elif attr_name == "register_for_execution" and isinstance(obj, ast.Name):
                tool = tools.setdefault(node.name, ToolInfo(node.name))
                tool.executor_var = obj.id

    # Pass 2: autogen.register_function() calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        fn = _call_func_name(call)
        if fn != "register_function":
            continue

        # First positional arg is the function
        if call.args and isinstance(call.args[0], ast.Name):
            func_name = call.args[0].id
            tool = tools.setdefault(func_name, ToolInfo(func_name))

            caller = _get_keyword(call, "caller")
            if isinstance(caller, ast.Name):
                tool.caller_var = caller.id

            executor = _get_keyword(call, "executor")
            if isinstance(executor, ast.Name):
                tool.executor_var = executor.id

            tool.description = _get_str_keyword(call, "description") or ""
            name = _get_str_keyword(call, "name")
            if name:
                tool.func_name = name

    # Pass 3: v0.4 FunctionTool() constructors
    # Also build a map from variable name → tool function name for resolving agent tools=[]
    tool_var_map: dict[str, str] = {}  # var_name → tool func_name
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if _call_func_name(node.value) not in TOOL_CLASSES:
            continue

        call = node.value
        var_name = ""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                break

        func_ref = ""
        if call.args and isinstance(call.args[0], ast.Name):
            func_ref = call.args[0].id

        desc = _get_str_keyword(call, "description")
        tool_name = func_ref or var_name
        tool = tools.setdefault(tool_name, ToolInfo(tool_name))
        if desc:
            tool.description = desc

        if var_name:
            tool_var_map[var_name] = tool_name

    return tools, tool_var_map


# ═══════════════════════════════════════════════════════════════
# AST Extraction: Chat Initiations
# ═══════════════════════════════════════════════════════════════

def _extract_chats(tree: ast.Module) -> list[ChatInfo]:
    """Extract initiate_chat() and initiate_chats() calls."""
    chats: list[ChatInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue

        method = call.func.attr
        caller_node = call.func.value

        if method == "initiate_chat" and isinstance(caller_node, ast.Name):
            caller_var = caller_node.id
            recipient_var = ""
            if call.args and isinstance(call.args[0], ast.Name):
                recipient_var = call.args[0].id

            chat = ChatInfo(caller_var, recipient_var)
            chat.message = _get_str_keyword(call, "message")
            chat.max_turns = _get_int_keyword(call, "max_turns")
            chat.summary_method = _get_str_keyword(call, "summary_method")
            chats.append(chat)

        elif method == "initiate_chats" and isinstance(caller_node, ast.Name):
            caller_var = caller_node.id
            if call.args and isinstance(call.args[0], ast.List):
                for elt in call.args[0].elts:
                    if isinstance(elt, ast.Dict):
                        recipient_var = ""
                        for k, v in zip(elt.keys, elt.values):
                            if k is None:
                                continue
                            ks = _get_str(k) if isinstance(k, ast.Constant) else ""
                            if ks == "recipient" and isinstance(v, ast.Name):
                                recipient_var = v.id
                        if recipient_var:
                            chat = ChatInfo(caller_var, recipient_var)
                            for k, v in zip(elt.keys, elt.values):
                                if k is None:
                                    continue
                                ks = _get_str(k) if isinstance(k, ast.Constant) else ""
                                if ks == "message":
                                    chat.message = _get_str(v)
                                elif ks == "max_turns":
                                    chat.max_turns = _get_int(v)
                                elif ks == "summary_method":
                                    chat.summary_method = _get_str(v)
                            chats.append(chat)

    return chats


def _extract_nested_chats(tree: ast.Module) -> list[NestedChatInfo]:
    """Extract register_nested_chats() calls."""
    nested: list[NestedChatInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "register_nested_chats":
            continue
        if not isinstance(call.func.value, ast.Name):
            continue

        nc = NestedChatInfo(call.func.value.id)

        trigger = _get_keyword(call, "trigger")
        if isinstance(trigger, ast.Name):
            nc.trigger_var = trigger.id

        chat_queue = _get_keyword(call, "chat_queue")
        if isinstance(chat_queue, ast.List):
            for elt in chat_queue.elts:
                if isinstance(elt, ast.Dict):
                    entry = {}
                    for k, v in zip(elt.keys, elt.values):
                        if k is None:
                            continue
                        ks = _get_str(k) if isinstance(k, ast.Constant) else ""
                        if ks in ("sender", "recipient") and isinstance(v, ast.Name):
                            entry[ks] = v.id
                        elif ks in ("message", "summary_method"):
                            entry[ks] = _get_str(v)
                        elif ks == "max_turns":
                            entry[ks] = _get_int(v)
                    nc.chat_queue.append(entry)

        nested.append(nc)

    return nested


# ═══════════════════════════════════════════════════════════════
# API Version Detection
# ═══════════════════════════════════════════════════════════════

def _detect_api_version(tree: ast.Module) -> str:
    """Detect whether code uses v0.2 or v0.4 API from imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "autogen_agentchat" in node.module:
                return "v0.4"
            if "autogen_core" in node.module:
                return "v0.4"
            if node.module.startswith("ag2"):
                return "v0.4"
    return "v0.2"


# ═══════════════════════════════════════════════════════════════
# Spec Assembly
# ═══════════════════════════════════════════════════════════════

def import_autogen(source_path: Path) -> dict:
    """Parse an AutoGen Python file and produce an Agent Ontology YAML spec."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    var_map = _build_var_map(tree)

    api_version = _detect_api_version(tree)

    agents = _extract_agents(tree, var_map)
    groups = _extract_group_chats(tree, var_map)
    managers = _extract_managers(tree, var_map)
    tools, tool_var_map = _extract_tools(tree)
    chats = _extract_chats(tree)
    nested = _extract_nested_chats(tree)

    # Resolve v0.4 agent tools=[] variable references to tool function names
    for var_name, ag in agents.items():
        resolved_tools = []
        for t in ag.tools:
            if t in tool_var_map:
                resolved_tools.append(tool_var_map[t])
            elif t in tools:
                resolved_tools.append(t)
        ag.tools = resolved_tools
        # Register as caller for these tools
        for tool_name in ag.tools:
            if tool_name in tools:
                tools[tool_name].caller_var = var_name

    # Build spec
    spec_name = source_path.stem.replace("_", " ").title()
    spec: dict[str, Any] = {
        "name": spec_name,
        "version": "1.0",
        "description": f"Imported from AutoGen ({api_version}) file: {source_path.name}",
        "entities": [],
        "processes": [],
        "edges": [],
        "schemas": [],
    }

    entities = spec["entities"]
    processes = spec["processes"]
    edges = spec["edges"]
    schemas = spec["schemas"]

    # Track var_name -> entity_id mapping
    var_to_id: dict[str, str] = {}
    # Track which agents have which tools (agent_var -> [tool_id, ...])
    agent_tools: dict[str, list[str]] = {}

    # ── Entities from agents ──
    for var_name, ag in agents.items():
        eid = ag.id
        var_to_id[var_name] = eid

        if ag.is_human:
            # Create both a human entity and an agent entity
            entities.append({
                "id": f"{eid}_human",
                "type": "human",
                "label": f"{_snake_to_label(ag.name)} (Human)",
                "role": "user",
            })
            entities.append({
                "id": eid,
                "type": "agent",
                "label": _snake_to_label(ag.name),
                "model": ag.model,
                **({"system_prompt": ag.system_message} if ag.system_message else {}),
                **({"description": ag.description} if ag.description else {}),
            })
        else:
            entities.append({
                "id": eid,
                "type": "agent",
                "label": _snake_to_label(ag.name),
                "model": ag.model,
                **({"system_prompt": ag.system_message} if ag.system_message else {}),
                **({"description": ag.description} if ag.description else {}),
            })

        # Code execution → tool entity (invoke edges added from steps later)
        if ag.code_execution:
            tool_id = f"{eid}_code_exec"
            entities.append({
                "id": tool_id,
                "type": "tool",
                "label": f"{_snake_to_label(ag.name)} Code Executor",
                "tool_type": "shell",
            })
            # Track as a tool for this agent
            agent_tools.setdefault(var_name, []).append(tool_id)

    # ── Tool entities ──
    for func_name, tool in tools.items():
        tid = tool.id
        entities.append({
            "id": tid,
            "type": "tool",
            "label": _snake_to_label(func_name),
            "tool_type": "function",
            **({"description": tool.description} if tool.description else {}),
        })

        # Track tool-to-agent mapping (invoke edges added later from steps)
        if tool.caller_var:
            agent_tools.setdefault(tool.caller_var, []).append(tid)

    # ── Team entities from GroupChat ──
    for var_name, gc in groups.items():
        team_id = _sanitize_id(var_name)
        var_to_id[var_name] = team_id

        # Map speaker_method to strategy
        strategy_map = {
            "auto": "dynamic",
            "round_robin": "round_robin",
            "random": "dynamic",
            "manual": "sequential",
        }
        strategy = strategy_map.get(gc.speaker_method, "sequential")

        member_ids = [var_to_id.get(v, _sanitize_id(v)) for v in gc.agent_vars]

        entities.append({
            "id": team_id,
            "type": "team",
            "label": _snake_to_label(var_name),
            "strategy": strategy,
            "members": member_ids,
            **({"max_iterations": gc.max_round} if gc.max_round else {}),
        })

        # Speaker transitions → handoff edges
        for src_var, dst_vars in gc.speaker_transitions.items():
            src_id = var_to_id.get(src_var, _sanitize_id(src_var))
            for dst_var in dst_vars:
                dst_id = var_to_id.get(dst_var, _sanitize_id(dst_var))
                edges.append({
                    "type": "handoff",
                    "from": src_id,
                    "to": dst_id,
                    "label": f"{_snake_to_label(src_var)} → {_snake_to_label(dst_var)}",
                })

    # ── Manager entities ──
    for var_name, mgr in managers.items():
        mgr_id = _sanitize_id(var_name)
        var_to_id[var_name] = mgr_id
        entities.append({
            "id": mgr_id,
            "type": "agent",
            "label": _snake_to_label(var_name),
            "model": mgr.model,
            "description": "GroupChat coordinator",
        })

    # ── v0.4 Handoff edges ──
    for var_name, ag in agents.items():
        for target_name in ag.handoffs:
            target_id = _sanitize_id(target_name)
            edges.append({
                "type": "handoff",
                "from": var_to_id.get(var_name, ag.id),
                "to": target_id,
                "label": f"Handoff to {_snake_to_label(target_name)}",
            })

    # ── Processes from chat initiations ──
    entry_point = None
    prev_step_id = None

    for i, chat in enumerate(chats):
        step_id = f"chat_{i}" if len(chats) > 1 else "main_chat"
        caller_id = var_to_id.get(chat.caller_var, _sanitize_id(chat.caller_var))
        recipient_id = var_to_id.get(chat.recipient_var, _sanitize_id(chat.recipient_var))

        proc: dict[str, Any] = {
            "id": step_id,
            "type": "step",
            "label": f"Chat: {_snake_to_label(chat.caller_var)} → {_snake_to_label(chat.recipient_var)}",
        }
        if chat.message:
            proc["description"] = f"Message: {chat.message[:100]}"

        processes.append(proc)

        # Invoke edge to recipient
        edges.append({
            "type": "invoke",
            "from": step_id,
            "to": recipient_id,
            "label": f"Talk to {_snake_to_label(chat.recipient_var)}",
        })

        # Wire tools for the recipient agent from this step
        for tool_id in agent_tools.get(chat.recipient_var, []):
            edges.append({
                "type": "invoke",
                "from": step_id,
                "to": tool_id,
                "label": f"Use {_snake_to_label(tool_id)}",
            })

        # Flow from previous step
        if prev_step_id:
            edges.append({
                "type": "flow",
                "from": prev_step_id,
                "to": step_id,
                "label": _snake_to_label(step_id),
            })

        if entry_point is None:
            entry_point = step_id

        prev_step_id = step_id

    # ── GroupChat loop processes ──
    for var_name, gc in groups.items():
        # If no chats reference the group, create a standalone loop
        gc_step_id = f"{_sanitize_id(var_name)}_loop"
        gate_id = f"{_sanitize_id(var_name)}_check"

        processes.append({
            "id": gc_step_id,
            "type": "step",
            "label": f"{_snake_to_label(var_name)} Round",
            "description": f"Execute one round of {_snake_to_label(var_name)}",
        })

        # Gate: check if group chat should continue
        condition = f"round < {gc.max_round}" if gc.max_round else "not terminated"
        processes.append({
            "id": gate_id,
            "type": "gate",
            "label": f"Continue {_snake_to_label(var_name)}?",
            "condition": condition,
            "branches": [
                {"condition": "continue", "target": gc_step_id},
                {"condition": "done", "target": "_done"},
            ],
        })

        # Flow: step → gate
        edges.append({
            "type": "flow",
            "from": gc_step_id,
            "to": gate_id,
            "label": "Check",
        })

        # Branch: gate → done
        edges.append({
            "type": "branch",
            "from": gate_id,
            "to": "_done",
            "label": "Finished",
        })

        # Loop: gate → step (back edge)
        edges.append({
            "type": "loop",
            "from": gate_id,
            "to": gc_step_id,
            "label": "Next Round",
        })

        # Invoke edges from the loop step to each member agent and their tools
        for agent_var in gc.agent_vars:
            agent_id = var_to_id.get(agent_var, _sanitize_id(agent_var))
            edges.append({
                "type": "invoke",
                "from": gc_step_id,
                "to": agent_id,
                "label": f"Call {_snake_to_label(agent_var)}",
            })
            # Wire tool invokes from the loop step
            for tool_id in agent_tools.get(agent_var, []):
                edges.append({
                    "type": "invoke",
                    "from": gc_step_id,
                    "to": tool_id,
                    "label": f"Use {_snake_to_label(tool_id)}",
                })

        if entry_point is None:
            entry_point = gc_step_id

        # Connect previous chat to group if exists
        if prev_step_id and prev_step_id != gc_step_id:
            edges.append({
                "type": "flow",
                "from": prev_step_id,
                "to": gc_step_id,
                "label": _snake_to_label(gc_step_id),
            })

    # ── Nested chat processes ──
    for nc in nested:
        for j, entry in enumerate(nc.chat_queue):
            nc_step_id = f"nested_{nc.agent_var}_{j}"
            recipient = entry.get("recipient", "")
            recipient_id = var_to_id.get(recipient, _sanitize_id(recipient))

            processes.append({
                "id": nc_step_id,
                "type": "step",
                "label": f"Nested Chat {j + 1}",
                **({"description": entry.get("message", "")[:100]} if entry.get("message") else {}),
            })

            edges.append({
                "type": "invoke",
                "from": nc_step_id,
                "to": recipient_id,
                "label": f"Talk to {_snake_to_label(recipient)}",
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

        # Connect last step to done (only if no GroupChat loop handles termination)
        if prev_step_id and not groups:
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

    # ── Schemas (basic) ──
    schemas.append({
        "name": "ChatMessage",
        "fields": [
            {"name": "content", "type": "string", "description": "Message content"},
            {"name": "role", "type": "string", "description": "Message role (user/assistant)"},
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
        description="Import an AutoGen Python file into an Agent Ontology YAML spec"
    )
    parser.add_argument("input", help="Path to AutoGen Python file")
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
        spec = import_autogen(input_path)
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
    sys.exit(main())
