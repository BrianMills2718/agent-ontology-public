#!/usr/bin/env python3
"""
LangGraph Importer — Parse LangGraph StateGraph Python files into Agent Ontology YAML specs.

Uses Python AST to extract:
  - TypedDict state class → schemas
  - add_node() calls → processes (step)
  - add_edge() calls → edges (flow)
  - add_conditional_edges() calls → processes (gate) + edges (branch)
  - set_entry_point() → entry_point
  - LLM calls in node functions → entities (agent) + edges (invoke)

Usage:
    python3 import_langgraph.py agent_lg.py
    python3 import_langgraph.py agent_lg.py -o specs/imported.yaml
    python3 import_langgraph.py agent_lg.py --validate
"""

import argparse
import ast
import sys
from pathlib import Path

import yaml


# ═══════════════════════════════════════════════════════════════
# AST Extraction
# ═══════════════════════════════════════════════════════════════

def _get_string(node):
    """Extract string value from AST Constant or JoinedStr."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _get_dict_literal(node):
    """Extract a dict literal from AST Dict node. Handles END variable as '__END__'."""
    if not isinstance(node, ast.Dict):
        return None
    result = {}
    for k, v in zip(node.keys, node.values):
        key = _get_string(k)
        val = _get_string(v)
        # Handle END variable reference
        if val is None and isinstance(v, ast.Name) and v.id == "END":
            val = "__END__"
        if key is not None and val is not None:
            result[key] = val
    return result


def _find_typed_dicts(tree):
    """Find TypedDict, BaseModel, and dataclass definitions -> list of (name, fields).

    Handles TypedDict, MessagesState, BaseModel, dataclass, and other state base classes.
    """
    # Known state base classes (extend TypedDict or similar)
    STATE_BASES = {"TypedDict", "MessagesState", "AgentState", "BaseModel"}
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Check if base class is TypedDict or a known state class
        is_typed_dict = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in STATE_BASES:
                is_typed_dict = True
            elif isinstance(base, ast.Attribute) and base.attr in STATE_BASES:
                is_typed_dict = True
        # Check for @dataclass decorator
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "dataclass":
                is_typed_dict = True
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "dataclass":
                is_typed_dict = True
            elif isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
                is_typed_dict = True
        for kw in node.keywords:
            pass  # total=False etc
        if not is_typed_dict:
            continue

        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fname = item.target.id
                ftype = _annotation_to_type(item.annotation)
                fields.append({"name": fname, "type": ftype})
        results.append((node.name, fields))
    return results


def _annotation_to_type(ann):
    """Convert a Python type annotation AST to a spec type string."""
    if isinstance(ann, ast.Name):
        mapping = {
            "str": "string", "int": "integer", "float": "float",
            "bool": "boolean", "dict": "object", "list": "list",
        }
        return mapping.get(ann.id, ann.id)
    if isinstance(ann, ast.Constant):
        return str(ann.value)
    if isinstance(ann, ast.Subscript):
        base = _annotation_to_type(ann.value)
        if base == "list":
            inner = _annotation_to_type(ann.slice)
            return f"list<{inner}>"
        if base == "dict":
            return "object"
        if base == "Annotated":
            # Annotated[type, reducer] — extract base type
            if isinstance(ann.slice, ast.Tuple) and ann.slice.elts:
                return _annotation_to_type(ann.slice.elts[0])
        return base
    if isinstance(ann, ast.Attribute):
        return ann.attr
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        # X | Y union type
        left = _annotation_to_type(ann.left)
        right = _annotation_to_type(ann.right)
        return f"{left} | {right}"
    return "object"


def _make_empty_graph():
    """Create an empty graph info dict."""
    return {
        "state_class": None,
        "entry_point": None,
        "nodes": [],
        "edges": [],
        "conditional_edges": [],
        "end_edges": [],
        "var_name": None,
        "subgraph_nodes": set(),
    }


def _find_all_graphs(tree, source_lines):
    """Find ALL StateGraph constructions and their builder API calls.

    Returns a list of graph info dicts, one per StateGraph.
    Each has: state_class, entry_point, nodes, edges, conditional_edges, end_edges, var_name.
    """
    # Map graph variable name → graph info
    graphs_by_var: dict[str, dict] = {}
    # Track order of graph creation
    graph_order: list[str] = []

    # First pass: find all StateGraph() constructions
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        val = node.value
        if not isinstance(val, ast.Call):
            continue
        func = val.func
        # Handle both StateGraph(...) and module.StateGraph(...)
        is_state_graph = False
        if isinstance(func, ast.Name) and func.id == "StateGraph":
            is_state_graph = True
        elif isinstance(func, ast.Attribute) and func.attr == "StateGraph":
            is_state_graph = True
        if not is_state_graph:
            continue

        # Handle both `graph = StateGraph(...)` and `self.graph = StateGraph(...)`
        var_name = None
        if isinstance(target, ast.Name):
            var_name = target.id
        elif isinstance(target, ast.Attribute):
            # self.graph = StateGraph(...) → use "self.graph" as key
            if isinstance(target.value, ast.Name):
                var_name = f"{target.value.id}.{target.attr}"

        if not var_name:
            continue

        g = _make_empty_graph()
        g["var_name"] = var_name
        if val.args:
            arg = val.args[0]
            if isinstance(arg, ast.Name):
                g["state_class"] = arg.id
        graphs_by_var[var_name] = g
        graph_order.append(var_name)

    if not graphs_by_var:
        # Fallback: return a single empty graph
        g = _make_empty_graph()
        return [g]

    # Track compiled graph variables: app = graph.compile() → app is a compiled graph
    compiled_graph_vars: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        val = node.value
        if not isinstance(val, ast.Call):
            continue
        func = val.func
        if isinstance(func, ast.Attribute) and func.attr == "compile":
            obj = func.value
            obj_name = None
            if isinstance(obj, ast.Name):
                obj_name = obj.id
            elif isinstance(obj, ast.Attribute) and isinstance(obj.value, ast.Name):
                obj_name = f"{obj.value.id}.{obj.attr}"
            if obj_name in graphs_by_var and isinstance(target, ast.Name):
                compiled_graph_vars.add(target.id)

    # Second pass: associate method calls with their graph variable
    for node in ast.walk(tree):
        call = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
        if call is None:
            continue

        func = call.func
        if not isinstance(func, ast.Attribute):
            continue

        method = func.attr
        obj = func.value
        obj_name = None
        if isinstance(obj, ast.Name):
            obj_name = obj.id
        elif isinstance(obj, ast.Attribute) and isinstance(obj.value, ast.Name):
            # self.graph.add_node(...) → "self.graph"
            obj_name = f"{obj.value.id}.{obj.attr}"

        if obj_name not in graphs_by_var:
            continue

        g = graphs_by_var[obj_name]

        if method == "set_entry_point" and call.args:
            ep = _get_string(call.args[0])
            if ep:
                g["entry_point"] = ep

        elif method == "add_node" and len(call.args) >= 1:
            name = _get_string(call.args[0])
            func_ref = call.args[1] if len(call.args) >= 2 else None
            func_name = None
            is_subgraph = False
            if func_ref is None:
                # add_node(name) with no function — might use 'action' kwarg
                for kw in call.keywords:
                    if kw.arg == "action":
                        if isinstance(kw.value, ast.Name):
                            func_name = kw.value.id
                        elif isinstance(kw.value, ast.Attribute):
                            func_name = kw.value.attr
            elif isinstance(func_ref, ast.Name):
                func_name = func_ref.id
                # Check if this variable is a compiled graph
                if func_ref.id in compiled_graph_vars:
                    is_subgraph = True
                    func_name = "__subgraph__"
            elif isinstance(func_ref, ast.Attribute):
                func_name = func_ref.attr
                # Detect compiled subgraph: sub_graph.compile() or self.sub.compile()
                if func_name == "compile":
                    is_subgraph = True
                    func_name = "__subgraph__"
            elif isinstance(func_ref, ast.Lambda):
                func_name = "__lambda__"
            elif isinstance(func_ref, ast.Call):
                # add_node("name", sub_graph.compile()) — Call node
                cfunc = func_ref.func
                if isinstance(cfunc, ast.Attribute) and cfunc.attr == "compile":
                    is_subgraph = True
                    func_name = "__subgraph__"
            if name:
                g["nodes"].append((name, func_name))
                if is_subgraph:
                    g.setdefault("subgraph_nodes", set()).add(name)

        elif method == "add_edge" and len(call.args) >= 2:
            src = _get_string(call.args[0])
            dst = call.args[1]
            dst_str = _get_string(dst)
            if dst_str is None and isinstance(dst, ast.Name) and dst.id == "END":
                dst_str = "__END__"
            # Also handle START
            if src is None and isinstance(call.args[0], ast.Name) and call.args[0].id == "START":
                src = "__START__"
            if src and dst_str:
                if dst_str == "__END__":
                    g["end_edges"].append(src)
                elif src == "__START__":
                    g["entry_point"] = dst_str
                else:
                    g["edges"].append((src, dst_str))

        elif method == "set_finish_point" and call.args:
            ep = _get_string(call.args[0])
            if ep:
                g["end_edges"].append(ep)

        elif method == "add_sequence" and call.args:
            # add_sequence([node1, node2, ...]) — adds nodes in order with edges
            arg = call.args[0]
            seq_names = []
            if isinstance(arg, ast.List):
                for elt in arg.elts:
                    name = _get_string(elt)
                    if name:
                        seq_names.append((name, None))
                    elif isinstance(elt, ast.Name):
                        seq_names.append((elt.id, elt.id))
            for name, func_name in seq_names:
                existing = {n for n, _ in g["nodes"]}
                if name not in existing:
                    g["nodes"].append((name, func_name))
            for i in range(1, len(seq_names)):
                g["edges"].append((seq_names[i - 1][0], seq_names[i][0]))
            if seq_names and not g["entry_point"]:
                g["entry_point"] = seq_names[0][0]

        elif method == "add_conditional_edges" and len(call.args) >= 2:
            src = _get_string(call.args[0])
            router = call.args[1]
            router_name = None
            if isinstance(router, ast.Name):
                router_name = router.id
            # Mapping can be arg[2] or omitted (router returns node names directly)
            mapping = None
            if len(call.args) >= 3:
                mapping = _get_dict_literal(call.args[2])
            if src and mapping:
                g["conditional_edges"].append((src, router_name, mapping))
            elif src and router_name and not mapping:
                # Router returns node names directly — we'll infer targets from function body
                g["conditional_edges"].append((src, router_name, None))

    return [graphs_by_var[v] for v in graph_order]


def _find_command_routes(tree):
    """Find Command(goto=...) return patterns in node functions.

    Returns dict: func_name -> list of target node names.
    """
    routes: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        targets = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            fname = None
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname != "Command":
                continue
            # Extract goto= keyword
            for kw in child.keywords:
                if kw.arg == "goto":
                    val = _get_string(kw.value)
                    if val:
                        targets.append(val)
                    elif isinstance(kw.value, ast.Name):
                        targets.append(kw.value.id)
                    elif isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            s = _get_string(elt)
                            if s:
                                targets.append(s)
        if targets:
            routes[node.name] = targets
    return routes


def _find_send_targets(tree):
    """Find Send(node_name, ...) calls in router functions.

    LangGraph's Send() enables dynamic map-reduce fan-out where conditional edges
    return [Send("node", data) for item in items].

    Returns dict: func_name -> list of target node names.
    """
    routes: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        targets = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            fname = None
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname != "Send":
                continue
            # Send(node_name, arg) — first arg is the target node
            if child.args:
                target = _get_string(child.args[0])
                if target and target not in targets:
                    targets.append(target)
        if targets:
            routes[node.name] = targets
    return routes


def _find_interrupt_nodes(tree):
    """Find interrupt() calls in node functions and compile(interrupt_before/after).

    Returns dict with:
      - 'interrupt_calls': {func_name: True} for functions that call interrupt()
      - 'interrupt_before': list of node names from compile(interrupt_before=[...])
      - 'interrupt_after': list of node names from compile(interrupt_after=[...])
    """
    result = {"interrupt_calls": {}, "interrupt_before": [], "interrupt_after": []}

    for node in ast.walk(tree):
        # Find interrupt() calls inside functions
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                fname = None
                if isinstance(func, ast.Name):
                    fname = func.id
                elif isinstance(func, ast.Attribute):
                    fname = func.attr
                if fname == "interrupt":
                    result["interrupt_calls"][node.name] = True
                    break

        # Find compile(interrupt_before=[...], interrupt_after=[...])
        if isinstance(node, ast.Call):
            func = node.func
            fname = None
            if isinstance(func, ast.Attribute):
                fname = func.attr
            if fname != "compile":
                continue
            for kw in node.keywords:
                if kw.arg in ("interrupt_before", "interrupt_after"):
                    names = []
                    if isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            s = _get_string(elt)
                            if s:
                                names.append(s)
                    result[kw.arg] = names

    return result


def _resolve_loop_add_nodes(tree, graph_var_names):
    """Resolve add_node() calls inside for-loops that iterate over tuple lists.

    Handles patterns like:
        nodes = [("name1", func1), ("name2", func2)]
        for i in range(len(nodes)):
            name, node = nodes[i]
            builder.add_node(name, node)

    Returns list of (graph_var, node_name, func_name) tuples.
    """
    # Find list assignments with string-tuple elements: var = [("str", name), ...]
    tuple_lists: dict[str, list[tuple[str, str]]] = {}
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        val = node.value
        if not isinstance(target, ast.Name) or not isinstance(val, ast.List):
            continue
        items = []
        for elt in val.elts:
            if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                first = elt.elts[0]
                second = elt.elts[1]
                name_str = _get_string(first)
                func_str = second.id if isinstance(second, ast.Name) else None
                if name_str:
                    items.append((name_str, func_str or "__unknown__"))
        if items:
            tuple_lists[target.id] = items

    # Find for-loops that call graph.add_node() using loop variables
    resolved = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # Check body for graph.add_node calls
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not isinstance(func, ast.Attribute) or func.attr != "add_node":
                continue
            obj = func.value
            if not isinstance(obj, ast.Name) or obj.id not in graph_var_names:
                continue
            # This is a graph.add_node call in a for-loop
            # Try to find which tuple list is being iterated
            # Look at the for-loop's iter for references to tuple_lists
            iter_var = None
            if isinstance(node.iter, ast.Call):
                # range(len(var)) pattern
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                    if node.iter.args:
                        arg = node.iter.args[0]
                        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "len":
                            if arg.args and isinstance(arg.args[0], ast.Name):
                                iter_var = arg.args[0].id
            elif isinstance(node.iter, ast.Name):
                iter_var = node.iter.id

            if iter_var and iter_var in tuple_lists:
                for name, func_name in tuple_lists[iter_var]:
                    resolved.append((obj.id, name, func_name))

    return resolved



def _find_node_functions(tree, source_lines):
    """Extract node function bodies for logic/LLM analysis.

    Returns dict: func_name -> {body_source, calls_llm, llm_model, system_prompt_var}
    """
    functions = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        fn_name = node.name
        # Get source of function body
        if node.body:
            start = node.body[0].lineno - 1
            end = node.end_lineno
            body_lines = source_lines[start:end]
            body_source = "\n".join(body_lines)
        else:
            body_source = ""

        info = {
            "body_source": body_source,
            "calls_llm": False,
            "model": None,
            "system_prompt": None,
            "invokes_func": None,
        }

        # Walk function body for LLM call patterns
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                cfunc = child.func
                cname = None
                if isinstance(cfunc, ast.Name):
                    cname = cfunc.id
                elif isinstance(cfunc, ast.Attribute):
                    cname = cfunc.attr

                if cname and ("call_llm" in cname or "invoke" in cname.lower()):
                    info["calls_llm"] = True
                    # Try to extract function name being called
                    if cname.startswith("invoke_"):
                        info["invokes_func"] = cname

        functions[fn_name] = info
    return functions


def _find_invoke_functions(tree):
    """Find invoke_* wrapper functions that call call_llm with a specific agent.

    Returns dict: func_name -> {model, agent_entity_id, system_prompt_snippet}
    """
    results = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("invoke_"):
            continue

        info = {"model": None, "system_prompt": None}

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                cfunc = child.func
                cname = None
                if isinstance(cfunc, ast.Name):
                    cname = cfunc.id
                if cname == "call_llm":
                    # call_llm(model, system_prompt, user_msg, ...)
                    if len(child.args) >= 2:
                        model_arg = child.args[0]
                        if isinstance(model_arg, ast.Constant):
                            info["model"] = model_arg.value
                        prompt_arg = child.args[1]
                        if isinstance(prompt_arg, ast.Constant):
                            info["system_prompt"] = prompt_arg.value
                        elif isinstance(prompt_arg, ast.Name):
                            info["system_prompt"] = f"<variable:{prompt_arg.id}>"

        results[node.name] = info
    return results


def _find_string_constants(tree):
    """Find module-level string constant assignments.

    Returns dict: var_name -> value
    """
    results = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and len(node.value.value) > 20:
                    results[target.id] = node.value.value
    return results


# ═══════════════════════════════════════════════════════════════
# Routing Function Analysis
# ═══════════════════════════════════════════════════════════════

def _analyze_router(tree, router_name, mapping):
    """Analyze a routing function to extract gate condition and branch conditions.

    Returns: (condition_str, branches)
    """
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == router_name:
            func_node = node
            break

    if mapping is None:
        # No explicit mapping — try to infer from return statements in router
        if func_node:
            inferred_mapping = {}
            for child in ast.walk(func_node):
                if isinstance(child, ast.Return) and child.value:
                    val = _get_string(child.value)
                    if val:
                        inferred_mapping[val] = val
            if inferred_mapping:
                mapping = inferred_mapping
            else:
                return "condition unknown", [], False
        else:
            return "condition unknown", [], False

    if not func_node:
        branches = []
        has_end = False
        for k, v in mapping.items():
            if v == "__END__" or v == "END":
                has_end = True
                branches.append({"condition": k, "target": "__terminal__"})
            else:
                branches.append({"condition": k, "target": v})
        return "condition unknown", branches, has_end

    # Extract condition from docstring or first if-statement
    condition = ""
    docstring = ast.get_docstring(func_node)
    if docstring:
        # Often format: "Gate: Label — condition"
        if "—" in docstring:
            condition = docstring.split("—", 1)[1].strip()
        elif "-" in docstring:
            condition = docstring.split("-", 1)[1].strip()
        else:
            condition = docstring.strip()

    # Fallback: analyze if-statements for condition
    if not condition:
        for child in ast.walk(func_node):
            if isinstance(child, ast.If):
                condition = _if_condition_to_str(child.test)
                break

    branches = []
    has_end = False
    for route_key, target in mapping.items():
        if target == "__END__" or target == "END":
            has_end = True
            branches.append({
                "condition": route_key,
                "target": "__terminal__",
            })
        else:
            branches.append({
                "condition": route_key,
                "target": target,
            })

    return condition, branches, has_end


def _if_condition_to_str(test):
    """Convert an if-test AST node to a human-readable condition string."""
    if isinstance(test, ast.Compare):
        left = _expr_to_str(test.left)
        ops = test.ops
        comparators = test.comparators
        if ops and comparators:
            op_str = _cmpop_to_str(ops[0])
            right = _expr_to_str(comparators[0])
            return f"{left} {op_str} {right}"
    if isinstance(test, ast.Call):
        return _expr_to_str(test)
    if isinstance(test, ast.BoolOp):
        op = "and" if isinstance(test.op, ast.And) else "or"
        parts = [_if_condition_to_str(v) for v in test.values]
        return f" {op} ".join(parts)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return f"not {_if_condition_to_str(test.operand)}"
    return _expr_to_str(test)


def _expr_to_str(node):
    """Simple AST expression to string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute):
        return f"{_expr_to_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        func = _expr_to_str(node.func)
        args = ", ".join(_expr_to_str(a) for a in node.args)
        return f"{func}({args})"
    if isinstance(node, ast.Subscript):
        val = _expr_to_str(node.value)
        sl = _expr_to_str(node.slice)
        return f"{val}[{sl}]"
    return "..."


def _cmpop_to_str(op):
    mapping = {
        ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
        ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
        ast.In: "in", ast.NotIn: "not in",
    }
    return mapping.get(type(op), "?")


# ═══════════════════════════════════════════════════════════════
# Spec Assembly
# ═══════════════════════════════════════════════════════════════

def _snake_to_label(s):
    """Convert snake_case ID to human-readable label."""
    return s.replace("_", " ").title()


def import_langgraph(source_path):
    """Parse a LangGraph Python file and produce an Agent Ontology spec dict."""
    path = Path(source_path)
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source)

    # Extract all components
    typed_dicts = _find_typed_dicts(tree)
    all_graphs = _find_all_graphs(tree, source_lines)
    command_routes = _find_command_routes(tree)
    send_targets = _find_send_targets(tree)
    interrupt_info = _find_interrupt_nodes(tree)
    node_funcs = _find_node_functions(tree, source_lines)
    invoke_funcs = _find_invoke_functions(tree)
    string_consts = _find_string_constants(tree)

    # Resolve loop-based add_node patterns
    graph_var_names = {g["var_name"] for g in all_graphs if g["var_name"]}
    loop_nodes = _resolve_loop_add_nodes(tree, graph_var_names)

    # Add loop-resolved nodes to their respective graphs
    for gvar, node_name, func_name in loop_nodes:
        for g in all_graphs:
            if g["var_name"] == gvar:
                # Avoid duplicates
                existing_names = {n for n, _ in g["nodes"]}
                if node_name not in existing_names:
                    g["nodes"].append((node_name, func_name))
                break

    # Also resolve loop-based add_edge patterns (linear chains from tuple lists)
    # Handle: if i > 0: builder.add_edge(nodes[i-1][0], name)
    for g in all_graphs:
        gvar = g["var_name"]
        if not gvar:
            continue
        loop_graph_nodes = [n for gv, n, f in loop_nodes if gv == gvar]
        if not loop_graph_nodes:
            continue
        # Create linear chain edges between consecutive loop nodes
        existing_edges = set(g["edges"])
        for i in range(1, len(loop_graph_nodes)):
            edge = (loop_graph_nodes[i - 1], loop_graph_nodes[i])
            if edge not in existing_edges:
                g["edges"].append(edge)
        # Infer entry/finish point if not resolved as string constants
        if not g["entry_point"]:
            g["entry_point"] = loop_graph_nodes[0]

    # Merge all graphs (preserving all nodes/edges from all sub-graphs)
    graph_info = _make_empty_graph()
    all_node_names = set()
    for g in all_graphs:
        graph_info["nodes"].extend(g["nodes"])
        graph_info["edges"].extend(g["edges"])
        graph_info["conditional_edges"].extend(g["conditional_edges"])
        graph_info["end_edges"].extend(g["end_edges"])
        graph_info["subgraph_nodes"].update(g.get("subgraph_nodes", set()))
        for name, _ in g["nodes"]:
            all_node_names.add(name)
        if g["state_class"] and not graph_info["state_class"]:
            graph_info["state_class"] = g["state_class"]

    # Entry point from last (main) graph
    for g in reversed(all_graphs):
        if g["entry_point"]:
            graph_info["entry_point"] = g["entry_point"]
            break

    # Add Command-based routing as edges
    # Map func_name → node_name
    func_to_node = {}
    for node_name, func_name in graph_info["nodes"]:
        if func_name:
            func_to_node[func_name] = node_name

    for func_name, targets in command_routes.items():
        src_node = func_to_node.get(func_name)
        if not src_node:
            continue
        for target in targets:
            if target == "END" or target == "__END__":
                graph_info["end_edges"].append(src_node)
            elif target in all_node_names:
                graph_info["edges"].append((src_node, target))

    # Add Send-based fan-out routing as edges
    # Send() targets come from router functions used in add_conditional_edges
    send_based_routers: set[str] = set()
    for func_name, targets in send_targets.items():
        send_based_routers.add(func_name)
        # Find the conditional edge that uses this router
        src_node = None
        for src, router_name, mapping in graph_info["conditional_edges"]:
            if router_name == func_name:
                src_node = src
                break
        if not src_node:
            # Also check if the function is a node function
            src_node = func_to_node.get(func_name)
        if src_node:
            for target in targets:
                if target in all_node_names:
                    graph_info["edges"].append((src_node, target))

    # Collect nodes that have interrupt() or compile(interrupt_before/after)
    interrupt_nodes = set()
    for func_name in interrupt_info["interrupt_calls"]:
        node_name = func_to_node.get(func_name)
        if node_name:
            interrupt_nodes.add(node_name)
    interrupt_nodes.update(interrupt_info["interrupt_before"])
    interrupt_nodes.update(interrupt_info["interrupt_after"])

    # Build spec
    spec = {
        "name": path.stem.replace("_agent_lg", "").replace("_agent", "").replace("_", " ").title(),
        "version": "1.0",
        "description": f"Imported from LangGraph file: {path.name}",
        "entry_point": graph_info["entry_point"] or (graph_info["nodes"][0][0] if graph_info["nodes"] else None),
    }

    entities = []
    processes = []
    edges = []
    schemas = []
    seen_agents = set()
    gate_nodes = set()  # Nodes that are actually gates (conditional edge sources)

    # Identify which nodes are gate sources
    for src, router, mapping in graph_info["conditional_edges"]:
        gate_nodes.add(src)

    # --- Schemas from TypedDicts ---
    for td_name, td_fields in typed_dicts:
        # Filter out control fields
        user_fields = [f for f in td_fields if not f["name"].startswith("_")]
        if user_fields:
            schemas.append({
                "name": td_name,
                "description": f"State schema from TypedDict {td_name}",
                "fields": user_fields,
            })

    # --- Agents from invoke_* functions ---
    for func_name, info in invoke_funcs.items():
        agent_id = func_name.replace("invoke_", "")
        if agent_id in seen_agents:
            continue
        seen_agents.add(agent_id)

        agent = {
            "id": agent_id,
            "type": "agent",
            "label": _snake_to_label(agent_id),
        }
        if info["model"]:
            agent["model"] = info["model"]
        else:
            agent["model"] = "unknown"

        # Resolve system prompt
        prompt = info.get("system_prompt", "")
        if prompt and prompt.startswith("<variable:"):
            var_name = prompt[10:-1]
            prompt = string_consts.get(var_name, f"<see {var_name}>")
        if prompt:
            agent["system_prompt"] = prompt

        entities.append(agent)

    # --- Processes from nodes ---
    subgraph_nodes = graph_info.get("subgraph_nodes", set())
    for node_name, func_name in graph_info["nodes"]:
        if node_name in gate_nodes:
            # This node is a conditional edge source — check if it's a pass-through
            is_passthrough = func_name == "__lambda__"
            # For pass-through gates, we'll create a gate process from
            # conditional_edges info later
            if is_passthrough:
                continue
            # Non-passthrough: it's a step that also feeds into a gate
            # Create a step process for it
            proc = {
                "id": node_name,
                "type": "step",
                "label": _snake_to_label(node_name),
            }

            # Check if this function calls an LLM
            func_info = node_funcs.get(func_name, {}) if func_name else {}
            if func_info.get("calls_llm"):
                invoked = func_info.get("invokes_func")
                if invoked and invoked.replace("invoke_", "") in seen_agents:
                    agent_id = invoked.replace("invoke_", "")
                    # Add invoke edge
                    edges.append({
                        "type": "invoke",
                        "from": node_name,
                        "to": agent_id,
                        "label": f"Call {_snake_to_label(agent_id)}",
                    })
            processes.append(proc)
        else:
            proc = {
                "id": node_name,
                "type": "step",
                "label": _snake_to_label(node_name),
            }
            if node_name in subgraph_nodes:
                proc["description"] = "Compiled subgraph (nested StateGraph)"

            func_info = node_funcs.get(func_name, {}) if func_name else {}
            if func_info.get("calls_llm"):
                invoked = func_info.get("invokes_func")
                if invoked and invoked.replace("invoke_", "") in seen_agents:
                    agent_id = invoked.replace("invoke_", "")
                    edges.append({
                        "type": "invoke",
                        "from": node_name,
                        "to": agent_id,
                        "label": f"Call {_snake_to_label(agent_id)}",
                    })

            processes.append(proc)

    # --- Create checkpoint processes for interrupt nodes ---
    for proc in processes:
        if proc["id"] in interrupt_nodes and proc["type"] == "step":
            # Create a checkpoint process after this step
            cp_id = f"{proc['id']}_checkpoint"
            cp_proc = {
                "id": cp_id,
                "type": "checkpoint",
                "label": f"{proc['label']} (Human Review)",
                "description": f"Human-in-the-loop checkpoint for {proc['id']}",
                "prompt": "Review and approve before continuing",
            }
            # Add human entity if not already present
            if not any(e.get("type") == "human" for e in entities):
                entities.append({
                    "id": "human_reviewer",
                    "type": "human",
                    "label": "Human Reviewer",
                    "role": "reviewer",
                })
            processes.append(cp_proc)
            # Wire: step → checkpoint, then rewire outgoing edges
            edges.append({
                "type": "flow",
                "from": proc["id"],
                "to": cp_id,
                "label": "Human Review",
            })

    # --- Create terminal step if any gate branches go to END ---
    needs_terminal = False
    for src, router_name, mapping in graph_info["conditional_edges"]:
        if mapping is None:
            continue
        for target in mapping.values():
            if target == "__END__" or target == "END":
                needs_terminal = True
                break

    terminal_id = "__done__"
    if needs_terminal:
        # Check if a terminal step already exists
        if not any(p["id"] == terminal_id for p in processes):
            processes.append({
                "id": terminal_id,
                "type": "step",
                "label": "Done",
                "description": "Terminal step (imported from LangGraph END)",
                "logic": 'state.data["_done"] = True\n',
            })

    # --- Gate processes from conditional edges ---
    for src, router_name, mapping in graph_info["conditional_edges"]:
        # Skip Send-based routers — they produce fan-out flow edges, not gates
        if router_name in send_based_routers:
            continue
        condition, branches, has_end = _analyze_router(tree, router_name, mapping)

        # Replace __terminal__ targets with actual terminal step
        for b in branches:
            if b["target"] == "__terminal__":
                b["target"] = terminal_id

        # Check if src already has a step process (non-passthrough)
        has_step = any(p["id"] == src for p in processes)

        if has_step:
            # Create a separate gate after the step
            gate_id = f"{src}_gate"
            gate_proc = {
                "id": gate_id,
                "type": "gate",
                "label": f"{_snake_to_label(src)} Decision",
                "condition": condition,
                "branches": branches,
            }
            processes.append(gate_proc)
            # Add flow edge from step to gate
            edges.append({
                "type": "flow",
                "from": src,
                "to": gate_id,
                "label": "Decide",
            })
        else:
            # Pass-through gate — src IS the gate
            gate_proc = {
                "id": src,
                "type": "gate",
                "label": _snake_to_label(src),
                "condition": condition,
                "branches": branches,
            }
            processes.append(gate_proc)

    # --- Flow edges ---
    for src, dst in graph_info["edges"]:
        # Skip edges that duplicate gate branches
        edges.append({
            "type": "flow",
            "from": src,
            "to": dst,
            "label": _snake_to_label(dst),
        })

    # --- End edges: mark terminal steps ---
    for src in graph_info["end_edges"]:
        # Find the process and add _done logic
        for proc in processes:
            if proc["id"] == src:
                if proc["type"] == "step":
                    proc["logic"] = proc.get("logic", "") + 'state.data["_done"] = True\n'
                break

    # --- Detect loop edges (back-edges where target appears earlier) ---
    process_order = {p["id"]: i for i, p in enumerate(processes)}
    flow_edges = [e for e in edges if e["type"] == "flow"]
    for e in flow_edges:
        src_idx = process_order.get(e["from"], -1)
        dst_idx = process_order.get(e["to"], -1)
        if dst_idx >= 0 and src_idx > dst_idx:
            e["type"] = "loop"

    # --- Deduplicate edges ---
    seen_edges: set[tuple] = set()
    unique_edges = []
    for e in edges:
        key = (e["type"], e["from"], e["to"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)
    edges = unique_edges

    # Ensure at least one agent entity exists
    has_agent = any(e.get("type") == "agent" for e in entities)
    if not has_agent:
        entities.insert(0, {"id": "llm", "type": "agent", "label": "LLM", "model": "unknown"})
    spec["entities"] = entities
    spec["processes"] = processes
    spec["edges"] = edges
    if schemas:
        spec["schemas"] = schemas

    return spec


# ═══════════════════════════════════════════════════════════════
# LLM-Augmented Import
# ═══════════════════════════════════════════════════════════════

_LLM_AUGMENT_SYSTEM = """\
You are an expert at understanding LangGraph agent code and converting it to \
Agent Ontology YAML specs. You are given:
1. The full source code of a LangGraph agent
2. An AST-extracted skeleton spec (incomplete — only captures graph topology)

Your job: produce a COMPLETE, ENRICHED Agent Ontology YAML spec that adds the \
semantic information the AST parser missed.

## What you must add or improve:
- **Tool entities**: Identify all tools used (search, calculator, etc.) and add \
  them as entities with type: tool
- **Agent descriptions**: Add meaningful descriptions for each agent entity \
  explaining its role
- **Process descriptions**: Add a description for each step explaining what it does
- **Model configuration**: Identify the actual model used (e.g. claude-3-sonnet, \
  gpt-4o) instead of "unknown"
- **Missing edges**: Add invoke edges from steps to agents/tools. Add any \
  flow/loop edges the AST parser missed.
- **Gate conditions**: Replace raw AST condition strings with human-readable \
  conditions referencing state.data fields
- **System prompts**: Extract and include system prompts for agents
- **Process logic**: Add concise logic descriptions for non-trivial steps

## Rules:
- Keep ALL processes and edges from the skeleton — only ADD, don't remove
- Every agent entity MUST have a model field
- Every step process MUST have a description
- The entry_point from the skeleton is correct — keep it
- Schemas from the skeleton are correct — keep them, optionally add more
- Output ONLY valid YAML. No markdown code fences. No explanatory text.
"""


def llm_augment_spec(source: str, skeleton_spec: dict, model: str = "gemini-2.0-flash") -> dict:
    """Use an LLM to enrich an AST-extracted skeleton spec with semantic information."""
    from .specgen import call_llm, extract_yaml

    skeleton_yaml = yaml.dump(skeleton_spec, default_flow_style=False, sort_keys=False)

    user_prompt = f"""\
## Source Code
```python
{source}
```

## AST-Extracted Skeleton Spec
```yaml
{skeleton_yaml}
```

Produce the enriched Agent Ontology YAML spec. Keep all structural elements \
from the skeleton (processes, edges, schemas) and add semantic information \
(tool entities, descriptions, model names, missing edges, system prompts).
Output ONLY the YAML."""

    try:
        response = call_llm(model, _LLM_AUGMENT_SYSTEM, user_prompt, temperature=0.2, max_tokens=8192)
    except Exception as exc:
        print(f"Warning: LLM augmentation failed ({exc}), returning AST-only spec", file=sys.stderr)
        return skeleton_spec

    enriched_yaml = extract_yaml(response)

    try:
        enriched = yaml.safe_load(enriched_yaml)
        if not isinstance(enriched, dict):
            print("Warning: LLM returned non-dict YAML, returning AST-only spec", file=sys.stderr)
            return skeleton_spec
    except yaml.YAMLError as exc:
        print(f"Warning: LLM returned invalid YAML ({exc}), returning AST-only spec", file=sys.stderr)
        return skeleton_spec

    # Merge: LLM-enriched spec takes precedence for content, but skeleton structure is preserved
    merged = _merge_specs(skeleton_spec, enriched)

    # Post-process: fix common LLM mistakes
    # 1. Add tool_type to tool entities missing it
    for entity in merged.get("entities", []):
        if entity.get("type") == "tool" and "tool_type" not in entity:
            entity["tool_type"] = "function"

    # 2. Remove edges referencing 'END' — add _done logic instead
    valid_ids = {e.get("id") for e in merged.get("entities", [])}
    valid_ids.update(p.get("id") for p in merged.get("processes", []))
    new_edges = []
    end_sources = []
    for e in merged.get("edges", []):
        if e.get("to") in ("END", "__END__", "end"):
            end_sources.append(e.get("from"))
        elif e.get("from") in ("END", "__END__", "end", "START", "__START__"):
            continue  # skip edges from START/END
        elif e.get("to") not in valid_ids or e.get("from") not in valid_ids:
            continue  # skip edges referencing unknown nodes
        else:
            new_edges.append(e)
    merged["edges"] = new_edges

    # Add _done logic to end sources
    for src in end_sources:
        for p in merged.get("processes", []):
            if p.get("id") == src and p.get("type") == "step":
                if '_done' not in p.get("logic", ""):
                    p["logic"] = p.get("logic", "") + 'state.data["_done"] = True\n'

    return merged


def _merge_specs(skeleton: dict, enriched: dict) -> dict:
    """Merge an LLM-enriched spec with the AST skeleton.

    Strategy: Use enriched as base, but ensure all skeleton structural elements
    (processes, edges) are present.
    """
    merged = dict(enriched)

    # Preserve skeleton's entry_point and name (only if non-None)
    skel_ep = skeleton.get("entry_point")
    if skel_ep is not None:
        merged["entry_point"] = skel_ep
    merged["name"] = skeleton.get("name") or merged.get("name")
    merged["version"] = skeleton.get("version", "1.0")

    # Ensure all skeleton processes exist in merged
    skeleton_proc_ids = {p["id"] for p in skeleton.get("processes", [])}
    merged_proc_ids = {p["id"] for p in merged.get("processes", [])}
    for p in skeleton.get("processes", []):
        if p["id"] not in merged_proc_ids:
            merged.setdefault("processes", []).append(p)

    # Ensure all skeleton edges exist in merged
    skeleton_edge_keys = {(e["type"], e["from"], e["to"]) for e in skeleton.get("edges", [])}
    merged_edge_keys = {(e.get("type"), e.get("from"), e.get("to")) for e in merged.get("edges", [])}
    for e in skeleton.get("edges", []):
        key = (e["type"], e["from"], e["to"])
        if key not in merged_edge_keys:
            merged.setdefault("edges", []).append(e)

    # Ensure all skeleton schemas exist in merged
    skeleton_schema_names = {s.get("name", s.get("id")) for s in skeleton.get("schemas", [])}
    merged_schema_names = {s.get("name", s.get("id")) for s in merged.get("schemas", [])}
    for s in skeleton.get("schemas", []):
        name = s.get("name", s.get("id"))
        if name not in merged_schema_names:
            merged.setdefault("schemas", []).append(s)

    return merged


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Import a LangGraph StateGraph Python file into an Agent Ontology YAML spec"
    )
    parser.add_argument("input", help="Path to LangGraph Python file")
    parser.add_argument("-o", "--output", help="Output YAML file path (default: stdout)")
    parser.add_argument("--validate", action="store_true", help="Validate the output spec")
    parser.add_argument("--quiet", action="store_true", help="Suppress info messages")
    parser.add_argument("--llm-augment", action="store_true",
                        help="Use LLM to enrich the AST-extracted skeleton with semantic information")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Model for LLM augmentation (default: gemini-2.0-flash)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        spec = import_langgraph(input_path)
    except SyntaxError as exc:
        print(f"Error: {input_path} is not valid Python: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.llm_augment:
        if not args.quiet:
            print("LLM-augmenting spec...", file=sys.stderr)
        source = input_path.read_text(encoding="utf-8")
        spec = llm_augment_spec(source, spec, model=args.model)

    # Count what we found
    n_ent = len(spec.get("entities", []))
    n_proc = len(spec.get("processes", []))
    n_edge = len(spec.get("edges", []))
    n_schema = len(spec.get("schemas", []))

    if not args.quiet:
        print(f"Imported: {n_ent} entities, {n_proc} processes, {n_edge} edges, {n_schema} schemas",
              file=sys.stderr)

    # Serialize
    yaml_str = yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    if args.output:
        Path(args.output).write_text(yaml_str, encoding="utf-8")
        if not args.quiet:
            print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(yaml_str)

    # Validate
    if args.validate:
        try:
            from .validate import validate_spec, load_ontology
            ontology = load_ontology()
            errors, warnings = validate_spec(spec, ontology)
            if errors:
                print(f"\n  {len(errors)} validation error(s):", file=sys.stderr)
                for e in errors:
                    print(f"    ✗ {e}", file=sys.stderr)
            if warnings:
                print(f"\n  {len(warnings)} warning(s):", file=sys.stderr)
                for w in warnings:
                    print(f"    ⚠ {w}", file=sys.stderr)
            if not errors:
                print(f"  ✓ Valid spec", file=sys.stderr)
        except ImportError:
            print("Warning: validate.py not found, skipping validation", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
