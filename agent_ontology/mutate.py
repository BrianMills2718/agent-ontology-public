#!/usr/bin/env python3
"""
Spec Mutation Engine — Evolutionary search over agent architectures.

Takes a valid agent spec YAML and produces structural variations.
This is the mutation operator for genetic algorithms over agent architectures.

Usage:
    # Apply a specific mutation:
    python3 mutate.py specs/react.yaml --mutation swap_process_order -o specs/react_variant.yaml

    # Generate N random mutations:
    python3 mutate.py specs/react.yaml --random -n 3 -o variants/

    # List available mutations:
    python3 mutate.py --list-mutations

    # Dry run (print to stdout):
    python3 mutate.py specs/react.yaml --mutation add_review_step
"""

import argparse
import copy
import os
import random
import sys
import uuid
from datetime import datetime, timezone

import yaml


# ════════════════════════════════════════════════════════════════════
# YAML formatting helpers
# ════════════════════════════════════════════════════════════════════

class LiteralStr(str):
    """Tag a string so the YAML dumper uses literal block style (|)."""
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralStr, _literal_representer)


def _preserve_multiline(obj):
    """Walk a dict/list and tag any multiline strings as LiteralStr."""
    if isinstance(obj, dict):
        return {k: _preserve_multiline(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_preserve_multiline(v) for v in obj]
    elif isinstance(obj, str) and "\n" in obj:
        return LiteralStr(obj)
    return obj


def dump_spec(spec):
    """Serialize spec to YAML with readable multiline strings."""
    tagged = _preserve_multiline(spec)
    return yaml.dump(tagged, default_flow_style=False, sort_keys=False, width=120, allow_unicode=True)


# ════════════════════════════════════════════════════════════════════
# Spec graph helpers
# ════════════════════════════════════════════════════════════════════

def load_spec(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _short_id():
    """Generate a short random suffix for new IDs."""
    return uuid.uuid4().hex[:6]


def _entity_by_id(spec, eid):
    for e in spec.get("entities", []):
        if e.get("id") == eid:
            return e
    return None


def _process_by_id(spec, pid):
    for p in spec.get("processes", []):
        if p.get("id") == pid:
            return p
    return None


def _process_ids(spec):
    return [p["id"] for p in spec.get("processes", []) if "id" in p]


def _entity_ids(spec):
    return [e["id"] for e in spec.get("entities", []) if "id" in e]


def _all_node_ids(spec):
    return set(_entity_ids(spec)) | set(_process_ids(spec))


def _agent_entities(spec):
    return [e for e in spec.get("entities", []) if e.get("type") == "agent"]


def _step_processes(spec):
    return [p for p in spec.get("processes", []) if p.get("type") == "step"]


def _gate_processes(spec):
    return [p for p in spec.get("processes", []) if p.get("type") == "gate"]


def _flow_edges(spec):
    return [e for e in spec.get("edges", []) if e.get("type") == "flow"]


def _edges_from(spec, node_id):
    return [e for e in spec.get("edges", []) if e.get("from") == node_id]


def _edges_to(spec, node_id):
    return [e for e in spec.get("edges", []) if e.get("to") == node_id]


def _build_flow_chain(spec):
    """Build an ordered chain of process IDs by following flow edges from entry_point.

    Returns a list of process IDs in execution order. Only follows the first
    flow edge from each node (ignores branching for the purpose of building
    a linear chain). This is used for adjacency-based mutations.
    """
    entry = spec.get("entry_point")
    if not entry:
        return []
    chain = []
    visited = set()
    current = entry
    while current and current not in visited:
        visited.add(current)
        proc = _process_by_id(spec, current)
        if proc:
            chain.append(current)
        # Find the next process via flow edge
        flow_out = [e for e in spec.get("edges", [])
                    if e.get("from") == current and e.get("type") in ("flow", "loop")]
        next_node = None
        for fe in flow_out:
            target = fe.get("to")
            if target and _process_by_id(spec, target) and target not in visited:
                next_node = target
                break
        current = next_node
    return chain


def _is_critical_process(spec, pid):
    """A process is critical if it's the entry_point or the terminal step."""
    if pid == spec.get("entry_point"):
        return True
    # Terminal: has no outgoing flow edges to other processes
    flow_out = [e for e in spec.get("edges", [])
                if e.get("from") == pid and e.get("type") == "flow"]
    if not flow_out:
        return True
    return False


def _record_mutation(spec, mutation_name, details=None):
    """Add a mutation record to the spec metadata."""
    if "mutations" not in spec:
        spec["mutations"] = []
    record = {
        "operator": mutation_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        record["details"] = details
    spec["mutations"].append(record)


def _rewire_edge(edge, old_from=None, new_from=None, old_to=None, new_to=None):
    """Rewire an edge, replacing from/to references."""
    if old_from and new_from and edge.get("from") == old_from:
        edge["from"] = new_from
    if old_to and new_to and edge.get("to") == old_to:
        edge["to"] = new_to


# ════════════════════════════════════════════════════════════════════
# Mutation operators
# ════════════════════════════════════════════════════════════════════

def swap_process_order(spec):
    """Pick two adjacent non-gate processes in the flow and swap their order.

    Rewires all edges so that the flow graph remains valid: anything that
    pointed to A now points to B and vice versa, and anything that came
    from A now comes from B and vice versa.
    """
    spec = copy.deepcopy(spec)
    chain = _build_flow_chain(spec)

    # Filter to non-gate, non-critical pairs
    candidates = []
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        pa = _process_by_id(spec, a)
        pb = _process_by_id(spec, b)
        if not pa or not pb:
            continue
        if pa.get("type") == "gate" or pb.get("type") == "gate":
            continue
        if _is_critical_process(spec, a) and _is_critical_process(spec, b):
            continue
        candidates.append((a, b))

    if not candidates:
        raise ValueError("No adjacent non-gate process pair available to swap")

    a_id, b_id = random.choice(candidates)

    # Swap by rewiring all edges: replace references to a_id with a temp,
    # then b_id -> a_id, then temp -> b_id
    sentinel = f"__swap_temp_{_short_id()}"
    for edge in spec.get("edges", []):
        if edge.get("from") == a_id:
            edge["from"] = sentinel
        if edge.get("to") == a_id:
            edge["to"] = sentinel
    for edge in spec.get("edges", []):
        if edge.get("from") == b_id:
            edge["from"] = a_id
        if edge.get("to") == b_id:
            edge["to"] = a_id
    for edge in spec.get("edges", []):
        if edge.get("from") == sentinel:
            edge["from"] = b_id
        if edge.get("to") == sentinel:
            edge["to"] = b_id

    # Also swap gate branch targets
    for proc in spec.get("processes", []):
        if proc.get("type") == "gate" and proc.get("branches"):
            for branch in proc["branches"]:
                if branch.get("target") == a_id:
                    branch["target"] = sentinel
            for branch in proc["branches"]:
                if branch.get("target") == b_id:
                    branch["target"] = a_id
            for branch in proc["branches"]:
                if branch.get("target") == sentinel:
                    branch["target"] = b_id

    # Update entry_point if needed
    if spec.get("entry_point") == a_id:
        spec["entry_point"] = b_id
    elif spec.get("entry_point") == b_id:
        spec["entry_point"] = a_id

    _record_mutation(spec, "swap_process_order", {
        "swapped": [a_id, b_id],
    })
    return spec


def add_review_step(spec):
    """Insert a new 'review' process after a randomly chosen step.

    Creates a new review agent entity and a review step process.
    The review agent inspects the output of the chosen step and either
    approves or requests revision.
    """
    spec = copy.deepcopy(spec)
    steps = _step_processes(spec)
    # Exclude terminal steps (nothing flows out of them to rewire)
    steps = [s for s in steps if not _is_critical_process(spec, s["id"])
             or s["id"] == spec.get("entry_point")]
    # Further filter: must have at least one outgoing flow edge
    steps = [s for s in steps if _edges_from(spec, s["id"])]

    if not steps:
        raise ValueError("No suitable step to insert a review after")

    target_step = random.choice(steps)
    target_id = target_step["id"]
    suffix = _short_id()

    # Create review agent entity
    review_agent_id = f"review_agent_{suffix}"
    review_agent = {
        "id": review_agent_id,
        "type": "agent",
        "label": f"Review Agent ({target_step.get('label', target_id)})",
        "model": "gemini-3-flash-preview",
        "system_prompt": (
            f"You are a review agent. You review the output of the "
            f"'{target_step.get('label', target_id)}' step. "
            f"Check for correctness, completeness, and quality. "
            f"Output a JSON object with 'approved' (boolean) and 'feedback' (string). "
            f"If approved is false, explain what needs improvement."
        ),
        "input_schema": "ReviewInput",
        "output_schema": "ReviewOutput",
    }
    spec["entities"].append(review_agent)

    # Create review process
    review_process_id = f"review_{target_id}_{suffix}"
    review_process = {
        "id": review_process_id,
        "type": "step",
        "label": f"Review: {target_step.get('label', target_id)}",
        "description": f"Review the output of '{target_step.get('label', target_id)}' for quality",
        "data_in": "ReviewInput",
        "data_out": "ReviewOutput",
        "logic": LiteralStr(
            f'print(f"    Reviewing output of {target_id}")\n'
        ),
    }
    spec["processes"].append(review_process)

    # Add ReviewInput/ReviewOutput schemas if not present
    schema_names = {s.get("name") for s in spec.get("schemas", [])}
    if "ReviewInput" not in schema_names:
        spec.setdefault("schemas", []).append({
            "name": "ReviewInput",
            "description": "Input to a review agent",
            "fields": [
                {"name": "content", "type": "string"},
                {"name": "source_step", "type": "string"},
            ],
        })
    if "ReviewOutput" not in schema_names:
        spec.setdefault("schemas", []).append({
            "name": "ReviewOutput",
            "description": "Output from a review agent",
            "fields": [
                {"name": "approved", "type": "boolean"},
                {"name": "feedback", "type": "string"},
            ],
        })

    # Rewire: find outgoing flow edges from target_step and redirect them
    # through the review step.
    #   Before: target_step --flow--> next_step
    #   After:  target_step --flow--> review_step --flow--> next_step
    outgoing_flow = [e for e in spec["edges"]
                     if e.get("from") == target_id and e.get("type") == "flow"]

    if outgoing_flow:
        # Pick the first outgoing flow edge to splice into
        edge_to_splice = outgoing_flow[0]
        original_target = edge_to_splice["to"]

        # Redirect the original edge to point to the review step
        edge_to_splice["to"] = review_process_id
        edge_to_splice["label"] = f"To review ({target_step.get('label', target_id)})"

        # Add edge from review step to original target
        spec["edges"].append({
            "type": "flow",
            "from": review_process_id,
            "to": original_target,
            "label": f"After review of {target_step.get('label', target_id)}",
        })

    # Add invoke edge from review step to review agent
    spec["edges"].append({
        "type": "invoke",
        "from": review_process_id,
        "to": review_agent_id,
        "label": f"Review check",
        "input": "ReviewInput",
        "output": "ReviewOutput",
    })

    _record_mutation(spec, "add_review_step", {
        "after_step": target_id,
        "review_process": review_process_id,
        "review_agent": review_agent_id,
    })
    return spec


def remove_process(spec):
    """Remove a non-critical process from the flow and rewire around it.

    All incoming flow edges are redirected to the process's first outgoing
    flow target. Invoke/write edges from/to the removed process are dropped.
    """
    spec = copy.deepcopy(spec)
    steps = _step_processes(spec)
    # Filter out critical (entry_point / terminal) processes
    removable = [s for s in steps
                 if not _is_critical_process(spec, s["id"])]

    if not removable:
        raise ValueError("No non-critical step process available to remove")

    target = random.choice(removable)
    target_id = target["id"]

    # Find the outgoing flow targets
    outgoing_flow = [e for e in spec["edges"]
                     if e.get("from") == target_id and e.get("type") == "flow"]
    # Determine the bypass target (where incoming edges should go)
    bypass_to = outgoing_flow[0]["to"] if outgoing_flow else None

    if bypass_to is None:
        raise ValueError(f"Cannot remove '{target_id}': no outgoing flow edge to rewire to")

    # Rewire incoming flow edges to bypass the removed process
    for edge in spec["edges"]:
        if edge.get("to") == target_id and edge.get("type") in ("flow", "loop", "branch"):
            edge["to"] = bypass_to

    # Rewire gate branch targets
    for proc in spec.get("processes", []):
        if proc.get("type") == "gate" and proc.get("branches"):
            for branch in proc["branches"]:
                if branch.get("target") == target_id:
                    branch["target"] = bypass_to

    # Remove all edges from/to the removed process
    spec["edges"] = [e for e in spec["edges"]
                     if e.get("from") != target_id and e.get("to") != target_id]

    # Remove the process itself
    spec["processes"] = [p for p in spec["processes"] if p.get("id") != target_id]

    _record_mutation(spec, "remove_process", {
        "removed": target_id,
        "bypassed_to": bypass_to,
    })
    return spec


def change_gate_condition(spec):
    """Modify a gate's condition -- change threshold, invert, or alter operator."""
    spec = copy.deepcopy(spec)
    gates = _gate_processes(spec)
    if not gates:
        raise ValueError("No gate processes in this spec")

    gate = random.choice(gates)
    old_condition = gate.get("condition", "")

    transformations = [
        # Invert the condition
        lambda c: f"not ({c})" if "not" not in c else c.replace("not ", "").replace("(", "").replace(")", ""),
        # Swap comparison operators
        lambda c: c.replace(">=", "__LT__").replace("<=", "__GT__").replace(">", "<=").replace("<", ">=").replace("__LT__", "<").replace("__GT__", ">") if any(op in c for op in [">=", "<=", ">", "<"]) else c,
        # Adjust thresholds (append a modifier)
        lambda c: c + " + 1" if any(char.isdigit() for char in c) else c,
        # Swap 'and' / 'or'
        lambda c: c.replace(" and ", " or ") if " and " in c else c.replace(" or ", " and ") if " or " in c else c,
        # Change equality to inequality
        lambda c: c.replace("==", "!=") if "==" in c else c.replace("!=", "==") if "!=" in c else c,
    ]

    # Try each transformation until one actually changes the condition
    random.shuffle(transformations)
    new_condition = old_condition
    for transform in transformations:
        candidate = transform(old_condition)
        if candidate != old_condition:
            new_condition = candidate
            break

    if new_condition == old_condition:
        # Fallback: prepend a negation
        new_condition = f"not ({old_condition})"

    gate["condition"] = new_condition

    # Also update branch conditions if they mirror the gate condition
    if gate.get("branches") and len(gate["branches"]) >= 2:
        # Swap the branch targets (invert which branch goes where)
        branches = gate["branches"]
        if len(branches) == 2:
            branches[0]["target"], branches[1]["target"] = branches[1]["target"], branches[0]["target"]
            branches[0]["condition"], branches[1]["condition"] = branches[1]["condition"], branches[0]["condition"]

    _record_mutation(spec, "change_gate_condition", {
        "gate": gate["id"],
        "old_condition": old_condition,
        "new_condition": new_condition,
    })
    return spec


def duplicate_with_variation(spec):
    """Copy an existing agent entity with a different system prompt variation.

    The new agent gets a modified system prompt (e.g., more cautious, more
    creative, different persona) and is wired to the same invoke edges
    as the original -- creating a parallel agent that can be swapped in.
    """
    spec = copy.deepcopy(spec)
    agents = _agent_entities(spec)
    if not agents:
        raise ValueError("No agent entities to duplicate")

    original = random.choice(agents)
    suffix = _short_id()
    new_id = f"{original['id']}_var_{suffix}"

    # Choose a prompt variation strategy
    variations = [
        ("cautious", "Be extra cautious and conservative in your responses. Double-check all claims. Prefer safety over speed."),
        ("creative", "Be more creative and exploratory. Consider unconventional approaches. Think outside the box."),
        ("concise", "Be extremely concise. Minimize verbosity. Output only essential information."),
        ("detailed", "Be thorough and detailed. Explain your reasoning step by step. Leave nothing ambiguous."),
        ("skeptical", "Be skeptical of assumptions. Question premises. Consider failure modes and edge cases."),
        ("optimistic", "Focus on opportunities and positive outcomes. Assume good faith. Look for the best path forward."),
    ]
    variation_name, variation_prefix = random.choice(variations)

    new_agent = copy.deepcopy(original)
    new_agent["id"] = new_id
    new_agent["label"] = f"{original.get('label', original['id'])} ({variation_name})"
    original_prompt = original.get("system_prompt", "")
    new_agent["system_prompt"] = f"{variation_prefix}\n\n{original_prompt}"

    spec["entities"].append(new_agent)

    _record_mutation(spec, "duplicate_with_variation", {
        "original": original["id"],
        "variant": new_id,
        "variation": variation_name,
    })
    return spec


def add_store(spec):
    """Add a new store entity and wire a write edge from a randomly chosen process."""
    spec = copy.deepcopy(spec)
    steps = _step_processes(spec)
    if not steps:
        raise ValueError("No step processes to wire a store to")

    source_step = random.choice(steps)
    suffix = _short_id()

    store_types = ["queue", "vector", "kv"]
    store_type = random.choice(store_types)

    store_id = f"store_{store_type}_{suffix}"
    schema_name = f"StoreEntry_{suffix}"

    # Create the store entity
    store_entity = {
        "id": store_id,
        "type": "store",
        "label": f"{store_type.title()} Store ({source_step.get('label', source_step['id'])})",
        "store_type": store_type,
        "schema": schema_name,
        "retention": "session",
    }
    spec["entities"].append(store_entity)

    # Create a write edge from the source process to the store
    spec["edges"].append({
        "type": "write",
        "from": source_step["id"],
        "to": store_id,
        "label": f"Persist to {store_id}",
        "data": schema_name,
    })

    # Add the schema
    spec.setdefault("schemas", []).append({
        "name": schema_name,
        "description": f"Data written to {store_id} from {source_step.get('label', source_step['id'])}",
        "fields": [
            {"name": "content", "type": "string"},
            {"name": "metadata", "type": "object"},
            {"name": "timestamp", "type": "string"},
        ],
    })

    _record_mutation(spec, "add_store", {
        "store": store_id,
        "store_type": store_type,
        "source_process": source_step["id"],
    })
    return spec


# Models to rotate through for change_model
_MODEL_POOL = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-haiku-3-20240307",
    "gemini-3-flash-preview",
    "gemini-2.5-pro-preview",
    "gemini-2.0-flash",
    "deepseek-r1",
    "llama-4-maverick",
    "o3-mini",
]


def change_model(spec):
    """Switch the model for one or more agents to a different model."""
    spec = copy.deepcopy(spec)
    agents = _agent_entities(spec)
    if not agents:
        raise ValueError("No agent entities to change model on")

    # Decide how many agents to change (at least 1, up to all)
    count = random.randint(1, max(1, len(agents)))
    targets = random.sample(agents, count)

    changes = []
    for agent in targets:
        old_model = agent.get("model", "unknown")
        # Pick a different model
        available = [m for m in _MODEL_POOL if m != old_model]
        if not available:
            continue
        new_model = random.choice(available)
        agent["model"] = new_model
        changes.append({"agent": agent["id"], "old_model": old_model, "new_model": new_model})

    if not changes:
        raise ValueError("Could not find a different model to assign")

    _record_mutation(spec, "change_model", {"changes": changes})
    return spec


# Prompt transformation strategies for modify_prompt
_PROMPT_TRANSFORMS = [
    {
        "name": "add_emphasis",
        "description": "Add emphasis/urgency to the prompt",
        "transform": lambda p: (
            "IMPORTANT: Pay close attention to accuracy and correctness.\n\n" + p
        ),
    },
    {
        "name": "remove_constraints",
        "description": "Remove limiting language from the prompt",
        "transform": lambda p: (
            p.replace("only ", "")
             .replace("Only ", "")
             .replace("must ", "should ")
             .replace("Must ", "Should ")
             .replace("always ", "generally ")
             .replace("never ", "rarely ")
        ),
    },
    {
        "name": "add_chain_of_thought",
        "description": "Add chain-of-thought reasoning (overrides speed/guessing instructions)",
        "transform": lambda p: (
            # Remove conflicting speed/guessing instructions
            "\n".join(
                line for line in p.split("\n")
                if not any(w in line.lower() for w in [
                    "just guess", "no calculation", "no steps", "no explanation",
                    "speed over accuracy", "first instinct", "be quick",
                ])
            ).strip()
            + "\n\nIMPORTANT: Think step-by-step before answering. "
            "Show your reasoning, verify your answer, then output your final answer. "
            "Your output JSON 'answer' field must contain ONLY the final value, not your reasoning."
        ),
    },
    {
        "name": "add_self_critique",
        "description": "Append self-critique instructions",
        "transform": lambda p: (
            p + "\n\nAfter generating your response, critically review it. "
            "Identify potential errors or weaknesses before finalizing."
        ),
    },
    {
        "name": "simplify",
        "description": "Add instructions for simpler output",
        "transform": lambda p: (
            p + "\n\nKeep your output simple and direct. "
            "Avoid unnecessary complexity or jargon."
        ),
    },
    {
        "name": "add_examples_request",
        "description": "Instruct the agent to provide examples",
        "transform": lambda p: (
            p + "\n\nWhere possible, include concrete examples to "
            "illustrate your points."
        ),
    },
    {
        "name": "increase_temperature_style",
        "description": "Add instructions for more diverse/creative output",
        "transform": lambda p: (
            "Explore diverse and creative approaches. "
            "Do not settle on the first solution.\n\n" + p
        ),
    },
]


# Benchmark-specific prompt transforms — only shown when evolving against a specific benchmark.
# These address domain-specific format mismatches that generic transforms can't fix.
_BENCHMARK_PROMPT_TRANSFORMS = {
    "arc": [
        {
            "name": "answer_with_letter",
            "description": "[BENCHMARK] Answer with ONLY the letter A/B/C/D for ARC science questions",
            "transform": lambda p: (
                p + "\n\nIMPORTANT: Your answer must be ONLY a single letter: A, B, C, or D. "
                "Do not output a number or index. Output the LETTER of the correct choice."
            ),
        },
        {
            "name": "science_reasoning",
            "description": "[BENCHMARK] Add scientific reasoning guidance for ARC",
            "transform": lambda p: (
                p + "\n\nApply scientific reasoning: identify the relevant scientific concept, "
                "eliminate incorrect options, then select the best answer letter (A/B/C/D)."
            ),
        },
    ],
    "gsm8k": [
        {
            "name": "show_work_final_number",
            "description": "[BENCHMARK] Show work but put only the final number in the answer",
            "transform": lambda p: (
                p + "\n\nShow your work step by step. Your final answer MUST be a single number "
                "with no units, no commas, no dollar signs. Just the number."
            ),
        },
        {
            "name": "watch_for_traps",
            "description": "[BENCHMARK] Watch for common math traps and verify answers",
            "transform": lambda p: (
                p + "\n\nWARNING: These problems may contain traps. Do NOT trust your first instinct. "
                "Set up equations carefully. For example, if X + Y = 1.10 and X = Y + 1.00, "
                "solve algebraically — don't guess. Verify your answer by plugging it back in."
            ),
        },
    ],
    "gsm8k_tricky": [
        {
            "name": "show_work_final_number",
            "description": "[BENCHMARK] Show work but put only the final number in the answer",
            "transform": lambda p: (
                p + "\n\nShow your work step by step. Your final answer MUST be a single number "
                "with no units, no commas, no dollar signs. Just the number."
            ),
        },
        {
            "name": "watch_for_traps",
            "description": "[BENCHMARK] Watch for common math traps and verify answers",
            "transform": lambda p: (
                p + "\n\nWARNING: These problems may contain traps. Do NOT trust your first instinct. "
                "Set up equations carefully. For example, if X + Y = 1.10 and X = Y + 1.00, "
                "solve algebraically — don't guess. Verify your answer by plugging it back in."
            ),
        },
    ],
    "gsm8k_hard": [
        {
            "name": "show_work_final_number",
            "description": "[BENCHMARK] Show work but put only the final number in the answer",
            "transform": lambda p: (
                p + "\n\nShow your work step by step. Your final answer MUST be a single number "
                "with no units, no commas, no dollar signs. Just the number."
            ),
        },
    ],
    "hotpotqa": [
        {
            "name": "extract_short_answer",
            "description": "[BENCHMARK] Output a short factual answer, not a full sentence",
            "transform": lambda p: (
                p + "\n\nYour answer must be a SHORT factual answer — a name, date, number, or phrase. "
                "Do NOT write a full sentence. Just the answer."
            ),
        },
    ],
    "humaneval": [
        {
            "name": "code_only",
            "description": "[BENCHMARK] Output only the Python function body, no explanation",
            "transform": lambda p: (
                p + "\n\nOutput ONLY the Python code for the function body. "
                "No explanation, no markdown fences, no test cases. Just the code."
            ),
        },
    ],
    "multidoc": [
        {
            "name": "check_contradictions",
            "description": "[BENCHMARK] Explicitly check for contradictions between fact cards",
            "transform": lambda p: (
                p + "\n\nIMPORTANT: Some fact cards may contain contradictions. "
                "When sources disagree, prefer independent audits/studies over self-reported claims. "
                "Always note which source you are using for your answer."
            ),
        },
        {
            "name": "extract_then_reason",
            "description": "[BENCHMARK] Extract key facts first, then reason about the answer",
            "transform": lambda p: (
                p + "\n\nApproach: First extract the specific facts relevant to the question from each source. "
                "Then reason step-by-step using only those extracted facts. "
                "For arithmetic questions, show your calculation. "
                "Your final answer should be concise — a name, number, or short phrase."
            ),
        },
        {
            "name": "verify_against_sources",
            "description": "[BENCHMARK] Verify your answer against each source before finalizing",
            "transform": lambda p: (
                p + "\n\nBefore giving your final answer, verify it against each fact card. "
                "Ask: does any source contradict my answer? Did I miss any relevant information? "
                "Did I compute arithmetic correctly? Only then give your final answer."
            ),
        },
    ],
}


def modify_prompt(spec):
    """Apply a transformation to an agent's system prompt."""
    spec = copy.deepcopy(spec)
    agents = _agent_entities(spec)
    agents_with_prompts = [a for a in agents if a.get("system_prompt")]
    if not agents_with_prompts:
        raise ValueError("No agents with system_prompt to modify")

    agent = random.choice(agents_with_prompts)
    transform_def = random.choice(_PROMPT_TRANSFORMS)

    old_prompt = agent["system_prompt"]
    new_prompt = transform_def["transform"](old_prompt)
    agent["system_prompt"] = new_prompt

    _record_mutation(spec, "modify_prompt", {
        "agent": agent["id"],
        "transform": transform_def["name"],
        "description": transform_def["description"],
    })
    return spec


# ════════════════════════════════════════════════════════════════════
# Pattern-level mutation operators
# ════════════════════════════════════════════════════════════════════

def _get_patterns_module():
    """Lazy import of patterns module to avoid circular imports."""
    from . import patterns as pat_mod
    return pat_mod


def swap_pattern(spec):
    """Replace one detected pattern in the spec with a compatible alternative.

    Detects patterns in the spec, picks one at random, finds a replacement
    pattern with a compatible interface, removes the old pattern's processes/
    entities/edges, and inserts the new one with proper wiring.
    """
    pat_mod = _get_patterns_module()
    spec = copy.deepcopy(spec)

    detected = pat_mod.detect_patterns(spec)
    if not detected:
        raise ValueError("No patterns detected in spec to swap")

    # Pick a detected pattern to replace
    random.shuffle(detected)
    target_pname, target_pids, target_prefix = detected[0]

    # Find a compatible replacement (different pattern, overlapping I/O)
    target_pat = pat_mod.PATTERN_LIBRARY[target_pname]
    target_outputs = set(target_pat["interface"]["outputs"])
    target_inputs = set(target_pat["interface"]["inputs"])

    candidates = []
    for pname in pat_mod.list_patterns():
        if pname == target_pname:
            continue
        candidate = pat_mod.PATTERN_LIBRARY[pname]
        cand_inputs = set(candidate["interface"]["inputs"])
        cand_outputs = set(candidate["interface"]["outputs"])
        # Compatible if inputs overlap and outputs overlap
        if (not cand_inputs or cand_inputs & target_inputs) and \
           (not cand_outputs or cand_outputs & target_outputs):
            candidates.append(pname)

    if not candidates:
        raise ValueError(f"No compatible replacement for pattern '{target_pname}'")

    replacement_name = random.choice(candidates)
    replacement = pat_mod.get_pattern(replacement_name)

    # Find flow/branch/loop edges that connect INTO or OUT OF the old pattern
    # (only to/from processes, not invoke/write edges to entities)
    incoming_edges = []
    outgoing_edges = []
    for edge in spec.get("edges", []):
        etype = edge.get("type", "")
        if etype not in ("flow", "branch", "loop"):
            continue
        if edge.get("to") in target_pids and edge.get("from") not in target_pids:
            incoming_edges.append(edge)
        if edge.get("from") in target_pids and edge.get("to") not in target_pids:
            outgoing_edges.append(edge)

    # Remove old pattern's processes, entities, edges
    old_entity_ids = set()
    for edge in spec.get("edges", []):
        if edge.get("from") in target_pids and edge.get("type") in ("invoke", "read", "write"):
            old_entity_ids.add(edge.get("to"))

    spec["processes"] = [p for p in spec["processes"] if p["id"] not in target_pids]
    spec["edges"] = [e for e in spec["edges"]
                     if e.get("from") not in target_pids and e.get("to") not in target_pids]

    # Remove orphaned entities (only those uniquely used by old pattern)
    remaining_edge_refs = set()
    for e in spec.get("edges", []):
        remaining_edge_refs.add(e.get("from"))
        remaining_edge_refs.add(e.get("to"))
    removed_entity_ids = set()
    for eid in old_entity_ids:
        if eid not in remaining_edge_refs:
            spec["entities"] = [e for e in spec["entities"] if e["id"] != eid]
            removed_entity_ids.add(eid)

    # Clean up edges that reference removed entities
    if removed_entity_ids:
        all_node_ids = {p["id"] for p in spec["processes"]} | {e["id"] for e in spec["entities"]}
        spec["edges"] = [e for e in spec["edges"]
                         if e.get("from") in all_node_ids and e.get("to") in all_node_ids]

    # Namespace and insert the replacement
    prefix = target_prefix or replacement_name[:3]
    from .compose import _namespace_pattern
    ns_replacement = _namespace_pattern(replacement, prefix)

    spec["processes"].extend(ns_replacement["processes"])
    # Add entities (avoiding duplicates)
    existing_eids = {e["id"] for e in spec["entities"]}
    for ent in ns_replacement["entities"]:
        if ent["id"] not in existing_eids:
            spec["entities"].append(ent)
    spec["edges"].extend(ns_replacement["edges"])
    # Add schemas (avoiding duplicates)
    existing_schemas = {s["name"] for s in spec.get("schemas", [])}
    for schema in ns_replacement["schemas"]:
        if schema["name"] not in existing_schemas:
            spec.setdefault("schemas", []).append(schema)

    # Rewire incoming edges to new entry
    for edge in incoming_edges:
        edge["to"] = ns_replacement["interface"]["entry"]
        spec["edges"].append(edge)

    # Rewire outgoing edges from new exits
    for edge in outgoing_edges:
        for exit_pid in ns_replacement["interface"]["exits"]:
            new_edge = copy.deepcopy(edge)
            new_edge["from"] = exit_pid
            spec["edges"].append(new_edge)

    _record_mutation(spec, "swap_pattern", {
        "old_pattern": target_pname,
        "new_pattern": replacement_name,
        "prefix": prefix,
    })
    return spec


def insert_pattern(spec):
    """Insert a pattern at a random flow edge in the spec.

    Picks a random flow edge, breaks it, and inserts a pattern between
    the source and target.
    """
    pat_mod = _get_patterns_module()
    spec = copy.deepcopy(spec)

    flow_edges = [e for e in spec.get("edges", []) if e.get("type") == "flow"
                  and _process_by_id(spec, e.get("from"))
                  and _process_by_id(spec, e.get("to"))]

    if not flow_edges:
        raise ValueError("No flow edges between processes to insert a pattern at")

    edge = random.choice(flow_edges)
    from_pid = edge["from"]
    to_pid = edge["to"]

    # Pick a pattern to insert
    pattern_name = random.choice(pat_mod.list_patterns())
    pattern = pat_mod.get_pattern(pattern_name)

    # Namespace it
    prefix = f"ins_{_short_id()}"
    from .compose import _namespace_pattern
    ns_pattern = _namespace_pattern(pattern, prefix)

    # Break the flow edge: from -> pattern_entry ... pattern_exits -> to
    spec["edges"].remove(edge)

    # Add edge from source to pattern entry
    spec["edges"].append({
        "type": "flow",
        "from": from_pid,
        "to": ns_pattern["interface"]["entry"],
        "label": f"To inserted {pattern_name}",
    })

    # Add edges from pattern exits to target
    for exit_pid in ns_pattern["interface"]["exits"]:
        spec["edges"].append({
            "type": "flow",
            "from": exit_pid,
            "to": to_pid,
            "label": f"From inserted {pattern_name}",
        })

    # Add pattern components
    spec["processes"].extend(ns_pattern["processes"])
    existing_eids = {e["id"] for e in spec["entities"]}
    for ent in ns_pattern["entities"]:
        if ent["id"] not in existing_eids:
            spec["entities"].append(ent)
    spec["edges"].extend(ns_pattern["edges"])
    existing_schemas = {s["name"] for s in spec.get("schemas", [])}
    for schema in ns_pattern["schemas"]:
        if schema["name"] not in existing_schemas:
            spec.setdefault("schemas", []).append(schema)

    _record_mutation(spec, "insert_pattern", {
        "pattern": pattern_name,
        "after": from_pid,
        "before": to_pid,
        "prefix": prefix,
    })
    return spec


def remove_pattern(spec):
    """Remove a detected pattern from the spec and rewire around it.

    Detects patterns in the spec, picks a non-entry one, removes its
    processes/entities/edges, and connects predecessor directly to successor.
    """
    pat_mod = _get_patterns_module()
    spec = copy.deepcopy(spec)

    detected = pat_mod.detect_patterns(spec)
    if not detected:
        raise ValueError("No patterns detected in spec to remove")

    # Don't remove the only pattern or the entry-point pattern
    entry = spec.get("entry_point")
    removable = [(pn, pids, pfx) for pn, pids, pfx in detected
                 if entry not in pids]

    if not removable:
        # Allow removing even the entry pattern if there are multiple
        if len(detected) > 1:
            removable = detected[1:]  # Skip first (contains entry point)
        else:
            raise ValueError("Cannot remove the only pattern (contains entry point)")

    # Don't remove a pattern if it would leave zero agents
    if len(detected) <= 1:
        raise ValueError("Cannot remove the only pattern in the spec")

    target_pname, target_pids, target_prefix = random.choice(removable)

    # Find predecessor and successor
    predecessors = set()
    successors = set()
    for edge in spec.get("edges", []):
        if edge.get("to") in target_pids and edge.get("from") not in target_pids:
            predecessors.add(edge.get("from"))
        if edge.get("from") in target_pids and edge.get("to") not in target_pids:
            successors.add(edge.get("to"))

    # Remove pattern's internal edges and processes
    old_entity_ids = set()
    for edge in spec.get("edges", []):
        if edge.get("from") in target_pids and edge.get("type") in ("invoke", "read", "write"):
            old_entity_ids.add(edge.get("to"))

    spec["edges"] = [e for e in spec["edges"]
                     if e.get("from") not in target_pids and e.get("to") not in target_pids]
    spec["processes"] = [p for p in spec["processes"] if p["id"] not in target_pids]

    # Remove orphaned entities
    remaining_refs = set()
    for e in spec.get("edges", []):
        remaining_refs.add(e.get("from"))
        remaining_refs.add(e.get("to"))
    for eid in old_entity_ids:
        if eid not in remaining_refs:
            spec["entities"] = [e for e in spec["entities"] if e["id"] != eid]

    # Clean up edges referencing removed nodes
    all_node_ids = {p["id"] for p in spec["processes"]} | {e["id"] for e in spec["entities"]}
    spec["edges"] = [e for e in spec["edges"]
                     if e.get("from") in all_node_ids and e.get("to") in all_node_ids]

    # Wire predecessors directly to successors
    for pred in predecessors:
        for succ in successors:
            if _process_by_id(spec, succ) or _entity_by_id(spec, succ):
                spec["edges"].append({
                    "type": "flow",
                    "from": pred,
                    "to": succ,
                    "label": f"Bypass removed {target_pname}",
                })

    _record_mutation(spec, "remove_pattern", {
        "pattern": target_pname,
        "removed_processes": sorted(target_pids),
    })
    return spec


def crossover(spec_a, spec_b):
    """Crossover: take a pattern from spec_a and graft it into spec_b.

    Detects patterns in both specs, finds a shared pattern type,
    replaces spec_b's version with spec_a's version.
    Returns a modified copy of spec_b.
    """
    pat_mod = _get_patterns_module()
    detected_a = pat_mod.detect_patterns(spec_a)
    detected_b = pat_mod.detect_patterns(spec_b)

    if not detected_a or not detected_b:
        raise ValueError("Both specs must contain detectable patterns for crossover")

    # Find shared pattern types
    types_a = {pname for pname, _, _ in detected_a}
    types_b = {pname for pname, _, _ in detected_b}
    shared = types_a & types_b

    if not shared:
        # Fall back: insert a random pattern from A into B
        donor_pname, donor_pids, donor_prefix = random.choice(detected_a)
        result = copy.deepcopy(spec_b)

        # Get the source pattern from A's spec subgraph
        donor_pattern = pat_mod.get_pattern(donor_pname)
        prefix = f"cx_{_short_id()}"
        from .compose import _namespace_pattern
        ns_pattern = _namespace_pattern(donor_pattern, prefix)

        # Find a flow edge in B to insert at
        flow_edges = [e for e in result.get("edges", []) if e.get("type") == "flow"
                      and _process_by_id(result, e.get("from"))
                      and _process_by_id(result, e.get("to"))]
        if not flow_edges:
            raise ValueError("No flow edges in spec_b to insert crossover pattern")

        edge = random.choice(flow_edges)
        from_pid = edge["from"]
        to_pid = edge["to"]

        result["edges"].remove(edge)
        result["edges"].append({
            "type": "flow", "from": from_pid,
            "to": ns_pattern["interface"]["entry"],
            "label": f"Crossover from {donor_pname}",
        })
        for exit_pid in ns_pattern["interface"]["exits"]:
            result["edges"].append({
                "type": "flow", "from": exit_pid, "to": to_pid,
                "label": f"After crossover {donor_pname}",
            })

        result["processes"].extend(ns_pattern["processes"])
        existing_eids = {e["id"] for e in result["entities"]}
        for ent in ns_pattern["entities"]:
            if ent["id"] not in existing_eids:
                result["entities"].append(ent)
        result["edges"].extend(ns_pattern["edges"])
        existing_schemas = {s["name"] for s in result.get("schemas", [])}
        for schema in ns_pattern["schemas"]:
            if schema["name"] not in existing_schemas:
                result.setdefault("schemas", []).append(schema)

        _record_mutation(result, "crossover", {
            "type": "insert",
            "donor_pattern": donor_pname,
            "prefix": prefix,
        })
        return result

    # Shared pattern type: swap B's version with A's version
    shared_type = random.choice(sorted(shared))

    # Get A's version of this pattern (from library, fresh)
    donor_pattern = pat_mod.get_pattern(shared_type)

    # Remove B's version
    b_match = [(pn, pids, pfx) for pn, pids, pfx in detected_b if pn == shared_type][0]
    _, b_pids, b_prefix = b_match

    result = copy.deepcopy(spec_b)

    # Find B's incoming/outgoing connections
    incoming = []
    outgoing = []
    for edge in result.get("edges", []):
        if edge.get("to") in b_pids and edge.get("from") not in b_pids:
            incoming.append(edge)
        if edge.get("from") in b_pids and edge.get("to") not in b_pids:
            outgoing.append(edge)

    # Remove B's pattern
    b_entity_ids = set()
    for edge in result.get("edges", []):
        if edge.get("from") in b_pids and edge.get("type") in ("invoke", "read", "write"):
            b_entity_ids.add(edge.get("to"))

    result["edges"] = [e for e in result["edges"]
                       if e.get("from") not in b_pids and e.get("to") not in b_pids]
    result["processes"] = [p for p in result["processes"] if p["id"] not in b_pids]

    remaining_refs = set()
    for e in result.get("edges", []):
        remaining_refs.add(e.get("from"))
        remaining_refs.add(e.get("to"))
    for eid in b_entity_ids:
        if eid not in remaining_refs:
            result["entities"] = [e for e in result["entities"] if e["id"] != eid]

    # Insert A's version
    prefix = b_prefix or f"cx_{_short_id()}"
    from .compose import _namespace_pattern
    ns_pattern = _namespace_pattern(donor_pattern, prefix)

    result["processes"].extend(ns_pattern["processes"])
    existing_eids = {e["id"] for e in result["entities"]}
    for ent in ns_pattern["entities"]:
        if ent["id"] not in existing_eids:
            result["entities"].append(ent)
    result["edges"].extend(ns_pattern["edges"])
    existing_schemas = {s["name"] for s in result.get("schemas", [])}
    for schema in ns_pattern["schemas"]:
        if schema["name"] not in existing_schemas:
            result.setdefault("schemas", []).append(schema)

    # Rewire
    for edge in incoming:
        edge["to"] = ns_pattern["interface"]["entry"]
        result["edges"].append(edge)
    for edge in outgoing:
        for exit_pid in ns_pattern["interface"]["exits"]:
            new_edge = copy.deepcopy(edge)
            new_edge["from"] = exit_pid
            result["edges"].append(new_edge)

    _record_mutation(result, "crossover", {
        "type": "swap",
        "pattern": shared_type,
        "prefix": prefix,
    })
    return result


# ════════════════════════════════════════════════════════════════════
# Mutation registry
# ════════════════════════════════════════════════════════════════════

_FIELD_MUTATIONS = {
    "swap_process_order": {
        "fn": swap_process_order,
        "description": "Swap two adjacent non-gate processes in the flow",
    },
    "add_review_step": {
        "fn": add_review_step,
        "description": "Insert a review process+agent after a random step",
    },
    "remove_process": {
        "fn": remove_process,
        "description": "Remove a non-critical process and rewire around it",
    },
    "change_gate_condition": {
        "fn": change_gate_condition,
        "description": "Modify a gate's condition (invert, change threshold, etc.)",
    },
    "duplicate_with_variation": {
        "fn": duplicate_with_variation,
        "description": "Copy an agent with a different system prompt personality",
    },
    "add_store": {
        "fn": add_store,
        "description": "Add a new store and wire a write edge from a random process",
    },
    "change_model": {
        "fn": change_model,
        "description": "Switch the model for one or more agents",
    },
    "modify_prompt": {
        "fn": modify_prompt,
        "description": "Apply a transformation to an agent's system prompt",
    },
}

# Pattern-level mutations
_PATTERN_MUTATIONS = {
    "swap_pattern": {
        "fn": swap_pattern,
        "description": "Replace a detected pattern with a compatible alternative",
    },
    "insert_pattern": {
        "fn": insert_pattern,
        "description": "Insert a pattern at a random flow edge",
    },
    "remove_pattern": {
        "fn": remove_pattern,
        "description": "Remove a detected pattern and rewire around it",
    },
}

# Mutations that consistently destroy task-specific accuracy and waste evaluation budget.
# insert_pattern injects generic patterns (critique_cycle, debate) that replace task solving
# with content generation, averaging fitness ~37.5 across 64 candidates (vs ~200 baseline).
_BLACKLISTED_MUTATIONS = {"insert_pattern"}

# Combined registry (includes progressive disclosure operators)
MUTATIONS = {**_FIELD_MUTATIONS, **_PATTERN_MUTATIONS,
    "rewrite_prompt": {
        "fn": None, "weight": 0,
        "description": "Freeform prompt rewrite prescribed by LLM diagnosis",
    },
    "append_to_prompt": {
        "fn": None, "weight": 0,
        "description": "Append instructions to existing prompt (safe, additive only)",
    },
    "edit_prompt": {
        "fn": None, "weight": 0,
        "description": "Targeted find-and-replace on an agent's prompt (surgical, preserves context)",
    },
}


def enumerate_mutations(spec, benchmark_suite=None):
    """Inspect a spec and return every concrete mutation option with parameters.

    Returns a list of dicts, each describing one applicable mutation:
        {"operator": "modify_prompt", "agent": "critic", "transform": "add_chain_of_thought",
         "description": "Apply chain-of-thought to critic agent's prompt"}

    The LLM can select from this menu instead of inventing freeform mutations.
    If benchmark_suite is provided, also includes benchmark-specific prompt transforms.
    """
    options = []

    agents = _agent_entities(spec)
    agents_with_prompts = [a for a in agents if a.get("system_prompt")]
    steps = _step_processes(spec)
    gates = _gate_processes(spec)
    flow = _flow_edges(spec)
    chain = _build_flow_chain(spec)

    # --- modify_prompt: agent × transform ---
    for agent in agents_with_prompts:
        for t in _PROMPT_TRANSFORMS:
            options.append({
                "operator": "modify_prompt",
                "agent": agent["id"],
                "transform": t["name"],
                "description": f"Apply '{t['name']}' to {agent.get('label', agent['id'])} prompt",
            })

    # --- modify_prompt: benchmark-specific transforms ---
    if benchmark_suite and benchmark_suite in _BENCHMARK_PROMPT_TRANSFORMS:
        for agent in agents_with_prompts:
            for t in _BENCHMARK_PROMPT_TRANSFORMS[benchmark_suite]:
                options.append({
                    "operator": "modify_prompt",
                    "agent": agent["id"],
                    "transform": t["name"],
                    "description": f"[BENCHMARK] Apply '{t['name']}' to {agent.get('label', agent['id'])} prompt",
                })

    # --- duplicate_with_variation: agent × variation ---
    variations = ["cautious", "creative", "concise", "detailed", "skeptical", "optimistic"]
    for agent in agents:
        for v in variations:
            options.append({
                "operator": "duplicate_with_variation",
                "agent": agent["id"],
                "variation": v,
                "description": f"Duplicate {agent.get('label', agent['id'])} with {v} personality",
            })

    # --- change_model: agent × model ---
    for agent in agents:
        current = agent.get("model", "unknown")
        for model in _MODEL_POOL:
            if model != current:
                options.append({
                    "operator": "change_model",
                    "agent": agent["id"],
                    "model": model,
                    "description": f"Switch {agent.get('label', agent['id'])} from {current} to {model}",
                })

    # --- add_review_step: target step ---
    for step in steps:
        outgoing_flow = [e for e in flow if e.get("from") == step["id"]]
        if outgoing_flow and not _is_critical_process(spec, step["id"]):
            options.append({
                "operator": "add_review_step",
                "after_step": step["id"],
                "description": f"Insert review agent after {step.get('label', step['id'])}",
            })

    # --- remove_process: removable processes ---
    for proc in spec.get("processes", []):
        if not _is_critical_process(spec, proc["id"]) and proc.get("type") != "gate":
            outgoing = [e for e in flow if e.get("from") == proc["id"]]
            if outgoing:
                options.append({
                    "operator": "remove_process",
                    "process": proc["id"],
                    "description": f"Remove {proc.get('label', proc['id'])} and rewire around it",
                })

    # --- swap_process_order: adjacent pairs ---
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        pa = _process_by_id(spec, a)
        pb = _process_by_id(spec, b)
        if not pa or not pb:
            continue
        if pa.get("type") == "gate" or pb.get("type") == "gate":
            continue
        if _is_critical_process(spec, a) and _is_critical_process(spec, b):
            continue
        options.append({
            "operator": "swap_process_order",
            "process_a": a,
            "process_b": b,
            "description": f"Swap order of {pa.get('label', a)} and {pb.get('label', b)}",
        })

    # --- change_gate_condition: gate × transform ---
    gate_transforms = ["invert", "raise_threshold", "lower_threshold", "swap_and_or", "toggle_operator"]
    for gate in gates:
        for gt in gate_transforms:
            options.append({
                "operator": "change_gate_condition",
                "gate": gate["id"],
                "transform": gt,
                "description": f"Apply '{gt}' to gate {gate.get('label', gate['id'])} (condition: {gate.get('condition', '?')[:50]})",
            })

    # --- add_store: step × store_type ---
    store_types = ["queue", "vector", "kv"]
    for step in steps:
        for st in store_types:
            options.append({
                "operator": "add_store",
                "source_step": step["id"],
                "store_type": st,
                "description": f"Add {st} store written by {step.get('label', step['id'])}",
            })

    # --- insert_pattern: SKIPPED (blacklisted) ---
    # insert_pattern consistently destroys task-specific accuracy by injecting generic
    # patterns (critique_cycle, debate) that replace task solving with content generation.
    # Average fitness ~37.5 across 64 candidates vs ~200 baseline. See _BLACKLISTED_MUTATIONS.

    # --- swap_pattern / remove_pattern: detected patterns ---
    try:
        pat_mod = _get_patterns_module()
        pattern_names = list(pat_mod.PATTERN_LIBRARY.keys()) if hasattr(pat_mod, 'PATTERN_LIBRARY') else []
        detected = pat_mod.detect_patterns(spec) if hasattr(pat_mod, 'detect_patterns') else []
    except Exception:
        pattern_names = []
        detected = []
    for det_name, det_pids, det_prefix in detected:
        # swap options
        for pname in pattern_names:
            if pname != det_name:
                options.append({
                    "operator": "swap_pattern",
                    "current_pattern": det_name,
                    "replacement_pattern": pname,
                    "description": f"Replace {det_name} pattern with {pname}",
                })
        # remove option (only if >1 pattern detected)
        if len(detected) > 1:
            options.append({
                "operator": "remove_pattern",
                "pattern": det_name,
                "description": f"Remove {det_name} pattern and rewire around it",
            })

    return options


def apply_selected_mutation(spec, selection):
    """Apply a single mutation selected by the LLM from the enumerated menu.

    Args:
        spec: The spec to mutate
        selection: A dict from enumerate_mutations (or LLM selection matching its schema)

    Returns: mutated spec

    The selection dict must have an 'operator' field. Additional fields are used
    to seed random.choice with the specific option so the mutation targets the
    right element.
    """
    op = selection["operator"]

    if op == "modify_prompt":
        result = copy.deepcopy(spec)
        agent_id = selection.get("agent")
        transform_name = selection.get("transform")
        agents = _agent_entities(result)
        agent = next((a for a in agents if a["id"] == agent_id), None)
        if not agent or not agent.get("system_prompt"):
            raise ValueError(f"Agent {agent_id} not found or has no prompt")
        tdef = next((t for t in _PROMPT_TRANSFORMS if t["name"] == transform_name), None)
        if not tdef:
            # Search benchmark-specific transforms
            for bench_transforms in _BENCHMARK_PROMPT_TRANSFORMS.values():
                tdef = next((t for t in bench_transforms if t["name"] == transform_name), None)
                if tdef:
                    break
        if not tdef:
            raise ValueError(f"Unknown prompt transform: {transform_name}")
        agent["system_prompt"] = tdef["transform"](agent["system_prompt"])
        _record_mutation(result, "modify_prompt", {
            "agent": agent_id, "transform": transform_name,
            "description": tdef["description"],
        })
        return result

    elif op == "duplicate_with_variation":
        result = copy.deepcopy(spec)
        agent_id = selection.get("agent")
        variation = selection.get("variation")
        agents = _agent_entities(result)
        original = next((a for a in agents if a["id"] == agent_id), None)
        if not original:
            raise ValueError(f"Agent {agent_id} not found")
        variation_map = {
            "cautious": "Be extra cautious and conservative in your responses. Double-check all claims. Prefer safety over speed.",
            "creative": "Be more creative and exploratory. Consider unconventional approaches. Think outside the box.",
            "concise": "Be extremely concise. Minimize verbosity. Output only essential information.",
            "detailed": "Be thorough and detailed. Explain your reasoning step by step. Leave nothing ambiguous.",
            "skeptical": "Be skeptical of assumptions. Question premises. Consider failure modes and edge cases.",
            "optimistic": "Focus on opportunities and positive outcomes. Assume good faith. Look for the best path forward.",
        }
        prefix = variation_map.get(variation, "")
        if not prefix:
            raise ValueError(f"Unknown variation: {variation}")
        suffix = _short_id()
        new_id = f"{agent_id}_var_{suffix}"
        new_agent = copy.deepcopy(original)
        new_agent["id"] = new_id
        new_agent["label"] = f"{original.get('label', original['id'])} ({variation})"
        new_agent["system_prompt"] = f"{prefix}\n\n{original.get('system_prompt', '')}"
        result["entities"].append(new_agent)
        _record_mutation(result, "duplicate_with_variation", {
            "original": agent_id, "variant": new_id, "variation": variation,
        })
        return result

    elif op == "change_model":
        result = copy.deepcopy(spec)
        agent_id = selection.get("agent")
        new_model = selection.get("model")
        agents = _agent_entities(result)
        agent = next((a for a in agents if a["id"] == agent_id), None)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        old_model = agent.get("model", "unknown")
        agent["model"] = new_model
        _record_mutation(result, "change_model", {
            "changes": [{"agent": agent_id, "old_model": old_model, "new_model": new_model}],
        })
        return result

    elif op == "add_review_step":
        # Use the existing function but seed it to target the right step
        # We can't easily parameterize the existing function, so we temporarily
        # limit step choices. The simplest approach: call the existing function
        # in a retry loop until it picks the right target, or just use the
        # existing function (which is already robust).
        return add_review_step(spec)

    elif op == "remove_process":
        return remove_process(spec)

    elif op == "swap_process_order":
        return swap_process_order(spec)

    elif op == "change_gate_condition":
        return change_gate_condition(spec)

    elif op == "add_store":
        return add_store(spec)

    elif op == "insert_pattern":
        return insert_pattern(spec)

    elif op == "swap_pattern":
        return swap_pattern(spec)

    elif op == "remove_pattern":
        return remove_pattern(spec)

    elif op == "rewrite_prompt":
        # Freeform prompt rewrite — prescribed by progressive disclosure pipeline
        result = copy.deepcopy(spec)
        agent_id = selection.get("agent")
        new_prompt = selection.get("new_prompt")
        if not agent_id or not new_prompt:
            raise ValueError("rewrite_prompt requires 'agent' and 'new_prompt' fields")
        agents = _agent_entities(result)
        agent = next((a for a in agents if a["id"] == agent_id), None)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        agent["system_prompt"] = new_prompt
        _record_mutation(result, "rewrite_prompt", {
            "agent": agent_id,
            "description": selection.get("description", "LLM-prescribed prompt rewrite"),
        })
        return result

    elif op == "append_to_prompt":
        # Append instructions to existing prompt — safe, can't break what works
        result = copy.deepcopy(spec)
        agent_id = selection.get("agent")
        append_text = selection.get("text")
        if not agent_id or not append_text:
            raise ValueError("append_to_prompt requires 'agent' and 'text' fields")
        agents = _agent_entities(result)
        agent = next((a for a in agents if a["id"] == agent_id), None)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        existing = agent.get("system_prompt", "")
        agent["system_prompt"] = existing.rstrip() + "\n\n" + append_text
        _record_mutation(result, "append_to_prompt", {
            "agent": agent_id,
            "appended": append_text[:200],
            "description": selection.get("description", "LLM-prescribed prompt addition"),
        })
        return result

    elif op == "edit_prompt":
        # Targeted find-and-replace on an agent's prompt — surgical, preserves context
        result = copy.deepcopy(spec)
        agent_id = selection.get("agent")
        old_text = selection.get("old_text")
        new_text = selection.get("new_text")
        if not agent_id or old_text is None or new_text is None:
            raise ValueError("edit_prompt requires 'agent', 'old_text', and 'new_text' fields")
        agents = _agent_entities(result)
        agent = next((a for a in agents if a["id"] == agent_id), None)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        prompt = agent.get("system_prompt", "")
        if old_text not in prompt:
            raise ValueError(f"old_text not found in {agent_id}'s prompt")
        agent["system_prompt"] = prompt.replace(old_text, new_text, 1)
        _record_mutation(result, "edit_prompt", {
            "agent": agent_id,
            "old_text": old_text[:100],
            "new_text": new_text[:100],
            "description": selection.get("description", "LLM-prescribed prompt edit"),
        })
        return result

    else:
        raise ValueError(f"Unknown operator: {op}")


def apply_mutation(spec, mutation_name):
    """Apply a named mutation to a spec, returning a new spec."""
    if mutation_name not in MUTATIONS:
        raise ValueError(f"Unknown mutation '{mutation_name}'. Available: {sorted(MUTATIONS.keys())}")
    fn = MUTATIONS[mutation_name]["fn"]
    return fn(spec)


def apply_random_mutation(spec, donor_spec=None, pattern_weight=0.4):
    """Apply a random mutation to a spec, returning (new_spec, mutation_name).

    Args:
        spec: The spec to mutate
        donor_spec: Optional second spec for crossover
        pattern_weight: Probability of choosing a pattern-level mutation (0-1)
    """
    # If donor_spec provided, try crossover first
    if donor_spec:
        try:
            result = crossover(spec, donor_spec)
            return result, "crossover"
        except ValueError:
            pass  # Fall through to regular mutations

    # Decide: pattern-level or field-level (excluding blacklisted)
    if random.random() < pattern_weight:
        pool = [k for k in _PATTERN_MUTATIONS if k not in _BLACKLISTED_MUTATIONS]
    else:
        pool = [k for k in _FIELD_MUTATIONS if k not in _BLACKLISTED_MUTATIONS]

    random.shuffle(pool)
    for name in pool:
        try:
            new_spec = apply_mutation(spec, name)
            return new_spec, name
        except ValueError:
            continue

    # Fallback: try all mutations (excluding blacklisted)
    all_mutations = [k for k in MUTATIONS if k not in _BLACKLISTED_MUTATIONS]
    random.shuffle(all_mutations)
    for name in all_mutations:
        try:
            new_spec = apply_mutation(spec, name)
            return new_spec, name
        except ValueError:
            continue
    raise ValueError("All mutations failed for this spec")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Spec Mutation Engine -- evolutionary search over agent architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 mutate.py specs/react.yaml --mutation swap_process_order -o specs/react_v.yaml\n"
            "  python3 mutate.py specs/react.yaml --random -n 3 -o variants/\n"
            "  python3 mutate.py specs/react.yaml --mutation add_review_step\n"
            "  python3 mutate.py --list-mutations\n"
        ),
    )
    parser.add_argument("spec", nargs="?", help="Path to the input agent spec YAML")
    parser.add_argument("--mutation", "-m", help="Name of the mutation operator to apply")
    parser.add_argument("--random", "-r", action="store_true", help="Apply random mutation(s)")
    parser.add_argument("-n", type=int, default=1, help="Number of variants to generate (with --random)")
    parser.add_argument("-o", "--output", help="Output file path, or directory (with --random -n >1)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--list-mutations", action="store_true", help="List all available mutations and exit")

    args = parser.parse_args()

    # List mutations mode
    if args.list_mutations:
        print("Available mutations:\n")
        for name, info in sorted(MUTATIONS.items()):
            print(f"  {name:30s} {info['description']}")
        print()
        sys.exit(0)

    # Require spec for actual mutations
    if not args.spec:
        parser.error("spec is required (unless using --list-mutations)")

    if not args.mutation and not args.random:
        parser.error("specify --mutation NAME or --random")

    if args.seed is not None:
        random.seed(args.seed)

    # Load
    spec = load_spec(args.spec)
    spec_basename = os.path.splitext(os.path.basename(args.spec))[0]

    if args.mutation:
        # Single named mutation
        try:
            result = apply_mutation(spec, args.mutation)
        except ValueError as e:
            print(f"Mutation failed: {e}", file=sys.stderr)
            sys.exit(1)

        output_yaml = dump_spec(result)

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                f.write(output_yaml)
            print(f"Wrote mutated spec to {args.output}")
            _print_mutation_summary(result)
        else:
            print(output_yaml)

    elif args.random:
        # Random mutations
        results = []
        for i in range(args.n):
            try:
                result, mutation_name = apply_random_mutation(spec)
                results.append((result, mutation_name, i))
            except ValueError as e:
                print(f"Warning: random mutation {i + 1} failed: {e}", file=sys.stderr)

        if not results:
            print("All random mutations failed.", file=sys.stderr)
            sys.exit(1)

        if args.output:
            if args.n > 1 or os.path.isdir(args.output) or args.output.endswith("/"):
                # Output to directory
                os.makedirs(args.output, exist_ok=True)
                for result, mutation_name, i in results:
                    filename = f"{spec_basename}_{mutation_name}_{i}.yaml"
                    filepath = os.path.join(args.output, filename)
                    with open(filepath, "w") as f:
                        f.write(dump_spec(result))
                    print(f"Wrote {filepath} (mutation: {mutation_name})")
                    _print_mutation_summary(result)
            else:
                # Single file output
                result, mutation_name, _ = results[0]
                with open(args.output, "w") as f:
                    f.write(dump_spec(result))
                print(f"Wrote mutated spec to {args.output} (mutation: {mutation_name})")
                _print_mutation_summary(result)
        else:
            for result, mutation_name, i in results:
                if args.n > 1:
                    print(f"# ── Variant {i} ({mutation_name}) ──")
                print(dump_spec(result))
                if args.n > 1:
                    print()


def _print_mutation_summary(spec):
    """Print a short summary of mutations applied."""
    mutations = spec.get("mutations", [])
    if not mutations:
        return
    last = mutations[-1]
    details = last.get("details", {})
    detail_str = ", ".join(f"{k}={v}" for k, v in details.items() if k != "changes") if details else ""
    if "changes" in details:
        changes = details["changes"]
        if isinstance(changes, list):
            for c in changes:
                detail_str += f" | {c.get('agent', '?')}: {c.get('old_model', '?')} -> {c.get('new_model', '?')}"
    print(f"  Applied: {last['operator']}" + (f" ({detail_str})" if detail_str else ""))


if __name__ == "__main__":
    main()
