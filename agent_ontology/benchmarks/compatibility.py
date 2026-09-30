#!/usr/bin/env python3
"""
Agent-dataset compatibility matrix and input formatters.

Defines which agents can run which benchmark suites, and how to format
dataset examples into the agent's expected input schema.
"""

# Agent -> list of compatible datasets
COMPATIBILITY = {
    "react":             ["hotpotqa", "gsm8k", "arc"],
    "rag":               ["hotpotqa"],
    "plan_and_solve":    ["gsm8k", "arc"],
    "self_refine":       ["gsm8k", "humaneval"],
    "tree_of_thought":   ["gsm8k", "arc"],
    "lats":              ["hotpotqa", "gsm8k"],
    "reflexion":         ["gsm8k"],
    "multi_agent_codegen": ["humaneval"],
    "minimal_solver":    ["gsm8k", "gsm8k_hard", "gsm8k_tricky", "arc"],
    "multidoc_baseline": ["multidoc"],
    "multidoc_structured": ["multidoc"],
    "kb_react": ["kb_tool"],
    "mutator": ["meta_improve"],
}

# Dataset -> list of compatible agents (derived from COMPATIBILITY)
DATASET_AGENTS = {}
for agent, datasets in COMPATIBILITY.items():
    for ds in datasets:
        DATASET_AGENTS.setdefault(ds, []).append(agent)


def get_compatible_agents(dataset_name):
    """Return list of agent names compatible with the given dataset."""
    return DATASET_AGENTS.get(dataset_name, [])


def get_compatible_datasets(agent_name):
    """Return list of dataset names compatible with the given agent."""
    return COMPATIBILITY.get(agent_name, [])


def format_input(agent_name, example, dataset_name):
    """Format a benchmark example into the agent's expected input dict.

    Each agent has different input schema expectations. This function
    maps the generic example format to what each agent needs.
    """
    question = example["question"]

    if agent_name == "react":
        return {"query": question}

    elif agent_name == "rag":
        return {"query": question}

    elif agent_name == "plan_and_solve":
        return {"problem": question}

    elif agent_name == "self_refine":
        return {"task": f"Solve this math problem step by step and give a final numeric answer: {question}"}

    elif agent_name == "tree_of_thought":
        return {"problem": question}

    elif agent_name == "lats":
        if dataset_name == "hotpotqa":
            return {"query": question}
        else:
            return {"query": f"Solve: {question}"}

    elif agent_name == "reflexion":
        return {"task": question}

    elif agent_name == "multi_agent_codegen":
        return {
            "project_description": question,
            "language": "Python",
        }

    elif agent_name == "minimal_solver":
        return {"problem": question}

    elif agent_name == "kb_react":
        return {"query": question}

    elif agent_name == "mutator":
        # Load the target spec YAML from file
        spec_yaml = example.get("spec_yaml", "")
        if not spec_yaml and example.get("target_spec"):
            import os
            spec_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "specs")
            spec_path = os.path.join(spec_dir, f"{example['target_spec']}.yaml")
            try:
                with open(spec_path) as _f:
                    spec_yaml = _f.read()
            except FileNotFoundError:
                spec_yaml = f"# Spec not found: {example['target_spec']}"
        return {
            "spec_yaml": spec_yaml,
            "failure_summary": example.get("failure_summary", question),
            "benchmark_description": example.get("benchmark_description", ""),
        }

    elif agent_name in ("multidoc_baseline", "multidoc_structured"):
        # MultiDoc benchmark: include fact cards as context
        fact_cards = example.get("fact_cards", [])
        context = "\n\n".join(
            f"[Source {card['id']}] {card['text']}" for card in fact_cards
        )
        return {
            "query": question,
            "context": context,
            "fact_cards": fact_cards,
        }

    else:
        # Generic fallback — also include fact_cards for multidoc if present
        result = {"query": question}
        if "fact_cards" in example:
            fact_cards = example["fact_cards"]
            context = "\n\n".join(
                f"[Source {card['id']}] {card['text']}" for card in fact_cards
            )
            result["context"] = context
            result["fact_cards"] = fact_cards
        return result
