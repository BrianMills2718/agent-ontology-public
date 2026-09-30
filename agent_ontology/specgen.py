#!/usr/bin/env python3
"""
Agent Ontology Spec Generator (docs → spec)
Takes a natural language description of an agent architecture and generates
a valid Agent Ontology YAML spec.

Usage:
  python3 specgen.py description.md -o specs/my_agent.yaml
  echo "A simple chatbot that..." | python3 specgen.py - -o specs/chatbot.yaml
  python3 specgen.py description.md --validate --model gpt-4.1-mini
"""

import sys
import os
import json
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Pattern keywords for text-based detection
_PATTERN_KEYWORDS = {
    "reasoning_loop": ["reason", "react", "think and act", "tool use loop", "observe", "action loop"],
    "critique_cycle": ["critique", "refine", "self-refine", "quality score", "iterative improvement"],
    "debate": ["debate", "argue", "pro and con", "judge", "opposing views"],
    "retrieval": ["retrieval", "rag", "retrieve", "vector search", "knowledge base", "document retrieval"],
    "decomposition": ["decompose", "sub-problem", "plan and solve", "break down", "step-by-step plan"],
    "fan_out_aggregate": ["parallel", "map reduce", "chunk", "aggregate", "fan-out", "concurrent"],
    "reflection": ["reflect", "episodic memory", "self-reflection", "learn from mistakes", "retry with memory"],
}

# Map pattern names to their source spec files
_PATTERN_SOURCE_SPECS = {
    "reasoning_loop": "react.yaml",
    "critique_cycle": "self_refine.yaml",
    "debate": "debate.yaml",
    "retrieval": "rag.yaml",
    "decomposition": "plan_and_solve.yaml",
    "fan_out_aggregate": "map_reduce.yaml",
    "reflection": "reflexion.yaml",
}


def _scan_description_for_patterns(description):
    """Detect which architectural patterns a description is asking for."""
    text_lower = description.lower()
    matches = []
    for pname, keywords in _PATTERN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matches.append(pname)
    return matches


def load_ontology():
    """Load the ONTOLOGY.yaml as text to include in the prompt."""
    path = os.path.join(SCRIPT_DIR, "ONTOLOGY.yaml")
    with open(path) as f:
        return f.read()


def load_examples(detected_patterns=None):
    """Load example specs. If patterns detected, prefer matching examples."""
    if detected_patterns:
        # Use pattern source specs as examples (up to 2)
        examples = []
        seen = set()
        for pname in detected_patterns[:2]:
            spec_file = _PATTERN_SOURCE_SPECS.get(pname)
            if spec_file and spec_file not in seen:
                path = os.path.join(SCRIPT_DIR, "specs", spec_file)
                if os.path.exists(path):
                    with open(path) as f:
                        examples.append((spec_file, f.read()))
                    seen.add(spec_file)
        # Fill remaining slots with defaults
        for name in ["react.yaml", "code_reviewer.yaml", "debate.yaml"]:
            if len(examples) >= 2:
                break
            if name not in seen:
                path = os.path.join(SCRIPT_DIR, "specs", name)
                if os.path.exists(path):
                    with open(path) as f:
                        examples.append((name, f.read()))
                    seen.add(name)
        return examples
    # Default: hardcoded examples
    examples = []
    for name in ["react.yaml", "code_reviewer.yaml", "debate.yaml"]:
        path = os.path.join(SCRIPT_DIR, "specs", name)
        if os.path.exists(path):
            with open(path) as f:
                examples.append((name, f.read()))
    return examples


def _build_pattern_context(detected_patterns):
    """Build pattern context string for the system prompt."""
    if not detected_patterns:
        return ""
    try:
        from .patterns import get_pattern, pattern_info
    except ImportError:
        return ""

    lines = ["\nDETECTED ARCHITECTURAL PATTERNS:"]
    lines.append("Your description matches these known patterns. Use them as structural guidance:\n")
    for pname in detected_patterns:
        p = get_pattern(pname)
        if not p:
            continue
        iface = p.get("interface", {})
        procs = [pr["id"] for pr in p.get("processes", [])]
        lines.append(f"  {pname}: {p.get('description', '')}")
        lines.append(f"    Entry: {iface.get('entry', '?')}, Exits: {iface.get('exits', [])}")
        lines.append(f"    Core processes: {', '.join(procs[:6])}")
        lines.append(f"    Inputs: {iface.get('inputs', [])}, Outputs: {iface.get('outputs', [])}")
        config = p.get("config", {})
        if config:
            lines.append(f"    Config params: {list(config.keys())}")
        lines.append("")
    lines.append("You should model your spec's structure on these patterns. Use similar")
    lines.append("process IDs, edge topology, and schema design where applicable.")
    return "\n".join(lines)


def build_prompt(description, ontology, examples, detected_patterns=None):
    """Build the system and user prompts for spec generation."""

    example_text = ""
    for name, content in examples[:2]:
        example_text += f"\n--- Example: {name} ---\n{content}\n"

    pattern_context = _build_pattern_context(detected_patterns) if detected_patterns else ""

    system = f"""You are an expert agent architecture analyst. You convert natural language
descriptions of AI agent systems into formal Agent Ontology YAML specifications.

An Agent Ontology spec defines an agent architecture using:
- entities: agents (LLM-based), stores (persistence), tools (capabilities), humans, configs,
            channels (pub/sub), teams (agent groups), conversations (multi-turn dialogue)
- processes: steps (computation), gates (decisions), checkpoints (human approval), spawns (sub-agents)
- edges: flow (sequence), invoke (agent/tool call), loop (cycle), branch (conditional),
         read/write (store access), publish/subscribe (channel messaging), handoff (agent transfer)
- schemas: data shapes flowing between components

RULES:
1. Every entity needs: id, type, label. Agents also need: model, system_prompt, input_schema, output_schema.
2. Every process needs: id, type, label. Steps can have logic (Python code).
3. Gates need: condition (human-readable), branches (2+ with condition and target). Use default for else.
4. Edges connect processes to processes (flow/loop) or processes to entities (invoke/read/write).
5. Schemas define the data contracts between components.
6. The spec needs: name, version, description, entry_point.
7. Tools need: tool_type (api, function, shell, browser, mcp).
8. Stores need: store_type (vector, file, kv, queue, relational, blackboard).
9. Gate conditions should reference flat state.data fields, not nested paths (e.g. use "round" not "state.round").
10. Use gemini-3-flash-preview as the default model unless the description specifies otherwise.
11. Add logic fields to processes that need data preparation before agent invocations.

FAN-OUT PATTERN (CRITICAL):
- When a process needs to invoke MULTIPLE agents in parallel (e.g. security review + style review + logic review), add MULTIPLE invoke edges from that same process to different agents.
- The code generator will automatically run all invocations from a single process and merge results.
- Each invoked agent MUST have a DIFFERENT output_schema to avoid field collisions.
- After fan-out, add a downstream "synthesis" process that reads from all the namespaced results.
- Example: code_reviewer.yaml has analyze_security, analyze_style, analyze_logic all invoked from the same fan-out step.

STATE NAMESPACING:
- Agent outputs are stored in state.data both flat (via .update()) and namespaced (under snake_case of output schema name).
- AVOID schema names where the snake_case version matches a field name in that schema.
  BAD: Schema "Plan" with field "plan" → both produce key "plan", causing collision.
  GOOD: Schema "PlanOutput" with field "plan" → "plan_output" vs "plan", no collision.
- For gate conditions, use flat field names from the output schema (e.g. "quality_score >= 7").

LOOP TERMINATION:
- Every loop MUST have a termination condition. Common patterns:
  - A counter (completed_count >= max_tasks) checked in logic
  - A state flag (_done = True) set when goal is achieved
  - A gate that checks completion and branches to exit
- Without termination, loops run forever. Always include a max iteration check.

PUB/SUB CHANNELS:
- Use channel entities + publish/subscribe edges when agents communicate via shared topics.
- publish edges go from a step to a channel (with data schema).
- subscribe edges go from a channel to a step (agent reads channel messages).
- Always specify data schema on publish edges to avoid state explosion.

HANDOFFS:
- Use handoff edges between agents for peer-to-peer control transfer.
- Handoff edges are metadata — actual routing is done by gate processes.
{pattern_context}
Output ONLY valid YAML. No markdown code fences. No explanatory text before or after.

Here is the ontology (type system):

{ontology}

Here are example specs for reference:
{example_text}"""

    user = f"""Convert the following agent architecture description into an Agent Ontology YAML spec:

---
{description}
---

Generate a complete, valid Agent Ontology spec YAML. Include all entities, processes, edges, and schemas.
Requirements:
- Every entity referenced in an edge MUST exist in entities
- Every schema referenced in entities/edges MUST exist in schemas
- entry_point MUST reference an existing process
- Add logic fields to processes that need data preparation before agent invocations
- If multiple agents run in parallel, use fan-out (multiple invoke edges from same process)
- Every loop MUST have a termination condition
- Avoid schema names whose snake_case matches a field name in the schema"""

    return system, user


def call_llm(model, system, user, temperature=0.3, max_tokens=8192):
    """Call the LLM to generate the spec."""
    if model.startswith("claude") or model.startswith("anthropic"):
        return _call_anthropic(model, system, user, temperature, max_tokens)
    elif model.startswith("gemini"):
        return _call_gemini(model, system, user, temperature, max_tokens)
    else:
        return _call_openai(model, system, user, temperature, max_tokens)


def _call_openai(model, system, user, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI()
    # Newer OpenAI models (gpt-5*, o3*, o4*) require max_completion_tokens
    # and may not support temperature != 1
    is_new_model = any(model.startswith(p) for p in ("gpt-5", "o3", "o4"))
    token_param = "max_completion_tokens" if is_new_model else "max_tokens"
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        token_param: max_tokens,
    }
    if not is_new_model:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_anthropic(model, system, user, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _call_gemini(model, system, user, temperature, max_tokens):
    from google import genai
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=f"{system}\n\n{user}",
        config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return response.text


def extract_yaml(response):
    """Extract YAML from the LLM response, handling code fences."""
    text = response.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip ```yaml
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def validate_spec(yaml_text):
    """Run the validator on the generated spec."""
    import yaml
    import tempfile
    import subprocess

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir=SCRIPT_DIR) as f:
        f.write(yaml_text)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "validate.py"), tmp_path],
            capture_output=True, text=True, cwd=SCRIPT_DIR
        )
        return result.returncode == 0, result.stdout + result.stderr
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Agent Ontology Spec Generator (docs → spec)")
    parser.add_argument("input", help="Path to description file, or '-' for stdin")
    parser.add_argument("-o", "--output", help="Output YAML file path")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="LLM model to use")
    parser.add_argument("--validate", action="store_true", help="Validate generated spec")
    parser.add_argument("--fix", action="store_true", help="Auto-fix validation errors (re-prompt)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt but don't call LLM")
    args = parser.parse_args()

    # Read input
    if args.input == "-":
        description = sys.stdin.read()
    else:
        with open(args.input) as f:
            description = f.read()

    if not description.strip():
        print("Error: empty description", file=sys.stderr)
        sys.exit(1)

    print(f"Generating spec from description ({len(description)} chars)...")
    print(f"Model: {args.model}")

    # Detect patterns in description
    detected_patterns = _scan_description_for_patterns(description)
    if detected_patterns:
        print(f"Detected patterns: {', '.join(detected_patterns)}")

    # Load ontology and examples (pattern-aware)
    ontology = load_ontology()
    examples = load_examples(detected_patterns)

    # Build prompt (with pattern context)
    system, user = build_prompt(description, ontology, examples, detected_patterns)

    if args.dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(system[:2000] + "..." if len(system) > 2000 else system)
        print("\n=== USER PROMPT ===")
        print(user)
        return

    # Generate spec
    t0 = time.time()
    response = call_llm(args.model, system, user)
    duration = time.time() - t0
    print(f"Generated in {duration:.1f}s")

    yaml_text = extract_yaml(response)

    # Validate if requested
    if args.validate or args.fix:
        valid, output = validate_spec(yaml_text)
        print(f"\nValidation: {'PASS' if valid else 'FAIL'}")
        if not valid:
            print(output)

            if args.fix:
                max_fix_attempts = 3
                for attempt in range(1, max_fix_attempts + 1):
                    print(f"\nAuto-fix attempt {attempt}/{max_fix_attempts}...")
                    fix_user = f"""The spec you generated has validation errors:

{output}

Here is the spec you generated:

{yaml_text}

Fix ALL validation errors and return the corrected YAML. Output ONLY valid YAML, no markdown fences.
Pay special attention to:
- Every entity referenced in edges must exist in entities
- Every process referenced in edges must exist in processes
- Every schema referenced in entities/edges must exist in schemas
- Gate branches must have valid target processes
- invoke edges need input and output schema references"""

                    response_fix = call_llm(args.model, system, fix_user)
                    yaml_text = extract_yaml(response_fix)

                    valid, output = validate_spec(yaml_text)
                    print(f"After fix {attempt}: {'PASS' if valid else 'FAIL'}")
                    if valid:
                        break
                    print(output)

    # Pattern compliance check
    if detected_patterns:
        try:
            import yaml
            from .patterns import detect_patterns
            spec = yaml.safe_load(yaml_text)
            if spec:
                found = detect_patterns(spec)
                found_names = {p[0] for p in found}
                matched = found_names & set(detected_patterns)
                if matched:
                    print(f"\nPattern compliance: {len(matched)}/{len(detected_patterns)} patterns detected in output")
                    for pn in matched:
                        print(f"  + {pn}")
                missed = set(detected_patterns) - found_names
                if missed:
                    print(f"  Patterns not detected (may use different process IDs): {', '.join(missed)}")
        except Exception:
            pass  # Pattern check is best-effort

    # Output
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(yaml_text)
        print(f"\nWritten to: {args.output}")
    else:
        print("\n" + yaml_text)


if __name__ == "__main__":
    main()
