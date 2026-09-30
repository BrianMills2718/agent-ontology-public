#!/usr/bin/env python3
"""
Agent Test Harness — Non-interactive test runner for all generated agents.

Runs each agent with canned inputs, captures traces, and reports pass/fail.
Patches input() to avoid blocking on checkpoint processes.

Usage:
    python3 test_agents.py                    # Run all agents (requires API keys)
    python3 test_agents.py --agent react      # Run specific agent
    python3 test_agents.py --dry-run          # Check imports only, no API calls
    python3 test_agents.py --timeout 60       # Set per-agent timeout (seconds)
    python3 test_agents.py --json             # Output results as JSON
    python3 test_agents.py --model gpt-4o     # Override model for all agents
    python3 test_agents.py --compare-models   # Compare across multiple models
"""

import argparse
import importlib
import json
import os
import sys
import time
import signal
import traceback
from contextlib import contextmanager
from io import StringIO

# Ensure project root is on sys.path so agents package is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ── Test fixtures: canned inputs for each agent ──

TEST_INPUTS = {
    "react": {
        "query": "What year was the Eiffel Tower built?",
    },
    "debate": {
        "topic": "Should remote work be mandatory?",
    },
    "babyagi": {
        "objective": "Plan a birthday party",
        "max_tasks": 2,
        "tasks": [
            {"id": 1, "description": "Choose a venue", "status": "pending", "priority": 1}
        ],
    },
    "babyagi_autogen": {
        "objective": "Plan a birthday party",
        "max_tasks": 2,
        "tasks": [
            {"id": 1, "description": "Choose a venue", "status": "pending", "priority": 1}
        ],
    },
    "autogpt": {
        "goal": "Write a haiku about the moon",
    },
    "rag": {
        "query": "What is the tallest building in the world?",
    },
    "code_reviewer": {
        "diff": '--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,5 @@\n+import os\n+PASSWORD = "secret123"\n def hello():\n     return "world"',
        "files": ["app.py"],
        "description": "Add password constant",
    },
    "crew": {
        "objective": "Write a short blog post about AI safety",
    },
    "tree_of_thought": {
        "problem": "What is the most efficient way to sort a million integers?",
        "branching_factor": 3,
        "max_depth": 2,
        "beam_width": 2,
    },
    "plan_and_solve": {
        "problem": "How do you build a basic web application with user authentication?",
        "max_retries": 1,
    },
    "self_refine": {
        "task": "Write a clear, concise explanation of how photosynthesis works for a 10-year-old.",
        "max_rounds": 2,
    },
    "lats": {
        "query": "What is the square root of 144?",
    },
    "voyager": {
        "max_iterations": 1,
    },
    "multi_agent_codegen": {
        "prompt": "Write a Python function that checks if a string is a palindrome",
    },
    "map_reduce": {
        "full_text": "The quick brown fox jumps over the lazy dog. " * 20,
        "task_description": "Extract all animals mentioned in this text.",
        "chunk_size": 50,
    },
    "socratic_tutor": {
        "topic": "What is 2+2?",
        "learning_objectives": ["Understand basic addition"],
        "difficulty": "beginner",
        "_canned_responses": [
            "2+2 equals 4 because you combine two groups of two",
            "Addition means combining quantities to get a sum",
            "The commutative property means 2+3 equals 3+2",
            "4 minus 2 equals 2, it reverses addition",
            "I understand addition is combining numbers together",
            "Addition is the inverse of subtraction",
        ],
    },
    "reflexion": {
        "task": "What is the capital of France?",
        "success_criteria": "The answer must name the capital city of France.",
        "max_trials": 2,
    },
    "mixture_of_agents": {
        "query": "Explain the pros and cons of renewable energy in 3 sentences.",
        "max_layers": 1,
    },
    "meta_prompting": {
        "task": "Summarize the key principles of good software design.",
        "max_depth": 1,
    },
    "customer_support_swarm": {
        "query": "I need a refund for my last purchase, order #12345.",
    },
    "software_team": {
        "query": "Build a simple URL shortener web application with a REST API.",
        "max_iterations": 1,
    },
    "self_improver": {
        "spec_yaml": open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent_ontology", "specs", "self_refine.yaml")).read(),
        "max_rounds": 1,
    },
    "kg_rag": {
        "query": "Who founded Nexara Technologies?",
        "_kg_triples": [
            ["Nexara Technologies", "foundedBy", "Dr. Elara Voss"],
            ["Nexara Technologies", "locatedIn", "Seattle"],
            ["Dr. Elara Voss", "worksAt", "Nexara Technologies"],
            ["Nexara Technologies", "type", "Organization"],
            ["Dr. Elara Voss", "type", "Person"],
            ["Seattle", "type", "Location"],
        ],
    },
    "pddl_planner": {
        "task": "Stack block A on block B",
    },
    "rap": {
        "problem": "What is 15 * 7?",
        "max_iterations": 2,
    },
    "alpha_geometry": {
        "problem": "Given triangle ABC where AB = AC, prove it is isosceles",
        "goal": "Prove triangle ABC is isosceles",
        "known_facts": ["AB = AC", "angle BAC = 60"],
        "max_iterations": 2,
    },
}

# ── Validation criteria per agent ──

def validate_react(state):
    """ReAct should produce a final answer."""
    issues = []
    if not state.data.get("answer") and not state.data.get("final_answer"):
        issues.append("No answer produced")
    # Trajectory check: only flag if there's also no answer (direct answers skip trajectory)
    trajectory = state.data.get("trajectory", [])
    has_answer = state.data.get("answer") or state.data.get("final_answer")
    if len(trajectory) < 1 and not has_answer:
        issues.append("No reasoning steps recorded and no answer produced")
    return issues

def validate_debate(state):
    """Debate should complete 3 rounds with a winner."""
    issues = []
    if not state.data.get("winner"):
        issues.append("No winner determined")
    history = state.data.get("debate_history", [])
    if len(history) < 4:  # at least 2 pro + 2 con
        issues.append(f"Only {len(history)} debate turns (expected >= 4)")
    if state.data.get("pro_score") is None:
        issues.append("No pro_score from judge")
    return issues

def validate_babyagi(state):
    """BabyAGI should complete at least 1 task loop."""
    issues = []
    if state.data.get("completed_count", 0) < 1:
        issues.append("No tasks completed")
    return issues

def validate_babyagi_autogen(state):
    """BabyAGI Autogen should complete at least 1 full loop."""
    issues = []
    if not state.data.get("result"):
        issues.append("No execution result produced")
    return issues

def validate_autogpt(state):
    """AutoGPT should produce output through its think/execute loop."""
    issues = []
    results = state.data.get("results", [])
    if not results and not state.data.get("result") and not state.data.get("output"):
        issues.append("No result produced")
    return issues

def validate_rag(state):
    """RAG should produce an answer (or explicit no-answer)."""
    issues = []
    if not state.data.get("answer"):
        issues.append("No answer produced")
    return issues

def validate_code_reviewer(state):
    """Code Reviewer should produce a synthesized review with quality score."""
    issues = []
    if state.data.get("quality_score") is None:
        issues.append("No quality_score produced")
    if not state.data.get("security_review"):
        issues.append("No security_review (namespacing may be broken)")
    if not state.data.get("recommendation"):
        issues.append("No recommendation")
    return issues

def validate_crew(state):
    """Crew should produce assignments and execute them."""
    issues = []
    if not state.data.get("assignments"):
        issues.append("No assignments created")
    return issues

def validate_tree_of_thought(state):
    """Tree of Thought should explore candidates and produce an answer."""
    issues = []
    if not state.data.get("answer") and not state.data.get("best_thought"):
        issues.append("No answer or best_thought produced")
    return issues

def validate_plan_and_solve(state):
    """Plan and Solve should decompose, solve, and synthesize."""
    issues = []
    if not state.data.get("sub_problems") and not state.data.get("plan"):
        issues.append("No sub_problems or plan produced")
    if (not state.data.get("final_answer") and not state.data.get("synthesis")
            and not state.data.get("solved_sub_solutions")):
        issues.append("No final_answer, synthesis, or solved_sub_solutions produced")
    return issues

def validate_self_refine(state):
    """Self Refine should produce output through generate-critique-refine loop."""
    issues = []
    if (not state.data.get("output") and not state.data.get("refined_output")
            and not state.data.get("draft") and not state.data.get("final_output")
            and not state.data.get("output_text") and not state.data.get("current_output")):
        issues.append("No output produced")
    return issues

def validate_lats(state):
    """LATS should run MCTS iterations and produce an answer."""
    issues = []
    if state.data.get("iteration", 0) < 1:
        issues.append("No MCTS iterations completed")
    if not state.data.get("final_solution") and not state.data.get("best_trajectory"):
        issues.append("No solution or best_trajectory produced")
    return issues

def validate_voyager(state):
    """Voyager should propose objectives and attempt skill generation."""
    issues = []
    if not state.data.get("objective"):
        issues.append("No objective proposed")
    return issues

def validate_multi_agent_codegen(state):
    """Multi-Agent Codegen should produce a final code package."""
    issues = []
    if not state.data.get("final_package") and not state.data.get("code_artifact_output"):
        issues.append("No code artifacts produced")
    return issues

def validate_map_reduce(state):
    """MapReduce should chunk, map, and reduce to a final output."""
    issues = []
    if not state.data.get("chunks"):
        issues.append("No chunks created")
    if not state.data.get("final_output") and not state.data.get("reduce_output_schema"):
        issues.append("No final output or reduce result produced")
    return issues

def validate_socratic_tutor(state):
    """Socratic tutor should plan curriculum and evaluate responses."""
    issues = []
    if not state.data.get("lesson_plan_output") and not state.data.get("concepts"):
        issues.append("No lesson plan created")
    if not state.data.get("transcript"):
        issues.append("No transcript recorded")
    return issues

def validate_reflexion(state):
    """Reflexion should attempt trials and produce a final answer."""
    issues = []
    if state.data.get("trial", 0) < 1:
        issues.append("No trials attempted")
    if not state.data.get("final_answer") and not state.data.get("answer"):
        issues.append("No final answer produced")
    return issues

def validate_mixture_of_agents(state):
    """Mixture of Agents should aggregate proposer responses."""
    issues = []
    if not state.data.get("final_response") and not state.data.get("synthesized_response"):
        issues.append("No final or synthesized response produced")
    return issues

def validate_meta_prompting(state):
    """Meta-Prompting should decompose task and produce a final answer."""
    issues = []
    if not state.data.get("final_answer") and not state.data.get("integrated_answer"):
        issues.append("No final or integrated answer produced")
    return issues

def validate_customer_support_swarm(state):
    """Customer Support Swarm should classify and respond to request."""
    issues = []
    if not state.data.get("intent"):
        issues.append("No intent classification produced")
    if not state.data.get("answer") and not state.data.get("response"):
        issues.append("No response produced")
    return issues

def validate_software_team(state):
    """Software Team should produce code through pub/sub pipeline."""
    issues = []
    if not state.data.get("code_module") and not state.data.get("final_output"):
        issues.append("No code output produced")
    return issues

def validate_self_improver(state):
    """Self Improver should analyze a spec and produce improvements or confirm no changes needed."""
    issues = []
    if not state.data.get("final_spec_yaml"):
        issues.append("No final spec YAML produced")
    if state.data.get("total_improvements") is None:
        issues.append("No improvement count")
    return issues

def validate_kg_rag(state):
    """KG RAG should produce an answer grounded in KG data."""
    issues = []
    answer = state.data.get("answer", "")
    if not answer:
        issues.append("No answer produced")
    else:
        # Check answer mentions KG entity (the fictional founder)
        answer_lower = answer.lower()
        if "elara" not in answer_lower and "voss" not in answer_lower:
            issues.append(f"Answer doesn't reference KG data (expected 'Dr. Elara Voss'): {answer[:100]}")
    # Check that results came from the KG (not empty)
    results = state.data.get("results", [])
    if results and all("No results" in str(r) for r in results):
        issues.append("KG query returned no results — KG was not used")
    return issues

def validate_pddl_planner(state):
    """PDDL planner should produce an explanation or handle unsolvable."""
    issues = []
    if not state.data.get("explanation") and not state.data.get("answer"):
        issues.append("No explanation or answer produced")
    return issues

def validate_rap(state):
    """RAP should produce an answer after search iterations."""
    issues = []
    if not state.data.get("answer"):
        issues.append("No answer produced")
    return issues

def validate_alpha_geometry(state):
    """AlphaGeometry should produce a proof or reach iteration limit."""
    issues = []
    if not state.data.get("answer"):
        issues.append("No answer produced")
    if state.data.get("proved") is None:
        issues.append("No proved flag set")
    return issues

VALIDATORS = {
    "react": validate_react,
    "debate": validate_debate,
    "babyagi": validate_babyagi,
    "babyagi_autogen": validate_babyagi_autogen,
    "autogpt": validate_autogpt,
    "rag": validate_rag,
    "code_reviewer": validate_code_reviewer,
    "crew": validate_crew,
    "tree_of_thought": validate_tree_of_thought,
    "plan_and_solve": validate_plan_and_solve,
    "self_refine": validate_self_refine,
    "lats": validate_lats,
    "voyager": validate_voyager,
    "multi_agent_codegen": validate_multi_agent_codegen,
    "map_reduce": validate_map_reduce,
    "socratic_tutor": validate_socratic_tutor,
    "reflexion": validate_reflexion,
    "mixture_of_agents": validate_mixture_of_agents,
    "meta_prompting": validate_meta_prompting,
    "customer_support_swarm": validate_customer_support_swarm,
    "software_team": validate_software_team,
    "self_improver": validate_self_improver,
    "kg_rag": validate_kg_rag,
    "pddl_planner": validate_pddl_planner,
    "rap": validate_rap,
    "alpha_geometry": validate_alpha_geometry,
}

# ── Timeout context manager ──

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ── Monkey-patch input() to avoid blocking ──

def fake_input(prompt=""):
    """Auto-respond to checkpoint prompts."""
    print(f"  [AUTO-INPUT] {prompt} → 'skip'")
    return "skip"

# ── Run a single agent test ──

def _not_runnable_reason(agent_name):
    """Unimplemented processes on the agent's path, or "" if it can run."""
    import yaml
    from agent_ontology.instantiate import unimplemented_reachable_processes
    specs_dir = os.path.join(PROJECT_ROOT, "agent_ontology", "specs")
    for stem in (agent_name, agent_name.replace("_", "-")):
        path = os.path.join(specs_dir, f"{stem}.yaml")
        if os.path.exists(path):
            with open(path) as f:
                blocked = unimplemented_reachable_processes(yaml.safe_load(f))
            return ", ".join(f"{pid} ({ptype})" for pid, ptype in blocked)
    return ""


def run_agent_test(agent_name, timeout_sec=120, dry_run=False):
    """Run a single agent test. Returns a result dict."""
    module_name = f"agents.{agent_name}_agent"
    result = {
        "agent": agent_name,
        "status": "unknown",
        "duration_ms": 0,
        "llm_calls": 0,
        "iterations": 0,
        "issues": [],
        "error": None,
    }

    # An agent whose path reaches a process the generator does not implement
    # raises NotImplementedError there; report that instead of running it.
    blocked = _not_runnable_reason(agent_name)
    if blocked:
        result["status"] = "NOT_RUNNABLE"
        result["error"] = blocked
        return result

    # Import check
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        result["status"] = "IMPORT_ERROR"
        result["error"] = str(e)
        return result

    if dry_run:
        result["status"] = "DRY_RUN_OK"
        return result

    # Get test inputs
    inputs = TEST_INPUTS.get(agent_name, {})
    if not inputs:
        result["status"] = "NO_TEST_DATA"
        result["issues"].append("No test inputs defined")
        return result

    # Patch input() and capture output
    import builtins
    original_input = builtins.input
    builtins.input = fake_input

    # Reset the module's TRACE
    if hasattr(mod, 'TRACE'):
        mod.TRACE.clear()

    t0 = time.time()
    try:
        with timeout(timeout_sec):
            state = mod.run(inputs.copy())

        result["duration_ms"] = int((time.time() - t0) * 1000)
        result["iterations"] = state.iteration if hasattr(state, 'iteration') else 0
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, 'TRACE') else 0

        # Validate
        validator = VALIDATORS.get(agent_name)
        if validator:
            result["issues"] = validator(state)

        if result["issues"]:
            result["status"] = "ISSUES"
        else:
            result["status"] = "PASS"

        # Save trace
        trace_path = os.path.join(PROJECT_ROOT, "traces", f"{agent_name}_trace.json")
        os.makedirs(os.path.join(PROJECT_ROOT, "traces"), exist_ok=True)
        if hasattr(mod, 'TRACE'):
            with open(trace_path, "w") as f:
                json.dump(mod.TRACE, f, indent=2)

    except TimeoutError as e:
        result["status"] = "TIMEOUT"
        result["error"] = str(e)
        result["duration_ms"] = int((time.time() - t0) * 1000)
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, 'TRACE') else 0
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["duration_ms"] = int((time.time() - t0) * 1000)
    finally:
        builtins.input = original_input

    return result


# ── Main ──

AGENT_NAMES = [
    "react", "debate", "babyagi", "babyagi_autogen",
    "autogpt", "rag", "code_reviewer", "crew",
    "tree_of_thought", "plan_and_solve", "self_refine",
    "lats", "voyager", "multi_agent_codegen",
    "map_reduce", "socratic_tutor",
    "reflexion", "mixture_of_agents", "meta_prompting",
    "customer_support_swarm", "software_team",
    "self_improver",
    "kg_rag", "pddl_planner", "rap", "alpha_geometry",
]

DEFAULT_COMPARE_MODELS = [
    "gemini-3-flash-preview",
    "gpt-4o-mini",
    "gpt-4o",
]


def run_model_comparison(models, agents, timeout_sec=120, as_json=False):
    """Run each agent with multiple models and output a comparison table."""
    all_results = {}  # model -> [result, ...]

    for model in models:
        os.environ["AGENT_ONTOLOGY_MODEL"] = model
        if not as_json:
            print(f"\n{'='*60}")
            print(f"  Model: {model}")
            print(f"{'='*60}")

        # Force reimport of all agent modules so they pick up new model
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("agents."):
                del sys.modules[mod_name]

        model_results = []
        for name in agents:
            if not as_json:
                print(f"  [TEST] {name}...", end=" ", flush=True)
            result = run_agent_test(name, timeout_sec=timeout_sec)
            result["model"] = model
            model_results.append(result)
            if not as_json:
                status = result["status"]
                calls = result["llm_calls"]
                dur = result["duration_ms"]
                icon = {"PASS": "✓", "ISSUES": "⚠", "ERROR": "✗", "TIMEOUT": "⏱"}.get(status, "?")
                print(f"{icon} {status} ({dur}ms, {calls} calls)")

        all_results[model] = model_results

    # Clean up env
    if "AGENT_ONTOLOGY_MODEL" in os.environ:
        del os.environ["AGENT_ONTOLOGY_MODEL"]

    if as_json:
        print(json.dumps(all_results, indent=2))
        return

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  Model Comparison Summary")
    print(f"{'='*70}")

    # Header
    col_w = max(len(m) for m in models) + 2
    header = f"  {'Agent':<20s}"
    for m in models:
        header += f"  {m:<{col_w}s}"
    print(header)
    print("  " + "-" * (20 + (col_w + 2) * len(models)))

    # Per-agent rows
    for i, name in enumerate(agents):
        row = f"  {name:<20s}"
        for model in models:
            r = all_results[model][i]
            if r["status"] == "PASS":
                cell = f"✓ {r['llm_calls']}c {r['duration_ms']/1000:.1f}s"
            else:
                cell = f"✗ {r['status']}"
            row += f"  {cell:<{col_w}s}"
        print(row)

    # Totals row
    print("  " + "-" * (20 + (col_w + 2) * len(models)))
    row = f"  {'TOTAL':<20s}"
    for model in models:
        mrs = all_results[model]
        passed = sum(1 for r in mrs if r["status"] == "PASS")
        total_calls = sum(r["llm_calls"] for r in mrs)
        total_s = sum(r["duration_ms"] for r in mrs) / 1000
        row += f"  {f'{passed}/{len(agents)} {total_calls}c {total_s:.0f}s':<{col_w}s}"
    print(row)
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Agent Test Harness — run agents with canned inputs and validate outputs"
    )
    parser.add_argument("--agent", "-a", choices=AGENT_NAMES, help="Run specific agent only")
    parser.add_argument("--dry-run", action="store_true", help="Check imports only, no API calls")
    parser.add_argument("--timeout", "-t", type=int, default=120, help="Per-agent timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Override model for all agents (sets AGENT_ONTOLOGY_MODEL env var)")
    parser.add_argument("--compare-models", nargs="*", metavar="MODEL",
                        help="Compare across models. No args = default set, or list models explicitly")
    args = parser.parse_args()

    # Model comparison mode
    if args.compare_models is not None:
        models = args.compare_models if args.compare_models else DEFAULT_COMPARE_MODELS
        run_model_comparison(
            models=models,
            agents=([args.agent] if args.agent else AGENT_NAMES),
            timeout_sec=args.timeout,
            as_json=args.json,
        )
        return

    # Single-model override
    if args.model:
        os.environ["AGENT_ONTOLOGY_MODEL"] = args.model

    agents_to_test = [args.agent] if args.agent else AGENT_NAMES
    results = []

    model_label = os.environ.get("AGENT_ONTOLOGY_MODEL", "spec default")
    if not args.json:
        print(f"\n{'='*60}")
        print(f"  Agent Test Harness")
        print(f"  Testing {len(agents_to_test)} agent(s), timeout={args.timeout}s")
        print(f"  Model: {model_label}")
        print(f"{'='*60}\n")

    for name in agents_to_test:
        if not args.json:
            print(f"[TEST] {name}...", end=" ", flush=True)

        result = run_agent_test(name, timeout_sec=args.timeout, dry_run=args.dry_run)
        results.append(result)

        if not args.json:
            status = result["status"]
            duration = result["duration_ms"]
            calls = result["llm_calls"]
            icon = {"PASS": "✓", "ISSUES": "⚠", "ERROR": "✗", "TIMEOUT": "⏱",
                    "DRY_RUN_OK": "○", "IMPORT_ERROR": "✗", "NO_TEST_DATA": "○"}.get(status, "?")
            print(f"{icon} {status} ({duration}ms, {calls} LLM calls)")
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"    - {issue}")
            if result["error"]:
                print(f"    ERROR: {result['error']}")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Summary
        passed = sum(1 for r in results if r["status"] == "PASS")
        total = len(results)
        total_calls = sum(r["llm_calls"] for r in results)
        total_ms = sum(r["duration_ms"] for r in results)
        print(f"\n{'='*60}")
        print(f"  Results: {passed}/{total} passed")
        print(f"  Total: {total_calls} LLM calls, {total_ms/1000:.1f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
