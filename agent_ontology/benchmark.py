#!/usr/bin/env python3
"""
Benchmark Suite for Agent Ontology

Runs multiple agents against standardized tasks and compares results.
Each task specifies which agents can run it (based on compatible input schemas).
Scoring: binary pass/fail from validators + LLM calls (fewer=better) + duration
(faster=better) + estimated token cost.

Usage:
    python3 benchmark.py                           # Run all applicable benchmarks
    python3 benchmark.py --agent react             # Run benchmarks for one agent
    python3 benchmark.py --task factual_qa         # Run one task across all compatible agents
    python3 benchmark.py --json                    # Output as JSON
    python3 benchmark.py --dry-run                 # Show what would run without running
    python3 benchmark.py --timeout 120             # Set per-run timeout (seconds)
    python3 benchmark.py --tasks-file path.json    # Use custom tasks file
"""

import argparse
import importlib
import json
import os
import re
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from io import StringIO

# ── Constants ──

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASKS_FILE = os.path.join(PROJECT_ROOT, "benchmarks", "tasks.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")

KNOWN_AGENTS = [
    "react", "debate", "babyagi", "babyagi_autogen",
    "autogpt", "rag", "code_reviewer", "crew",
    "plan_and_solve", "self_refine", "tree_of_thought",
    "lats", "reflexion", "minimal_solver",
    "kb_react", "mutator",
]

DATASETS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "datasets")

# Approximate cost per token (USD) — rough estimates for Gemini Flash
EST_COST_PER_INPUT_TOKEN = 0.000000075   # $0.075 / 1M tokens
EST_COST_PER_OUTPUT_TOKEN = 0.0000003    # $0.30 / 1M tokens


# ── Timeout ──

class BenchmarkTimeout(Exception):
    pass


@contextmanager
def timeout_ctx(seconds):
    """Signal-based timeout context manager (Unix only)."""
    def handler(signum, frame):
        raise BenchmarkTimeout(f"Timed out after {seconds}s")
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── Monkey-patch input() to avoid blocking ──

def fake_input(prompt=""):
    """Auto-respond to checkpoint prompts during benchmarking."""
    return "skip"


# ── Task loading ──

def load_tasks(tasks_file):
    """Load benchmark task definitions from JSON file."""
    with open(tasks_file, "r") as f:
        data = json.load(f)
    return data.get("tasks", [])


# ── Validators ──

def validate_result(task, state):
    """
    Run the task's validator against the agent's final state.
    Returns (passed: bool, reason: str).
    """
    validator = task.get("validator", {})
    vtype = validator.get("type", "")
    state_data = state.data if hasattr(state, "data") else {}

    # Collect all text from state for text-based validators
    all_text = _extract_all_text(state_data)

    if vtype == "contains":
        value = validator["value"].lower()
        target_field = validator.get("target_field")
        if target_field:
            field_text = str(state_data.get(target_field, "")).lower()
            if value in field_text:
                return True, f"Found '{validator['value']}' in field '{target_field}'"
            return False, f"'{validator['value']}' not found in field '{target_field}'"
        else:
            if value in all_text.lower():
                return True, f"Found '{validator['value']}' in output"
            return False, f"'{validator['value']}' not found in any output field"

    elif vtype == "contains_any":
        values = [v.lower() for v in validator["values"]]
        target_field = validator.get("target_field")
        search_text = (
            str(state_data.get(target_field, "")).lower()
            if target_field
            else all_text.lower()
        )
        found = [v for v in values if v in search_text]
        if found:
            return True, f"Found terms: {found}"
        return False, f"None of {validator['values']} found in output"

    elif vtype == "min_length":
        min_chars = validator.get("min_chars", 100)
        if len(all_text) >= min_chars:
            return True, f"Output length {len(all_text)} >= {min_chars}"
        return False, f"Output length {len(all_text)} < {min_chars}"

    elif vtype == "min_steps":
        min_count = validator.get("min_count", 3)
        step_count = _count_steps(state_data)
        if step_count >= min_count:
            return True, f"Found {step_count} steps (>= {min_count})"
        return False, f"Found only {step_count} steps (need >= {min_count})"

    elif vtype == "debate_structure":
        require_winner = validator.get("require_winner", True)
        min_history = validator.get("min_history", 4)
        issues = []
        if require_winner and not state_data.get("winner"):
            issues.append("No winner determined")
        history = state_data.get("debate_history", [])
        if len(history) < min_history:
            issues.append(f"Only {len(history)} debate turns (need >= {min_history})")
        if issues:
            return False, "; ".join(issues)
        return True, f"Winner: {state_data.get('winner')}, {len(history)} turns"

    else:
        return False, f"Unknown validator type: {vtype}"


def _extract_all_text(data, depth=0):
    """Recursively extract all string values from a dict/list into one blob."""
    if depth > 10:
        return ""
    parts = []
    if isinstance(data, str):
        parts.append(data)
    elif isinstance(data, dict):
        for v in data.values():
            parts.append(_extract_all_text(v, depth + 1))
    elif isinstance(data, (list, tuple)):
        for item in data:
            parts.append(_extract_all_text(item, depth + 1))
    return " ".join(parts)


def _count_steps(state_data):
    """
    Count distinct steps/tasks in agent output.
    Looks for task lists, numbered steps, or result entries.
    """
    count = 0

    # Check for tasks list (babyagi style)
    tasks = state_data.get("tasks", [])
    if isinstance(tasks, list) and tasks:
        return max(len(tasks), count)

    # Check for completed tasks count
    completed = state_data.get("completed_count", 0)
    if completed:
        count = max(count, completed)

    # Check for results list (autogpt style)
    results = state_data.get("results", [])
    if isinstance(results, list):
        count = max(count, len(results))

    # Check for result field that may contain numbered steps
    result_text = state_data.get("result", "")
    if isinstance(result_text, str) and result_text:
        # Count numbered items like "1.", "2.", "3." or "Step 1", etc.
        numbered = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+|task\s+\d+)', result_text, re.IGNORECASE)
        count = max(count, len(numbered))

    # Check all text for step patterns as fallback
    all_text = _extract_all_text(state_data)
    if all_text:
        numbered = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+|task\s+\d+)', all_text, re.IGNORECASE)
        count = max(count, len(numbered))

    return count


# ── Token estimation ──

def estimate_tokens_from_trace(trace):
    """
    Estimate input/output tokens from TRACE entries.
    Uses the ~4 chars per token heuristic (same as the agents use).
    """
    est_input = 0
    est_output = 0
    for entry in trace:
        est_input += len(entry.get("user_message", "")) // 4
        est_input += len(entry.get("system_prompt", "")) // 4
        est_output += len(entry.get("response", "")) // 4
    return est_input, est_output


def estimate_cost(input_tokens, output_tokens):
    """Estimate USD cost from token counts."""
    return (input_tokens * EST_COST_PER_INPUT_TOKEN +
            output_tokens * EST_COST_PER_OUTPUT_TOKEN)


# ── Single benchmark run ──

def run_single_benchmark(agent_name, task, timeout_sec=120, dry_run=False):
    """
    Run one agent on one task.
    Returns a result dict with status, metrics, and validation info.
    """
    module_name = f"agents.{agent_name}_agent"

    result = {
        "agent": agent_name,
        "task": task["id"],
        "task_name": task["name"],
        "status": "unknown",
        "passed": False,
        "validation_reason": "",
        "llm_calls": 0,
        "duration_s": 0.0,
        "est_input_tokens": 0,
        "est_output_tokens": 0,
        "est_total_tokens": 0,
        "est_cost_usd": 0.0,
        "iterations": 0,
        "error": None,
    }

    # ── Import ──
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        result["status"] = "IMPORT_ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    if dry_run:
        result["status"] = "DRY_RUN"
        return result

    # ── Prepare inputs ──
    inputs = task["input"].copy()

    # For autogpt, map "objective" -> "goal" if the task uses objective
    if agent_name == "autogpt" and "objective" in inputs and "goal" not in inputs:
        inputs["goal"] = inputs.pop("objective")

    # ── Patch input() ──
    import builtins
    original_input = builtins.input
    builtins.input = fake_input

    # Reset the module's TRACE
    if hasattr(mod, "TRACE"):
        mod.TRACE.clear()

    # ── Run ──
    t0 = time.time()
    try:
        with timeout_ctx(timeout_sec):
            state = mod.run(inputs)

        elapsed = time.time() - t0
        result["duration_s"] = round(elapsed, 2)
        result["iterations"] = state.iteration if hasattr(state, "iteration") else 0
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, "TRACE") else 0

        # Token estimation
        if hasattr(mod, "TRACE"):
            inp_tok, out_tok = estimate_tokens_from_trace(mod.TRACE)
            result["est_input_tokens"] = inp_tok
            result["est_output_tokens"] = out_tok
            result["est_total_tokens"] = inp_tok + out_tok
            result["est_cost_usd"] = round(estimate_cost(inp_tok, out_tok), 6)

        # Validate
        passed, reason = validate_result(task, state)
        result["passed"] = passed
        result["validation_reason"] = reason
        result["status"] = "PASS" if passed else "FAIL"

    except BenchmarkTimeout as e:
        elapsed = time.time() - t0
        result["status"] = "TIMEOUT"
        result["error"] = str(e)
        result["duration_s"] = round(elapsed, 2)
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, "TRACE") else 0
        if hasattr(mod, "TRACE"):
            inp_tok, out_tok = estimate_tokens_from_trace(mod.TRACE)
            result["est_input_tokens"] = inp_tok
            result["est_output_tokens"] = out_tok
            result["est_total_tokens"] = inp_tok + out_tok
    except Exception as e:
        elapsed = time.time() - t0
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["duration_s"] = round(elapsed, 2)
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, "TRACE") else 0
    finally:
        builtins.input = original_input

    return result


# ── Plan what to run ──

def build_run_plan(tasks, agent_filter=None, task_filter=None):
    """
    Build a list of (task, agent_name) pairs to run.
    Respects agent and task filters.
    """
    plan = []
    for task in tasks:
        if task_filter and task["id"] != task_filter:
            continue
        for agent_name in task["agents"]:
            if agent_name not in KNOWN_AGENTS:
                continue
            if agent_filter and agent_name != agent_filter:
                continue
            plan.append((task, agent_name))
    return plan


# ── Output formatting ──

def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s)}s"


def format_tokens(count):
    """Format token count with ~ prefix."""
    if count >= 1000:
        return f"~{count//1000}k"
    return f"~{count}"


def print_table(all_results):
    """
    Print a formatted comparison table grouped by task.

    Task: factual_qa
      react:     PASS  3 calls  7.1s  ~500 tokens
      rag:       PASS  2 calls  5.2s  ~300 tokens
    """
    # Group results by task
    tasks_seen = []
    by_task = {}
    for r in all_results:
        tid = r["task"]
        if tid not in by_task:
            by_task[tid] = []
            tasks_seen.append(tid)
        by_task[tid].append(r)

    # Find max agent name length for alignment
    max_agent_len = max((len(r["agent"]) for r in all_results), default=8)

    for tid in tasks_seen:
        results = by_task[tid]
        task_name = results[0].get("task_name", tid)
        print(f"\nTask: {tid} ({task_name})")

        for r in results:
            agent = r["agent"]
            status = r["status"]
            calls = r["llm_calls"]
            dur = format_duration(r["duration_s"])
            tokens = format_tokens(r["est_total_tokens"])

            # Status indicator
            if status == "PASS":
                indicator = "PASS"
            elif status == "FAIL":
                indicator = "FAIL"
            elif status == "DRY_RUN":
                indicator = "DRY_RUN"
            elif status == "TIMEOUT":
                indicator = "TIMEOUT"
            elif status == "ERROR":
                indicator = "ERROR"
            elif status == "IMPORT_ERROR":
                indicator = "IMPORT_ERR"
            else:
                indicator = status

            padding = " " * (max_agent_len - len(agent))
            if status in ("DRY_RUN",):
                print(f"  {agent}:{padding}  {indicator}")
            else:
                print(f"  {agent}:{padding}  {indicator:<8}  {calls} call{'s' if calls != 1 else ' '}  {dur:<8}  {tokens} tokens")

            # Show error or validation reason on failure
            if status == "FAIL":
                print(f"    reason: {r['validation_reason']}")
            elif r.get("error"):
                print(f"    error: {r['error']}")


def print_summary(all_results):
    """Print a summary line at the bottom."""
    total = len(all_results)
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    errors = sum(1 for r in all_results if r["status"] in ("ERROR", "TIMEOUT", "IMPORT_ERROR"))
    dry = sum(1 for r in all_results if r["status"] == "DRY_RUN")

    total_calls = sum(r["llm_calls"] for r in all_results)
    total_dur = sum(r["duration_s"] for r in all_results)
    total_tokens = sum(r["est_total_tokens"] for r in all_results)
    total_cost = sum(r["est_cost_usd"] for r in all_results)

    print(f"\n{'='*64}")
    if dry:
        print(f"  Dry run: {dry} benchmark(s) would execute")
    else:
        parts = [f"{passed} passed"]
        if failed:
            parts.append(f"{failed} failed")
        if errors:
            parts.append(f"{errors} error(s)")
        print(f"  Results: {', '.join(parts)} / {total} total")
        print(f"  Totals:  {total_calls} LLM calls, {format_duration(total_dur)}, {format_tokens(total_tokens)} tokens")
        if total_cost > 0:
            print(f"  Est cost: ${total_cost:.4f}")
    print(f"{'='*64}")


def save_results_json(all_results, output_path=None):
    """Save results to a JSON file in benchmarks/results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": all_results,
            "summary": {
                "total": len(all_results),
                "passed": sum(1 for r in all_results if r["status"] == "PASS"),
                "failed": sum(1 for r in all_results if r["status"] == "FAIL"),
                "errors": sum(1 for r in all_results if r["status"] in ("ERROR", "TIMEOUT", "IMPORT_ERROR")),
                "total_llm_calls": sum(r["llm_calls"] for r in all_results),
                "total_duration_s": round(sum(r["duration_s"] for r in all_results), 2),
                "total_tokens": sum(r["est_total_tokens"] for r in all_results),
                "total_est_cost_usd": round(sum(r["est_cost_usd"] for r in all_results), 6),
            },
        }, f, indent=2)
    return output_path


# ── Suite mode: dataset-driven evaluation ──

def load_dataset(name):
    """Load a benchmark dataset JSON by name (hotpotqa, gsm8k)."""
    path = os.path.join(DATASETS_DIR, f"{name}.json")
    with open(path, "r") as f:
        return json.load(f)


def run_suite_benchmark(agent_name, example, dataset_name, dataset_meta,
                        timeout_sec=120, dry_run=False):
    """
    Run one agent on one dataset example.
    Returns a result dict with scoring info.
    """
    from benchmarks.compatibility import format_input
    from benchmarks.scoring import (
        extract_answer, score_hotpotqa, score_gsm8k,
        score_arc, score_humaneval, extract_code, score_kb_tool,
        score_meta_improve,
    )

    module_name = f"agents.{agent_name}_agent"

    result = {
        "agent": agent_name,
        "dataset": dataset_name,
        "example_id": example["id"],
        "question": example["question"],
        "expected": example["answer"],
        "predicted": "",
        "scores": {},
        "status": "unknown",
        "llm_calls": 0,
        "duration_s": 0.0,
        "est_input_tokens": 0,
        "est_output_tokens": 0,
        "est_total_tokens": 0,
        "est_cost_usd": 0.0,
        "iterations": 0,
        "error": None,
    }

    # Import
    try:
        mod = importlib.import_module(module_name)
        # Force reload to reset TRACE
        importlib.reload(mod)
    except Exception as e:
        result["status"] = "IMPORT_ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    if dry_run:
        result["status"] = "DRY_RUN"
        return result

    # Monkey-patch tool functions for kb_tool benchmark
    if dataset_name == "kb_tool":
        from benchmarks.kb_tools import patch_agent_tools
        patch_agent_tools(mod)

    # Build input
    inputs = format_input(agent_name, example, dataset_name)

    # Patch input()
    import builtins
    original_input = builtins.input
    builtins.input = fake_input

    # Reset TRACE
    if hasattr(mod, "TRACE"):
        mod.TRACE.clear()

    # Run
    t0 = time.time()
    try:
        with timeout_ctx(timeout_sec):
            state = mod.run(inputs)

        elapsed = time.time() - t0
        result["duration_s"] = round(elapsed, 2)
        result["iterations"] = state.iteration if hasattr(state, "iteration") else 0
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, "TRACE") else 0

        # Token estimation
        if hasattr(mod, "TRACE"):
            inp_tok, out_tok = estimate_tokens_from_trace(mod.TRACE)
            result["est_input_tokens"] = inp_tok
            result["est_output_tokens"] = out_tok
            result["est_total_tokens"] = inp_tok + out_tok
            result["est_cost_usd"] = round(estimate_cost(inp_tok, out_tok), 6)

        # Extract predicted answer
        state_data = state.data if hasattr(state, "data") else {}
        predicted = extract_answer(state_data, inputs=inputs)
        result["predicted"] = predicted

        # Score based on dataset type
        scoring_types = dataset_meta.get("scoring", ["em"])
        expected = example["answer"]

        if dataset_name == "hotpotqa":
            result["scores"] = score_hotpotqa(predicted, str(expected))
        elif dataset_name in ("gsm8k", "gsm8k_hard", "gsm8k_tricky"):
            result["scores"] = score_gsm8k(predicted, expected)
        elif dataset_name == "arc":
            result["scores"] = score_arc(predicted, expected)
        elif dataset_name == "humaneval":
            result["scores"] = score_humaneval(predicted, example)
        elif dataset_name == "kb_tool":
            result["scores"] = score_kb_tool(predicted, str(expected))
        elif dataset_name == "meta_improve":
            result["scores"] = score_meta_improve(predicted, example)
        else:
            result["scores"] = {"em": 1.0 if str(expected).lower() in predicted.lower() else 0.0}

        result["status"] = "DONE"

    except BenchmarkTimeout as e:
        elapsed = time.time() - t0
        result["status"] = "TIMEOUT"
        result["error"] = str(e)
        result["duration_s"] = round(elapsed, 2)
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, "TRACE") else 0
    except Exception as e:
        elapsed = time.time() - t0
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["duration_s"] = round(elapsed, 2)
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, "TRACE") else 0
    finally:
        builtins.input = original_input

    return result


def run_suite(dataset_name, agent_filter=None, n_runs=1, max_examples=None,
              timeout_sec=120, dry_run=False, json_output=False):
    """
    Run a full benchmark suite: all compatible agents on a dataset.
    Returns list of all result dicts.
    """
    from benchmarks.compatibility import get_compatible_agents

    dataset_meta = load_dataset(dataset_name)
    examples = dataset_meta["examples"]
    if max_examples and max_examples < len(examples):
        examples = examples[:max_examples]

    agents = get_compatible_agents(dataset_name)
    if agent_filter:
        if agent_filter not in agents:
            print(f"Warning: {agent_filter} is not in the compatibility list for {dataset_name}", file=sys.stderr)
            agents = [agent_filter]
        else:
            agents = [agent_filter]

    if not agents:
        print(f"No compatible agents for dataset {dataset_name}", file=sys.stderr)
        return []

    if not json_output:
        print(f"\n{'='*64}")
        print(f"  Suite: {dataset_name.upper()} ({len(examples)} examples)")
        print(f"  Agents: {', '.join(agents)}")
        print(f"  Runs per example: {n_runs}")
        if dry_run:
            print(f"  Mode: DRY RUN")
        print(f"{'='*64}")

    all_results = []
    for agent_name in agents:
        for run_idx in range(n_runs):
            for example in examples:
                if not json_output and not dry_run:
                    run_label = f" (run {run_idx+1}/{n_runs})" if n_runs > 1 else ""
                    print(f"  {agent_name} | {example['id']}{run_label}...", end="", flush=True)

                result = run_suite_benchmark(
                    agent_name, example, dataset_name, dataset_meta,
                    timeout_sec=timeout_sec, dry_run=dry_run,
                )
                result["run_index"] = run_idx
                all_results.append(result)

                if not json_output and not dry_run:
                    status = result["status"]
                    if status == "DONE":
                        em = result["scores"].get("em", 0)
                        f1 = result["scores"].get("f1")
                        pass1 = result["scores"].get("pass_at_1")
                        score_str = f"EM={em:.0f}"
                        if f1 is not None:
                            score_str += f" F1={f1:.2f}"
                        if pass1 is not None:
                            score_str = f"pass@1={pass1:.0f}"
                        print(f" {score_str} ({result['llm_calls']} calls, {format_duration(result['duration_s'])})")
                    elif status == "TIMEOUT":
                        print(f" TIMEOUT")
                    elif status == "ERROR":
                        print(f" ERROR: {result['error'][:80]}")
                    elif status == "IMPORT_ERROR":
                        print(f" IMPORT_ERROR: {result['error'][:80]}")

    return all_results


def print_suite_summary(all_results, dataset_name):
    """Print statistical summary for a suite run."""
    from collections import defaultdict
    import math

    # Group by agent
    by_agent = defaultdict(list)
    for r in all_results:
        if r["status"] in ("DONE",):
            by_agent[r["agent"]].append(r)

    if not by_agent:
        print("\n  No completed results to summarize.")
        return

    print(f"\n{'='*64}")
    print(f"  {dataset_name.upper()} Results Summary")
    print(f"{'='*64}")

    # Table header
    has_f1 = any("f1" in r["scores"] for results in by_agent.values() for r in results)
    header = f"  {'Agent':<20} {'EM':>6}"
    if has_f1:
        header += f" {'F1':>8}"
    header += f" {'LLM calls':>10} {'Cost':>8} {'Duration':>10}"
    print(header)
    print(f"  {'-'*18}  {'-'*6}", end="")
    if has_f1:
        print(f" {'-'*8}", end="")
    print(f" {'-'*10} {'-'*8} {'-'*10}")

    agent_scores = []
    for agent_name in sorted(by_agent.keys()):
        results = by_agent[agent_name]
        n = len(results)

        # EM scores
        em_scores = [r["scores"].get("em", 0) for r in results]
        em_mean = sum(em_scores) / n if n else 0
        em_std = math.sqrt(sum((s - em_mean)**2 for s in em_scores) / n) if n > 1 else 0

        # F1 scores
        f1_scores = [r["scores"].get("f1", 0) for r in results if "f1" in r["scores"]]
        f1_mean = sum(f1_scores) / len(f1_scores) if f1_scores else None
        f1_std = math.sqrt(sum((s - f1_mean)**2 for s in f1_scores) / len(f1_scores)) if f1_scores and len(f1_scores) > 1 else 0

        # Metrics
        avg_calls = sum(r["llm_calls"] for r in results) / n if n else 0
        avg_cost = sum(r["est_cost_usd"] for r in results) / n if n else 0
        avg_dur = sum(r["duration_s"] for r in results) / n if n else 0

        line = f"  {agent_name:<20} {em_mean:>5.1%}"
        if em_std > 0:
            line = f"  {agent_name:<20} {em_mean:>5.1%}"
        if has_f1:
            if f1_mean is not None:
                line += f" {f1_mean:>7.1%}"
            else:
                line += f" {'n/a':>8}"
        line += f" {avg_calls:>10.1f} ${avg_cost:>6.4f} {format_duration(avg_dur):>10}"
        print(line)

        agent_scores.append((agent_name, em_mean))

    # Total cost
    total_cost = sum(r["est_cost_usd"] for r in all_results if r["status"] == "DONE")
    total_calls = sum(r["llm_calls"] for r in all_results if r["status"] == "DONE")
    errors = sum(1 for r in all_results if r["status"] in ("ERROR", "TIMEOUT", "IMPORT_ERROR"))

    print(f"\n  Total: {total_calls} LLM calls, ${total_cost:.4f} est. cost")
    if errors:
        print(f"  Errors/timeouts: {errors}")
    print(f"{'='*64}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Suite -- run agents against standardized tasks and compare results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 benchmark.py                        Run all applicable benchmarks
  python3 benchmark.py --agent react          Run benchmarks for one agent
  python3 benchmark.py --task factual_qa      Run one task across all compatible agents
  python3 benchmark.py --json                 Output results as JSON to stdout
  python3 benchmark.py --dry-run              Show what would run without executing
        """,
    )
    parser.add_argument(
        "--agent", "-a",
        choices=KNOWN_AGENTS,
        help="Run benchmarks for a specific agent only",
    )
    parser.add_argument(
        "--task", "-t",
        help="Run a specific task across all compatible agents",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without actually executing agents",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-benchmark timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--tasks-file",
        default=DEFAULT_TASKS_FILE,
        help=f"Path to tasks JSON file (default: {DEFAULT_TASKS_FILE})",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to benchmarks/results/ as JSON",
    )

    # Suite mode args
    parser.add_argument(
        "--suite",
        choices=["hotpotqa", "gsm8k", "gsm8k_hard", "gsm8k_tricky", "arc", "humaneval", "multidoc", "kb_tool", "meta_improve", "all"],
        help="Run dataset-driven benchmark suite",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of runs per (agent, example) pair for statistical reporting (default: 1)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Limit number of examples per dataset (default: all)",
    )
    args = parser.parse_args()

    # Ensure we can import agents from project root
    _repo_root = os.path.dirname(PROJECT_ROOT)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # ── Suite mode ──
    if args.suite:
        datasets = ["hotpotqa", "gsm8k", "gsm8k_hard", "gsm8k_tricky", "arc", "humaneval", "multidoc", "kb_tool", "meta_improve"] if args.suite == "all" else [args.suite]
        all_suite_results = []

        for ds_name in datasets:
            results = run_suite(
                ds_name,
                agent_filter=args.agent,
                n_runs=args.n_runs,
                max_examples=args.examples,
                timeout_sec=args.timeout,
                dry_run=args.dry_run,
                json_output=args.json,
            )
            all_suite_results.extend(results)

            if not args.json and not args.dry_run:
                print_suite_summary(results, ds_name)

        if args.json:
            print(json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode": "suite",
                "datasets": datasets,
                "results": all_suite_results,
            }, indent=2))

        if args.save and not args.dry_run:
            path = save_results_json(all_suite_results)
            if not args.json:
                print(f"\n  Results saved to {path}")

        sys.exit(0)

    # Load tasks
    try:
        tasks = load_tasks(args.tasks_file)
    except FileNotFoundError:
        print(f"Error: Tasks file not found: {args.tasks_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in tasks file: {e}", file=sys.stderr)
        sys.exit(1)

    if not tasks:
        print("Error: No tasks defined in tasks file", file=sys.stderr)
        sys.exit(1)

    # Build run plan
    plan = build_run_plan(tasks, agent_filter=args.agent, task_filter=args.task)

    if not plan:
        filters = []
        if args.agent:
            filters.append(f"agent={args.agent}")
        if args.task:
            filters.append(f"task={args.task}")
        filter_desc = ", ".join(filters) if filters else "no filters"
        print(f"No benchmarks to run ({filter_desc}). Check task/agent compatibility.", file=sys.stderr)
        sys.exit(1)

    # Header
    if not args.json:
        task_count = len(set(t["id"] for t, _ in plan))
        agent_count = len(set(a for _, a in plan))
        print(f"\n{'='*64}")
        print(f"  Agent Ontology Benchmark Suite")
        print(f"  {len(plan)} benchmark(s): {task_count} task(s) x {agent_count} agent(s)")
        print(f"  Timeout: {args.timeout}s per run")
        if args.dry_run:
            print(f"  Mode: DRY RUN (no agents will execute)")
        print(f"{'='*64}")

    # Run benchmarks
    all_results = []
    for task, agent_name in plan:
        if not args.json and not args.dry_run:
            print(f"\n  Running {agent_name} on {task['id']}...", end="", flush=True)

        result = run_single_benchmark(
            agent_name,
            task,
            timeout_sec=args.timeout,
            dry_run=args.dry_run,
        )
        all_results.append(result)

        if not args.json and not args.dry_run:
            status = result["status"]
            if status == "PASS":
                print(f" PASS ({result['llm_calls']} calls, {format_duration(result['duration_s'])})")
            elif status == "FAIL":
                print(f" FAIL ({result['validation_reason']})")
            elif status == "TIMEOUT":
                print(f" TIMEOUT")
            elif status == "ERROR":
                print(f" ERROR: {result['error']}")
            elif status == "IMPORT_ERROR":
                print(f" IMPORT_ERROR: {result['error']}")

    # Output
    if args.json:
        print(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": all_results,
        }, indent=2))
    else:
        print_table(all_results)
        print_summary(all_results)

    # Save if requested
    if args.save and not args.dry_run:
        path = save_results_json(all_results)
        if not args.json:
            print(f"\n  Results saved to {path}")

    # Exit code: 0 if all passed (or dry run), 1 otherwise
    if args.dry_run:
        sys.exit(0)
    all_passed = all(r["status"] == "PASS" for r in all_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
