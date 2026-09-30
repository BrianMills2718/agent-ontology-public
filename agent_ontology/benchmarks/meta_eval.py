#!/usr/bin/env python3
"""
Nested evaluation logic for the meta_improve benchmark.

Provides:
  - evaluate_improvement: parse mutator output → validate → benchmark → compare to baseline
  - score_meta_improve: scoring function called from compute_fitness_benchmark
"""

import os
import re
import sys
import yaml


def _extract_yaml_from_text(text):
    """Extract YAML content from mutator output text.

    Tries in order:
      1. Content between ```yaml ... ``` fences
      2. Content between ``` ... ``` fences
      3. Raw text if it looks like YAML (starts with name: or entities:)
      4. The entire text as-is
    """
    text = str(text).strip()

    # Pattern 1: ```yaml block
    m = re.search(r'```yaml\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Pattern 2: ``` block
    m = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Pattern 3: looks like YAML
    if text.lstrip().startswith(('name:', 'entities:', '#')):
        return text

    return text


def _load_target_spec(target_spec_name):
    """Load the target spec YAML file by name."""
    spec_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "specs")
    spec_path = os.path.join(spec_dir, f"{target_spec_name}.yaml")
    if not os.path.exists(spec_path):
        return None, f"Spec file not found: {spec_path}"
    with open(spec_path) as f:
        return yaml.safe_load(f), None


def evaluate_improvement(mutator_output, task):
    """Parse mutator output → validate → benchmark → return fitness delta.

    Args:
        mutator_output: text output from the mutator agent (should contain YAML)
        task: dict with target_benchmark, baseline_fitness, sub_examples, target_spec

    Returns:
        dict with:
          - em: 1.0 if improved, 0.0 otherwise
          - fitness: the achieved fitness (0.0 if failed)
          - baseline_fitness: the baseline to compare against
          - fitness_delta: fitness - baseline (negative if worse)
          - status: "improved", "no_change", "worse", "invalid", "error"
          - error: error message if any
    """
    result = {
        "em": 0.0,
        "fitness": 0.0,
        "baseline_fitness": task.get("baseline_fitness", 0.0),
        "fitness_delta": 0.0,
        "status": "unknown",
        "error": None,
    }

    # 1. Extract YAML from mutator output
    yaml_text = _extract_yaml_from_text(mutator_output)
    if not yaml_text:
        result["status"] = "invalid"
        result["error"] = "No YAML content found in mutator output"
        return result

    # 2. Parse YAML
    try:
        spec = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        result["status"] = "invalid"
        result["error"] = f"YAML parse error: {e}"
        return result

    if not isinstance(spec, dict):
        result["status"] = "invalid"
        result["error"] = "Parsed YAML is not a dict"
        return result

    # 3. Check required sections
    required = ("entities", "processes", "edges")
    missing = [k for k in required if k not in spec]
    if missing:
        result["status"] = "invalid"
        result["error"] = f"Missing required sections: {missing}"
        return result

    # 4. Run nested benchmark via benchmark_candidate
    target_benchmark = task.get("target_benchmark", "gsm8k")
    sub_examples = task.get("sub_examples", 5)
    target_spec = task.get("target_spec", "")
    baseline_fitness = task.get("baseline_fitness", 0.0)

    try:
        # Import benchmark_candidate from evolve (lazy to avoid circular imports)
        # Try relative import first (when loaded as agent_ontology.benchmarks.meta_eval),
        # fall back to absolute package import (when loaded from benchmark.py)
        try:
            from ..evolve import benchmark_candidate
        except (ImportError, ValueError):
            from agent_ontology.evolve import benchmark_candidate

        bench_result = benchmark_candidate(
            spec,
            benchmark_suite=target_benchmark,
            benchmark_examples=sub_examples,
            timeout_sec=180,
            base_agent_type=target_spec,
            verbose=False,
        )

        achieved_fitness = bench_result.get("fitness", 0.0)
        result["fitness"] = achieved_fitness
        result["fitness_delta"] = achieved_fitness - baseline_fitness

        if not bench_result.get("ok"):
            result["status"] = "error"
            result["error"] = bench_result.get("error", "Benchmark failed")
            return result

        if achieved_fitness > baseline_fitness:
            result["em"] = 1.0
            result["status"] = "improved"
        elif achieved_fitness == baseline_fitness:
            result["status"] = "no_change"
        else:
            result["status"] = "worse"

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def score_meta_improve(predicted, example):
    """Scoring function for meta_improve benchmark.

    Unlike other scorers, this runs a nested benchmark evaluation.
    Called from compute_fitness_benchmark in evolve.py.

    Args:
        predicted: the mutator's output text (should contain improved YAML)
        example: the full benchmark example dict with task metadata

    Returns:
        dict with em (0.0 or 1.0) and eval_details
    """
    task = {
        "target_spec": example.get("target_spec", ""),
        "target_benchmark": example.get("target_benchmark", "gsm8k"),
        "sub_examples": example.get("sub_examples", 5),
        "baseline_fitness": example.get("baseline_fitness", 0.0),
        "failure_summary": example.get("failure_summary", ""),
        "benchmark_description": example.get("benchmark_description", ""),
    }

    eval_result = evaluate_improvement(predicted, task)

    return {
        "em": eval_result["em"],
        "eval_details": eval_result,
    }
