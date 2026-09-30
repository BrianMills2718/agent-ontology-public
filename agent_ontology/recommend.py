#!/usr/bin/env python3
"""
Agent Ontology Architecture Recommender

Suggests agent architectures for given task descriptions based on:
1. OWL pattern classification of existing specs
2. Benchmark results (if available)
3. Structural complexity analysis
4. Pattern compatibility

Usage:
  python3 recommend.py "multi-hop question answering"
  python3 recommend.py "code review with quality gates" --top 3
  python3 recommend.py "customer support routing" --verbose
  python3 recommend.py --list-patterns
  python3 recommend.py --list-capabilities
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# Knowledge Base: task→pattern→spec mappings
# ═══════════════════════════════════════════════════════════════

# Task keywords → relevant architectural patterns
TASK_PATTERN_MAP: dict[str, list[str]] = {
    # Reasoning & problem solving
    "question answering": ["reasoning_loop", "retrieval"],
    "multi-hop": ["reasoning_loop", "retrieval", "decomposition"],
    "reasoning": ["reasoning_loop", "reflection"],
    "math": ["reasoning_loop", "decomposition"],
    "arithmetic": ["reasoning_loop"],
    "logic": ["reasoning_loop", "decomposition"],
    "planning": ["decomposition"],
    "problem solving": ["decomposition", "reasoning_loop"],

    # Code & engineering
    "code review": ["critique_cycle", "fan_out_aggregate"],
    "code generation": ["fan_out_aggregate", "critique_cycle"],
    "debugging": ["reasoning_loop", "reflection"],
    "refactoring": ["critique_cycle"],
    "testing": ["critique_cycle", "fan_out_aggregate"],
    "software": ["fan_out_aggregate", "critique_cycle"],

    # Content & writing
    "writing": ["critique_cycle"],
    "editing": ["critique_cycle"],
    "content": ["critique_cycle", "fan_out_aggregate"],
    "summarization": ["fan_out_aggregate"],
    "translation": ["critique_cycle"],

    # Research & retrieval
    "research": ["reasoning_loop", "retrieval"],
    "search": ["reasoning_loop", "retrieval"],
    "retrieval": ["retrieval"],
    "rag": ["retrieval"],
    "knowledge": ["retrieval"],
    "document": ["fan_out_aggregate", "retrieval"],

    # Multi-agent & collaboration
    "debate": ["debate"],
    "discussion": ["debate"],
    "consensus": ["debate", "fan_out_aggregate"],
    "collaboration": ["fan_out_aggregate", "debate"],
    "team": ["fan_out_aggregate"],
    "parallel": ["fan_out_aggregate"],

    # Iterative improvement
    "refine": ["critique_cycle"],
    "improve": ["critique_cycle", "reflection"],
    "optimize": ["critique_cycle", "reflection"],
    "quality": ["critique_cycle"],
    "feedback": ["critique_cycle"],

    # Customer service & routing
    "customer support": ["fan_out_aggregate"],
    "routing": ["fan_out_aggregate"],
    "handoff": ["fan_out_aggregate"],
    "triage": ["fan_out_aggregate"],

    # Exploration & learning
    "exploration": ["reasoning_loop", "reflection"],
    "learning": ["reflection"],
    "tutoring": ["critique_cycle"],
    "teaching": ["critique_cycle"],
    "education": ["critique_cycle"],

    # Autonomous
    "autonomous": ["reasoning_loop", "reflection"],
    "self-improving": ["reflection", "critique_cycle"],
    "agent": ["reasoning_loop"],
}

# Pattern → reference spec mappings (best exemplar for each pattern)
PATTERN_SPECS: dict[str, list[str]] = {
    "reasoning_loop": ["react", "autogpt"],
    "critique_cycle": ["self_refine", "plan_and_solve"],
    "debate": ["debate"],
    "retrieval": ["rag"],
    "decomposition": ["plan_and_solve", "map_reduce"],
    "fan_out_aggregate": ["map_reduce", "code_reviewer", "multi_agent_codegen"],
    "reflection": ["reflexion", "voyager"],
}

# Pattern descriptions
PATTERN_INFO: dict[str, dict[str, str]] = {
    "reasoning_loop": {
        "name": "Reasoning Loop",
        "description": "Think → Act → Observe cycle. Best for tasks requiring tool use and iterative reasoning.",
        "strengths": "Flexible, handles unknown tool chains, self-correcting",
        "weaknesses": "Can be slow (many LLM calls), may loop without progress",
    },
    "critique_cycle": {
        "name": "Critique Cycle",
        "description": "Generate → Evaluate → Refine loop. Best for quality-sensitive output.",
        "strengths": "High quality output, measurable improvement per iteration",
        "weaknesses": "Requires good evaluation criteria, fixed overhead per refinement",
    },
    "debate": {
        "name": "Multi-Agent Debate",
        "description": "Multiple agents argue different positions, judge decides. Best for contentious or nuanced topics.",
        "strengths": "Explores multiple perspectives, reduces bias",
        "weaknesses": "Expensive (many agents × many rounds), may not converge",
    },
    "retrieval": {
        "name": "Retrieval-Augmented Generation",
        "description": "Query → Retrieve → Generate pipeline. Best for knowledge-intensive tasks.",
        "strengths": "Grounded in facts, reduces hallucination, works with external knowledge",
        "weaknesses": "Quality depends on retrieval, can't reason beyond retrieved context",
    },
    "decomposition": {
        "name": "Decomposition",
        "description": "Break problem into sub-problems → solve each → combine. Best for complex multi-step tasks.",
        "strengths": "Handles complexity, each sub-task is simpler, parallelizable",
        "weaknesses": "Plan quality is critical, error compounds across sub-tasks",
    },
    "fan_out_aggregate": {
        "name": "Fan-Out / Aggregate",
        "description": "Distribute work to parallel workers → collect and merge results. Best for batch processing.",
        "strengths": "Parallelizable, scales to large inputs, independent failure",
        "weaknesses": "Aggregation can lose nuance, requires good chunking strategy",
    },
    "reflection": {
        "name": "Reflection",
        "description": "Execute → Evaluate → Reflect → Retry with memory. Best for learning from mistakes.",
        "strengths": "Self-improving, builds episodic memory, handles failure gracefully",
        "weaknesses": "Slow convergence, memory management overhead",
    },
}

# Benchmark results knowledge base
BENCHMARK_RESULTS: dict[str, dict[str, Any]] = {
    "react": {"gsm8k": {"em": 0.0, "notes": "Needs tools for math"}, "hotpotqa": {"f1": 0.099}},
    "self_refine": {"gsm8k": {"em": 1.0, "notes": "Strong on arithmetic"}},
    "plan_and_solve": {"gsm8k": {"em": 0.67, "notes": "Good decomposition for multi-step math"}},
}


# ═══════════════════════════════════════════════════════════════
# Recommendation Engine
# ═══════════════════════════════════════════════════════════════

def _match_task_to_patterns(task_description: str) -> dict[str, float]:
    """Score patterns by relevance to task description."""
    task_lower = task_description.lower()
    scores: dict[str, float] = defaultdict(float)

    for keyword, patterns in TASK_PATTERN_MAP.items():
        if keyword in task_lower:
            weight = len(keyword.split()) * 1.5  # Multi-word matches score higher
            for i, pattern in enumerate(patterns):
                # First pattern in list is most relevant
                scores[pattern] += weight * (1.0 - i * 0.2)

    return dict(scores)


def _load_spec_catalog(specs_dir: str) -> dict[str, dict]:
    """Load all specs from directory for analysis."""
    catalog: dict[str, dict] = {}
    specs_path = Path(specs_dir)
    if not specs_path.is_dir():
        return catalog

    for f in sorted(specs_path.glob("*.yaml")):
        try:
            spec = yaml.safe_load(f.read_text(encoding="utf-8"))
            if spec:
                catalog[f.stem] = spec
        except Exception:
            continue
    return catalog


def _get_spec_complexity(spec: dict) -> dict[str, int]:
    """Quick complexity metrics for a spec."""
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])
    schemas = spec.get("schemas", [])
    return {
        "entities": len(entities),
        "processes": len(processes),
        "edges": len(edges),
        "schemas": len(schemas),
        "agents": sum(1 for e in entities if isinstance(e, dict) and e.get("type") == "agent"),
        "stores": sum(1 for e in entities if isinstance(e, dict) and e.get("type") == "store"),
        "loops": sum(1 for e in edges if isinstance(e, dict) and e.get("type") == "loop"),
        "gates": sum(1 for p in processes if isinstance(p, dict) and p.get("type") == "gate"),
    }


class Recommendation:
    def __init__(self, spec_name: str, patterns: list[str], score: float,
                 complexity: dict[str, int], reasoning: str,
                 benchmark_info: str = ""):
        self.spec_name = spec_name
        self.patterns = patterns
        self.score = score
        self.complexity = complexity
        self.reasoning = reasoning
        self.benchmark_info = benchmark_info


def _get_knowledge_store_data(benchmark: str | None = None) -> dict[str, Any]:
    """Query the knowledge store for evidence-backed pattern performance data."""
    try:
        from .knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        data: dict[str, Any] = {}

        # Best genotypes per benchmark
        for suite in (["gsm8k", "hotpotqa", "arc", "humaneval"] if not benchmark else [benchmark]):
            best = store.best_genotypes(suite, limit=10)
            if best:
                data.setdefault("best_by_benchmark", {})[suite] = best

        # Pattern performance across benchmarks
        for suite in (["gsm8k", "hotpotqa", "arc", "humaneval"] if not benchmark else [benchmark]):
            perf = store.pattern_performance(suite)
            if perf:
                data.setdefault("pattern_performance", {})[suite] = perf

        # Mutation effectiveness
        mut_eff = store.mutation_effectiveness()[:10]
        if mut_eff:
            data["mutation_effectiveness"] = mut_eff

        store.close()
        return data
    except Exception:
        return {}


def recommend(task_description: str, specs_dir: str | None = None,
              top_n: int = 3, verbose: bool = False,
              use_knowledge_store: bool = True) -> list[Recommendation]:
    """Generate architecture recommendations for a task description.

    If use_knowledge_store=True and a knowledge store exists, recommendations
    are boosted by actual benchmark performance data from evolution runs.
    """
    # Score patterns by task match
    pattern_scores = _match_task_to_patterns(task_description)

    if not pattern_scores:
        # Fallback: suggest reasoning_loop as default
        pattern_scores = {"reasoning_loop": 0.5}

    # Load spec catalog
    if specs_dir is None:
        specs_dir = os.path.join(SCRIPT_DIR, "specs")
    catalog = _load_spec_catalog(specs_dir)

    # Query knowledge store for evidence
    knowledge_data: dict[str, Any] = {}
    if use_knowledge_store:
        # Determine relevant benchmarks from task keywords
        task_lower = task_description.lower()
        relevant_benchmark = None
        if any(kw in task_lower for kw in ["math", "arithmetic", "calculation"]):
            relevant_benchmark = "gsm8k"
        elif any(kw in task_lower for kw in ["question", "qa", "knowledge", "multi-hop"]):
            relevant_benchmark = "hotpotqa"
        elif any(kw in task_lower for kw in ["code", "programming"]):
            relevant_benchmark = "humaneval"
        elif any(kw in task_lower for kw in ["science", "multiple choice"]):
            relevant_benchmark = "arc"
        knowledge_data = _get_knowledge_store_data(relevant_benchmark)

    # Build evidence-backed pattern scores from knowledge store
    ks_pattern_scores: dict[str, float] = {}
    ks_spec_scores: dict[str, tuple[float, str]] = {}

    for suite, perf_list in knowledge_data.get("pattern_performance", {}).items():
        for perf in perf_list:
            patterns_str = perf.get("detected_patterns", "")
            avg_fitness = perf.get("avg_fitness", 0)
            if patterns_str and avg_fitness > 0:
                for pat in patterns_str.split(","):
                    pat = pat.strip().lower().replace(" ", "_")
                    if pat:
                        ks_pattern_scores[pat] = max(ks_pattern_scores.get(pat, 0), avg_fitness / 100.0)

    for suite, best_list in knowledge_data.get("best_by_benchmark", {}).items():
        for entry in best_list:
            spec_name = entry.get("spec_name", "")
            fitness = entry.get("fitness", 0)
            patterns = entry.get("detected_patterns", "")
            # Extract base spec name (strip gen prefixes)
            base = spec_name.split("_mut")[0].split("_llm")[0].split("_cx")[0]
            base = re.sub(r"^gen\d+_", "", base)
            if base and fitness > 0:
                prev_fitness, prev_info = ks_spec_scores.get(base, (0, ""))
                if fitness > prev_fitness:
                    ks_spec_scores[base] = (fitness, f"{suite}: fitness={fitness:.1f}")

    # Detect patterns using combined OWL structural + ID-based methods
    owl_patterns: dict[str, list[str]] = {}
    try:
        from .patterns import detect_patterns_combined
        for spec_name, spec_data in catalog.items():
            spec_path = os.path.join(specs_dir, f"{spec_name}.yaml")
            combined = detect_patterns_combined(spec_data, spec_path)
            owl_patterns[spec_name] = [r["name"] for r in combined]
    except ImportError:
        try:
            from .patterns import detect_patterns
            for spec_name, spec_data in catalog.items():
                detected = detect_patterns(spec_data)
                owl_patterns[spec_name] = [name for name, _, _ in detected]
        except ImportError:
            pass

    # Score each spec
    recommendations: list[Recommendation] = []

    for spec_name, spec in catalog.items():
        spec_score = 0.0
        matched_patterns: list[str] = []
        reasons: list[str] = []

        # Pattern matching score
        spec_patterns = owl_patterns.get(spec_name, [])
        # Also check PATTERN_SPECS mapping
        for pattern, ref_specs in PATTERN_SPECS.items():
            if spec_name in ref_specs:
                if pattern not in spec_patterns:
                    spec_patterns.append(pattern)

        for pattern in spec_patterns:
            # Normalize pattern name for matching
            pattern_key = pattern.lower().replace(" ", "_")
            if pattern_key in pattern_scores:
                spec_score += pattern_scores[pattern_key]
                matched_patterns.append(pattern)
                reasons.append(f"matches '{pattern}' pattern")

            # Evidence boost from knowledge store
            if pattern_key in ks_pattern_scores:
                boost = ks_pattern_scores[pattern_key]
                spec_score += boost
                reasons.append(f"'{pattern}' scored {boost:.1f} in evolution")

        if not matched_patterns:
            continue

        # Benchmark bonus from hardcoded results
        benchmark_info = ""
        if spec_name in BENCHMARK_RESULTS:
            for dataset, results in BENCHMARK_RESULTS[spec_name].items():
                task_lower = task_description.lower()
                if any(kw in task_lower for kw in ["math", "arithmetic", "calculation"]) and dataset == "gsm8k":
                    em = results.get("em", 0)
                    spec_score += em * 2.0
                    benchmark_info += f"GSM8K EM={em:.0%} "
                elif any(kw in task_lower for kw in ["question", "qa", "knowledge"]) and dataset == "hotpotqa":
                    f1 = results.get("f1", 0)
                    spec_score += f1 * 2.0
                    benchmark_info += f"HotpotQA F1={f1:.1%} "

        # Evidence bonus from knowledge store (spec-level)
        if spec_name in ks_spec_scores:
            ks_fitness, ks_info = ks_spec_scores[spec_name]
            spec_score += ks_fitness / 50.0  # normalize to comparable range
            benchmark_info += f"[KS: {ks_info}] "
            reasons.append(f"knowledge store: {ks_info}")

        complexity = _get_spec_complexity(spec)
        reasoning = "; ".join(reasons)

        recommendations.append(Recommendation(
            spec_name=spec_name,
            patterns=matched_patterns,
            score=spec_score,
            complexity=complexity,
            reasoning=reasoning,
            benchmark_info=benchmark_info.strip(),
        ))

    # Sort by score descending
    recommendations.sort(key=lambda r: r.score, reverse=True)
    return recommendations[:top_n]


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _format_recommendations(recs: list[Recommendation], task: str, verbose: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"\n  Architecture Recommendations for: \"{task}\"")
    lines.append("  " + "═" * 58)

    if not recs:
        lines.append("  No matching architectures found. Try a more specific description.")
        return "\n".join(lines)

    for i, rec in enumerate(recs, 1):
        lines.append(f"\n  {i}. {rec.spec_name}")
        lines.append(f"     Score: {rec.score:.1f}")
        lines.append(f"     Patterns: {', '.join(rec.patterns)}")
        lines.append(f"     Why: {rec.reasoning}")
        if rec.benchmark_info:
            lines.append(f"     Benchmarks: {rec.benchmark_info}")
        if verbose:
            c = rec.complexity
            lines.append(f"     Complexity: {c['entities']} entities, {c['processes']} processes, "
                         f"{c['agents']} agents, {c['loops']} loops, {c['gates']} gates")

    # Add pattern explanations for top match
    if recs:
        top_patterns = recs[0].patterns
        lines.append(f"\n  Pattern Details:")
        lines.append("  " + "─" * 58)
        for pat in top_patterns:
            pat_key = pat.lower().replace(" ", "_")
            info = PATTERN_INFO.get(pat_key, {})
            if info:
                lines.append(f"  {info.get('name', pat)}:")
                lines.append(f"    {info.get('description', '')}")
                if verbose:
                    lines.append(f"    Strengths: {info.get('strengths', '')}")
                    lines.append(f"    Weaknesses: {info.get('weaknesses', '')}")

    lines.append("")
    return "\n".join(lines)


def _format_patterns_list() -> str:
    lines = ["\n  Available Architectural Patterns", "  " + "═" * 50]
    for key, info in PATTERN_INFO.items():
        lines.append(f"\n  {info['name']} ({key})")
        lines.append(f"    {info['description']}")
        lines.append(f"    Reference specs: {', '.join(PATTERN_SPECS.get(key, []))}")
    lines.append("")
    return "\n".join(lines)


def _format_capabilities() -> str:
    lines = ["\n  Task Capabilities", "  " + "═" * 50]
    # Group by category
    categories: dict[str, list[str]] = defaultdict(list)
    for keyword in sorted(TASK_PATTERN_MAP.keys()):
        patterns = TASK_PATTERN_MAP[keyword]
        categories[patterns[0]].append(keyword)

    for pattern, keywords in sorted(categories.items()):
        info = PATTERN_INFO.get(pattern, {})
        name = info.get("name", pattern)
        lines.append(f"\n  {name}:")
        lines.append(f"    Keywords: {', '.join(keywords)}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Agent Ontology Architecture Recommender — suggest architectures for task descriptions",
    )
    parser.add_argument("task", nargs="?", help="Task description (in quotes)")
    parser.add_argument("--top", type=int, default=3, help="Number of recommendations (default 3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")
    parser.add_argument("--specs-dir", default=None, help="Path to specs directory")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    parser.add_argument("--list-patterns", action="store_true", help="List available patterns")
    parser.add_argument("--list-capabilities", action="store_true", help="List task capabilities")
    args = parser.parse_args(argv)

    if args.list_patterns:
        print(_format_patterns_list())
        return

    if args.list_capabilities:
        print(_format_capabilities())
        return

    if not args.task:
        parser.error("Please provide a task description")

    recs = recommend(args.task, specs_dir=args.specs_dir, top_n=args.top, verbose=args.verbose)

    if args.json_output:
        results = [{
            "spec": r.spec_name,
            "score": r.score,
            "patterns": r.patterns,
            "reasoning": r.reasoning,
            "benchmark_info": r.benchmark_info,
            "complexity": r.complexity,
        } for r in recs]
        print(json.dumps(results, indent=2))
    else:
        print(_format_recommendations(recs, args.task, args.verbose))


if __name__ == "__main__":
    main()
