#!/usr/bin/env python3
"""
ao-design: End-to-end agent architecture design pipeline.

One command that closes the full loop:
  1. Query knowledge store for evidence
  2. Recommend architectures (or generate from scratch)
  3. Validate + benchmark the best candidate
  4. Optionally evolve over multiple generations
  5. Store results back to knowledge store
  6. Output the best spec + generated agent code

Usage:
    ao-design "multi-hop question answering agent"
    ao-design "math problem solver" --benchmark gsm8k --evolve
    ao-design "code review pipeline" --generations 5 --population 8
    ao-design "RAG agent for medical documents" --output agents/medical_rag.py
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPECS_DIR = os.path.join(SCRIPT_DIR, "specs")


def query_knowledge(task_description, benchmark_suite=None, verbose=False):
    """Query knowledge store for evidence relevant to this task."""
    try:
        from .knowledge_store import KnowledgeStore
    except ImportError:
        from agent_ontology.knowledge_store import KnowledgeStore

    evidence = {
        "best_specs": [],
        "pattern_performance": [],
        "mutation_insights": [],
        "failure_lessons": [],
    }

    try:
        store = KnowledgeStore()
        if benchmark_suite:
            evidence["best_specs"] = store.best_genotypes(benchmark_suite, limit=5)
            evidence["pattern_performance"] = store.pattern_performance(benchmark_suite)
            evidence["failure_lessons"] = store.failure_lessons(benchmark_suite, limit=10)
        evidence["mutation_insights"] = store.mutation_effectiveness()[:10]
        total = store.count()
        store.close()

        if verbose:
            print(f"\n  Knowledge store: {total} candidates recorded")
            if evidence["best_specs"]:
                print(f"  Best for {benchmark_suite}:")
                for s in evidence["best_specs"][:3]:
                    print(f"    {s['spec_name']}: fitness={s.get('fitness', 0):.1f}")
            if evidence["pattern_performance"]:
                print(f"  Pattern performance ({benchmark_suite}):")
                for p in evidence["pattern_performance"][:3]:
                    print(f"    {p.get('detected_patterns', '?')}: avg={p.get('avg_fitness', 0):.1f}")
    except Exception as e:
        if verbose:
            print(f"  Knowledge store unavailable: {e}")

    return evidence


def get_recommendations(task_description, top_n=3, verbose=False):
    """Get architecture recommendations for a task."""
    try:
        from .recommend import recommend
    except ImportError:
        from agent_ontology.recommend import recommend

    recs = recommend(
        task_description,
        specs_dir=SPECS_DIR,
        top_n=top_n,
        verbose=verbose,
        use_knowledge_store=True,
    )
    return recs


def generate_spec(task_description, model="gemini-3-flash-preview",
                  evidence=None, verbose=False):
    """Generate a new spec from task description using specgen."""
    try:
        from .specgen import (
            _scan_description_for_patterns, load_ontology, load_examples,
            build_prompt, call_llm, extract_yaml, validate_spec
        )
    except ImportError:
        from agent_ontology.specgen import (
            _scan_description_for_patterns, load_ontology, load_examples,
            build_prompt, call_llm, extract_yaml, validate_spec
        )

    patterns = _scan_description_for_patterns(task_description)
    ontology = load_ontology()
    examples = load_examples(patterns)

    # Enrich prompt with knowledge store evidence if available
    evidence_context = ""
    if evidence and evidence.get("best_specs"):
        evidence_context = "\n\nEvidence from previous experiments:\n"
        for s in evidence["best_specs"][:3]:
            evidence_context += f"- {s['spec_name']}: fitness={s.get('fitness', 0):.1f}\n"
    if evidence and evidence.get("pattern_performance"):
        evidence_context += "Pattern performance:\n"
        for p in evidence["pattern_performance"][:3]:
            evidence_context += (
                f"- {p.get('detected_patterns', '?')}: "
                f"avg_fitness={p.get('avg_fitness', 0):.1f}\n"
            )

    enriched_description = task_description
    if evidence_context:
        enriched_description += evidence_context

    system, user = build_prompt(enriched_description, ontology, examples, patterns)

    max_attempts = 3
    for attempt in range(max_attempts):
        if verbose:
            print(f"\n  Generating spec (attempt {attempt + 1}/{max_attempts})...")

        response = call_llm(model, system, user, temperature=0.3, max_tokens=8192)
        yaml_text = extract_yaml(response)

        is_valid, output = validate_spec(yaml_text)
        if is_valid:
            spec = yaml.safe_load(yaml_text)
            if verbose:
                print(f"  Valid spec generated: {spec.get('name', 'unnamed')}")
            return spec, yaml_text

        if verbose:
            print(f"  Validation failed: {output[:200]}")

        # Feed errors back for retry
        user += f"\n\nThe previous attempt had validation errors:\n{output}\nPlease fix and regenerate."

    return None, None


def select_or_generate(task_description, benchmark_suite=None,
                       model="gemini-3-flash-preview", verbose=False):
    """Select best existing spec or generate a new one."""
    evidence = query_knowledge(task_description, benchmark_suite, verbose)
    recs = get_recommendations(task_description, top_n=3, verbose=verbose)

    if verbose:
        print(f"\n  Recommendations:")
        for r in recs:
            print(f"    {r.spec_name}: score={r.score:.2f}, "
                  f"patterns={r.patterns}")

    # If top recommendation has a high score, use that spec
    if recs and recs[0].score >= 0.5:
        best_rec = recs[0]
        spec_path = os.path.join(SPECS_DIR, f"{best_rec.spec_name}.yaml")
        if os.path.exists(spec_path):
            with open(spec_path) as f:
                spec = yaml.safe_load(f)
            with open(spec_path) as f:
                yaml_text = f.read()
            if verbose:
                print(f"\n  Selected existing spec: {best_rec.spec_name} "
                      f"(score={best_rec.score:.2f})")
            return spec, yaml_text, spec_path, "selected"

    # Generate from scratch
    if verbose:
        print(f"\n  No strong match found, generating new spec...")
    spec, yaml_text = generate_spec(
        task_description, model=model, evidence=evidence, verbose=verbose
    )
    if spec:
        # Write to temp file for evolve.py compatibility
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', prefix='design_',
            dir=SPECS_DIR, delete=False
        )
        tmp.write(yaml_text)
        tmp.close()
        return spec, yaml_text, tmp.name, "generated"

    return None, None, None, "failed"


def benchmark_spec(spec, benchmark_suite="gsm8k", benchmark_examples=5,
                   timeout_sec=120, verbose=False):
    """Benchmark a spec and return results."""
    try:
        from .evolve import benchmark_candidate
    except ImportError:
        from agent_ontology.evolve import benchmark_candidate

    if verbose:
        print(f"\n  Benchmarking on {benchmark_suite} "
              f"({benchmark_examples} examples)...")

    result = benchmark_candidate(
        spec,
        benchmark_suite=benchmark_suite,
        benchmark_examples=benchmark_examples,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )

    if verbose:
        if result["ok"]:
            print(f"  Result: fitness={result['fitness']:.1f}, "
                  f"EM={result.get('score_em', 0):.0%}, "
                  f"F1={result.get('score_f1', 0):.0%}, "
                  f"calls={result['llm_calls']}, "
                  f"duration={result['duration_ms']}ms")
        else:
            print(f"  Benchmark failed: {result.get('error', 'unknown')}")

    return result


def evolve_spec(spec_path, benchmark_suite="gsm8k", benchmark_examples=5,
                generations=3, population=5, timeout_sec=120,
                verbose=False):
    """Evolve a spec over multiple generations."""
    try:
        from .evolve import evolve
        from .knowledge_store import KnowledgeStore
    except ImportError:
        from agent_ontology.evolve import evolve
        from agent_ontology.knowledge_store import KnowledgeStore

    store = KnowledgeStore()

    if verbose:
        print(f"\n  Evolving {os.path.basename(spec_path)} "
              f"({generations} generations, population {population})...")

    history = evolve(
        spec_path,
        test_inputs={"query": "test", "max_rounds": 2},
        generations=generations,
        population=population,
        timeout_sec=timeout_sec,
        benchmark_suite=benchmark_suite,
        benchmark_examples=benchmark_examples,
        llm_guided=True,
        knowledge_store=store,
        verbose=verbose,
    )

    store.close()
    return history


def instantiate_spec(spec, output_path=None, backend="custom", verbose=False):
    """Instantiate a spec into runnable code."""
    import subprocess

    # Write spec to temp file
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False
    )
    yaml.dump(spec, tmp, default_flow_style=False)
    tmp.close()

    try:
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "instantiate.py"), tmp.name]
        if output_path:
            cmd.extend(["-o", output_path])
        if backend == "langgraph":
            cmd.extend(["--backend", "langgraph"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            if verbose:
                if output_path:
                    print(f"  Agent code written to: {output_path}")
                else:
                    print(f"  Agent code generated (stdout)")
            return True, result.stdout
        else:
            if verbose:
                print(f"  Instantiation failed: {result.stderr[:300]}")
            return False, result.stderr
    finally:
        os.unlink(tmp.name)


def design(task_description, benchmark_suite=None, benchmark_examples=5,
           do_evolve=False, generations=3, population=5,
           output_path=None, output_spec=None, backend="custom",
           model="gemini-3-flash-preview", timeout_sec=120,
           verbose=True):
    """Full design pipeline: recommend/generate → benchmark → evolve → output.

    Returns dict with design results.
    """
    results = {
        "task": task_description,
        "benchmark": benchmark_suite,
        "source": None,
        "spec_name": None,
        "initial_benchmark": None,
        "evolution_history": None,
        "final_spec": None,
        "output_path": None,
    }

    print(f"\n{'═' * 60}")
    print(f"  ao-design: {task_description}")
    print(f"{'═' * 60}")

    # Step 1: Select or generate
    print(f"\n[Step 1] Finding best architecture...")
    spec, yaml_text, spec_path, source = select_or_generate(
        task_description,
        benchmark_suite=benchmark_suite,
        model=model,
        verbose=verbose,
    )

    if spec is None:
        print(f"\n  FAILED: Could not find or generate a suitable architecture.")
        results["source"] = "failed"
        return results

    results["source"] = source
    results["spec_name"] = spec.get("name", "unnamed")
    results["final_spec"] = spec

    # Step 2: Benchmark (if suite specified)
    if benchmark_suite:
        print(f"\n[Step 2] Benchmarking initial architecture...")
        bench_result = benchmark_spec(
            spec,
            benchmark_suite=benchmark_suite,
            benchmark_examples=benchmark_examples,
            timeout_sec=timeout_sec,
            verbose=verbose,
        )
        results["initial_benchmark"] = bench_result

    # Step 3: Evolve (if requested)
    if do_evolve and spec_path:
        print(f"\n[Step 3] Evolving architecture...")
        history = evolve_spec(
            spec_path,
            benchmark_suite=benchmark_suite or "gsm8k",
            benchmark_examples=benchmark_examples,
            generations=generations,
            population=population,
            timeout_sec=timeout_sec,
            verbose=verbose,
        )
        results["evolution_history"] = history

        # Find the best evolved spec
        if history:
            last_gen = history[-1]
            best_name = last_gen.get("best_name")
            best_fitness = last_gen.get("best_fitness", 0)
            if verbose:
                print(f"\n  Best evolved: {best_name} "
                      f"(fitness={best_fitness:.1f})")

            # Try to load the best evolved spec if it was saved
            for gen_data in reversed(history):
                for r in gen_data.get("results", []):
                    if r.get("name") == best_name and r.get("spec"):
                        results["final_spec"] = r["spec"]
                        break

    # Step 4: Record to knowledge store
    if benchmark_suite:
        print(f"\n[Step 4] Recording results to knowledge store...")
        try:
            from .knowledge_store import KnowledgeStore
        except ImportError:
            from agent_ontology.knowledge_store import KnowledgeStore

        try:
            store = KnowledgeStore()
            bench = results.get("initial_benchmark", {})
            store.record_candidate(
                spec_name=results["spec_name"],
                spec=results["final_spec"],
                spec_yaml=yaml_text,
                benchmark=benchmark_suite,
                score_em=bench.get("score_em"),
                score_f1=bench.get("score_f1"),
                fitness=bench.get("fitness", 0),
                llm_calls=bench.get("llm_calls", 0),
                duration_ms=bench.get("duration_ms", 0),
                status=bench.get("status", "UNKNOWN"),
            )
            store.close()
            if verbose:
                print(f"  Recorded to knowledge store.")
        except Exception as e:
            if verbose:
                print(f"  Could not record: {e}")

    # Step 5: Output spec and/or agent code
    if output_spec:
        print(f"\n[Step 5] Writing spec to {output_spec}...")
        with open(output_spec, 'w') as f:
            yaml.dump(results["final_spec"], f, default_flow_style=False)
        if verbose:
            print(f"  Spec written to: {output_spec}")

    if output_path:
        step = 5 if not output_spec else 6
        print(f"\n[Step {step}] Generating agent code...")
        ok, output = instantiate_spec(
            results["final_spec"],
            output_path=output_path,
            backend=backend,
            verbose=verbose,
        )
        if ok:
            results["output_path"] = output_path

    # Clean up temp spec if generated
    if source == "generated" and spec_path and os.path.exists(spec_path):
        os.unlink(spec_path)

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  Design Complete")
    print(f"{'─' * 60}")
    print(f"  Task:     {task_description}")
    print(f"  Source:   {source}")
    print(f"  Spec:     {results['spec_name']}")
    if results.get("initial_benchmark"):
        b = results["initial_benchmark"]
        print(f"  Fitness:  {b.get('fitness', 0):.1f}")
        if b.get("score_em") is not None:
            print(f"  EM:       {b['score_em']:.0%}")
    if results.get("evolution_history"):
        last = results["evolution_history"][-1]
        print(f"  Evolved:  {last.get('best_fitness', 0):.1f} "
              f"(gen {len(results['evolution_history'])})")
    if results.get("output_path"):
        print(f"  Output:   {results['output_path']}")
    print(f"{'═' * 60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ao-design: End-to-end agent architecture design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ao-design "multi-hop question answering agent"
  ao-design "math problem solver" --benchmark gsm8k
  ao-design "code review pipeline" --evolve --generations 5
  ao-design "RAG agent for medical docs" -o agents/medical_rag.py
        """,
    )
    parser.add_argument("task", help="Task description (natural language)")
    parser.add_argument("--benchmark", "-b",
                        help="Benchmark suite (gsm8k, hotpotqa, arc, humaneval)")
    parser.add_argument("--benchmark-examples", type=int, default=5,
                        help="Number of benchmark examples (default: 5)")
    parser.add_argument("--evolve", "-e", action="store_true",
                        help="Evolve the architecture over multiple generations")
    parser.add_argument("--generations", "-g", type=int, default=3,
                        help="Number of evolution generations (default: 3)")
    parser.add_argument("--population", "-p", type=int, default=5,
                        help="Population size per generation (default: 5)")
    parser.add_argument("--output", "-o",
                        help="Output path for generated agent code")
    parser.add_argument("--output-spec", "-s",
                        help="Output path for the best spec YAML")
    parser.add_argument("--backend", choices=["custom", "langgraph"],
                        default="custom",
                        help="Code gen backend (default: custom)")
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="Model for spec generation (default: gemini-3-flash-preview)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per agent run in seconds (default: 120)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    results = design(
        task_description=args.task,
        benchmark_suite=args.benchmark,
        benchmark_examples=args.benchmark_examples,
        do_evolve=args.evolve,
        generations=args.generations,
        population=args.population,
        output_path=args.output,
        output_spec=args.output_spec,
        backend=args.backend,
        model=args.model,
        timeout_sec=args.timeout,
        verbose=not args.quiet,
    )

    if args.json:
        # Serialize, removing non-serializable parts
        out = {k: v for k, v in results.items()
               if k not in ("final_spec",)}
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
