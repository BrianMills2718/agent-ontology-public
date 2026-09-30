# Vision: Agent Ontology as Formal Representation of Agency

## The Core Idea

Agent Ontology is a **formal representation of agent architecture** that is simultaneously machine-readable, LLM-legible, evolvable, and scale-invariant.

It is not "a YAML format for describing agents." It is the substrate through which agent systems can be analyzed, reasoned about, composed, and evolved — by humans and by agents themselves.

## Four Pillars

### 1. Multi-Scale Agency Analysis

In biological systems, agency exists at every scale. A cell is an agent. An organ is an agent. An organism is an agent. A social group is an agent. Michael Levin's work on multi-scale competency architecture shows that the same analytical framework can describe agency at any level — what changes is not the framework but the resolution.

The same is true for artificial agent systems:

- A single LLM call with state read/write is the simplest agent
- A ReAct loop (reason → act → observe → repeat) is an agent made of components
- A BabyAGI (task decomposition + execution + memory) is a more complex agent
- A firm (multiple specialized agents coordinating via contracts) is an agent
- An ecology (firms competing and cooperating in a market) is an agent

Agent Ontology provides the analytical lens that identifies and describes agency at any level. You point it at a subgraph of component interactions and it tells you: this is a ReAct pattern, this is a hierarchical delegation pattern, this cluster has the structure of a firm with specialized roles.

The spec format is the same at every scale — entities, processes, edges, schemas. The CoALA cognitive taxonomy (memory types, action types, decision strategies) provides a complementary classification layer.

### 2. Self-Legible Architecture for Agent Self-Improvement

For an agent to improve itself, it must be able to reason about its own structure. Raw Python code is hard for LLMs to reason about reliably. A constrained, validated YAML spec is not.

If an agent wants to rewrite its own behavior, the path is:

1. Read the spec describing its current architecture
2. Analyze it (pattern detection, lint rules, complexity scoring)
3. Identify weaknesses (missing error handling, inefficient topology, dead stores)
4. Mutate the spec (swap a pattern, add a reflection loop, restructure control flow)
5. Validate the mutation (23 structural rules, 10 anti-pattern checks)
6. Generate new code from the modified spec
7. Test the generated code against benchmarks
8. Deploy if fitness improves

Every step except the LLM call in step 4 is deterministic and verifiable. The symbolic structure constrains the neural creativity, ensuring that mutations produce valid architectures.

### 3. Evolutionary Optimization with Real Feedback

The spec is the genome. We already have:

- **Mutation operators**: field-level mutations (swap model, add store, rewire edges) and pattern-level mutations (swap reasoning pattern, insert reflection loop, remove dead branches)
- **Crossover**: exchange detected patterns between specs
- **Composition**: combine patterns from a reusable library (7 patterns: reasoning loop, critique cycle, debate, retrieval, decomposition, fan-out/aggregate, reflection)
- **Validation**: 23 structural rules ensure every mutation produces a valid spec
- **Benchmarks**: HotpotQA, GSM8K, ARC, HumanEval for fitness evaluation

What closes the loop:

- **Real fitness functions**: not just "does it validate" but "how well does it perform on tasks" — scrip earned, benchmark accuracy, task completion rate
- **Prompt optimization**: DSPy-style optimization of system prompts and few-shot examples within the spec's schema constraints
- **Agent-driven evolution**: agents in the ecology running evolutionary search over their own architectures — an agent that evolves other agents, or that evolves itself
- **Selection pressure from resource scarcity**: in an agent ecology with finite resources, better architectures earn more scrip, survive longer, and reproduce

### 4. Neurosymbolic AI

Most current AI is either:
- **Pure neural**: LLMs generating free-form code and text, flexible but unconstrained
- **Pure symbolic**: formal verification, type systems, logic programming, precise but brittle

Agent Ontology sits at the intersection:

- **Symbolic**: typed entity/process/edge schemas, JSON Schema validation, OWL Description Logic reasoning, 23 structural rules, 10 anti-pattern detectors, graph topology classification
- **Neural**: LLMs generating specs from natural language, LLMs populating logic blocks, LLMs making runtime decisions within the spec's control flow
- **The bridge**: formal specs generated and manipulated by neural systems, validated by symbolic rules, instantiated into running agents where neural systems operate within symbolic constraints

The OWL bridge already enables Description Logic reasoning over agent architectures — classifying patterns, checking subsumption, detecting structural properties that pure graph analysis misses.

The hypothesis: **the path to reliable, improvable, composable AI agents runs through neurosymbolic representations** — not through pure neural generation of unconstrained code, and not through rigid symbolic programming that can't adapt. Agent Ontology is an experiment in finding the right interface between neural flexibility and symbolic rigor.

## Connection to Agent Ecology

Agent Ecology (github.com/BrianMills2718/agent_ecology2) is a mechanism design experiment for emergent collective intelligence. It provides:

- **Runtime physics**: resource scarcity, contracts, markets, auctions
- **The execution substrate**: artifact storage, safe execution, LLM routing, rate limiting
- **Emergent coordination**: agents form firms, trade, and self-modify

Agent Ontology provides:

- **Formal specification**: what an agent IS (structure, behavior, data flow)
- **Analysis tools**: what an agent DOES (patterns, complexity, topology)
- **Evolution machinery**: how an agent IMPROVES (mutations, crossover, fitness)
- **Multi-scale lens**: where agency EXISTS in a system (pattern detection at any resolution)

The integration: Agent Ontology specs describe the patterns of artifact interaction within the ecology. The ecology provides the runtime and selection pressure. Together, they enable agents that can formally reason about their own architecture, evolve through validated mutations, and be analyzed at any scale from individual components to system-wide organization.

## What This Is Not

- Not "OpenAPI for agents" (the original tagline was too narrow)
- Not a competitor to LangGraph or CrewAI (those are runtimes; this is a representation)
- Not an academic taxonomy like CoALA (CoALA describes; this generates and validates)
- Not just YAML files (the toolchain — validate, instantiate, evolve, analyze — is the product)

## Current State

- 23 agent specs covering major architectural patterns (including self-improver)
- Import from LangGraph and CrewAI source code (AST-based + LLM-augmented)
- Code generation targeting 2 backends (custom executor, LangGraph)
- **22/22 runnable specs pass automated tests** (+ 1 description-only spec)
- OWL Description Logic bridge with 9 structural pattern classifiers, lossless round-trip
- Evolutionary search with mutation, crossover, and benchmark fitness
- Analysis suite: validate, lint, topology, complexity, similarity, patterns, coverage, verify, recommend
- PyPI-ready package with 18 CLI entry points
- 182/182 property tests, 8 round-trip tests

### Pillar 2 Proof: Self-Modifying Agent

The `self_improver` agent (spec: `specs/self_improver.yaml`) demonstrates Pillar 2 in practice:

1. **Reads** a target agent spec (any YAML spec from the library)
2. **Analyzes** it using the toolchain: `lint_spec()` finds 8 warnings, `verify_spec()` finds 8 issues, `detect_patterns()` identifies the critique_cycle pattern
3. **Diagnoses** via LLM: correctly identifies data plumbing as the primary weakness ("spec never persists or maps agent outputs into shared state")
4. **Proposes** a structured mutation: `add_store` with specific implementation details
5. **Mutates** via LLM: generates a complete modified YAML spec
6. **Validates** symbolically: the validator catches that the LLM dropped the schemas section (9 errors)
7. **Rejects** the invalid mutation and finalizes with the original spec

This is the neurosymbolic feedback loop in action: neural creativity (LLM proposes improvements) constrained by symbolic validation (23 structural rules catch mistakes). The agent successfully analyzed a real spec, identified genuine architectural weaknesses, and the validation correctly caught a flawed mutation.

### Pillar 3 Proof: Evolutionary Optimization

The `evolve.py` tool runs evolutionary search over agent architectures:

- **3 generations**, 17 total candidates evaluated on `self_refine` base spec
- **Best variant**: `insert_pattern(reflexion)` achieved fitness 168.3 (+5.1% over base at 160.1)
  - Survived all 3 generations as the fittest individual
  - The reflexion pattern adds trial-evaluate-reflect loops, making the agent more robust
- **Crossover worked**: combining reflexion + debate variants (from different parents) produced viable offspring at fitness 165.1
- **Pattern diversity**: critique_cycle, reflection, and debate patterns all appeared in surviving candidates
- **Failure modes were informative**: debate insertion scored only 136.0 because the debate agents received no topic (data plumbing — exactly the issue our analysis tools detect)
- **Mutation types exercised**: swap_process_order, insert_pattern, remove_pattern, remove_process, change_model, modify_prompt, duplicate_with_variation

## Open Questions

- Can agents reliably self-modify via spec manipulation? Initial evidence: the self-improver correctly identifies weaknesses but LLMs struggle to produce valid large YAML mutations. Smaller, targeted mutations may work better.
- What DSPy integration looks like — optimize prompts within spec constraints, or treat the whole spec as a DSPy program?
- How to close the loop between evolution and real-world fitness? Current pass/fail + efficiency scoring is coarse. Benchmark-based evolution (GSM8K, HotpotQA) provides ground truth.
- Should mutations be constrained to one operation at a time (add one store, change one gate condition) for higher validity rates?
- How to measure "level of agency" in a subgraph of artifact interactions — is pattern detection sufficient, or do we need information-theoretic measures?
