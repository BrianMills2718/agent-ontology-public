# Landscape: Automated Agent Architecture Search

**Date:** 2026-02-10
**Purpose:** Step 0 of our roadmap — understand what exists so we build on prior art.

---

## The Field

This is a rapidly maturing area. The term ADAS (Automated Design of Agentic Systems) was coined by Hu, Lu & Clune in August 2024. By mid-2025, there are at least 8 distinct systems and 2 comprehensive surveys. The field is analogous to Neural Architecture Search (NAS) but for LLM agent workflows instead of neural network layers.

---

## Key Systems (Chronological)

### 1. DSPy / MIPRO (Stanford, 2023-2025)
- **What**: Declarative framework for LLM programs. Optimizes prompts and few-shot examples via Bayesian optimization.
- **Search space**: Prompts + demonstrations (not architecture/topology)
- **Method**: MIPROv2 uses data-aware instruction generation + Bayesian optimization over prompt space
- **Limitation**: Optimizes *within* a fixed workflow topology, doesn't search over topologies
- **Relevance**: Complementary to our work — we search over architectures, DSPy optimizes prompts within them
- **Link**: https://dspy.ai/

### 2. FunSearch (Google DeepMind, Dec 2023)
- **What**: Evolutionary search over short code snippets for mathematical discoveries
- **Search space**: Python functions (short snippets)
- **Method**: LLM generates candidates → programmatic evaluator scores → evolutionary selection
- **Result**: Discovered new solutions to the cap set problem (published in Nature)
- **Limitation**: Short code snippets only, not full programs
- **Link**: https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/

### 3. ADAS / Meta Agent Search (Hu, Lu, Clune — ICLR 2025)
- **What**: Framed "automated design of agentic systems" as a research problem. Meta agent programs ever-better agents in code.
- **Search space**: Python code defining agent systems (Turing-complete)
- **Method**: Meta agent iteratively programs new agents → tests on tasks → adds to archive → archive informs next iteration
- **Three components**: (1) search space, (2) search algorithm, (3) evaluation function
- **Result**: Discovered agents outperformed hand-designed ones on benchmarks
- **Key insight**: Code is the search space because it's Turing-complete, so any agentic system is representable
- **Limitation**: Searching over raw code is expensive and unstructured. No validation before evaluation.
- **Link**: https://arxiv.org/abs/2408.08435

### 4. AFlow (ICLR 2025)
- **What**: Automated workflow optimization using Monte Carlo Tree Search
- **Search space**: Workflow topologies built from predefined operators
- **Method**: MCTS explores operator combinations → evaluates on benchmarks → selects best topologies
- **Result**: Outperformed all manually designed methods by 5.7% average, 80.3% across 6 datasets
- **Limitation**: Predefined operator set limits novelty. Can't invent new building blocks.
- **Link**: https://arxiv.org/pdf/2410.10762

### 5. AlphaEvolve (Google DeepMind, May 2025)
- **What**: Next-gen FunSearch. Evolutionary search over full programs (hundreds of lines).
- **Search space**: Complete Python programs
- **Method**: Gemini Flash (broad, cheap) + Gemini Pro (deep, quality) → evaluator scores → evolutionary selection
- **Result**: Enhanced Google data center efficiency, chip design, AI training. Found new matrix multiplication algorithms.
- **Key innovation**: Flash/Pro model cascade — breadth vs depth tradeoff
- **Limitation**: Searches over raw code, not structured representations
- **Link**: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/

### 6. EvoAgent (NAACL 2025)
- **What**: Extends single expert agents to multi-agent systems via evolutionary algorithms
- **Search space**: Agent configurations (roles, prompts, tools) within a fixed framework
- **Method**: Mutation + crossover + selection over agent populations
- **Result**: Significantly enhanced task-solving capability across various tasks
- **Limitation**: Evolves agent *configurations*, not workflow *topologies*
- **Link**: https://arxiv.org/abs/2406.14228

### 7. EvoFlow (Feb 2025)
- **What**: Niching evolutionary algorithm for diverse agentic workflows
- **Search space**: Heterogeneous workflows using operator nodes (LLM-invoking nodes)
- **Method**: Tag-based retrieval → crossover → mutation → niching-based selection for diversity
- **Key innovation**: Evolves *diverse* workflows (simple to complex), not just one optimal one
- **Result**: Outperformed handcrafted workflows by 1.2-29.9%, surpassed o1-preview at 12.4% cost
- **Limitation**: Operator nodes are predefined; can't invent new primitives
- **Link**: https://arxiv.org/abs/2502.07373

### 8. EvoAgentX (EMNLP 2025)
- **What**: Unified open-source platform integrating TextGrad + AFlow + MIPRO
- **Search space**: Agent prompts + tool configs + workflow topologies (all three)
- **Method**: Modular 5-layer architecture. Iterative refinement via feedback loops.
- **Result**: +7.4% HotPotQA F1, +10% MBPP pass@1, +10% MATH, up to +20% GAIA
- **Key innovation**: Unified framework that optimizes prompts AND topology AND tools together
- **Limitation**: Still searches over code/configs, not a formal typed representation
- **Link**: https://arxiv.org/abs/2507.03616

### 9. LoongFlow (Baidu, Dec 2025)
- **What**: Directed evolutionary search with cognitive Plan-Execute-Summarize paradigm
- **Search space**: Code (algorithms and ML pipelines)
- **Method**: LLM-guided evolution with "PES" — Plan (strategic), Execute (verified), Summarize (reflect + write to memory)
- **Key innovation**: Hybrid evolutionary memory (Multi-Island + MAP-Elites + Boltzmann selection). Long-term memory accumulation.
- **Result**: Outperformed AlphaEvolve baselines by up to 60% in evolutionary efficiency
- **Limitation**: Focused on algorithm/ML pipeline optimization, not agent architectures specifically
- **Link**: https://arxiv.org/abs/2512.24077

---

## Surveys

### "A Comprehensive Survey of Self-Evolving AI Agents" (Aug 2025)
- Unified framework: System Inputs → Agent System → Environment → Optimisers
- Reviews evolution across: foundation models, prompts, memory, tools, workflows, inter-agent communication
- https://arxiv.org/abs/2508.07407

### "A Survey of Self-Evolving Agents" (Jul 2025)
- Categorizes adaptation by stage: intra-test-time, inter-test-time
- Examines evolution of: models, memory, tools, architecture
- https://arxiv.org/abs/2507.21046

---

## Landscape Map

| System | Searches Over | Representation | Validates Before Eval? | Accumulates Knowledge? | Model Cascade? |
|--------|--------------|----------------|----------------------|----------------------|----------------|
| DSPy/MIPRO | Prompts only | Signatures/Modules | N/A | No | No |
| FunSearch | Short code snippets | Python strings | No | Archive (in-memory) | No |
| ADAS | Full Python code | Code strings | No | Archive (in-memory) | No |
| AFlow | Workflow topologies | Operator graphs | Partial (type checks) | No | No |
| AlphaEvolve | Full programs | Code strings | No | Programs DB | **Yes (Flash/Pro)** |
| EvoAgent | Agent configs | Framework-specific | No | No | No |
| EvoFlow | Diverse workflows | Operator nodes | No | Population (in-memory) | No |
| EvoAgentX | Prompts+tools+topology | Framework-specific | No | No | No |
| LoongFlow | Code | Code strings | Verification contracts | **Yes (long-term memory)** | No |
| **Agent Ontology** | **Typed graph specs** | **Validated YAML** | **Yes (23 rules)** | **Not yet** | **Not yet** |

---

## Where We're Novel

### 1. Typed, Validated Search Space
Every other system searches over code (strings) or framework-specific configs. We search over **typed graphs** with 8 entity types, 7 process types, 12 edge types, and 23 validation rules. Invalid candidates are rejected *before* expensive LLM evaluation. This is structurally analogous to how NAS constrains the search space to valid architectures — but nobody has done this for agent workflows.

### 2. Formal Ontology Backing
Our specs project into OWL/RDF for formal reasoning. No other system has a formal ontological model of agent architectures. This enables:
- Label-independent pattern classification (DL reasoner, not string matching)
- Structural property verification (termination, information flow, policy coverage)
- Compositional search (constraint satisfaction over typed interfaces)

### 3. Framework-Independent Representation
5 importers (LangGraph, CrewAI, AutoGen, OpenAI Agents, Google ADK) prove the format is universal. Other systems are locked to one framework or use raw code. Our specs are the interchange layer.

### 4. Multi-Level Abstraction
Pattern → Spec → Code → Runtime. Mutations happen at the spec level but can be analyzed at the pattern level. Nobody else has this compression hierarchy for agent architectures.

---

## Where We Should Borrow Ideas

### From AlphaEvolve: Flash/Pro Model Cascade
Use cheap model for broad generation, capable model for deep refinement. This is our Step 2.

### From LoongFlow: Persistent Evolutionary Memory
Their hybrid memory (Multi-Island + MAP-Elites + Boltzmann selection) prevents optimization stagnation. Their "Summarize" step writes knowledge to long-term memory. This directly maps to our Step 1 (knowledge store).

### From AFlow: MCTS for Topology Search
Monte Carlo Tree Search over operator combinations is a smarter search strategy than random mutation. Could replace or augment our mutation operators.

### From EvoAgentX: Joint Optimization
They optimize prompts + tools + topology together. We currently only mutate topology (spec structure). Adding prompt optimization (system_prompt fields in specs) and tool selection to our mutation operators would be more complete.

### From EvoFlow: Diversity Preservation
Niching-based selection maintains a diverse population. Our evolve.py just takes top-K by fitness, which can converge prematurely. Adding diversity pressure (e.g., topology distance as a niche metric) would improve exploration.

### From DSPy/MIPRO: Bayesian Optimization for Prompts
Once we have the architecture right, optimizing the prompts within it via Bayesian optimization is the natural next step. This is orthogonal to architecture search but composes well.

---

## What Nobody Has Done (Our Opportunity)

1. **Formal ontology + evolutionary search** — Combining DL reasoning with population-based evolution over agent architectures. The reasoner constrains the search space; the evolution explores it.

2. **Cross-framework architecture knowledge base** — Accumulated facts like "CritiqueCycle achieves 85% on GSM8K but 30% on multi-hop QA" that are queryable and inform future designs. LoongFlow has memory but it's code-level, not architecture-level.

3. **Validated mutations** — Every candidate is structurally verified before evaluation. 23 validation rules + 9 semantic checks + 10 lint rules mean we reject bad candidates for free. Nobody else has this.

4. **Architecture-level transfer learning** — "This pattern worked on math benchmarks, so try it on code benchmarks with retrieval added." Reasoning at the pattern level, not the code level.

---

## Risk: Are We Too Late?

The field is moving fast. EvoAgentX (EMNLP 2025) and LoongFlow (Dec 2025) are close to what we're building. But:

- They search over **code/configs**, not **typed graphs**. Our search space is fundamentally different (more constrained, more analyzable).
- They don't have a **formal ontology**. Pattern classification, compositional search, and property verification are unique to us.
- They don't have **cross-framework interop**. Their workflows are locked to one framework.
- The validation-before-evaluation advantage compounds: at scale, rejecting 80% of candidates for free means 5x more efficient search.

The risk is not that someone builds exactly what we're building — it's that "good enough" code-level evolution makes the formal approach seem like overengineering. The counter-argument: NAS also seemed like overengineering compared to random search, until the search spaces got large enough that structure mattered. Agent architectures are getting complex enough that structured search will win.

---

## Recommended Reading (Priority Order)

1. **ADAS** (Hu et al., ICLR 2025) — The foundational framing. Read first. https://arxiv.org/abs/2408.08435
2. **AlphaEvolve** (DeepMind, 2025) — Flash/Pro cascade, our direct inspiration. https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
3. **EvoAgentX** (EMNLP 2025) — Closest to our unified vision. https://arxiv.org/abs/2507.03616
4. **LoongFlow** (Baidu, 2025) — Long-term memory + PES paradigm. https://arxiv.org/abs/2512.24077
5. **EvoFlow** (Feb 2025) — Diversity-preserving evolution. https://arxiv.org/abs/2502.07373
6. **AFlow** (ICLR 2025) — MCTS for topology search. https://arxiv.org/pdf/2410.10762
7. **Survey: Self-Evolving AI Agents** (Aug 2025) — Comprehensive taxonomy. https://arxiv.org/abs/2508.07407
