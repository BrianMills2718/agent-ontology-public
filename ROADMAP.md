# Agent Ontology Roadmap

**Date:** 2026-02-12
**Current state:** v0.2 ontology, 32 specs, 34 tools, 2 code gen backends, OWL dual representation, PyPI package, real-world-tested importers (LangGraph + CrewAI + AutoGen + OpenAI Agents SDK + Google ADK), spec-level state schema support, benchmark evolution, meta-evolution (mutator agent + nested fitness)

---

## Where We Are

### Assets
- **Ontology** (ONTOLOGY.yaml v0.2): 9 entity types, 10 process types, 13 edge types, 25+ validation rules
- **Spec catalog**: 32 agent specs covering major patterns (ReAct, debate, RAG, plan-and-solve, self-refine, tree-of-thought, LATS, voyager, multi-agent codegen, map-reduce, socratic tutor, reflexion, mixture-of-agents, meta-prompting, customer-support swarm, software team, self-improver, KG RAG, PDDL planner, RAP, alpha-geometry, minimal-solver, multidoc, kb-react, mutator)
- **Full pipeline**: docs -> spec -> {validate, visualize, instantiate, test, analyze, mutate, evolve, benchmark}
- **Two code gen backends**: custom state machine + LangGraph (32 agents each)
- **Pattern library**: 7 reusable patterns with composition, mutation, crossover
- **Benchmarks**: 9 datasets (HotpotQA, GSM8K, GSM8K-Hard, GSM8K-Tricky, ARC, HumanEval, MultiDoc, KB-Tool, Meta-Improve)
- **Property tests**: 254/254 pass across all 32 specs
- **Meta-evolution**: Mutator agent that improves other specs, evaluated via nested fitness on meta_improve benchmark

### Core Thesis
Agent architectures are typed graphs. If you make the graph formal, you unlock validation, visualization, code generation, testing, analysis, comparison, mutation, and evolution for free.

---

## Ontology Completeness Assessment

### Confirmed Complete
These areas were evaluated against 10+ frameworks (LangGraph, CrewAI, AutoGen, MetaGPT, CAMEL, OpenAI Agents SDK, Google ADK, DSPy, Magentic-One, MCP) and are fully expressible:

| Pattern | How it's expressed | Frameworks covered |
|---------|-------------------|-------------------|
| Reasoning loops (ReAct) | step + gate + loop edge | All |
| Critique/refine cycles | step + gate + loop edge | Self-refine, DSPy |
| Fan-out/fan-in | Multiple invoke edges from one step | Code reviewer, map-reduce |
| Pub/sub messaging | channel entity + publish/subscribe edges | MetaGPT, AutoGen |
| Agent handoffs | handoff edge + gate routing | OpenAI Swarm, CrewAI |
| Team execution | team entity with strategy field | CrewAI, MetaGPT |
| Conversations | conversation entity | CAMEL, OpenAI threads |
| Nested loops | Multiple loop edges at different scopes | Magentic-One dual-loop |
| Recursive composition | spawn with template: 'self' | Meta-prompting |
| Human-in-the-loop | checkpoint process | Socratic tutor |
| Guardrails/policies | policy process with targets + effect | OpenAI Agents SDK, Google ADK |
| Error handling | error_handler process + error edges | All production frameworks |
| Store-backed memory | store entity (vector, kv, queue, etc.) | RAG, BabyAGI, AutoGPT |
| Composable termination | termination_condition structured type | AutoGen |
| Checkpointing | checkpointing config on spec_shape | LangGraph |

### Mild Gaps (Optional Improvements)

#### 1. Multimodal Schema Types
**What**: Schema field types are string/integer/float/object/list. No image/audio/video types.
**Impact**: Can't distinguish "this agent receives a screenshot" from "this agent receives text" at the spec level. Both would be `type: string` (base64) or `type: object`.
**Fix**: Add `image`, `audio`, `video` as recognized schema field types. Purely additive — no structural change.
**Verdict**: Nice-to-have. Doesn't block any current patterns. Modern LLMs handle multimodal via their API, not via schema types.

#### 2. Memory Consolidation (Episodic -> Semantic)
**What**: No way to express "periodically consolidate raw episodic traces into summarized semantic memory."
**Impact**: Reflexion spec has episodic memory, but the consolidation step is manual logic, not a declarable pattern.
**Fix**: Already expressible as a maintenance flow (step with logic that reads episodic store, summarizes, writes to semantic store). Not a new ontology type — just a spec pattern.
**Verdict**: Not an ontology gap. Could add as a pattern in patterns.py.

#### 3. MCP Tool Discovery
**What**: MCP tools are discovered at runtime, not declared statically.
**Impact**: Our tool entities are statically wired in specs.
**Analysis**: This is a runtime concern, not an architecture concern. `tool_type: mcp` already declares "this agent uses MCP." Which specific tools are available is determined at runtime by the MCP server — same as a database query returning different rows. Adding a "discovery flag" would mix runtime semantics into the structural ontology.
**Verdict**: Not an ontology gap. Already handled by `tool_type: mcp`.

#### 4. DSPy Prompt Optimization
**What**: DSPy compiles/optimizes prompts against metrics. No way to declare optimization targets in specs.
**Impact**: Can't express "optimize this agent's system prompt for exact_match on GSM8K."
**Analysis**: Optimization is a meta-process that operates on specs/agents, not part of the architecture itself. Like compiler optimizations aren't part of the source language.
**Verdict**: Not an ontology gap. Would be a separate tool (`dspy_optimize.py`) in the pipeline, not an ontology extension.

### Not Gaps (Confirmed Out of Scope)
- **Deployment topology** (which machine agents run on) — runtime concern
- **Cost/budget constraints** — runtime concern
- **Authentication/authorization** — runtime concern
- **Streaming/async execution details** — runtime concern
- **Distributed runtime** — runtime concern

---

## Strategic Direction

### Value Proposition
"OpenAPI for Agents" — a universal spec format for describing AI agent architectures, with a complete toolchain for validation, visualization, code generation, and analysis.

### The unique asset is the ontology + spec format, not the code generator.
Code generators are commodity (LangGraph, CrewAI will always run their own frameworks better). But nobody has the interchange format.

### Three audiences (in priority order)

1. **Framework interop** (highest leverage): Import from LangGraph/CrewAI/AutoGen -> YAML spec -> Export to any target. The "Rosetta Stone for agent frameworks."

2. **Architecture documentation/audit**: Point at existing agent code, get a validated architecture diagram + complexity analysis + lint warnings. Useful TODAY for teams with agents.

3. **Rapid prototyping**: Describe agent in English -> get running code. Useful but competing with native frameworks.

---

## Plan: Order of Attack

### Phase 1: Solidify the Ontology — DONE
**Goal**: Declare the ontology complete for v1.0 and write the formal specification.

#### 1.1 Formal Spec Document — DONE (a26ee15)
`SPEC.md` — ~945 lines covering all 8 entity types, 7 process types, 12 edge types, schema format, 24 validation rules, conventions, and a complete Self-Refine example. Cross-validated against all 22 specs. A developer who has never seen this repo can write a valid spec from SPEC.md alone.

#### 1.2 Multimodal Schema Types (Optional)
Add `image`, `audio`, `video` as recognized schema field types.

- **Satisfaction criteria**: Can write a spec with `{ name: screenshot, type: image }` and it validates. Generated code handles multimodal inputs.
- **Uncertainties**: Do we need to change the code generator? Multimodal is handled by the LLM API, not by schema types.
- **Decision**: Defer unless a spec needs it.

### Phase 1.3: JSON Schema for Spec Format — DONE (spec_schema.json)
`spec_schema.json` — JSON Schema draft 2020-12, discriminated unions via `if/then` for all 8 entity types, 7 process types, 12 edge types. 22/22 specs validate. Enables VS Code autocomplete and cross-language validation.

### Phase 2: Import Bridge (Highest Leverage)
**Goal**: Prove the interchange story by importing real agent code into specs.

#### 2.1 LangGraph Importer — DONE (import_langgraph.py)
AST-based parser extracts StateGraph definitions → valid YAML specs. Tested on 5 real-world open-source LangGraph agents:
- Canonical agent (96 lines): 2 nodes, 2 edges, 2 schemas — validates clean
- Research assistant (148 lines): 4 nodes, 3 edges — validates clean
- STORM/647 lines (langchain-ai/langgraph): 9 processes, 7 edges, 12 schemas — multi-StateGraph + loop resolution
- deer-flow/473 lines (bytedance): 10 processes, 7 edges — Command routing extraction
- SWE agent/567 lines (langtalks): 17 processes, 11 edges, 14 schemas — 3 sub-graphs merged

Supports: multiple StateGraphs, `Command(goto=...)` routing, for-loop add_node resolution, `set_finish_point`, `START` import, edge deduplication. `--llm-augment` flag enriches with tool entities, descriptions, model config, missing edges.

#### 2.2 CrewAI Importer — DONE (import_crewai.py)
AST-based parser extracts Agent/Task/Crew definitions → valid YAML specs. Tested on 3 real-world CrewAI projects:
- Instagram post/364 lines (crewAI-examples): 5 agents, 6 tasks — validates clean
- Meeting prep/245 lines (crewAI-examples): 4 agents, 4 async tasks — fan-out detected, validates clean
- Financial analysis/221 lines (third-party): 4 agents, 5 tools, 4 tasks — validates clean

Supports: `async_execution` fan-out, task-specific tool entities + invoke edges, checkpoint for `human_input`, sequential + hierarchical process types, Pydantic output schemas.

#### 2.3 AutoGen Importer — DONE (import_autogen.py)
AST-based parser extracts AutoGen agent code → valid YAML specs. Supports both pyautogen v0.2 and autogen-agentchat v0.4 APIs:
- v0.2: AssistantAgent, UserProxyAgent, ConversableAgent, GroupChat, GroupChatManager, `register_for_llm`/`register_for_execution` tool decorators, `register_function`, `initiate_chat`/`initiate_chats`, nested chats
- v0.4: AssistantAgent (autogen_agentchat), FunctionTool, RoundRobinGroupChat/SelectorGroupChat/Swarm teams, Handoff, `model_client` extraction
- Variable resolution for llm_config dicts, config_list model extraction, FunctionTool variable references
- GroupChat → team entity + loop/gate processes with termination detection
- `--llm-augment` flag for LLM-enhanced import
- CLI entry point: `ao-import-autogen`

#### 2.4 OpenAI Agents SDK Importer — DONE (import_openai_agents.py)
AST-based parser extracts OpenAI Agents SDK (`from agents import ...`) code → valid YAML specs:
- `Agent()` definitions with name/instructions/model/tools/handoffs/output_type
- `@function_tool` decorated functions, hosted tools (WebSearchTool, FileSearchTool, CodeInterpreterTool, etc.)
- `agent.as_tool()` orchestrator pattern (sub-agent delegation)
- `@input_guardrail`/`@output_guardrail` → policy processes with targets
- `Runner.run()`/`Runner.run_sync()` → sequential step processes + flow edges
- `asyncio.gather(Runner.run(...), ...)` → fan-out step processes
- `.handoffs.append()` circular handoff references
- Pydantic BaseModel/TypedDict/dataclass → output schemas
- `--llm-augment` flag for LLM-enhanced import
- CLI entry point: `ao-import-openai-agents`

#### 2.5 Google ADK Importer — DONE (import_google_adk.py)
AST-based parser extracts Google Agent Development Kit (`from google.adk.agents import ...`) code → valid YAML specs:
- `Agent`/`LlmAgent` → agent entity + step process with output_key logic
- `SequentialAgent` → flow edges between sub_agent steps
- `ParallelAgent` → fan-out step invoking all sub_agents
- `LoopAgent` → gate + loop edge around sub_agent steps (with max_iterations)
- Function tools → tool entity + invoke edge
- `google_search`, `exit_loop` built-in tools → tool entities
- `AgentTool(agent=...)` → invoke edge to sub-agent
- `output_key` → step logic for state data flow
- `output_schema` (Pydantic BaseModel) → schema extraction
- `Runner(agent=...)` → root agent detection
- Recursive `_flatten_agent_tree()` returns (entry, exit) tuples for proper sequential→loop→parallel wiring
- `--llm-augment` flag for LLM-enhanced import
- CLI entry point: `ao-import-google-adk`

### Phase 3: Production Code Generation
**Goal**: Generated agents that actually work with real tools and persistent memory.

#### 3.1 Real Store Backends — DONE (`--persistent-stores` flag)
`instantiate.py --persistent-stores` generates real backends: kv/blackboard → SQLite, vector → ChromaDB PersistentClient, file → JSONL I/O, queue → in-memory. Both custom and LangGraph backends supported. 22/22 agents generate + syntax check.

#### 3.2 MCP Tool Integration — DONE (`tool_type: mcp` + `tool_type: shell`)
`tool_type: mcp` generates JSON-RPC stdio client code with `MCP_SERVER_{NAME}` env var config. `tool_type: shell` generates subprocess execution. Verified on claude-code.yaml spec.

#### 3.3 DSPy Optimization Backend
Add a `dspy_optimize.py` tool that takes a generated agent + training data -> optimized prompts.

- **Input**: Generated agent file + benchmark dataset + metric
- **Output**: Optimized system prompts written back to the agent file
- **Satisfaction criteria**: ReAct agent on HotpotQA improves EM score by >10% after DSPy optimization.
- **Uncertainties**:
  - Does DSPy work with our generated code structure, or do we need to emit DSPy modules directly?
  - How to map our schemas to DSPy signatures?
- **Questions**: Should this be a new code gen backend (`--backend dspy`) or a post-processing tool?

### Phase 4: Polish & Ecosystem
**Goal**: Make it easy for others to adopt.

#### 4.1 JSON Schema for Spec Format — Moved to Phase 1.3
Machine-readable validation of YAML specs, usable by any language. Pulled forward as prerequisite for importer work.

#### 4.2 Web-based Spec Editor
Extend spec-viewer.html with editing capabilities.

#### 4.3 VS Code Extension
YAML autocompletion and inline validation for spec files.

#### 4.4 PyPI Package — DONE (pyproject.toml + agent_ontology/)
`pip install -e .` works. 18 CLI entry points (`ao-validate`, `ao-instantiate`, `ao-import-langgraph`, etc.). Optional extras: `[owl]`, `[rdf]`, `[langgraph]`, `[agents]`, `[all]`, `[dev]`. All source in `agent_ontology/` package with relative imports.

---

## Long-Term Vision: Formal Ontology & Neuro-Symbolic Reasoning

### The Gap Between What We Have and What's Possible

Our current system **describes** agent architectures (YAML schemas, string matching, structural validation). A formal ontology would let us **reason** about them (logical inference, automated classification, property verification, compositional search).

This is the difference between a database schema and a knowledge base. A schema says what shape data can be. A knowledge base draws conclusions from data.

### The Three-Layer Architecture

```
Layer 0: Reified Hypergraph Substrate
         Any node, any edge, any property. The raw graph.
         Represented in RDF/OWL. Turing-complete with rewriting rules.
              |
Layer 1: Formal Ontology (OWL/DL)
         Class hierarchy, property constraints, inference rules.
         Defines what "agent", "step", "flow" MEAN — not just their fields,
         but their logical relationships. A reasoner operates here.
              |
Layer 2: Standard Vocabulary + YAML Surface
         The 27 types users write specs with today.
         Generated FROM the ontology, not the source of truth.
         Keeps the simple YAML interface for humans.
              |
Layer 3: Tooling (validate, instantiate, visualize, analyze)
         Operates on Layer 2 YAML for simple tasks.
         Calls down to Layer 1 reasoner for complex tasks.
```

Users who don't care about DL stay at Layer 2 (YAML specs, same as today).
Power users and automated tools operate at Layer 1 (reasoning, composition, verification).
The formal ontology grows at Layer 0-1 without breaking Layer 2 compatibility.

### What Reasoning Unlocks

#### Capability 1: Automatic Pattern Classification
**Current**: `detect_patterns()` checks if process IDs match hardcoded names. Fragile, label-dependent.
**With DL**: Define patterns structurally:
```
SelfRefineAgent ≡ Agent ⊓
  ∃hasProcess.(GenerationStep ⊓ ∃flowsTo.QualityGate) ⊓
  ∃hasProcess.(QualityGate ⊓ ∃loopsTo.GenerationStep)
```
The reasoner classifies ANY spec as self-refine if it has this structure — regardless of naming.

#### Capability 2: Compositional Architecture Search
**Current**: `compose.py` combines patterns from a handwritten library. Manual, limited.
**With DL**: "Give me an architecture with retrieval AND critique AND human oversight."
The reasoner knows the interface of each pattern (inputs, outputs, entry, exit) and finds compatible compositions automatically. This is constraint satisfaction over the ontology.

#### Capability 3: Property Verification
**Current**: `validate.py` checks structural rules (references exist, types match).
**With DL**: Prove properties about the architecture:
- "Does every path eventually reach a terminal node?" (termination)
- "Can agent A's output reach agent B?" (information flow)
- "Is there a path that bypasses the safety policy?" (policy coverage)
These are graph-theoretic properties that a reasoner derives from the formal structure.

#### Capability 4: Cross-Framework Ontology Alignment
**Current**: Hand-coded importers (planned). One per framework, maintained manually.
**With DL**: Express both the Agent Ontology and LangGraph's ontology in OWL. Ontology alignment algorithms (LogMap, AML) find correspondences automatically:
- `langgraph:StateGraphNode ≡ agent_ontology:Step`
- `langgraph:ConditionalEdge ⊑ agent_ontology:Branch`
This means interop emerges from the formal definitions rather than requiring bespoke code per framework.

#### Capability 5: Learning from Execution (Neuro-Symbolic Loop)
**Current**: Benchmark scores are numbers in a report. No feedback into the ontology.
**With DL**: Feed execution traces back as assertions:
```
:react_agent_run_42 rdf:type :ExecutionTrace ;
    :achievedScore 0.85 ;
    :onBenchmark :gsm8k ;
    :usedPattern :reasoning_loop ;
    :failedOn :multi_hop_reasoning .
```
The knowledge base accumulates. The reasoner infers: "Reasoning loops achieve >80% on arithmetic but <30% on multi-hop. For multi-hop tasks, prefer architectures with retrieval AND decomposition."

An LLM can query this knowledge base when designing new agents. The symbolic system constrains and validates what the LLM proposes. This is the neuro-symbolic loop:
```
LLM generates architecture → Reasoner verifies/classifies →
Executor runs benchmarks → Results feed back as facts →
Reasoner updates recommendations → LLM uses updated knowledge → ...
```

### Phased Approach

#### Phase A: OWL Prototype (exploratory) — DONE
- Translate core ontology types to OWL — `ontology_owl.py` (owlready2)
- Define 9 patterns as DL concepts (ReasoningLoop, CritiqueCycle, RetrievalAugmented, FanOut, HumanInLoop, MultiAgentDebate, PubSub, Handoff, MemoryBacked)
- Load all 22 specs as OWL instances
- Structural pattern matching (Java/Pellet not available in current env)
- **Result**: 21/22 correct classifications. All canonical matches work (ReAct→ReasoningLoop, Self-Refine→CritiqueCycle, RAG→RetrievalAugmented, Debate→MultiAgentDebate, etc.)
- **Key finding — the ceiling of structural matching**: BabyAGI is incorrectly classified as CritiqueCycle because its `pull_task → enrich` subgraph is topologically identical to Self-Refine's `generate → critique`. The structural matcher sees two agent-invoking steps in sequence inside a loop — it can't distinguish "generate then critique" from "execute then enrich" because it doesn't understand what the steps *do*, only how they connect. This is a concrete motivating example for Phase B-C: semantic annotations (step roles, data-flow constraints) or richer DL axioms would let the reasoner distinguish these. Pure topology has a ceiling; semantics is what the formal ontology adds.

#### Phase A.5: RDF Experiment — does formalism actually help? — DONE

**Question**: Does modeling specs as RDF with reified edges and querying with schema semantics solve problems that structural matching can't?

**Answer**: YES. Three levels of CritiqueCycle detection on 22 specs:

| Level | Matches | False positives |
|-------|---------|-----------------|
| Structural (topology only) | 12 | 7 (babyagi, autogpt, plan_and_solve, etc.) |
| Semantic (+ eval field names in schemas) | 5 | 0 |
| Data-flow (+ shared schema fields) | 2 | 0 |

**Key findings**:
1. **No manual annotation needed.** The semantic signal was already in the YAML schema field names (`quality_score`, `weaknesses`, `specific_feedback` vs `result`, `context`). Reified edges make this information queryable.
2. **rdflib SPARQL is too slow** for multi-join pattern queries (>5min on 695 triples). Python graph traversal over the same RDF data model runs in 13ms. A real triplestore (Jena Fuseki) would handle SPARQL fine, but rdflib is not viable for this.
3. **The formal data model adds value.** The reified edge model (edges as first-class objects with properties including data schemas and field names) enables queries that flat graph topology cannot express.
4. **Three tiers of confidence**: structural (broad), semantic (precise), data-flow (strict). Each tier adds signal without requiring more annotation.

**Decision**: Formal data modeling is justified. The next question is infrastructure — do we invest in a triplestore (for SPARQL), or keep Python-native queries over the RDF data model?

#### Phase B: Dual Representation — DONE (owl_bridge.py)
`owl_bridge.py` — Bidirectional YAML <-> OWL conversion with lossless round-trip. 22/22 specs pass YAML→OWL→YAML round-trip with zero information loss. Round-tripped specs also pass ontology validation (22/22).

Architecture: structural model (OWL classes + object properties) enables reasoning and pattern classification (9 patterns across 22 specs). Raw data (JSON-encoded YAML dicts in data properties) enables lossless reconstruction. Both coexist on the same OWL instances.

Features: `--round-trip` (test all specs), `--classify` (pattern classification via OWL), `--export` (reconstruct YAML from OWL). Uses isolated owlready2 worlds to prevent state leakage. Extends ontology_owl.py's class hierarchy with Schema class, round-trip data properties, and order-preservation properties.

#### Phase C: Reasoning-Powered Tools
- `compose.py` replaced by constraint-based composition over OWL
- `detect_patterns()` replaced by DL classification
- New tool: `verify.py` — prove properties about architectures
- New tool: `recommend.py` — suggest architectures for tasks based on accumulated knowledge
- **Goal**: Tools that reason, not just pattern-match
- **Success criteria**: Given a new task description, the system recommends an architecture with justification based on past benchmark results

#### Phase D: Neuro-Symbolic Agent Designer
- LLM + Reasoner loop for agent design
- LLM proposes architectures; reasoner verifies them
- Execution results feed back into knowledge base
- Self-improving system that gets better at designing agents over time
- **Goal**: The system designs agents better than specgen.py does today
- **Success criteria**: Agents designed by the neuro-symbolic loop outperform LLM-only specgen on benchmarks

### Abstraction Hierarchy

The system operates at multiple levels of abstraction. Each level compresses the one below it and serves different audiences and use cases:

```
Level 4: Patterns           "CritiqueCycle"                (~3 words)
         ↕ detect/compose/mutate
Level 3: Specs              self_refine.yaml               (~100 lines)
         ↕ instantiate
Level 2: Generated Code     self_refine_agent.py           (~300 lines)
         ↕ execute
Level 1: Running System     Python process + LLM + state   (unbounded)
```

**Each level answers different questions:**

| Level | Audience | Questions it answers |
|-------|----------|---------------------|
| Pattern | Architect, evolution engine | "What family is this?" / "What if we swap the reasoning strategy?" |
| Spec | Developer, analysis tools | "Does data flow correctly?" / "Are there dead stores?" / "Which schemas connect?" |
| Generated code | Runtime, debugger | "Does it produce the right answer?" / "Where does it fail?" |
| Running system | End user | "Did I get a useful result?" |

The ontology's job is to keep projections between levels **consistent** — a pattern claiming "CritiqueCycle has an evaluator" should correspond to a spec step with evaluation-related output fields, which should generate code that actually calls an LLM that critiques.

### Multiple Viewpoints (Architecture Framework Analogy)

This multi-level structure is well-established in architecture frameworks:

- **ArchiMate** (Open Group): Formal modeling language with metamodel, multiple viewpoints (Business/Application/Technology), and aspects (Active Structure = our entities, Behavior = our processes, Passive Structure = our schemas). We're building an ArchiMate for agent architectures.
- **C4 Model**: Four zoom levels (Context → Container → Component → Code). Our pattern → spec → code → runtime is the same idea.
- **4+1 View Model** (Kruchten): Logical, Process, Development, Physical, Scenarios. We have Logical (graph), Process (runtime behavior), Scenarios (test cases).

One spec projects into multiple consistent views:

| View | Audience | What's visible | What's hidden |
|------|----------|---------------|---------------|
| Analytical | Evolution/mutation engine | Graph topology, edge types, schemas | Logic blocks, prompts, model names |
| Executable | Runtime/instantiator | Everything | Nothing |
| Expert diagram | Agent architect | Processes, data flow, schemas, edge semantics | Logic block internals |
| Newcomer diagram | Learner | "ReAct = think → act → observe loop" | Everything else |
| Formal (RDF/OWL) | Reasoner/classifier | Classes, reified edges, schema fields | Implementation details |

The formal ontology (OWL/RDF) is what guarantees these projections are **coherent** — that the newcomer diagram isn't lying about the structure and the analytical view preserves the properties that matter for evolution.

### Turing Completeness and DSL Design

**Is the spec format Turing complete?** Yes — trivially, because logic blocks contain arbitrary Python. A loop + gate + `state.data["x"] += 1` is a counter machine.

**Is the graph structure (without logic blocks) Turing complete?** No. It's a workflow net: finite process nodes, conditional routing (gates), loops (backward edges), state tokens flowing through schemas. Closer to a Petri net than a programming language.

**Is Turing completeness desirable?** No. The value of a DSL is in what it *can't* express. A Turing-complete specification language is just a general-purpose programming language with extra syntax. Our tools work precisely because specs are constrained:

- **Analyzability**: Pattern detection, similarity, topology classification work because the graph structure is finite and inspectable
- **Composability**: `compose.py` grafts patterns because interfaces (entry, exits, inputs, outputs) are well-defined. Arbitrary code doesn't compose.
- **Verifiability**: `validate.py` checks 21+ structural rules. You can't validate arbitrary programs.
- **Mutation/evolution**: `mutate.py` and `evolve.py` work because the search space is structured. Mutating arbitrary code is just fuzzing.

**Logic blocks are the escape hatch.** Every logic block is an admission that the graph structure couldn't express something, so we fell back to Python. Logic blocks are opaque to every analysis tool. Minimizing them maximizes the abstraction's value. The edge-driven-stores plan is an example: moving store access from logic blocks (opaque) to edges (analyzable).

**Are we creating abstractions simpler than what they model?** Yes — and that's the whole point. A spec captures the *essential structure* (entities, communication patterns, data flow, decisions) and hides the *accidental complexity* (API clients, JSON parsing, retry logic, state plumbing). The generated code contains both; the spec contains only the essential. This is what makes the pipeline (analyze, compose, mutate, evolve) possible.

### Why YAML + Formal Backing (Not Pure Formalism)

**Why not write specs directly in OWL/RDF?** Ergonomics. YAML is human-readable, diffable, easy to author. Nobody wants to write RDF by hand.

**Why not stay with ad-hoc YAML?** Because we've proven that formal data modeling adds analytical power. The Phase A.5 RDF experiment showed that reified edges + schema field semantics eliminate false positives that structural matching can't handle — and the semantic signal was already in the YAML, just not queryable.

**The right architecture:**

1. **Author in YAML** (Layer 2) — human-friendly, same as today
2. **Formal meta-model in OWL** (Layer 1) — class hierarchy, property constraints, inference rules. Makes ONTOLOGY.yaml machine-checkable. This is the missing piece.
3. **Export to RDF for analysis** (Layer 0) — reified edges, schema fields, SPARQL or Python graph queries. Already prototyped in `ontology_rdf.py`.

The YAML format is not "ad-hoc" — it's a well-typed DSL backed by a formal ontology. The formalism lives underneath, not in front of the user.

### Uncertainties

1. **OWL expressiveness vs complexity**: OWL-DL is decidable but limited. OWL-Full is more expressive but undecidable. Which fragment do we need?
2. **Tooling maturity**: OWL reasoners (HermiT, Pellet) are mature but not actively developed. Python bindings (owlready2) work but are not production-grade.
3. **Performance**: Reasoning over 22 specs is fast. Over 10,000 specs? Unknown.
4. **User adoption**: Will developers use OWL-powered tools, or is this too academic?
5. **The right DL fragment**: We need enough expressiveness for pattern definitions but decidability for verification. This is a research question.
6. **Logic block minimization**: How much computation can we push from logic blocks into declarative graph structure? `query_key` on read edges is one example. Schema mapping declarations could be another. Where's the diminishing-returns point?

### Why This Matters Beyond Agent Architectures

If this works — formal ontology + LLM + symbolic reasoning for agent design — it's a template for any domain where you want to combine neural creativity with logical rigor. Software architecture, business process design, drug discovery pipeline design, circuit design. The agent ontology is the proving ground.

---

## Open Questions

1. **Naming**: Renamed to "Agent Ontology" to convey "agent architecture standard."
2. **Versioning**: When do we declare v1.0? After the formal spec? After the first importer?
3. **Community**: Should we write a paper? Post on HN? Create a Discord?
4. **Governance**: If this becomes a standard, who maintains it? Just you, or a broader group?
5. **Licensing**: Current repo is public on GitHub. MIT? Apache 2.0?
6. **Ontology additions vs stability**: Every addition to the ontology breaks backward compatibility for importers/exporters. When do we freeze?

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-09 | MCP discovery is NOT an ontology gap | Runtime concern — `tool_type: mcp` already captures the architectural fact |
| 2026-02-09 | DSPy optimization is NOT an ontology concern | Meta-process on specs, not part of agent architecture |
| 2026-02-09 | Multimodal types deferred | No current spec needs them; LLM APIs handle multimodal transparently |
| 2026-02-09 | Memory consolidation is a spec pattern, not ontology gap | Expressible as a maintenance flow with existing types |
| 2026-02-09 | Ontology assessed as substantially complete for v1.0 | Evaluated against 10 frameworks, no structural gaps found |
| 2026-02-09 | Pursue formal OWL ontology as long-term path | Reasoning > pattern matching. Unlocks composition, verification, cross-framework alignment, neuro-symbolic loop |
| 2026-02-09 | YAML stays as human authoring format | OWL is the formal layer underneath. Users don't need to know about DL. |
| 2026-02-09 | Turing completeness is NOT a goal | DSL value comes from constraint, not power. Logic blocks are an escape hatch, not a feature. Minimize them. |
| 2026-02-09 | Three-layer authoring model: YAML + OWL + RDF | Author in YAML (ergonomic), meta-model in OWL (formal), export to RDF (queryable). Each layer serves different tools. |
| 2026-02-09 | Abstraction hierarchy is the core value proposition | Patterns compress specs, specs compress code, code compresses runtime. Each level enables different analysis. |
| 2026-02-09 | Architecture framework analogy (ArchiMate/C4) is relevant | Multiple viewpoints on one model is a solved problem in enterprise architecture. Learn from it. |
| 2026-02-09 | Phase 1.1 (SPEC.md) complete | ~945 lines, cross-validated against all 22 specs, 3 gaps found and fixed |
| 2026-02-09 | 5 bugs fixed from Codex audit | trace format mismatch, migrate loop→back_edge, fan-out transitions, crew output, quote escaping. 174/174 property tests pass. |
| 2026-02-09 | JSON Schema (Phase 1.3) before LangGraph importer | Quick win that enables tooling; importer is highest leverage but larger effort |
| 2026-02-09 | OWL dual representation (Phase B) complete | 22/22 lossless round-trip. Structural model for reasoning, JSON data properties for reconstruction. Isolated worlds prevent state leakage. |
| 2026-02-10 | Rename OpenClaw → Agent Ontology | 79 files, 277 replacements. Env vars: AGENT_ONTOLOGY_*. OWL URI: agent-ontology.org. All 22 specs validate, 174/174 properties pass, 22/22 OWL round-trip. |
| 2026-02-10 | AutoGen importer (Phase 2.3) complete | AST-based, supports v0.2 + v0.4 APIs, GroupChat to team+loop, tool registration patterns. Validates clean on both test files. |
| 2026-02-10 | Benchmark evolution fix | Rewrote compute_fitness_benchmark to run inline instead of subprocess. All candidates now score properly (was: all BENCH_FAIL). |
| 2026-02-10 | Self-improver spec added | 23rd spec: self-improvement loop with meta-review, strategy adjustment, re-execution, and convergence gate. |
| 2026-02-10 | Spec-level state schema support | instantiate.py now honors state field: initial values, channel reducers as Annotated in LangGraph TypedDict. 182/182 property tests pass. |
| 2026-02-10 | OpenAI Agents SDK importer (Phase 2.4) complete | AST-based, handles handoffs, guardrails→policy processes, agent.as_tool(), asyncio.gather fan-out, output schemas. 0 errors on all test files. |
| 2026-02-10 | Google ADK importer (Phase 2.5) complete | AST-based, handles Agent/SequentialAgent/ParallelAgent/LoopAgent composition, recursive _flatten_agent_tree with (entry,exit) tuples. 15/15 round-trip tests pass. |
| 2026-02-10 | All 5 major framework importers complete | LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, Google ADK. The "Rosetta Stone" vision is realized for import. |
