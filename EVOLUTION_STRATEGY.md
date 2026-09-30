# Evolution Strategy v1

**The one approach we're building. Alternatives documented at the bottom, deferred.**

---

## Why Specs Beat Code as a Search Medium

The spec isn't just a validation layer — it's the **language the LLM reasons through** at every stage of the loop. This is the core advantage over code-level search (ADAS, AlphaEvolve, EvoAgentX):

**Legibility.** The LLM reads a 100-line typed YAML spec, not 300 lines of Python with state plumbing. It sees "CritiqueCycle feeding into a ReasoningLoop" — the architecture is transparent at the right level of abstraction.

**Diagnosis.** When a candidate fails, the LLM gets structured errors pointing at spec components: "L005: schema mismatch between `execute_action` and `record_step`", "V001: no termination path from `refine_loop`", "benchmark: 0% — output field `answer` was empty." Not "line 147: KeyError 'answer'."

**Reasoning.** The LLM thinks in the abstraction: "the CritiqueCycle has no termination gate, so it loops forever — add a quality_gate with `score >= 7` after the evaluate step." It reasons about architecture, not code.

**Editing.** Surgical edits to a typed graph: "add this process, add this edge, modify this gate condition." Not rewriting functions and hoping the indentation is right.

**Accumulation.** The knowledge store accumulates **genotype-level** facts: "CritiqueCycle + RetrievalAugmented scores 85% on multi-hop QA." "Adding a debate step before final output improved ambiguous-task accuracy by 12%." Over time, a structured map of what architectures work where — something impossible to extract from raw code variants.

```
Code-level search:  LLM sees code → crashes → stack trace → guesses fix → tries again
Spec-level search:  LLM sees architecture → structured errors → reasons about fix → edits typed graph
                    + accumulates genotype knowledge over time
```

---

## The Loop

```
┌─────────────────────────────────────────────────────┐
│  1. GENERATE (gemini-3-flash, broad)                │
│     - N mutations from top-K parents                │
│     - All LLM-guided: Flash sees spec + errors +    │
│       analysis from previous generation             │
│     - Flash proposes structural changes in YAML     │
│                                                      │
│  2. VALIDATE (free, no LLM cost)                    │
│     - 23 structural rules (validate.py)             │
│     - 10 lint checks (lint.py)                      │
│     - Schema flow verification (verify.py V005)     │
│     - Reject invalid candidates before benchmark    │
│                                                      │
│  3. EVALUATE (benchmark.py, main cost)              │
│     - Instantiate surviving specs to Python          │
│     - Run on benchmark dataset (e.g. GSM8K)         │
│     - Score: accuracy + efficiency (calls, time)     │
│     - Collect structured error info for failures     │
│                                                      │
│  4. ANALYZE (gpt-5-mini, deep, top-K only)          │
│     - Sees: parent spec, mutated spec, diff,         │
│       validation results, benchmark scores,          │
│       error messages, detected patterns              │
│     - Explains: why top candidates scored well,      │
│       why promising mutations failed                 │
│     - Proposes: targeted mutations for next gen      │
│                                                      │
│  5. STORE (knowledge store, persistent)              │
│     - Spec genotype (patterns, topology features)    │
│     - Benchmark phenotype (scores, failures)         │
│     - Mini's analysis (lessons, insights)            │
│     - Lineage (parents, generation, mutations)       │
│                                                      │
│  6. SELECT parents for next generation               │
│     - Top-K by fitness                               │
│     - Feed Mini's analysis + knowledge store         │
│       context into next generation's Flash prompts   │
│     - Loop back to step 1                            │
└─────────────────────────────────────────────────────┘
```

## Model Roles

| Role | Model | What it sees | What it produces |
|------|-------|-------------|-----------------|
| Mutation generator | gemini-3-flash | Parent spec + lint/verify errors + previous analysis + knowledge store context | N mutated YAML specs |
| Agent execution | gemini-3-flash | Benchmark inputs (the agents themselves run on Flash) | Agent outputs for scoring |
| Deep analyst | gpt-5-mini | Top-K specs + diffs + benchmark scores + error details + detected patterns | Diagnosis, lessons learned, targeted mutation suggestions |

## Mutation Approach (v2: Progressive Disclosure)

### The Problem with v1 (Menu Selection)
v1 gave Flash a menu of ~150 pre-defined mutations and asked it to pick 1-3. This was safe (every mutation goes through battle-tested operators) but limited:
- Flash always picked the same combo (add_chain_of_thought + insert_pattern)
- It couldn't *generate* new fixes — only select from existing ones
- Diagnosis and prescription were smashed into one prompt
- Domain-specific fixes (e.g., "output letters not numbers") had to be hand-coded

### v2: Multi-Call Pipeline (Diagnose → Prescribe → Apply)
Each LLM-guided mutation uses 2-3 focused calls instead of 1 overloaded call:

**Call 1 — Diagnose (Flash, cheap):**
Input: Failure details (5 failed examples with expected vs predicted, error types)
Output: A 1-2 sentence diagnosis of the failure pattern
Example: "The agent outputs numeric indices (1, 2, 3) instead of choice letters (A, B, C). This is a format mismatch between the agent's output and the expected answer format."

**Call 2 — Prescribe (Flash, cheap):**
Input: The diagnosis + the parent spec + the enumerated mutation menu
Output: Either (a) a menu selection, or (b) a freeform prompt rewrite with specific text
Example: `{"action": "rewrite_prompt", "agent": "solver", "new_prompt": "...Answer with ONLY the letter A, B, C, or D..."}`

This is the key difference: the LLM can now *generate* prompt text it hasn't seen before, informed by the specific failure pattern. It's not limited to pre-coded transforms.

**Call 3 — Analyze (Mini, after generation):**
Same as v1: Mini analyzes top-K candidates, explains why the best worked, proposes next-generation mutations. This already exists.

### When Progressive Disclosure is Used
- When `benchmark_results` has `failure_summary` (i.e., there are failures to diagnose) → use progressive pipeline
- When no failure data exists (first generation, or parent scored 100%) → fall back to menu selection

### Freeform Prompt Rewrite
The prescription call can output a new action type: `rewrite_prompt`. This bypasses the pre-coded `_PROMPT_TRANSFORMS` menu and lets Flash write arbitrary prompt text. The rewritten prompt:
- Replaces the agent's `system_prompt` entirely (not append-only)
- Must preserve the output format instructions (JSON schema)
- Is validated by re-running the full pipeline (validate → instantiate → benchmark)

### Programmatic Diversity (20%)
Unchanged from v1: ~20% of mutations use random `mutate.py` operators (crossover, swap_process_order, change_model, etc.) to explore parts of the space the LLM wouldn't think of. `insert_pattern` is blacklisted (consistently destroys accuracy, avg fitness 37.5).

### Why This is Better Than a DSL
We considered encoding algorithmic choices as typed spec fields (e.g., `reasoning_strategy: chain_of_thought` vs `tree_search`). This would make mutations more structured but requires designing a complete DSL for agent behavior — a massive effort.

The progressive disclosure approach gets 80% of the benefit:
- The LLM reasons about what's wrong (diagnosis) at the semantic level
- The LLM generates domain-specific fixes (prescription) using natural language
- Validation ensures structural correctness regardless of what the LLM generates
- No new ontology types or fields needed

## Selection

**Simple top-K.** Take the top K candidates by fitness. K = max(2, population // 2).

Fitness v2 = accuracy² × 200 (max 200, quadratic rewards perfection) + call_efficiency (max 15) + speed (max 5).
Going from 96%→100% accuracy is worth +15.7 fitness points, which outweighs any efficiency penalty from added reasoning steps.

## Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Population per generation | 12 | Enough diversity, affordable |
| Generations | 5 | Enough to see improvement |
| Parents (K) | 4 | Keeps diversity |
| Benchmark examples | 10 | Balance noise vs cost |
| Benchmark timeout per agent | 120s | Generous for complex agents |

## Knowledge Store Schema (Step 1)

SQLite with one table:

```sql
CREATE TABLE evolution_results (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    -- Genotype (what the agent IS)
    spec_name TEXT,
    spec_yaml TEXT,
    base_spec TEXT,
    detected_patterns TEXT,  -- JSON list: ["critique_cycle", "retrieval"]
    topology_hash TEXT,      -- Hash of graph structure for dedup
    entity_count INTEGER,
    process_count INTEGER,
    edge_count INTEGER,
    -- Lineage (where it came from)
    generation INTEGER,
    parents TEXT,            -- JSON list
    mutation_description TEXT,-- What Flash proposed
    -- Phenotype (how it performed)
    benchmark TEXT,
    score_em REAL,
    score_f1 REAL,
    fitness REAL,
    llm_calls INTEGER,
    duration_ms INTEGER,
    status TEXT,             -- PASS, ERROR, TIMEOUT, INVALID
    error_details TEXT,      -- Structured errors if failed
    -- Analysis (from gpt-5-mini)
    analysis TEXT,           -- Why this scored how it did
    lessons TEXT             -- Reusable insights for future generations
);
```

Query examples:
- "Best genotypes for GSM8K" → `SELECT detected_patterns, AVG(fitness) FROM ... WHERE benchmark='gsm8k' GROUP BY detected_patterns ORDER BY AVG(fitness) DESC`
- "What mutations improved self_refine?" → `SELECT mutation_description, fitness FROM ... WHERE base_spec='self_refine' AND fitness > 100 ORDER BY fitness DESC`
- "Lessons from failed candidates" → `SELECT lessons FROM ... WHERE status='ERROR' AND benchmark='gsm8k' ORDER BY timestamp DESC LIMIT 10`

---

## Deferred Alternatives (Not Building Now)

### Search strategy alternatives
- **MCTS (AFlow style)**: Tree search over operator combinations. Smarter than random but requires defining an operator vocabulary. Defer until LLM-guided proves insufficient.
- **MAP-Elites (LoongFlow style)**: Quality-diversity archive maintaining best specimen per niche. Defer until premature convergence becomes a measured problem.
- **TextGrad (EvoAgentX)**: Gradient-based prompt optimization via natural language feedback. Defer — orthogonal to architecture search.
- **Bayesian optimization (MIPRO)**: Surrogate model over prompt space. Defer — useful for prompt tuning after architecture is fixed.

### Selection alternatives
- **Niching (EvoFlow)**: Diversity-preserving selection using topology distance. Simple to add later if top-K converges too fast.
- **Boltzmann selection (LoongFlow)**: Temperature-based probabilistic selection. Prevents greedy convergence.
- **Pareto selection**: Multi-objective (accuracy vs efficiency vs complexity). Add when we have enough signal to optimize multiple objectives.

### Scope expansion
- **Joint prompt + topology optimization**: Mutate system_prompt fields alongside graph structure. Add after the loop works on topology alone.
- **Tool selection optimization**: Mutate which tools are wired to which steps. Add after basic loop proves out.
- **Plumbing-safe programmatic mutations**: Harden mutate.py so pattern swaps/inserts auto-rewire schemas. Worth doing if programmatic diversity proves valuable.
