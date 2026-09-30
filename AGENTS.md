# Agent Ontology

## What This Is
A formal representation of agent architecture: YAML specs that are machine-readable, LLM-legible, evolvable, and scale-invariant. The toolchain validates, generates, analyzes, and evolves agent architectures.

## Long-Term Vision: AlphaEvolve for Agent Architectures

The north star is a **self-improving system that designs agents better over time** — an AlphaEvolve-style loop over agent architectures instead of algorithms:

```
Flash diagnoses failures and prescribes targeted mutations (progressive disclosure)
    → Automated evaluator scores each (benchmark + verify + early stop)
    → Pro analyzes top candidates deeply (diagnosis + refinement)
    → Results feed back as persistent knowledge
    → Next iteration uses accumulated evidence + failure details
    → Repeat until convergence
```

This is Phase D in ROADMAP.md. Everything else is infrastructure serving this goal.

### Why Specs Beat Code as a Search Space
AlphaEvolve searches over raw code (hundreds of lines, unstructured). We search over typed graphs (~100 lines of validated YAML). Our advantage:
- **Mutations are semantic**: swap a pattern, add a review step, change a gate — not random text edits
- **Validation is instant**: every candidate is structurally checked before expensive LLM evaluation
- **The search space is navigable**: patterns, topology, similarity all work because specs are constrained
- **Knowledge is structured**: "CritiqueCycle + RetrievalAugmented scores 85% on multi-hop" is a queryable fact, not a number in stdout

### What Exists Today
| Component | Tool | Status |
|-----------|------|--------|
| Broad mutation | `mutate.py` (field + pattern level, insert_pattern blacklisted) | Working |
| LLM-guided mutation | `evolve.py --llm-guided` (structured menu selection) | Working |
| Progressive disclosure | `evolve.py` (diagnose → prescribe via edit_prompt/menu) | Working |
| Deep analysis | `evolve.py` Mini (gpt-5-mini) analyzes top-K per generation | Working |
| Population evolution | `evolve.py` (selection, crossover, lineage) | Working |
| Knowledge store | `knowledge_store.py` (SQLite, persistent across runs) | Working |
| Benchmark fitness | `benchmark.py` + 9 datasets (incl. multidoc, kb_tool, meta_improve) | Working |
| Meta-evolution | `mutator.yaml` agent + `meta_improve` benchmark (nested fitness) | Working |
| Error analysis | Per-example failure tracking, failure_summary in LLM context | Working |
| Early stop | Skip remaining examples after N failures (saves ~40% budget) | Working |
| Multi-run eval | `--eval-runs N` averages fitness across runs | Working |
| Benchmark transforms | Domain-specific prompt fixes (ARC letters, GSM8K traps) | Working |
| Structural verification | `verify.py` (9 checks), `lint.py` (10 rules) | Working |
| OWL reasoning | `owl_bridge.py` (round-trip + pattern classification) | Working |

### What's Actually Established

Read this section before citing any number from this repository.

**The representation works.** 37 specs validate against the ontology. Both code
generators produce importable Python for all 37. 294/294 structural property
checks pass. `tests/test_cli_surface.py` runs every offline command the README
advertises. This part is verified and reproducible from a clean clone.

**The evolution results are not established, and were previously written up here
as though they were.** Every benchmark holds 25 examples (meta_improve holds 5),
so four percentage points is one question. The reported wins were each one
question, measured once:

| Benchmark | Reported | In questions |
|---|---|---|
| GSM8K-Tricky | 96% → 100% EM | 1 of 25 |
| ARC | 32% → 36% EM | 1 of 25 |
| MultiDoc | 72% → 76% EM | 1 of 25 |
| KB-Tool | 88% → 92% EM | 1 of 25 |

Worse, `evolution_results.json` re-scores survivors each generation, so it
contains the same unchanged genotype scored three times: `gen1_base_mut2` at
166.5, then 168.3, then 164.6. Two other survivors drift comparably (162.7 →
159.6; 165.1 → 161.5). Re-running an identical spec moves the score by up to
3.7. The improvement celebrated in that same file is 160.1 → 168.3, or 8.2.

The effect being measured is roughly twice the noise floor, from one
measurement each. Downstream artifacts inherit this: the 75-candidate knowledge
store, the ranking of mutation operators, and the blacklisting of
`insert_pattern` on a bad average. `insert_pattern` may genuinely be bad; the
data as recorded cannot establish it.

Note also that one candidate stayed best across all three generations — after
generation 1 the search found nothing. That was previously read as convergence
and recorded below as weakness #6, "search stagnation." Regression toward the
mean is the simpler explanation.

**Before running evolution again**, establish the noise floor: score one fixed
spec with `--eval-runs 10`, take the standard deviation, and treat anything
smaller as the instrument rather than a result. That flag has existed the whole
time and none of the numbers above used it.

### Known Weaknesses (and fixes)
1. ~~**Pattern insertion breaks task-specific agents**~~ FIXED: `insert_pattern` blacklisted (Session 24).
2. ~~**LLM-guided mutation selection is too narrow**~~ FIXED: Progressive disclosure pipeline (diagnose → prescribe) with `edit_prompt` operator for surgical find-and-replace on prompts. Menu selection preferred when pre-validated transforms match.
3. ~~**Prompt transforms can contradict**~~ FIXED: `add_chain_of_thought` strips conflicting instructions before appending.
4. ~~**Near-saturated accuracy limits evolution**~~ FIXED: Custom multi-doc benchmark has genuine headroom (baseline 72% EM, structured 76% EM, hard questions consistently wrong).
5. ~~**Freeform prompt rewrite over-corrects**~~ FIXED: Replaced `rewrite_prompt` with `edit_prompt` (surgical find-and-replace). LLM provides `old_text` and `new_text` for targeted edits. Verified working in progressive disclosure pipeline.
6. ~~**Search stagnation in later generations**~~ SUPERSEDED: this was read as greedy selection narrowing to a local optimum. See "What's Actually Established" — regression toward the mean after generation 1 crowns the luckiest roll explains it without any stagnation to fix.
7. ~~**No error analysis**~~ FIXED: Per-example failure details + failure_summary fed to LLM mutations (Session 24).
8. ~~**Prescriptive mutation prompts**~~ FIXED (Session 26): Removed "Key insights" prescriptive guidance from mutation selection prompt. Now context-driven: provides goal + failure data, lets LLM reason about what to fix. Base spec pre-evaluated before gen 1 so failure data exists from the start.
9. **Several process types have no generated runtime** — `spawn`, `formal_translation`, `symbolic_inference`, `search`, `protocol` and `error_handler` are emitted as `raise NotImplementedError(...)` naming the agent and process (they used to print a placeholder or do nothing and let the run report success). `policy` nodes are pass-through and print `NOT ENFORCED`, recording themselves in `_unenforced_policies`. Agents whose path reaches an unimplemented process (currently alpha_geometry, kg_rag, map_reduce, meta_prompting, pddl_planner, sota_superset_runtime) cannot be run or scored; `instantiate.unimplemented_reachable_processes(spec)` lists them, and a test keeps them out of the benchmark lists. Only `knowledge_graph` stores have real runtime (networkx).
10. **Mutation selector could be a full agent** — Currently a single LLM call with a menu. Could be an agent itself (with tool use for spec analysis, memory for mutation history, multi-step reasoning). Deferred — current pipeline works, and making the selector an agent would be a significant architecture change. The key principle is: provide context, not prescriptions. The LLM should reason from failure data, not follow hardcoded heuristics.

### Strategic Direction (Session 24)

**Problem**: Existing benchmarks (GSM8K, ARC) don't differentiate architectures — a single LLM call saturates performance. Evolution works but has little headroom to show improvement.

**Solution — Two parallel tracks**:

**Track 1: Progressive disclosure mutations** — Replace single-shot menu selection with multi-call pipeline:
1. **Diagnose** (Flash): "Here are 5 failed examples. What pattern explains the failures?"
2. **Prescribe** (Flash): "Given that diagnosis, here's the spec. Write a specific fix."
3. **Analyze** (Mini): "Here are the top candidates. Why did the best one work?"

This lets the LLM *generate* fixes from error patterns (e.g., "outputs numbers instead of letters" → add format instruction) instead of picking from a pre-coded menu.

**Track 2: Custom multi-doc reasoning benchmark** — A benchmark where architecture genuinely matters:
- 25 questions requiring cross-referencing 3-5 fact cards
- Trap types: contradictions, misleading details, arithmetic aggregation, negation, temporal reasoning
- A single LLM call can't reliably handle cross-reference + trap combinations
- A retrieve → verify → reason → check architecture should do measurably better
- This creates genuine headroom for evolution to search within

### Model Choices (During Development)
- **Broad exploration**: `gemini-3-flash-preview` — cheap, fast, generates many candidates
- **Deep analysis**: `gpt-5-mini` — higher quality reasoning for top-K refinement
- Upgrade to Pro-tier models once the loop is proven and we're optimizing for quality

### Roadmap to Phase D (Ordered Steps)

**Steps 0-6**: DONE. See git history.

**Step 7: Evolution efficiency** — DONE (Session 24). Blacklisted `insert_pattern`, added early stop (skip after 3 failures), per-example error analysis with failure_summary, multi-run evaluation (`--eval-runs`), benchmark-specific prompt transforms.

**Step 8: Progressive disclosure mutations** — DONE. Multi-call pipeline: diagnose failures → prescribe fix via `edit_prompt` (surgical find-and-replace) or menu selection → apply. Falls back to menu selection when no failure data exists.

**Step 9: Architecture-sensitive benchmark** — DONE. Custom multi-doc reasoning benchmark (25 questions) where multi-step architecture outperforms single-shot. Baseline 72% EM, structured 76% EM.

**Step 10: Multi-tool benchmark** — DONE. KB-Tool benchmark (25 questions over fictional knowledge base, 2-4 chained tool calls). All entities fictional → 0% without tools. Monkey-patches tool functions at runtime. `kb_react.yaml` spec, `kb_tools.py` module, `score_kb_tool` scoring.

**Step 11: Compelling experiment** — DONE (Session 26). Evolution on kb_tool (25 examples, 2 gen, pop 4): base 161.4 fitness (88% EM) → best evolved 176.2 (+9%). Progressive disclosure diagnosed `format_mismatch` from failure data, prescribed surgical prompt edit. Context-driven prompts (no prescriptive guidance) were the key fix — previous run with prescriptive prompts found no improvement.

**Step 12: Meta-evolution** — DONE (Session 27). The mutator itself is an agent spec (`mutator.yaml`) that can be evolved. `meta_improve` benchmark evaluates mutator quality via nested fitness: run mutator on target spec → validate output → benchmark improved spec → compare to baseline. 5 improvement tasks across kb_tool, gsm8k_tricky, hotpotqa, gsm8k, multidoc. Base mutator: 60% EM (3/5 tasks improved). Evolution run (2 gen, pop 4): base tied at 102.4 fitness — mutations were too shallow to improve the already-effective diagnose→edit architecture.

### What NOT to Prioritize
- **More framework exporters** — 5 importers + 2 code gen backends prove the interchange story
- **Polish/ecosystem** (VS Code extension, web editor) — doesn't advance the core vision
- **DSPy optimization** — prompt optimization is orthogonal to architecture search
- **Richer ontology / DSL for process logic** — tempting but massive effort; lean into LLM-driven mutation instead

## Project Structure
```
agent_ontology/          Python package (39 modules, pip install -e .)
  ONTOLOGY.yaml          Entity/process/edge type definitions (v0.2+, 9 entity types, 10 process types, 13 edge types, verify edge added)
  specs/                 37 agent spec YAMLs (36 runnable + 1 description-only)
  instantiate.py         Spec → Python agent code (custom + LangGraph backends)
  validate.py            23 structural validation rules
  design.py              End-to-end agent design pipeline (ao-design)
  evolve.py              Evolutionary search over architectures
  mutate.py              Mutation operators (field-level + pattern-level)
  ingest.py              Corpus ingestion for ChromaDB vector stores (ao-ingest)
  knowledge_store.py     SQLite store for evolution results (ao-knowledge-store)
  recommend.py           Evidence-backed architecture recommender (ao-recommend)
  owl_bridge.py          Bidirectional YAML↔OWL with DL pattern classification
  import_*.py            5 framework importers (LangGraph, CrewAI, AutoGen, OpenAI Agents, Google ADK)
  benchmarks/            9 datasets (GSM8K, GSM8K-Hard, GSM8K-Tricky, HotpotQA, ARC, HumanEval, MultiDoc, KB-Tool, Meta-Improve) + scoring
  ...                    lint, verify, patterns, compose, benchmark, specgen, etc.
agents/                  37 generated custom-backend agents
agents_lg/               37 generated LangGraph-backend agents
tests/                   test_agents.py (26 E2E), test_properties.py (294), test_cli_surface.py, test_roundtrip.py
```

## Key Commands
```bash
export $(cat .env.local | xargs)                          # Load API keys
python3 agent_ontology/validate.py agent_ontology/specs/*.yaml  # Validate all specs
python3 agent_ontology/instantiate.py SPEC -o agents/X.py       # Generate agent
python3 agent_ontology/instantiate.py --all agent_ontology/specs/ -o agents/  # Batch regen
python3 tests/test_agents.py --agent react --timeout 120        # Test one agent
python3 tests/test_agents.py --timeout 120                      # Test all runnable agents
python3 tests/test_properties.py                                # 294 property tests
PYTHONPATH=. python3 -m agent_ontology.v2.api_server --host 127.0.0.1 --port 8012 --root .  # Clean-room v2 server
# Open http://127.0.0.1:8012/agent_ontology/v2/spec-viewer-v2.html
# v2 session APIs: /api/v2/session/new, /api/v2/session/<id>, /api/v2/session/<id>/turn
# v2 file API is snapshot-at-cursor first: /api/v2/file/<run_id>?path=...&source=snapshot&at_event_index=...
# v2 prune API: POST /api/v2/prune/<run_id> (body: {"force": true|false})
# v2 snapshot retention env: AGENT_ONTOLOGY_V2_SNAPSHOT_KEEP_EVENTS, AGENT_ONTOLOGY_V2_SNAPSHOT_KEEP_VERSIONS_PER_FILE
python3 -m agent_ontology.evolve specs/X.yaml -g 3 -p 5 --benchmark multidoc --llm-guided  # Evolution
python3 -m agent_ontology.evolve specs/X.yaml --eval-runs 3 --benchmark gsm8k  # Multi-run eval
python3 -m agent_ontology.knowledge_store stats              # Query knowledge store
python3 -m agent_ontology.knowledge_store best gsm8k         # Best genotypes for benchmark
python3 -m agent_ontology.recommend "multi-hop QA" --use-knowledge-store  # Get recommendations
```

## Design Principles
1. **Fail Loud** — No silent fallbacks
2. **Simplest thing that works** — Don't overengineer
3. **Delete > Comment** — Remove unused code
4. **Flat > Nested** — Prefer flat structures
5. **Search > Design** — Prefer evolving architectures over hand-designing them
6. **LLM > DSL** — Let LLMs generate fixes from errors rather than hand-coding every transform

## Key Technical Details
- State model: `state.data` dict with flat keys
- Gate conditions: parsed by `_generate_gate_check()` in instantiate.py
- Env var prefix: `AGENT_ONTOLOGY_` (MODEL, MAX_ITER)
- LLM routing: `call_llm()` routes by model prefix (claude→Anthropic, gemini→Google, else→OpenAI)
- Generated agents go to `agents/` (custom) and `agents_lg/` (LangGraph)
- All tool source is in the `agent_ontology/` package — use relative imports within
- Evolution improvements (Session 24): `insert_pattern` blacklisted, early stop after 3 failures, per-example error analysis, multi-run eval, benchmark-specific prompt transforms


## Multi-Agent Coordination

This repo uses worktree-based isolation for concurrent AI instances.

**Before starting work:**
1. Check existing claims: `python scripts/meta/worktree-coordination/check_claims.py --list`
2. Claim your work if needed to inspect scope conflicts: `python scripts/meta/worktree-coordination/check_claims.py --claim --feature <name> --task "description"`
3. Create a worktree through the sanctioned entrypoint: `make worktree BRANCH=plan-N-desc TASK="description" [PLAN=N]`
4. Work in the worktree, not the main directory

**Before committing:**
- Commits must use prefixes: `[Plan #N]`, `[Trivial]`, or `[Unplanned]`
- Release claims when done: `python scripts/meta/worktree-coordination/check_claims.py --release`

**Check for messages from other instances:**
`python scripts/meta/worktree-coordination/check_messages.py`

## Commands
```bash
export $(cat .env.local | xargs)
python3 agent_ontology/validate.py agent_ontology/specs/*.yaml
python3 tests/test_agents.py --timeout 120
python3 tests/test_properties.py
make worktree BRANCH=plan-42-feature TASK="Describe the task" PLAN=42
make worktree-list
make worktree-remove BRANCH=plan-42-feature
```

## Workflow
1. Validate scope and success criteria in the relevant plan doc before editing code.
2. Claim and create work through the sanctioned worktree path, not raw `git worktree add`.
3. Regenerate agents or runtimes only after the schema/spec change is validated.
4. Run focused verification first, then broader property or benchmark coverage for behavioral changes.
5. Commit verified increments with the required prefix and release claims when the branch is done.

## References
- `docs/plans/AGENTS.md` — active plan index and execution status
- `docs/meta-patterns/worktree-coordination/README.md` — local mirror path for coordination docs
- `scripts/relationships.yaml` — machine-readable governance and coupling graph
