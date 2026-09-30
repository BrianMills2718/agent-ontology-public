> **Public snapshot.** This repository is a single-commit snapshot of a private working repository (development history is private). Specs and generated agents derived from third-party proprietary code were removed, so a few catalog counts below may be off by a few rows.

# Agent Ontology

**OpenAPI for Agents** — a universal format for describing AI agent architectures.

```
docs  -->  spec  -->  { validate, visualize, instantiate, analyze }
```

Given any agent's documentation, Agent Ontology lets you:
1. **Decompose** it into a structured YAML spec
2. **Validate** the spec against a formal ontology (structural + graph analysis)
3. **Visualize** it as an interactive flowchart
4. **Instantiate** it as a runnable Python agent with real LLM calls
5. **Analyze** complexity, traces, and runtime metrics
6. **Test** agents automatically with canned inputs and validators

Nobody does all six today. This is the gap.

## Quick Start

```bash
# Install as a package
pip install -e .

# 1. Validate a spec
ao-validate agent_ontology/specs/react.yaml

# 2. Generate a runnable agent (custom backend)
ao-instantiate agent_ontology/specs/react.yaml -o agents/react_agent.py

# 3. Generate a LangGraph agent
ao-instantiate agent_ontology/specs/react.yaml --backend langgraph -o agents_lg/react_agent_lg.py

# 4. Batch-generate all agents (both backends)
ao-instantiate --all agent_ontology/specs/ -o agents/
ao-instantiate --all agent_ontology/specs/ -o agents_lg/ --backend langgraph

# 5. Run it (requires API keys in .env.local)
export $(cat .env.local | xargs)
python3 agents/react_agent.py

# 6. View specs in the browser
python3 -m http.server 8000
# Open http://localhost:8000/agent_ontology/spec-viewer.html

# 7. Analyze the trace
ao-trace trace.json

# 8. Score spec complexity
ao-complexity --all agent_ontology/specs/

# 9. Run all agent tests (Claude Code is description-only)
python3 tests/test_agents.py

# 10. Override model for all agents
python3 tests/test_agents.py --model gpt-4o --agent react

# 11. Compare models side-by-side
python3 tests/test_agents.py --compare-models gemini-3-flash-preview gpt-4o-mini gpt-4o

# 12. Mutate and evolve a spec
ao-mutate agent_ontology/specs/react.yaml --random -n 3
ao-evolve agent_ontology/specs/react.yaml --generations 2 --population 4

# 13. Evolve with crossover and benchmark fitness
ao-evolve agent_ontology/specs/react.yaml --generations 3 --population 6 --crossover --benchmark gsm8k

# 14. E2E specgen pipeline test
python3 tests/test_specgen.py --fix

# 15. Analyze spec coverage of ontology features
ao-coverage --all agent_ontology/specs/

# 16. Lint specs for anti-patterns
ao-lint --all agent_ontology/specs/ --severity warn

# 17. Classify agent topologies
ao-topology --all agent_ontology/specs/

# 18. Compare two spec versions
ao-spec-diff agent_ontology/specs/react.yaml agent_ontology/specs/autogpt.yaml

# 19. Project health dashboard
ao-dashboard agent_ontology/specs/

# 20. Export Mermaid flowcharts
ao-mermaid agent_ontology/specs/react.yaml

# 21. Spec similarity and clustering
ao-similarity --all agent_ontology/specs/ --top 5 --clusters 5

# 22. Run property-based tests (no API keys needed, 294 tests)
python3 tests/test_properties.py

# 23. Generate comparative analysis report
ao-report --all agent_ontology/specs/

# 24. Compose patterns into new agents
ao-compose compose_specs/react_refine.yaml -o agent_ontology/specs/react_refine.yaml --validate

# 25. Import a LangGraph agent into an Agent Ontology spec
ao-import-langgraph agents_lg/react_agent_lg.py -o agent_ontology/specs/imported_react.yaml --validate

# 26. Import a CrewAI agent
ao-import-crewai crew_app.py -o agent_ontology/specs/imported_crew.yaml --validate

# 27. Run benchmark suites
ao-benchmark --suite gsm8k --agent self_refine --examples 3
ao-benchmark --suite hotpotqa --agent react --examples 5

# 28. OWL bridge: round-trip test (YAML -> OWL -> YAML)
ao-owl-bridge --round-trip

# 29. Verify semantic properties
ao-verify agent_ontology/specs/react.yaml

# 30. Get architecture recommendations
ao-recommend "multi-hop question answering with retrieval"

# 31. Run live viewer backend (from workspace root that contains /agent_ontology)
PYTHONPATH="$(pwd)" \
python3 -m agent_ontology.live_viewer_server --host 127.0.0.1 --port 8000 --root "$(pwd)"
# Open http://127.0.0.1:8000/spec-viewer.html

# 31b. Run clean-room v2 live viewer backend (new UI + /api/v2/* contract)
PYTHONPATH="$(pwd)" \
python3 -m agent_ontology.v2.api_server --host 127.0.0.1 --port 8012 --root "$(pwd)"
# Open http://127.0.0.1:8012/agent_ontology/v2/spec-viewer-v2.html
# v2 now uses a native runtime session manager (no dependency on v1 RunSession internals)
# v2 persists chat sessions in .live_runs/_sessions/*.json and binds runs -> session_id
# v2 file reads are snapshot-at-cursor first (fallback to workspace when snapshots are absent)
# v2 snapshot retention knobs: AGENT_ONTOLOGY_V2_SNAPSHOT_KEEP_EVENTS, AGENT_ONTOLOGY_V2_SNAPSHOT_KEEP_VERSIONS_PER_FILE
# v2 manual prune endpoint for active runs: POST /api/v2/prune/<run_id>
# v2 runtime lenses (Timeline/State/Dataflow/Sequence) are live and event-driven

# If port 8000 is in use, switch ports:
PYTHONPATH="$(pwd)" \
python3 -m agent_ontology.live_viewer_server --host 127.0.0.1 --port 8011 --root "$(pwd)"
# Open http://127.0.0.1:8011/spec-viewer.html

# If you're already in the repo root:
PYTHONPATH=. python3 -m agent_ontology.live_viewer_server --host 127.0.0.1 --port 8000 --root .

# v2 CLI entrypoint (after `pip install -e .`):
ao-live-viewer-v2 --host 127.0.0.1 --port 8012 --root .

# 32. Optional runtime mode controls for live runs
export AGENT_ONTOLOGY_CAPABILITY_PROFILE=external_enabled  # unrestricted default in live viewer
export AGENT_ONTOLOGY_MOCK_MODE=live                       # or deterministic / llm_mock
export AGENT_ONTOLOGY_DEFAULT_MODEL=openrouter/deepseek/deepseek-chat

# 33. Backend E2E with LLM review of logs + outputs
ao-live-e2e-review \
  --spec-path agent_ontology/specs/babyagi.yaml \
  --text "write me code for a todo task api. full code dont ask any questions just work until its done" \
  --mock-mode live \
  --capability-profile restricted \
  --review-profile babyagi_strict \
  --judge-model openrouter/deepseek/deepseek-chat \
  --json-out .live_runs/latest_e2e_review.json

# 34. Run fixture-pack E2E suites (deterministic/external lanes)
ao-live-e2e-fixtures \
  --fixtures agent_ontology/benchmarks/live_e2e_fixtures.json \
  --tag ci \
  --no-judge-default \
  --json-out .live_runs/latest_e2e_fixture_pack.json
```

Note: for `babyagi*` specs, live runs now auto-seed an initial task from the input text so the loop does real work instead of terminating immediately on an empty task queue.
If `model_override` is blank in the Live Run panel/API, the backend now defaults to `openrouter/deepseek/deepseek-chat` (or `AGENT_ONTOLOGY_DEFAULT_MODEL` if set).
Live viewer now defaults to `AutoGPT` on initial load, and auto-applies `max_iter=220` for AutoGPT specs (other specs keep the standard `80` default).
When `workspace_dir` is omitted, live runs now default to `agent_ontology/workspaces/<run_id>` so generated files are outside `.live_runs` and visible as tracked artifacts.
AutoGPT now forces progress from critique retry loops (no retry-reset deadlock) and falls back to a safe scaffold command for build intents when executor output omits a command, ensuring concrete file generation in workspace.
For build intents, AutoGPT also auto-replaces privileged/package-manager command drift (for example `sudo apt-get ...`) and missing-tool command failures with the same local scaffold path, so runs keep producing files instead of looping on environment setup.
For “full todo app” objectives, AutoGPT now rewrites environment-setup-first plans (git/venv/install front-loaded steps) into file-first implementation plans, so artifacts appear earlier in the run.
AutoGPT command execution now includes a build-intent file-quality gate: if generated candidate source/docs files are all zero-byte placeholders, the step is marked failed and retried (`quality_gate_reason=all_candidate_files_empty`).
Live panel now includes a persistent chat session transcript (`New Chat` reset, session picker, `Ctrl/Cmd+Enter` send shortcut). Chat sessions are stored in browser local storage and restored after refresh; each run carries `chat_session_id`, `current_user_message`, `conversation_history`, and `messages` into backend initial data for iterative prompting behavior.
`ao-live-e2e-review` runs a full backend flow (`/api/live/run` + SSE stream + status + trace) and then uses `llm_client` as a judge against a fixed rubric for objective completion, artifact quality, runtime reliability, observability quality, and risk posture.
Use `--review-profile babyagi_strict` for stronger post-judge gating (judge required, clean trace exit, LLM call presence, artifact signal, and final-data presence checks).
`babyagi_strict` now also gates on file-quality outcomes from `final_data_summary`: runs fail profile checks when `quality_gate_failed=true` or when candidate file count is nonzero and all candidates are empty.
Judge score gating uses a canonical normalized score (`overall_score_normalized` in `[0,1]`) with raw model-provided values preserved as `overall_score_raw`.
For multi-fixture replay/regression checks, use `ao-live-e2e-fixtures` with tag/id filters and assertion checks (`min_event_counts`, `min_process_counts`, `require_executor_interfaces`, etc.).

## The Pipeline

### Step 1: Write or generate a spec

Specs are YAML files following the ontology. You can write one by hand or generate one from documentation:

```bash
# Auto-generate from a natural language description
python3 specgen.py description.md -o specs/my_agent.yaml --validate --fix
```

The `specgen.py` pipeline loads the ontology + example specs as context for an LLM, generates a spec, validates it, and optionally auto-fixes errors. Pattern-aware: scans descriptions for architectural patterns (reasoning loop, critique cycle, debate, retrieval, etc.), selects matching example specs, and includes pattern context in the prompt.

### Step 2: Validate

```bash
python3 validate.py specs/my_agent.yaml
# Output: errors (must fix) and warnings (informational)
```

Validates against `ONTOLOGY.yaml` with 25+ rules:
- Entity/process/edge type validation with required fields
- Schema reference resolution
- Graph connectivity (unreachable processes, disconnected chains)
- Fan-out without join detection
- Empty process shells (no logic, invocations, or store access)
- Schema field collision warnings in fan-out patterns
- Error handler scope and retry configuration validation

### Step 3: Visualize

Open `spec-viewer.html` in a browser (via HTTP server). Five views:

- **Overview** — Simplified architecture diagram with BFS spine layout. Steps and agents are collapsed into single nodes. Entity connections shown as single lines per (process, entity) pair — click for sidebar details showing edge types, labels, and return schemas. Draggable nodes.
- **Graph** — Interactive canvas flowchart. Drag nodes, click for details, hover for tooltips.
- **State Machine** — Linear process flow with gates, branches, loops, and agent invocations.
- **Schemas** — All data schemas with field types and cross-references.
- **Compare All** — Side-by-side comparison table across all agent specs.

Supports trace overlay: load a `trace.json` to see execution counts, durations, and LLM call heat-maps on the State Machine view.
Live Run panel streams execution logs, per-step/store/file diffs, and now delegated `task local_run` trace events (`delegate_run` + `delegate_trace_file`) when child agents emit trace metadata.
Live Run now uses a consolidated `Trace` pane with internal modes: `Execution`, `Payload`, `Step Diffs`, `Store Diffs`, `File Diffs`, `Store + File`, `Data Flow`, and `Sequence` (UML). These modes are projection views derived from canonical `liveEvents` (not pane-local event arrays), so cross-pane cursor sync is deterministic. Verify/retry/checkpoint/resume transitions are surfaced as causal hints directly in flow/sequence summaries. The tracked-files tree (folder hierarchy + click-to-open files) remains first-class. The `File Snapshot` pane has controls for `Refresh Files`, `Load Snapshot @ Cursor`, `Load Workspace`, and per-file snapshot version pinning, so you can inspect full file text (snapshot-at-event or workspace-now) side-by-side with patch/preview context. The `State Machine` view renders live packet dots, clickable sink nodes (`Input/LLM/State/Stores/Files/Output/Done`), and a payload trace panel so you can watch data move through the graph (including plan/output/file payload previews); sink clicks use explicit modes (`click` = highlight, `shift-click` = strict filter).
Live Run explorer now includes a first-class chat transcript. Each `Run Live` click records a user turn, captures assistant output on completion, and keeps turn history visible/inspectable while preserving all existing trace/diff/file inspection behavior.
The panel now includes a VSCode-style `Workspace + Memory Explorer` tree that lets you inspect input payload, runtime state/store memory, tracked files, cursor/event context, routing telemetry, and final output from one place.
The live options row now includes a `Quality Filter` toggle that narrows trace/timeline panes to quality-gate related events (`quality_gate_*`, `quality_file_check`, `all_candidate_files_empty`), and cursor chips now show quality status (`pass/fail/unknown`) with non-empty/candidate file counts when available.
UI now defaults to a simplified mode (single timeline + explorer/inspector). Use the `Advanced` toggle in the live panel to reveal all trace modes/tabs and jump panes.
Top tabs are now spec-analysis focused (`Schemas`, `Compare`).
`Explorer → Views` is the primary dynamic switcher for live work: `Canvas Views` (`Graph`, `Overview`, `State Machine`) plus runtime `Lenses` (`Timeline`, `Flow`, `Sequence`, `Execution`, `Files`, `State`, `Output`), all cursor-synced.
Runtime lens switching works in both simple and advanced modes; simple mode hides extra controls but does not coerce lens selection.
Live panel visibility is stable across canvas views, so you can keep runs active while switching visualizations.
In `Graph` view, clicking a `store` entity now resolves to live runtime store keys (exact/case-insensitive/normalized match) and shows the current store value snapshot in the inspector, with a one-click `Focus In Explorer` action to jump the memory tree to that store.
Store inspector now decodes nested JSON-string payloads and surfaces `Text Excerpts` plus `Vector Data` summaries (vector count/dimensions) before the full decoded snapshot, so text-heavy memories are readable without manual JSON parsing.
When vector arrays are present, the same inspector also renders a lightweight 2D `Vector Map` (deterministic projection + heuristic clustering) with per-cluster labels inferred from nearby text snippets.
Vector maps are interactive: choose dataset/projection (`random`, `first_two_dims`, `top_variance_dims`), set cluster count (`auto` or explicit), hover to preview linked text, and click to pin point-level details; point clicks also attempt to focus the corresponding snippet leaf in Explorer memory paths.
Simple mode now defaults to a narrower left dock with chat + run controls + explorer + core trace/log context; high-density controls and deep trace lenses (`Flow`, `Sequence`, `E2E`, jump panes) remain in `Advanced`.
Cursor chips now include explicit `canvas`, `pane`, and `trace` context so it is clear whether you are viewing Graph/Overview/State Machine and which runtime lens is active.
Store-flow cards and graph store selection are now bidirectionally linked: selecting flow-store cards syncs graph sidebar selection, and selected graph stores are highlighted in flow map state.
Trace toolbar now includes cursor replay controls (`Prev`, `Play/Pause`, `Next`) for deterministic event-index walkthroughs, and Explorer `Quick Lenses` presets (`Execution`, `Timeline`, `Files`, `Output`) for low-friction navigation.
File snapshot view is now snapshot-first by default: entering `Files` auto-loads `snapshot@cursor`, the source badge reports `snapshot@<event_index>`, and workspace text remains an explicit fallback action.
Payload detail now surfaces `related_run_id` links for delegate lineage. Clicking a related run loads that child run timeline directly into the live trace view.
Live backend events now use a non-breaking `EventEnvelope v1` (`schema_version`, `event_index`, `facet`, `entity_id`, `actor`, `span_id`, `parent_span_id`, `related_run_id`, `related_span_id`, `causal_link`, etc.) while preserving legacy payload fields, and each run persists a canonical append-only ledger at `.live_runs/<run_id>/events.jsonl`.
Runtime artifact directories used by observability (notably `.live_runs/**`) are excluded from `file_diff` tracking so file diffs stay focused on agent-produced workspace artifacts without globally suppressing same-named files elsewhere in the workspace.
Live backend now exposes cursor-aware file inspection APIs:
- `GET /api/live/files/<run_id>` (tracked file manifest)
- `GET /api/live/file_snapshot/<run_id>?path=<key>&at_event_index=<n>` (snapshot at-or-before cursor)
- `GET /api/live/file_workspace/<run_id>?path=<key>` (current workspace read for known tracked paths)
- `GET /api/live/events/<run_id>?since_event_index=<n>&limit=<k>` (load/replay event timeline; serves active-memory events, persisted `events.jsonl`, or trace-derived fallback events when ledger is absent)
The live panel also has an `E2E Replay` tab that loads `ao-live-e2e-review` JSON artifacts for post-run inspection of deterministic/judge outcomes, trace metrics, and workspace file snapshots.
The replay tab supports loading local JSON files and backend-powered loading by latest/selected run id (`/api/live/reviews`, `/api/live/review/latest`, `/api/live/review/{run_id}`).
CI also publishes deterministic fixture-pack artifacts (`ci-live-e2e-fixtures-deterministic`, `ci-live-e2e-review-external-deterministic`) and an optional strict judged artifact (`ci-live-e2e-review-strict-judged`).
AutoGPT routing now records explicit telemetry for policy-driven external routing, heuristic semantic fallback overrides, interface repair, and external-call semantic fallback dispatches (`state.data["routing_telemetry"]`, `semantic_fallback_count`, `semantic_fallback_events`).
Trace now includes a routing telemetry strip that binds to the shared event cursor and shows `action_type`, `interface/executed_interface`, override reason, fallback counters, and latest fallback route summary in real time.

### Step 4: Instantiate

```bash
# Custom backend (state machine)
python3 instantiate.py specs/react.yaml

# LangGraph backend (StateGraph)
python3 instantiate.py specs/react.yaml --backend langgraph
```

Two backends available:

**Custom backend** (default) generates a complete Python agent with:
- State machine (PROCESSES dict + TRANSITIONS dict)
- **Fan-out support**: Multiple outgoing flow edges run all branches sequentially
- **Namespaced state**: Agent outputs stored under `state.data["schema_name"]` and `state.data["process_id_result"]` to prevent field collisions
- **Edge-driven store access**: Store reads generated from `read` edges (before logic blocks), writes from `write` edges (after invocations). Supports `query_key` for parameterized reads.
- Schema-aware LLM calls (input extraction, output parsing, structured prompts)
- Gate condition evaluation (parsed from human-readable conditions)
- Trace logging with metrics (LLM calls, duration, schema compliance, token estimates)
- Store abstractions (queue, vector, buffer, log)
- Multi-model routing: `claude*` → Anthropic, `gemini*` → Google genai, else → OpenAI
- Runtime model override via `AGENT_ONTOLOGY_MODEL` env var
- LLM retry with exponential backoff (3 attempts)
- Runtime schema validation (field presence + type checking)
- Configurable `MAX_ITERATIONS` via `AGENT_ONTOLOGY_MAX_ITER` env var
- Shared fallback tool runtime (`agent_ontology/tool_runtime.py`) for generic tools (read/write/edit/glob/grep/webfetch/search/lookup/mcp/filesystem/calculate/apply_patch/update_plan/parallel/exec_command/write_stdin/notebook_edit/rpc/environment/task)
  - Includes semantic unknown-tool fallback routing (filesystem/search/mcp/rpc/task/notebook/calc/exec), so common alias/tool-name drift resolves to typed adapters instead of `tool ... not implemented`.
  - Includes `task` delegation modes:
    - `deterministic` (default contract-only delegation)
    - `local_run` (executes a delegate generated agent via `run(initial)`)
      - accepts `agent_path` directly, or `spec_path` to auto-generate the child agent via `instantiate.py`
    - `llm` (LLM-backed delegated summary)
  - Useful env vars:
    - `AGENT_ONTOLOGY_TASK_TOOL_MODE` (`deterministic|local_run|llm|disabled`)
    - `AGENT_ONTOLOGY_TASK_TOOL_AGENT_PATH` (default delegate agent file for `local_run`)
    - `AGENT_ONTOLOGY_TASK_TOOL_SPEC_PATH` (default delegate spec file when using `spec_path`)
    - `AGENT_ONTOLOGY_TASK_TOOL_INSTANTIATE_SCRIPT` (override path to `instantiate.py` for delegate generation)
    - `AGENT_ONTOLOGY_TASK_TOOL_GENERATED_DIR` (output directory for generated delegate agents)

**LangGraph backend** (`--backend langgraph`) generates agents using LangGraph's `StateGraph`:
- Steps → nodes, gates → routing functions + conditional edges, flow → edges, loops → back-edges
- LangChain ChatModels (`ChatGoogleGenerativeAI`, `ChatAnthropic`, `ChatOpenAI`)
- TypedDict-based state with fields derived from schemas + logic block analysis
- Gate chaining support (gate → gate inserts pass-through node)
- Channel support: publish/subscribe edges generate channel read/write code
- All 37 specs generate valid Python; verified E2E on multiple agents

## Agent Catalog

37 agent specs, all validated. 36 are instantiable and runnable with `gemini-3-flash-preview` (1 description-only).
The counts, the catalog contents and the property-test total in this file are asserted by
`tests/test_cli_surface.py`, because every one of them had drifted before that test existed.

| Spec | Type | Ent | Proc | Sch | Complexity | Status |
|------|------|-----|------|-----|------------|--------|
| `claude-code` | Tool-use agent | 22 | 19 | 40 | 83.2 (very complex) | Description only |
| `codex` | Tool-use agent | 17 | 16 | 27 | 75.2 (complex) | Generated |
| `sota_superset_runtime` | Runtime feature superset | 31 | 29 | 32 | 74.8 (complex) | Generated |
| `meta_prompting` | Dynamic delegation | 6 | 15 | 11 | 70.4 (complex) | Working |
| `mixture_of_agents` | Parallel proposers | 6 | 11 | 7 | 68.8 (complex) | Working |
| `plan_and_solve` | Decompose+verify | 4 | 9 | 9 | 65.7 (complex) | Working |
| `socratic_tutor` | Human-in-loop tutoring | 7 | 12 | 13 | 65.5 (complex) | Working |
| `reflexion` | Self-reflection loop | 5 | 11 | 10 | 62.1 (complex) | Working |
| `crew` | Multi-agent | 6 | 17 | 9 | 61.4 (complex) | Working |
| `react` | Reason+Act | 6 | 8 | 8 | 60.7 (complex) | Working |
| `code_reviewer` | Parallel analysis | 7 | 10 | 9 | 60.6 (complex) | Working |
| `babyagi` | Task-driven autonomous | 5 | 5 | 9 | 59.5 (moderate) | Working |
| `autogpt` | Goal-driven | 6 | 11 | 10 | 59.2 (moderate) | Working |
| `babyagi_autogen` | Task-driven | 5 | 3 | 9 | 58.4 (moderate) | Working |
| `debate` | Multi-agent debate | 6 | 13 | 7 | 57.7 (moderate) | Working |
| `voyager` | Open-ended exploration | 5 | 12 | 9 | 55.5 (moderate) | Working |
| `tree_of_thought` | Tree search | 3 | 7 | 7 | 55.4 (moderate) | Working |
| `self_refine` | Generate-critique | 2 | 7 | 7 | 55.0 (moderate) | Working |
| `lats` | MCTS tree search | 5 | 8 | 8 | 51.2 (moderate) | Working |
| `rag` | Retrieval-augmented | 5 | 10 | 10 | 50.4 (moderate) | Working |
| `map_reduce` | Parallel chunk processing | 4 | 6 | 11 | 48.6 (moderate) | Working |
| `multi_agent_codegen` | Code generation pipeline | 5 | 5 | 7 | 47.4 (moderate) | Working |
| `customer_support_swarm` | Handoff-based routing | 8 | 7 | 7 | — | Working |
| `software_team` | Pub/sub team pipeline | 10 | 8 | 10 | — | Working |
| `self_improver` | Meta-agent spec optimizer | 3 | 16 | 8 | — | Working |
| `kg_rag` | Knowledge graph RAG | 3 | 6 | 4 | — | Working |
| `pddl_planner` | NL→PDDL→plan→NL | 3 | 8 | 5 | — | Working |
| `rap` | Reasoning via Planning (MCTS) | 5 | 7 | 5 | — | Working |
| `alpha_geometry` | LLM+symbolic deduction | 4 | 8 | 6 | — | Working |
| `minimal_solver` | Minimal single-LLM baseline | 1 | 2 | 3 | — | Working |
| `multidoc_baseline` | Single-shot multi-doc QA | 1 | 3 | 4 | — | Working |
| `multidoc_structured` | Structured multi-doc QA | 2 | 8 | 7 | — | Working |
| `kb_react` | KB tool-using ReAct | 6 | 8 | 8 | — | Working |
| `mutator` | Meta-evolution agent | 2 | 5 | 7 | — | Working |

`Working` means the generated agent has been run end-to-end against a live model. `Generated`
means it validates, generates and imports, but has not been run E2E. `Description only` means
the spec documents an architecture that is not instantiable.

Complexity scores computed by `complexity.py` using weighted graph metrics (entities, edges, fan-out, loops, schema count, graph depth, invocation density).

## Spec Format

A spec has four sections:

```yaml
name: "My Agent"
version: "1.0"
description: "What this agent does"
entry_point: first_step

entities:       # Things that exist (agents, stores, tools, humans, channels, teams)
processes:      # Things that happen (steps, gates, checkpoints)
edges:          # Connections (flow, invoke, loop, branch, read, write, publish, subscribe, handoff)
schemas:        # Data shapes flowing between components
```

### Entity types
- `agent` — LLM-backed component with system prompt, model, I/O schemas
- `store` — Persistent state (queue, vector, buffer, log, knowledge_graph, ontology, logic_program, pddl_domain)
- `tool` — External capability (API, function, system)
- `human` — Human-in-the-loop participant
- `config` — Static configuration
- `channel` — Named pub/sub communication channel with message schema and reducer
- `team` — Agent group with strategy (sequential, hierarchical, consensus, round-robin)
- `conversation` — Multi-turn dialogue with history and persistence
- `world_model` — State transition model T(s'|s,a) for planning agents

### Process types
- `step` — Do something (may include inline `logic:` as Python)
- `gate` — Decision point with `condition:` and `branches:`
- `checkpoint` — Human approval required
- `spawn` — Create sub-agents (supports `recursive: true`)
- `protocol` — Multi-party interaction
- `policy` — Cross-cutting constraint
- `error_handler` — Structured error handling with retry, fallback, timeout
- `symbolic_inference` — Deterministic non-LLM reasoning (Z3, Prolog, DL classification)
- `formal_translation` — NL↔formal language translation (NL→SPARQL, NL→PDDL, etc.)
- `search` — Systematic search over possibility spaces (MCTS, beam search, A*)

### Edge types
- `flow` — Sequential control flow
- `invoke` — Call an agent/tool and get a response
- `loop` — Conditional back-edge
- `branch` — Conditional forward-edge (from gates)
- `read` / `write` — Store access
- `publish` / `subscribe` — Channel-based messaging
- `handoff` — Agent-to-agent control transfer
- `error` — Error flow routing to handler
- `modify` / `observe` — Policy interactions

Full type system: `ONTOLOGY.yaml` | Machine-readable: `spec_schema.json` (JSON Schema)

## Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `validate.py` | Check spec against ontology | `python3 validate.py spec.yaml` |
| `instantiate.py` | Generate runnable agent (custom or LangGraph) | `python3 instantiate.py spec.yaml [--backend langgraph]` |
| `specgen.py` | Generate spec from description (pattern-aware) | `python3 specgen.py desc.md -o spec.yaml --validate --fix` |
| `spec-viewer.html` | Interactive visualization (5 views) | Serve via HTTP, open in browser |
| `test_agents.py` | Automated agent testing | `python3 test_agents.py --agent react` |
| `analyze_trace.py` | Trace analysis and comparison | `python3 analyze_trace.py trace.json` |
| `complexity.py` | Spec complexity scoring | `python3 complexity.py --all specs/` |
| `mutate.py` | Spec mutation engine (field + pattern-level) | `python3 mutate.py spec.yaml --random -n 5` |
| `evolve.py` | Evolutionary search with crossover | `python3 evolve.py spec.yaml --generations 3 --crossover` |
| `benchmark.py` | Benchmark suite (9 datasets) | `python3 benchmark.py --suite gsm8k --agent self_refine` |
| `patterns.py` | Pattern library (7 patterns) | `from patterns import detect_patterns, PATTERNS` |
| `compose.py` | Compose patterns into new specs | `python3 compose.py compose_spec.yaml -o spec.yaml` |
| `test_specgen.py` | Specgen E2E testing | `python3 test_specgen.py --fix` |
| `spec_diff.py` | Structured spec comparison | `python3 spec_diff.py old.yaml new.yaml` |
| `coverage.py` | Ontology feature coverage | `python3 coverage.py --all specs/` |
| `lint.py` | Anti-pattern detection (10 rules) | `python3 lint.py --all specs/` |
| `topology.py` | Control-flow topology classifier | `python3 topology.py --all specs/` |
| `dashboard.py` | Unified project health report | `python3 dashboard.py specs/` |
| `migrate.py` | Spec version migration | `python3 migrate.py --all specs/ --to 2.0 --dry-run` |
| `mermaid.py` | Mermaid flowchart export | `python3 mermaid.py specs/react.yaml` |
| `similarity.py` | Spec similarity & clustering | `python3 similarity.py --all specs/ --clusters 5` |
| `test_properties.py` | Property-based structural tests (294) | `python3 test_properties.py` |
| `comparative_report.py` | Cross-spec comparative analysis | `python3 comparative_report.py --all specs/` |
| `import_langgraph.py` | Import LangGraph StateGraph → YAML spec | `python3 import_langgraph.py agent.py -o spec.yaml` |
| `import_crewai.py` | Import CrewAI → YAML spec | `python3 import_crewai.py crew.py -o spec.yaml` |
| `import_autogen.py` | Import AutoGen → YAML spec | `python3 import_autogen.py autogen_app.py -o spec.yaml` |
| `import_openai_agents.py` | Import OpenAI Agents SDK → YAML spec | `python3 import_openai_agents.py app.py -o spec.yaml` |
| `import_google_adk.py` | Import Google ADK → YAML spec | `python3 import_google_adk.py app.py -o spec.yaml` |
| `design.py` | End-to-end agent design pipeline | `python3 -m agent_ontology.design "task" --benchmark gsm8k --evolve` |
| `ingest.py` | Corpus ingestion for ChromaDB vector stores | `python3 -m agent_ontology.ingest docs/ --store my_kb` |
| `knowledge_store.py` | SQLite store for evolution results | `python3 -m agent_ontology.knowledge_store stats` |
| `verify.py` | Semantic property verification (9 checks) | `python3 verify.py spec.yaml` |
| `recommend.py` | Evidence-backed architecture recommender | `python3 -m agent_ontology.recommend "task description"` |
| `owl_bridge.py` | OWL dual representation: YAML<->OWL round-trip + classification | `python3 owl_bridge.py --round-trip` |
| `ontology_owl.py` | OWL/DL pattern classification | `python3 ontology_owl.py` |
| `ontology_rdf.py` | RDF export + semantic pattern queries | `python3 ontology_rdf.py` |

## Architecture

```
ONTOLOGY.yaml          # Type system (entity types, edge types, constraints)
     |
     v
specs/*.yaml           # Agent specifications (32 agents)
     |
     +---> validate.py       # Validation (25+ rules, graph analysis)
     +---> instantiate.py    # Code generation -> agents/*.py or agents_lg/*.py
     +---> complexity.py     # Complexity scoring (10 metrics)
     +---> mutate.py         # Spec mutation engine (field + pattern-level)
     +---> evolve.py         # Evolutionary search (mutate → test → select)
     +---> design.py         # End-to-end agent design pipeline
     +---> knowledge_store.py # Persistent evolution results (SQLite)
     +---> recommend.py      # Evidence-backed architecture recommender
     +---> patterns.py       # Pattern library (7 reusable patterns)
     +---> compose.py        # Compose patterns into new specs
     +---> verify.py         # Semantic property verification (9 checks)
     +---> coverage.py       # Ontology feature coverage report
     +---> lint.py           # Anti-pattern detection (10 rules)
     +---> topology.py       # Control-flow topology classifier
     +---> spec_diff.py      # Structured spec comparison
     +---> dashboard.py      # Unified project health report
     +---> mermaid.py        # Mermaid flowchart export
     +---> similarity.py     # Spec similarity & clustering
     +---> migrate.py        # Spec version migration
     +---> comparative_report.py  # Cross-spec comparative analysis
     +---> test_properties.py     # Property-based structural tests (294)
     +---> spec-viewer.html  # Visualization (5 views + trace overlay)

agents/*.py            # Generated runnable agents (custom backend)
agents_lg/*.py         # Generated runnable agents (LangGraph backend)
     |
     +---> test_agents.py    # Automated testing + multi-model comparison
     +---> benchmark.py      # Benchmark suite (9 datasets)
     +---> trace.json        # Runtime traces
              |
              v
         analyze_trace.py    # Trace analysis + comparison

compose_specs/*.yaml   # Pattern composition recipes
     |
     v
compose.py             # Pattern → spec composition
     |
     v
specs/*.yaml           # Composed specs (validated)

test_descriptions/*.md # Natural language agent descriptions
     |
     v
specgen.py             # LLM-powered spec generation
     |
     v
specs/*.yaml           # Generated specs (validated + auto-fixed)
     |
     v
test_specgen.py        # E2E pipeline testing
```

## Formal Foundation

Agent Ontology uses a **reified hypergraph** constrained by an **ontology**:

- **Reified hypergraph**: Any relationship can be a node, any node can participate in any number of relationships. This means templates, dynamic topology, conditional wiring, self-modification — all expressible.
- **Ontology**: Constrains the hypergraph so code generators know what to expect. Defines types, required fields, and validation rules.

See `SPEC.md` for the standalone format specification, `spec_schema.json` for machine-readable JSON Schema validation, `ROADMAP.md` for the long-term vision, and `ONTOLOGY.yaml` for the internal type system definition.
For live runtime event semantics, see `LIVE_EVENT_ENVELOPE_V1.md`.

## Requirements

- Python 3.8+
- Modern browser (for spec-viewer)

```bash
pip install -e .                   # Core package
pip install -e ".[agents]"         # + LLM providers for running agents
pip install -e ".[langgraph]"      # + LangGraph backend
pip install -e ".[owl]"            # + OWL/DL reasoning (owlready2)
pip install -e ".[all]"            # Everything
pip install -e ".[dev]"            # + dev/test tools
```

API keys go in `.env.local`:
```bash
GEMINI_API_KEY=...
OPENAI_API_KEY=...     # optional, for specgen
ANTHROPIC_API_KEY=...  # optional, for Claude model agents
```

## Project Structure

```
agent_ontology/         # Python package (33 modules, pip install -e .)
  ONTOLOGY.yaml         #   The type system (9 entity types, 10 process types, 13 edge types)
  specs/                #   Agent specifications (31 YAML specs)
  benchmarks/           #   Benchmark datasets (HotpotQA, GSM8K, GSM8K-Hard, GSM8K-Tricky, ARC, HumanEval, MultiDoc, KB-Tool) + scoring
  spec-viewer.html      #   Interactive multi-view visualization (5 views + trace overlay)
  validate.py           #   Spec validator (25+ rules)
  instantiate.py        #   Code generator (custom + LangGraph backends)
  specgen.py            #   Description-to-spec pipeline (pattern-aware)
  design.py             #   End-to-end agent design pipeline (ao-design)
  evolve.py             #   Evolutionary search with crossover + benchmark fitness
  knowledge_store.py    #   SQLite store for evolution results
  recommend.py          #   Evidence-backed architecture recommender
  ingest.py             #   Corpus ingestion for ChromaDB vector stores
  patterns.py           #   Pattern library (7 reusable architectural patterns)
  compose.py            #   Pattern composition operator
  verify.py             #   Semantic property verification (9 checks)
  mutate.py             #   Spec mutation engine (field + pattern-level)
  import_langgraph.py   #   Import LangGraph StateGraph → YAML spec
  import_crewai.py      #   Import CrewAI → YAML spec
  import_autogen.py     #   Import AutoGen → YAML spec
  import_openai_agents.py # Import OpenAI Agents SDK → YAML spec
  import_google_adk.py  #   Import Google ADK → YAML spec
  owl_bridge.py         #   OWL dual representation: YAML<->OWL round-trip
  complexity.py         #   Spec complexity scoring
  coverage.py           #   Ontology feature coverage report
  lint.py               #   Anti-pattern linter (10 rules)
  topology.py           #   Control-flow topology classifier
  similarity.py         #   Spec similarity & clustering
  migrate.py            #   Spec version migration
  mermaid.py            #   Mermaid flowchart export
  ...                   #   + analyze_trace, spec_diff, dashboard, comparative_report, etc.
tests/                  # Test suite
  test_agents.py        #   Automated agent testing
  test_properties.py    #   Property-based structural tests (294)
  test_specgen.py       #   E2E specgen pipeline testing
  test_roundtrip.py     #   LangGraph import round-trip tests (8 tests)
agents/                 # Generated runnable agents (custom backend, 31 agents)
agents_lg/              # Generated runnable agents (LangGraph backend, 27 agents)
compose_specs/          # Pattern composition recipes (3 examples)
test_descriptions/      # Natural language descriptions for specgen testing
pyproject.toml          # Package config with CLI entry points (ao-*)
SPEC.md                 # Formal spec format specification (standalone)
spec_schema.json        # JSON Schema for machine-readable spec validation
ROADMAP.md              # Strategic roadmap and long-term vision
VISION.md               # Project vision: formal representation of agency
.github/workflows/      # CI: validate, lint, syntax-check, smoke-test
```
