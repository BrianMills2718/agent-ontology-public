# Stabilization Review (Pre-Next-Phase)

Date: 2026-02-21

## Scope

This pass focused on:

1. Runtime stability for live execution and visualization
2. Typed external action path in AutoGPT
3. Test/validation hygiene before adding more capability surface

## What Was Verified

1. Targeted runtime tests:
   - `PYTHONPATH=. pytest -q tests/test_autogpt_runtime.py tests/test_codex_runtime.py tests/test_external_interfaces.py tests/test_live_viewer_server.py`
   - Result: `19 passed`
2. Spec validation:
   - `python3 agent_ontology/validate.py agent_ontology/specs/autogpt.yaml agent_ontology/specs/codex.yaml agent_ontology/specs/sota_superset_runtime.yaml`
   - Result: all valid with `0 errors, 0 warnings`
3. Full spec sweep:
   - `python3 agent_ontology/validate.py agent_ontology/specs/*.yaml`
   - Result: no errors; multiple legacy warnings remain in older specs (documented below)

## Cleanup Applied

1. `.gitignore` now excludes runtime artifacts:
   - `.live_runs/`
   - `trace_overlay*.json`
   - `autogpt_workspace/`
2. `tests/test_properties.py` now supports clean pytest runs:
   - added `ontology` pytest fixture
   - renamed `TestResults` helper class to `_ResultsTracker` to avoid pytest class collection warnings
3. `SOTA_RUNTIME_MAPPING.md` updated to reflect current state:
   - `external_call` in AutoGPT marked implemented
   - immediate next steps now focus on routing reliability, verify/retry semantics, handoff/checkpoint hardening

## Current Uncertainties

1. Action selection reliability:
   - AutoGPT now supports typed `external_call`, but model output can still choose shell command paths for some retrieval-style tasks.
2. Tool completeness:
   - Some generated agents intentionally include stubbed tool handlers (`not implemented`) depending on spec coverage.
3. Legacy spec warnings:
   - Several older benchmark/imported specs still emit validation warnings (orphan/unreachable/empty-shell patterns). These are non-fatal but add noise.
4. Live viewer UX:
   - The backend supports state/store/file diff streaming, but readability/layout behavior is still the main UX risk area.

## Update (Follow-up Pass, 2026-02-21)

Resolved in this pass:

1. Tool completeness for declared tool IDs:
   - Expanded `agent_ontology/tool_runtime.py` alias/adapter coverage to include remaining spec-declared tool IDs (`bash`, `shell_tool`, `filesystem_tool`, `search_tool`, `lookup_tool`, `semantic_search_tool`, `web_search`, `mcp_tool`, `mcp_tools`, `calculate_tool`).
   - Added runtime adapters for search/lookup/mcp/filesystem/calculate and verified coverage against all spec tool IDs (`missing 0`).
2. Live delegated trace observability:
   - Added delegated runtime markers (`__AO_DELEGATE_TRACE__`, `__AO_DELEGATE_RUN__`) in task local-run flow.
   - Live viewer server now emits `delegate_trace_file` and `delegate_run` SSE events.
   - `spec-viewer.html` surfaces delegated trace/run events in the live log.
   - Added golden event-sequence test coverage.

Remaining:

1. Planner/action preference quality can still be model-sensitive in edge cases, even with deterministic post-executor routing.
2. Legacy spec warning noise still exists in non-production benchmark/imported specs.
3. Live viewer readability/panel ergonomics may still need iterative UX tuning based on user runs.

## Recommendations Before Expanding External Capability

1. Add deterministic action-routing policy:
   - pre-execution classifier that can override model choice to `external_call` for known retrieval intents.
2. Harden typed-action contract tests:
   - add property tests that all branches (`tool_calls`, `external_call`, `respond`, `none`) are schema-valid and executable.
3. Separate "production specs" from "research/legacy specs":
   - split validation gates to reduce warning noise in CI.
4. Keep default mode deterministic:
   - use `deterministic` mocks by default and gate `live` mode behind explicit opt-in.
5. Add golden run fixtures for live viewer:
   - store expected event sequence (`step_diff`, `store_diff`, `file_diff`, `final_channel`) and replay in tests.

## 24-Hour Autonomous Execution Queue (2026-02-23)

This is the current no-intervention backlog used to drive live-viewer usability and observability work.

### Completed in this cycle

1. Graph store inspector now has interactive vector exploration:
   - dataset selector (when multiple vector arrays are present)
   - projection selector (`deterministic_random`, `first_two_dims`, `top_variance_dims`)
   - cluster selector (`auto` or explicit count)
   - point hover/click inspection with linked text snippet preview
   - point click deep-link attempts into Explorer `stores/<key>/snippet_*` leaves
2. Graph-store runtime mapping was hardened:
   - exact, case-insensitive, and normalized key matching from graph node → live store key
   - one-click `Focus In Explorer` still synchronized to `stores/<key>`
3. Simple/advanced UX pass:
   - simple mode dock narrowed for canvas visibility
   - advanced-only runtime lenses (`Flow`, `Sequence`, `E2E`) hidden from simple mode
   - simple mode defaults to `Execution` trace lens
4. Cursor/selection clarity:
   - cursor chips now include explicit `canvas`, `pane`, and `trace` labels
   - advanced mode still exposes extra diagnostics (`source`, `sink`, selected event index)
5. Memory explorer detail readability:
   - typed detail formatter for input/output text, store values (decoded), and file metadata
   - details include type summary and path-stable rendering
6. Validation:
   - `PYTHONPATH=. pytest -q tests/test_spec_viewer_live_contract.py` → pass
   - `PYTHONPATH=. pytest -q tests/test_live_viewer_server.py` → pass
   - backend smoke run on `minimal_solver` via `/api/live/run` and `/api/live/events/<run_id>` → pass
7. Store-flow cohesion:
   - selecting a store card in flow now syncs graph sidebar selection
   - flow-card hover highlights corresponding graph store node in Graph view
8. Lens ergonomics:
   - added `Quick Lenses` preset buttons (`Execution`, `Timeline`, `Files`, `Output`)
   - preset active state tracks current pane/trace context
9. Story-mode cursor replay:
   - trace toolbar now includes `Prev`, `Play/Pause`, `Next` controls over `event_index`
   - replay is deterministic and stops automatically at stream end
10. Snapshot-first file inspection:
   - entering `Files` pane auto-loads `snapshot@cursor` when a tracked file is selected
   - source badge now reports `snapshot@<event_index>` instead of a generic source label
11. Golden fixture coverage:
   - added deterministic memory/vector golden fixture test in `tests/test_live_viewer_server.py` to lock store-diff payload shape used by Explorer/Vector rendering paths

### Remaining 24-hour tasks (ordered)

1. No blocking items remain in this tranche.
   - next pass can focus on scale/perf tuning and richer semantic labeling (optional enhancement work).

### Active uncertainties / assumptions

1. Vector map labels are heuristic:
   - current cluster labels are token-frequency guesses from nearby text snippets, not semantic labeling.
2. Point-to-path precision:
   - nested path extraction from arbitrary store payloads is improved via snippet leaves, but still best-effort for non-snippet vector rows.
3. Large payload performance:
   - decoding/pretty-printing large store payloads is bounded, but extreme payloads can still be expensive in-browser.
4. Live-run completion variability:
   - complex specs (for example AutoGPT with high `max_iter`) may legitimately remain running in backend smoke windows; quick smoke should use a short-horizon spec (`minimal_solver`) for deterministic completion checks.
