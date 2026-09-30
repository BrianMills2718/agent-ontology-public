# SOTA Runtime Mapping

This document maps the current `agent_ontology` runtime to a SOTA-style superset architecture represented in:

- `agent_ontology/specs/sota_superset_runtime.yaml`

## Why This Superset

The goal is to represent modern agent systems in one graph:

1. Typed action selection (`tool_calls`, `external_call`, `handoff`, `respond`, `checkpoint`)
2. Guardrailized execution (input + output + tool policies)
3. External capability plane with switchable modes (`deterministic`, `live`, `llm_mock`)
4. Verification + critic loop before finalize
5. Checkpoint/resume and thread continuity
6. Multi-agent handoffs
7. First-class trace spans and artifacts

## Current Runtime -> Superset Mapping

| Superset Node | Current Implementation | Status |
|---|---|---|
| `receive_request` / `hydrate_context` | Implemented in generated agents + live viewer bootstrap | Implemented |
| Typed `ActionDecision` gate | Present in `codex.yaml` and `autogpt.yaml` specs | Partial |
| `tool_calls` path | Present via generated tool execution paths | Implemented |
| `external_call` typed path | First-class in `autogpt.yaml` via `execution_mode_gate` + `execute_external_call`, backed by `external_interfaces.py` | Implemented |
| External mode routing (`deterministic/live/llm_mock`) | `AGENT_ONTOLOGY_MOCK_MODE` + `dispatch_external_interface(...)` | Implemented |
| Capability profile gating | `restricted` vs `external_enabled` wired through live runner + AutoGPT command policy | Implemented |
| Handoff protocol / spawn | Represented in ontology; runtime behavior not standardized across agents | Partial |
| Input/output guardrail policy nodes | Representable in specs; enforcement mostly prompt/policy-level today | Partial |
| Checkpoint save/restore | Representable; store plumbing exists; full resume semantics vary per agent | Partial |
| Verification gate before finalize | Present in some benchmark flows; not universal default in generated agents | Partial |
| Trace spans + live observability | Live viewer streams logs, state/store diffs, file diffs | Implemented |
| Artifact persistence and inspection | `.live_runs/<run_id>/...` + viewer file panes | Implemented |

## Concrete Runtime Anchors

1. External adapters:
   - `agent_ontology/external_interfaces.py`
2. Live execution and mode wiring:
   - `agent_ontology/live_viewer_server.py`
3. AutoGPT capability profile + command policy:
   - `agent_ontology/specs/autogpt.yaml`
4. Live visual inspection (state/store/file diffs):
   - `spec-viewer.html`

## Immediate Next Steps (to converge with SOTA behavior)

1. Improve planner/executor routing reliability so retrieval tasks prefer `external_call` over shell fallback.
2. Add explicit `verify_outcome` and `retry` semantics to AutoGPT default flow.
3. Add typed `handoff` action path and minimal delegate-return contract.
4. Persist checkpoint records with deterministic resume tests.
5. Add policy tests that assert capability guardrails for every typed action branch.

## Long-Term Target

A single runtime where each agent execution is:

1. Typed (all actions/tool calls/handoffs are structured)
2. Policy-enforced (capabilities + guardrails)
3. Observable (trace spans + memory/file diffs)
4. Replayable (checkpoint + deterministic mock mode)
5. Evolvable (benchmarked mutations over explicit graph primitives)
