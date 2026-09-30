# Live EventEnvelope v1

Date: 2026-02-22
Status: Implemented (non-breaking)
Primary implementation: `agent_ontology/live_viewer_server.py`

## Purpose

This document defines the live runtime event contract used by the viewer and replay tooling.

The contract is designed to be:
1. Non-breaking for existing viewer logic.
2. Causality-aware for debugging and analysis.
3. Persistable as an append-only ledger for deterministic replay.

## Core Principles

1. **Flattened envelope, no wrapper**
- Envelope metadata is top-level on every event.
- Existing payload fields remain unchanged.
- Existing consumers that read `type` + payload keep working.

2. **Single ordering key**
- `event_index` is the canonical monotonically increasing order key.
- `seq` remains present as a backward-compatible alias (`event_index == seq`).

3. **Canonical ledger parity**
- The exact event object streamed via SSE is also appended to `events.jsonl`.
- No separate transform layer is used for on-disk events.

## Event Shape

Every emitted event includes these envelope fields:

```json
{
  "schema_version": "1.0",
  "run_id": "<run-id>",
  "event_id": "<run-id>:<event-index>",
  "event_index": 42,
  "seq": 42,
  "time": 1771775333.57,
  "ts": "2026-02-22T15:08:53.570000+00:00",
  "type": "state",
  "facet": "state",
  "entity_id": "process:receive_request",
  "actor": "runtime",
  "span_id": "span.state.<run-id>.42",
  "parent_span_id": "span.run.<run-id>",

  "...": "existing legacy payload fields remain unchanged"
}
```

## Field Definitions

- `schema_version`: Version string for event contract.
- `run_id`: Stable run identifier.
- `event_id`: Stable per-event identifier (`run_id:event_index`).
- `event_index`: Canonical monotonic ordering key.
- `seq`: Backward-compatible alias to `event_index`.
- `time`: UNIX epoch seconds (float).
- `ts`: ISO-8601 UTC timestamp.
- `type`: Existing event type (e.g., `state`, `file_diff`).
- `facet`: Coarse category for filtering/highlighting.
- `entity_id`: Primary object touched by the event (best-effort normalized ID).
- `actor`: Emitting subsystem (`runtime`, `agent:<name>`, `delegate:<id>`).
- `span_id`: Causal span identifier for this event.
- `parent_span_id`: Immediate causal parent span identifier or `null`.

## Event Type -> Facet Mapping

- `started`, `runtime_info` -> `input`
- `state`, `step_diff` -> `state`
- `llm_call` -> `llm`
- `store_diff` -> `stores`
- `file_diff` -> `files`
- `final_channel`, `final_data`, `final_keys` -> `output`
- `completed`, `stopped` -> `done`
- `error` -> `error`
- all others -> `control`

## Span Semantics (v1)

- Run root span: `span.run.<run_id>`
- `started`: uses run span (`parent_span_id = null`)
- `state`: new transition span (`span.state.<run_id>.<event_index>`, parent = run span)
- non-state operational events (`llm_call`, `step_diff`, `store_diff`, `file_diff`, `trace_file`, delegate events, output events):
  - parent = active transition span if available, else run span
- `completed` / `stopped`: `span.done.<run_id>.<event_index>`, parent = run span
- `error`: `span.error.<run_id>.<event_index>`, parent = active transition span if available, else run span

This is intentionally simple and deterministic for v1.

## Entity ID Semantics (v1)

Best-effort normalization rules:

- `state`: `process:<process_id>`
- `llm_call`: `llm:<process>:<agent>:<event_index>`
- `store_diff`: first touched store (`store:<name>`) else `store-process:<process>` else `store:unknown`
- `file_diff`: first touched file (`file:<path>`) else `file-process:<process>` else `file:unknown`
- `trace_file`: `trace:<path>`
- `delegate_trace_file`: `delegate-trace:<delegate_id>[:<path>]`
- `delegate_run`: `delegate-run:<delegate_id>`
- output events: `output:<run_id>`
- run-level events: `run:<run_id>`

## Persistence Contract

Per run, ledger path:

```text
.live_runs/<run_id>/events.jsonl
```

Properties:
- Append-only, one JSON object per line.
- Event object is identical to SSE payload object.
- Write failure is best-effort and does not stop run execution.

## File-Diff Artifact Exclusions

File-diff snapshots intentionally ignore runtime observability artifact directories so diffs reflect agent outputs, not viewer/runtime bookkeeping.

Current behavior:
1. Ignore `.live_runs/**` by path scope.
2. Ignore standard cache/environment directories (`.git`, `node_modules`, `.venv`, etc.).
3. Do not globally ignore files by basename (for example, `debug/trace.json` remains visible in diffs).

## Streaming Contract

SSE stream now emits both:
1. `id: <event_index>` line
2. `data: <event-json>` line

Resume query supports:
- `?since=<n>` (legacy)
- `?since_event_index=<n>` (preferred)

Resume precedence:
1. `since_event_index`
2. `since`
3. `Last-Event-ID` request header
4. fallback to `0`

## File Snapshot APIs

The live backend persists text snapshots for `file_diff` added/changed files and serves them through cursor-aware APIs:

1. `GET /api/live/files/<run_id>`
- Returns tracked file manifest with version counts, latest event index, and available snapshot event indexes (`event_indexes`) per tracked file.

2. `GET /api/live/file_snapshot/<run_id>?path=<path_key>&at_event_index=<n>`
- Returns file content from the latest snapshot where `snapshot_event_index <= at_event_index`.
- If `at_event_index` is omitted, latest snapshot is returned.

3. `GET /api/live/file_workspace/<run_id>?path=<path_key>`
- Returns current on-disk content for known tracked file paths.
- Path resolution is constrained to tracked file roots to prevent root escape.

## Run Events API

`GET /api/live/events/<run_id>?since_event_index=<n>&limit=<k>`

- Returns a replayable event timeline for a specific run.
- Source precedence:
1. Active in-memory run session (`RUNS`) when available.
2. Persisted per-run ledger (`.live_runs/<run_id>/events.jsonl`).
3. `trace.json` synthetic fallback (emits `started` + `llm_call*` + `completed`) when ledger is absent.
- Supports cursor-windowing via `since_event_index` and bounded response size via `limit`.

## Backward Compatibility Guarantees

1. Existing payload keys are not moved or renamed.
2. Existing `type` field remains unchanged.
3. Existing `seq` and `time` remain present.
4. `seq` is a legacy compatibility alias and is always equal to `event_index` in v1.
5. Viewer can continue using old behavior while gradually adopting `event_index`/`facet`/`entity_id`.

## Example (Todo App Run)

`file_diff` event (representative shape):

```json
{
  "schema_version": "1.0",
  "run_id": "e83b43cb0f52",
  "event_id": "e83b43cb0f52:67",
  "event_index": 67,
  "seq": 67,
  "time": 1771775368.11,
  "ts": "2026-02-22T15:09:28.110000+00:00",
  "type": "file_diff",
  "facet": "files",
  "entity_id": "file:autogpt_workspace/todo_app.py",
  "actor": "runtime",
  "span_id": "span.file_diff.e83b43cb0f52.67",
  "parent_span_id": "span.state.e83b43cb0f52.61",

  "payload": {
    "process": "execute_command",
    "added_files": ["autogpt_workspace/todo_app.py"],
    "changed_files": [],
    "removed_files": [],
    "patches": {"autogpt_workspace/todo_app.py": "..."}
  }
}
```

## Validation / Tests

Coverage in `tests/test_live_viewer_server.py` now includes:
1. Required envelope fields present.
2. Monotonic `event_index` (`[1,2,3,...]`).
3. `event_index == seq` in v1.
4. Basic span parent linkage (`llm_call` parent = latest `state` span).
5. `events.jsonl` persistence parity with emitted events.

## Immediate Follow-on (Planned)

1. Frontend canonical single event store (`liveEvents`) with projections for State Machine / Data Flow / UML.
2. Cross-pane cursor semantics hardening over the global `event_index` cursor (selection persistence + replay friendliness).
3. External-call routing quality improvements for retrieval intents in mixed build/research objectives.
