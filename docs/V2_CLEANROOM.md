# Live Viewer v2 (Clean-Room Reboot)

## Goal
Build a fresh live runtime + viewer with strict contracts and low coupling to v1.

v1 remains unchanged (`/api/live/*`, `spec-viewer.html`).

## Scope in v2
- Server module: `agent_ontology/v2/api_server.py`
- Viewer page: `agent_ontology/v2/spec-viewer-v2.html`
- CLI entrypoint: `ao-live-viewer-v2`
- Tests: `tests/v2/test_api_server_v2.py`

## Architecture
v2 runtime is native and decoupled from `agent_ontology/live_viewer_server.py` internals.

Core runtime object:
- `V2RunSession`

Responsibilities:
- launch generated agent subprocess
- parse runtime stdout into semantic events
- persist canonical event ledger (`events.jsonl`)
- maintain in-memory event stream for SSE
- index file snapshots by `event_index`
- serve file manifest + snapshot-at-cursor reads
- bind run completion output back into session turns

## Event model
All events are normalized with envelope fields:
- `schema_version`
- `run_id`
- `event_index`
- `seq` (compat alias of `event_index`)
- `event_id`
- `ts`
- `type`
- `facet`
- `entity_id`
- `span_id`
- `parent_span_id`
- `actor`

Additional semantic events emitted from runtime logs include:
- `state`
- `step_action`
- `plan_step`
- `routing_decision`
- `llm_call`
- `llm_summary`
- `tool_result`
- `observation`
- `schema_violation`
- `failure`
- `runtime_info`
- `file_diff`
- `file_snapshot_scan`
- `final_data`
- `final_channel`
- `completed`

## Session model (iterative prompting)
Sessions are persisted under:
- `.live_runs/_sessions/<session_id>.json`

Session payload tracks:
- `session_id`
- `created_at`
- `updated_at`
- `run_ids[]`
- `turns[]` (`role`, `text`, `time`, `run_id`, `event_index`)

`POST /api/v2/run` accepts `session_id` and auto-wires:
- user prompt -> session turn
- run completion/final output -> assistant turn

## Snapshot-at-cursor model
Snapshots are stored under:
- `.live_runs/<run_id>/file_snapshots/<event_index>/<workspace-relative-path>`

`file_diff` events carry:
- `path`
- `change_type` (`added|modified|deleted`)
- `snapshot_rel`
- `snapshot_event_index`
- `size`
- `is_text`

Manifest and file reads are snapshot-first when available.

Retention/prune controls:
- `AGENT_ONTOLOGY_V2_SNAPSHOT_KEEP_EVENTS` (default `160`)
- `AGENT_ONTOLOGY_V2_SNAPSHOT_KEEP_VERSIONS_PER_FILE` (default `24`)
- automatic pruning after snapshot capture
- manual pruning endpoint for active runs:
  - `POST /api/v2/prune/<run_id>` with optional JSON body `{ "force": true }`

## API contract (`/api/v2/*`)

### Health/spec discovery
- `GET /api/v2/health`
- `GET /api/v2/specs`

### Session APIs
- `GET /api/v2/sessions?limit=<n>`
- `POST /api/v2/session/new`
- `GET /api/v2/session/<session_id>`
- `POST /api/v2/session/<session_id>/turn`

### Run APIs
- `GET /api/v2/runs?limit=<n>&session_id=<optional>`
- `POST /api/v2/run`
- `POST /api/v2/stop/<run_id>`
- `POST /api/v2/prune/<run_id>`
- `GET /api/v2/status/<run_id>`

### Event APIs
- `GET /api/v2/events/<run_id>?since_event_index=<n>&limit=<k>`
- `GET /api/v2/stream/<run_id>`
  - SSE supports `since_event_index`, `since`, and `Last-Event-ID`

### File APIs
- `GET /api/v2/files/<run_id>`
- `GET /api/v2/file/<run_id>?path=<rel>&source=snapshot|workspace&at_event_index=<n>&start=<n>&limit=<n>`
  - snapshot-first with workspace fallback if needed

## Viewer v2 UX model
Layout:
- Left: full-height Explorer (one section visible at a time)
- Center: Runtime canvas + bottom chat/run console
- Right: Inspector + file preview

Explorer sections:
- `Runs`
- `Files`
- `Memory`
- `Views`
- `Settings` (advanced mode)

Viewer controls:
- `Simple` vs `Advanced` mode toggle
- Explorer show/hide toggle
- Resizable left panel drag handle

Runtime lenses (all live and event-driven):
- `Timeline`
- `State`
- `Dataflow`
- `Sequence`

Graph lenses support:
- click node -> select event
- drag nodes -> reposition
- drag blank canvas -> pan

## Run commands
From repo root:

```bash
PYTHONPATH=. python3 -m agent_ontology.v2.api_server --host 127.0.0.1 --port 8012 --root .
```

Open:

```text
http://127.0.0.1:8012/agent_ontology/v2/spec-viewer-v2.html
```

After editable install:

```bash
ao-live-viewer-v2 --host 127.0.0.1 --port 8012 --root .
```

## Tests

```bash
cd "$(pwd)"
PYTHONPATH=. pytest -q tests/v2/test_api_server_v2.py
```

Current v2 tests validate:
- persisted run discovery
- workspace manifest fallback
- snapshot fallback behavior
- session binding + turn persistence
- snapshot-at-cursor version selection
- HTTP session endpoint round-trip (`/api/v2/session/new`, `/api/v2/session/<id>`, `/api/v2/session/<id>/turn`, `/api/v2/sessions`)
- run listing filter by `session_id`
- SSE resume precedence in stream handler (`since_event_index` > `since` > `Last-Event-ID`)
- snapshot retention pruning by version window and event window
- manual prune API behavior (`POST /api/v2/prune/<run_id>`)

## Remaining milestones
1. Expand semantic parsing coverage for tool/external/store-specific events
2. Add explicit lane overlays for external-call traces in runtime canvas
3. Add run replay UX controls for cursor stepping/pinning across all lenses
4. Add prune observability in viewer (snapshot stats and manual prune action)
