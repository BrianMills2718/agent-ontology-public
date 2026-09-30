from __future__ import annotations

import io
import json
import subprocess
import pytest
import sys
import tempfile
from pathlib import Path

from agent_ontology.live_viewer_server import (
    RunSession,
    LiveViewerHandler,
    _build_initial_data_for_run,
    _build_runner_code,
    _hydrate_persisted_run_session,
    _is_valid_run_id,
    _load_persisted_run_events,
    _list_review_artifacts,
    _run_artifact_dir,
    _load_review_artifact,
    _review_panel_summary,
    _resolve_stream_since,
    _resolve_model_override,
)


def _llm_client_has_call_llm() -> bool:
    """Is a *usable* llm_client importable?

    The live-viewer runner refuses to execute without `llm_client.call_llm`,
    by design -- it will not silently fall back to a non-llm_client path. CI
    installs `llm-client` from PyPI and gets a distribution whose top-level
    module has no `call_llm`, so these six tests fail there with a bare
    `assert 1 == 0` while passing locally against an editable install of the
    real package.

    Skipping is the honest state: the dependency is genuinely absent, and a
    skip that names why is worth more than a red job everyone learns to
    ignore. It is NOT a fix -- see the note in agents/HAND_EDITED.md's sibling
    discussion in the PR: CI needs a route to the real package before this
    runtime path is covered there at all.
    """
    try:
        import llm_client
    except Exception:
        return False
    return callable(getattr(llm_client, "call_llm", None))


requires_llm_client = pytest.mark.skipif(
    not _llm_client_has_call_llm(),
    reason=(
        "llm_client.call_llm unavailable; the live-viewer runner refuses to run "
        "without it. The PyPI 'llm-client' distribution does not provide it."
    ),
)


def _run_runner(agent_source: str, initial_data: dict[str, object]) -> tuple[int, str]:
    """Run the live-viewer runner over an inline agent and return (rc, stdout).

    stderr is folded into the returned stdout on failure. These tests used to
    discard it, so a non-zero exit surfaced only as `assert 1 == 0` with no
    indication of why -- which is exactly how six of them stayed unexplained
    across a Python version difference.
    """
    with tempfile.TemporaryDirectory() as td:
        agent_path = Path(td) / "fake_agent.py"
        agent_path.write_text(agent_source)
        code = _build_runner_code(agent_path, initial_data)
        proc = subprocess.run(
            [sys.executable, "-u", "-c", code],
            capture_output=True,
            text=True,
            check=False,
            cwd=td,
        )
        if proc.returncode != 0:
            return proc.returncode, (
                f"{proc.stdout}\n--- runner exited {proc.returncode}; stderr ---\n{proc.stderr}"
            )
        return proc.returncode, proc.stdout


def _parse_prefixed_json(stdout: str, prefix: str) -> object:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise AssertionError(f"Missing {prefix} in output:\n{stdout}")


class _FakeStreamHandler:
    def __init__(self, last_event_id: str | None = None):
        self.headers: dict[str, str] = {}
        if last_event_id is not None:
            self.headers["Last-Event-ID"] = last_event_id
        self.wfile = io.BytesIO()
        self.status_code: int | None = None
        self.response_headers: list[tuple[str, str]] = []

    def send_response(self, code: int) -> None:
        self.status_code = code

    def send_header(self, key: str, value: str) -> None:
        self.response_headers.append((key, value))

    def end_headers(self) -> None:
        return


def _parse_sse_ids_and_events(raw_text: str) -> tuple[list[int], list[dict[str, object]]]:
    event_ids: list[int] = []
    events: list[dict[str, object]] = []
    for line in raw_text.splitlines():
        if line.startswith("id: "):
            event_ids.append(int(line.split(":", 1)[1].strip()))
            continue
        if line.startswith("data: "):
            payload = json.loads(line[len("data: ") :])
            assert isinstance(payload, dict)
            events.append(payload)
    return event_ids, events


@requires_llm_client
def test_runner_injects_fallback_channel_for_emit_final() -> None:
    source = """
PROCESSES = {}

class AgentState:
    def __init__(self):
        self.data = {}
        self.iteration = 0
        self.schema_violations = 0

def process_emit_final(state):
    # No final_channel on AgentState: wrapper should inject fallback channel.
    state.final_channel.write({"answer": "ok from fallback"})
    state.data["answer"] = "ok from fallback"
    return state

PROCESSES["emit_final"] = process_emit_final

def run(initial=None):
    state = AgentState()
    if isinstance(initial, dict):
        state.data.update(initial)
    PROCESSES["emit_final"](state)
    return state
"""
    rc, stdout = _run_runner(source, {"text": "hello"})
    assert rc == 0, stdout
    final_channel = _parse_prefixed_json(stdout, "__AO_FINAL_CHANNEL__")
    assert isinstance(final_channel, dict)
    assert final_channel.get("source") == "state.final_channel"
    value = final_channel.get("value")
    assert isinstance(value, dict)
    assert value.get("answer") == "ok from fallback"


@requires_llm_client
def test_runner_single_turn_marks_done_after_emit_final() -> None:
    source = """
PROCESSES = {}

class AgentState:
    def __init__(self):
        self.data = {}
        self.iteration = 0
        self.schema_violations = 0

def process_emit_final(state):
    state.data["answer"] = "single turn"
    return state

PROCESSES["emit_final"] = process_emit_final

def run(initial=None):
    state = AgentState()
    if isinstance(initial, dict):
        state.data.update(initial)
    loops = 0
    while loops < 20 and not state.data.get("_done"):
        PROCESSES["emit_final"](state)
        loops += 1
    state.data["loop_count"] = loops
    return state
"""
    rc, stdout = _run_runner(source, {"text": "hello"})
    assert rc == 0, stdout
    final_data = _parse_prefixed_json(stdout, "__AO_FINAL_DATA__")
    assert isinstance(final_data, dict)
    assert final_data.get("_done") is True
    assert final_data.get("loop_count") == 1


@requires_llm_client
def test_runner_emits_file_diff_for_workspace_writes() -> None:
    source = """
import tempfile
from pathlib import Path

PROCESSES = {}

class AgentState:
    def __init__(self):
        self.data = {}
        self.iteration = 0
        self.schema_violations = 0

def process_write_file(state):
    ws = Path(state.data["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "todo.txt").write_text("line-one\\nline-two\\n", encoding="utf-8")
    state.data["written"] = str(ws / "todo.txt")
    return state

PROCESSES["write_file"] = process_write_file

def run(initial=None):
    state = AgentState()
    ws = tempfile.mkdtemp(prefix="ao_ws_")
    state.data["workspace_dir"] = ws
    if isinstance(initial, dict):
        state.data.update(initial)
    PROCESSES["write_file"](state)
    return state
"""
    rc, stdout = _run_runner(source, {"text": "hello"})
    assert rc == 0, stdout
    file_diff = _parse_prefixed_json(stdout, "__AO_FILE_DIFF__")
    assert isinstance(file_diff, dict)
    added_files = file_diff.get("added_files")
    assert isinstance(added_files, list)
    assert any(str(path).endswith("/todo.txt") for path in added_files)
    patches = file_diff.get("patches")
    assert isinstance(patches, dict)
    patch_text = next(iter(patches.values()), "")
    assert isinstance(patch_text, str)
    assert "todo.txt" in patch_text


@requires_llm_client
def test_runner_emits_file_diff_with_cwd_fallback_without_workspace_dir() -> None:
    source = """
from pathlib import Path

PROCESSES = {}

class AgentState:
    def __init__(self):
        self.data = {}
        self.iteration = 0
        self.schema_violations = 0

def process_write_file(state):
    Path("cwd_written.txt").write_text("hello-cwd\\n", encoding="utf-8")
    state.data["written"] = "cwd_written.txt"
    return state

PROCESSES["write_file"] = process_write_file

def run(initial=None):
    state = AgentState()
    if isinstance(initial, dict):
        state.data.update(initial)
    PROCESSES["write_file"](state)
    return state
"""
    rc, stdout = _run_runner(source, {"text": "hello"})
    assert rc == 0, stdout
    file_diff = _parse_prefixed_json(stdout, "__AO_FILE_DIFF__")
    assert isinstance(file_diff, dict)
    added_files = file_diff.get("added_files")
    assert isinstance(added_files, list)
    assert any(str(path).endswith("/cwd_written.txt") for path in added_files)


@requires_llm_client
def test_runner_emits_file_diff_when_workspace_is_under_live_runs_dir() -> None:
    source = """
from pathlib import Path

PROCESSES = {}

class AgentState:
    def __init__(self):
        self.data = {}
        self.iteration = 0
        self.schema_violations = 0

def process_write_file(state):
    ws = Path(state.data["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "under_live_runs.txt").write_text("hello-live-runs\\n", encoding="utf-8")
    state.data["written"] = str(ws / "under_live_runs.txt")
    return state

PROCESSES["write_file"] = process_write_file

def run(initial=None):
    state = AgentState()
    if isinstance(initial, dict):
        state.data.update(initial)
    state.data["workspace_dir"] = str(Path(".live_runs/run123/autogpt_workspace").resolve())
    PROCESSES["write_file"](state)
    return state
"""
    rc, stdout = _run_runner(source, {"text": "hello"})
    assert rc == 0, stdout
    file_diff = _parse_prefixed_json(stdout, "__AO_FILE_DIFF__")
    assert isinstance(file_diff, dict)
    added_files = file_diff.get("added_files")
    assert isinstance(added_files, list)
    assert any(str(path).endswith("/under_live_runs.txt") for path in added_files)


@requires_llm_client
def test_runner_ignores_live_runs_subtree_but_keeps_regular_trace_json_in_file_diff() -> None:
    source = """
from pathlib import Path

PROCESSES = {}

class AgentState:
    def __init__(self):
        self.data = {}
        self.iteration = 0
        self.schema_violations = 0

def process_write_files(state):
    Path("todo.txt").write_text("task-one\\n", encoding="utf-8")
    Path("debug").mkdir(parents=True, exist_ok=True)
    Path("debug/trace.json").write_text("{\\"trace\\": true}\\n", encoding="utf-8")
    Path(".live_runs/abc").mkdir(parents=True, exist_ok=True)
    Path(".live_runs/abc/events.jsonl").write_text("{\\"run\\": \\"meta\\"}\\n", encoding="utf-8")
    Path(".live_runs/abc/trace.json").write_text("{\\"trace\\": true}\\n", encoding="utf-8")
    state.data["written"] = "todo.txt"
    return state

PROCESSES["write_files"] = process_write_files

def run(initial=None):
    state = AgentState()
    if isinstance(initial, dict):
        state.data.update(initial)
    PROCESSES["write_files"](state)
    return state
"""
    rc, stdout = _run_runner(source, {"text": "hello"})
    assert rc == 0, stdout
    file_diff = _parse_prefixed_json(stdout, "__AO_FILE_DIFF__")
    assert isinstance(file_diff, dict)
    added_files = file_diff.get("added_files")
    assert isinstance(added_files, list)
    assert any(str(path).endswith("/todo.txt") for path in added_files)
    assert any(str(path).endswith("/debug/trace.json") for path in added_files)
    assert not any("/.live_runs/" in str(path) for path in added_files)


def _build_stream_test_session(tmp_path: Path, run_id: str = "stream001") -> RunSession:
    session = RunSession(
        run_id=run_id,
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="stream",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    ledger = tmp_path / f"{run_id}_events.jsonl"
    session._events_log_path = ledger
    session._events_log_path.write_text("", encoding="utf-8")
    session.emit("started", input={"text": "build me a full app that handles todos"})
    session.emit("state", process="receive_request", iteration=1, process_metrics={})
    session.emit(
        "llm_call",
        process="receive_request",
        agent="Planner",
        model="openrouter/deepseek/deepseek-chat",
        duration_ms=12,
        process_metrics={},
    )
    session.done = True
    return session


def test_resolve_stream_since_precedence_and_fallback() -> None:
    assert _resolve_stream_since({"since_event_index": ["7"]}, {"Last-Event-ID": "2"}) == 7
    assert _resolve_stream_since({"since_event_index": ["bad"], "since": ["5"]}, {"Last-Event-ID": "2"}) == 5
    assert _resolve_stream_since({"since": ["3"]}, {"Last-Event-ID": "9"}) == 3
    assert _resolve_stream_since({}, {"Last-Event-ID": "4"}) == 4
    assert _resolve_stream_since({}, {"Last-Event-ID": "bad"}) == 0


def test_run_id_validation_for_events_api() -> None:
    assert _is_valid_run_id("abc123")
    assert _is_valid_run_id("run_1.test-2")
    assert not _is_valid_run_id("")
    assert not _is_valid_run_id("../escape")
    assert not _is_valid_run_id("bad/slash")
    assert not _is_valid_run_id("bad id")


def test_load_persisted_run_events_from_ledger(tmp_path: Path) -> None:
    live_runs = tmp_path / ".live_runs" / "ledger123"
    live_runs.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event_index": 1, "type": "started", "facet": "input", "run_id": "ledger123"},
        {"event_index": 2, "type": "llm_call", "facet": "llm", "agent": "Planner", "run_id": "ledger123"},
        {"event_index": 3, "type": "completed", "facet": "done", "run_id": "ledger123"},
    ]
    (live_runs / "events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    events, source = _load_persisted_run_events(tmp_path, "ledger123")
    assert source == "ledger"
    assert [int(evt.get("event_index", -1)) for evt in events] == [1, 2, 3]
    assert [str(evt.get("type", "")) for evt in events] == ["started", "llm_call", "completed"]


def test_load_persisted_run_events_from_trace_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / ".live_runs" / "trace123"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_payload = {
        "metrics": {"total_llm_calls": 1, "clean_exit": True},
        "trace": [
            {
                "timestamp": "2026-02-23T10:00:00Z",
                "agent": "Planner",
                "model": "openrouter/deepseek/deepseek-chat",
                "duration_ms": 42,
                "response": "{\"action_type\":\"respond\"}",
            }
        ],
    }
    (run_dir / "trace.json").write_text(json.dumps(trace_payload), encoding="utf-8")
    events, source = _load_persisted_run_events(tmp_path, "trace123")
    assert source == "trace_fallback"
    assert [str(evt.get("type", "")) for evt in events] == ["started", "llm_call", "completed"]
    assert int(events[1].get("event_index", 0)) == 2
    assert str(events[1].get("agent", "")) == "Planner"


def test_stream_events_resumes_from_last_event_id_header(tmp_path: Path) -> None:
    session = _build_stream_test_session(tmp_path, run_id="streamhdr")
    handler = _FakeStreamHandler(last_event_id="1")
    LiveViewerHandler._stream_events(handler, session, "")
    text = handler.wfile.getvalue().decode("utf-8")
    ids, events = _parse_sse_ids_and_events(text)
    assert handler.status_code == 200
    assert ids == [2, 3]
    assert [int(evt.get("event_index", -1)) for evt in events] == [2, 3]


def test_stream_events_query_overrides_last_event_id_header(tmp_path: Path) -> None:
    session = _build_stream_test_session(tmp_path, run_id="streamquery")
    handler = _FakeStreamHandler(last_event_id="1")
    LiveViewerHandler._stream_events(handler, session, "since_event_index=2")
    ids, _events = _parse_sse_ids_and_events(handler.wfile.getvalue().decode("utf-8"))
    assert ids == [3]


def test_stream_events_match_ledger_structural_envelope(tmp_path: Path) -> None:
    session = _build_stream_test_session(tmp_path, run_id="streamparity")
    handler = _FakeStreamHandler()
    LiveViewerHandler._stream_events(handler, session, "")
    stream_ids, stream_events = _parse_sse_ids_and_events(handler.wfile.getvalue().decode("utf-8"))

    ledger_path = session._events_log_path
    assert ledger_path is not None
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stream_events) == len(rows)
    assert stream_ids == [int(row["event_index"]) for row in rows]

    envelope_keys = (
        "schema_version",
        "run_id",
        "event_index",
        "type",
        "facet",
        "entity_id",
        "span_id",
        "parent_span_id",
    )
    for stream_evt, row in zip(stream_events, rows):
        for key in envelope_keys:
            assert stream_evt.get(key) == row.get(key)
        assert int(stream_evt.get("seq", -1)) == int(stream_evt.get("event_index", -2))
        assert int(row.get("seq", -1)) == int(row.get("event_index", -2))


def test_run_session_file_snapshot_manifest_and_cursor_lookup(tmp_path: Path) -> None:
    session = RunSession(
        run_id="filesnap1",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="files",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    session.run_dir = run_dir
    session._snapshot_dir = run_dir / "file_snapshots"
    session._snapshot_dir.mkdir(parents=True, exist_ok=True)

    workspace = tmp_path / "workspace"
    target = workspace / "debug" / "trace.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"version": 1}\n', encoding="utf-8")
    path_key = "ws/debug/trace.json"

    payload_added = {
        "process": "act",
        "added_files": [path_key],
        "changed_files": [],
        "removed_files": [],
        "previews": {path_key: {"path": str(target)}},
    }
    payload_changed = {
        "process": "act",
        "added_files": [],
        "changed_files": [path_key],
        "removed_files": [],
        "previews": {path_key: {"path": str(target)}},
    }

    session._handle_line("__AO_FILE_DIFF__" + json.dumps(payload_added))
    target.write_text('{"version": 2}\n', encoding="utf-8")
    session._handle_line("__AO_FILE_DIFF__" + json.dumps(payload_changed))

    manifest = session.file_manifest()
    files = manifest.get("files")
    assert isinstance(files, list)
    match = next((f for f in files if f.get("path") == path_key), None)
    assert isinstance(match, dict)
    assert match.get("versions") == 2
    assert match.get("latest_event_index") == 2
    assert match.get("content_available") is True
    assert match.get("event_indexes") == [1, 2]

    status_v1, snap_v1 = session.get_file_snapshot(path_key, 1)
    assert status_v1 == 200
    assert snap_v1.get("snapshot_event_index") == 1
    assert snap_v1.get("source") == "snapshot"
    assert '"version": 1' in str(snap_v1.get("content", ""))

    status_v2, snap_v2 = session.get_file_snapshot(path_key, 2)
    assert status_v2 == 200
    assert snap_v2.get("snapshot_event_index") == 2
    assert '"version": 2' in str(snap_v2.get("content", ""))

    status_ws, ws_payload = session.get_workspace_file(path_key)
    assert status_ws == 200
    assert ws_payload.get("source") == "workspace"
    assert '"version": 2' in str(ws_payload.get("content", ""))


def test_run_session_file_snapshot_deleted_file_behavior(tmp_path: Path) -> None:
    session = RunSession(
        run_id="filesnap2",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="files",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    session.run_dir = run_dir
    session._snapshot_dir = run_dir / "file_snapshots"
    session._snapshot_dir.mkdir(parents=True, exist_ok=True)

    workspace = tmp_path / "workspace"
    target = workspace / "todo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("first\n", encoding="utf-8")
    path_key = "ws/todo.txt"

    payload_added = {
        "process": "act",
        "added_files": [path_key],
        "changed_files": [],
        "removed_files": [],
        "previews": {path_key: {"path": str(target)}},
    }
    payload_removed = {
        "process": "act",
        "added_files": [],
        "changed_files": [],
        "removed_files": [path_key],
        "previews": {path_key: {"path": str(target)}},
    }

    session._handle_line("__AO_FILE_DIFF__" + json.dumps(payload_added))
    target.unlink()
    session._handle_line("__AO_FILE_DIFF__" + json.dumps(payload_removed))

    status_latest, payload_latest = session.get_file_snapshot(path_key, None)
    assert status_latest == 404
    assert "deleted" in str(payload_latest.get("error", "")).lower()

    status_old, payload_old = session.get_file_snapshot(path_key, 1)
    assert status_old == 200
    assert "first" in str(payload_old.get("content", ""))


def test_run_session_file_snapshot_and_workspace_support_chunked_reads(tmp_path: Path) -> None:
    session = RunSession(
        run_id="filesnap3",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="files",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    session.run_dir = run_dir
    session._snapshot_dir = run_dir / "file_snapshots"
    session._snapshot_dir.mkdir(parents=True, exist_ok=True)

    workspace = tmp_path / "workspace"
    target = workspace / "todo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    path_key = "ws/todo.txt"

    payload_added = {
        "process": "act",
        "added_files": [path_key],
        "changed_files": [],
        "removed_files": [],
        "previews": {path_key: {"path": str(target)}},
    }
    session._handle_line("__AO_FILE_DIFF__" + json.dumps(payload_added))

    status_chunk, chunk_payload = session.get_file_snapshot(path_key, 1, start=0, limit=8)
    assert status_chunk == 200
    assert chunk_payload.get("content") == "abcdefgh"


def test_run_session_file_lookup_normalizes_run_id_prefixed_paths(tmp_path: Path) -> None:
    session = RunSession(
        run_id="normrun",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="files",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    session.run_dir = run_dir
    session._snapshot_dir = run_dir / "file_snapshots"
    session._snapshot_dir.mkdir(parents=True, exist_ok=True)

    workspace = tmp_path / "workspace"
    target = workspace / "notes.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("line 1\nline 2\n", encoding="utf-8")

    path_key = "normrun/notes.txt"
    payload_added = {
        "process": "act",
        "added_files": [path_key],
        "changed_files": [],
        "removed_files": [],
        "previews": {path_key: {"path": str(target)}},
    }
    session._handle_line("__AO_FILE_DIFF__" + json.dumps(payload_added))

    manifest = session.file_manifest()
    files = manifest.get("files", [])
    assert any(row.get("path") == "notes.txt" for row in files)

    status_snapshot, snapshot = session.get_file_snapshot(path_key, None)
    assert status_snapshot == 200
    assert "line 2" in str(snapshot.get("content", ""))

    status_snapshot_legacy, snapshot_legacy = session.get_file_snapshot("notes.txt", None)
    assert status_snapshot_legacy == 200
    assert "line 1" in str(snapshot_legacy.get("content", ""))
    status_ws, ws_payload = session.get_workspace_file("notes.txt", start=0, limit=6)
    assert status_ws == 200
    assert ws_payload.get("content") == "line 1"


def test_run_session_file_manifest_fallback_uses_workspace_artifacts_without_file_diff(tmp_path: Path) -> None:
    session = RunSession(
        run_id="filesnap_fallback",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="files",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    run_dir = tmp_path / ".live_runs" / "filesnap_fallback"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "autogpt_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "hello_world.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    session.run_dir = run_dir
    session._events_log_path = run_dir / "events.jsonl"
    session.events = [{"event_index": 9, "type": "completed"}]
    session.final_data = {
        "workspace_dir": str(workspace),
        "artifact_files": ["hello_world.py"],
    }

    manifest = session.file_manifest()
    files = manifest.get("files")
    assert isinstance(files, list)
    path_key = "autogpt_workspace/hello_world.py"
    match = next((row for row in files if row.get("path") == path_key), None)
    assert isinstance(match, dict)
    assert match.get("deleted") is False
    assert match.get("content_available") is True

    status_snap, snap_payload = session.get_file_snapshot(path_key, None)
    assert status_snap == 200
    assert snap_payload.get("source") == "snapshot"
    assert "print('hello')" in str(snap_payload.get("content", ""))

    status, payload = session.get_workspace_file(path_key)
    assert status == 200
    assert payload.get("source") == "workspace"
    assert "print('hello')" in str(payload.get("content", ""))


def test_hydrate_persisted_run_session_enables_workspace_file_manifest(tmp_path: Path) -> None:
    run_id = "persisted_files_001"
    run_dir = tmp_path / ".live_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "autogpt_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "demo.py").write_text("x = 1\n", encoding="utf-8")

    events = [
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "event_index": 1,
            "seq": 1,
            "type": "started",
            "facet": "input",
            "input": {"workspace_dir": str(workspace)},
        },
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "event_index": 2,
            "seq": 2,
            "type": "final_data",
            "facet": "output",
            "data": {"workspace_dir": str(workspace), "artifact_files": ["demo.py"]},
        },
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "event_index": 3,
            "seq": 3,
            "type": "completed",
            "facet": "done",
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events) + "\n",
        encoding="utf-8",
    )

    session = _hydrate_persisted_run_session(tmp_path, run_id)
    assert session is not None
    manifest = session.file_manifest()
    files = manifest.get("files")
    assert isinstance(files, list)
    path_key = "autogpt_workspace/demo.py"
    row = next((item for item in files if str(item.get("path")) == path_key), None)
    assert isinstance(row, dict)
    assert row.get("content_available") is True
    status_snap, snap_payload = session.get_file_snapshot(path_key, None)
    assert status_snap == 200
    assert snap_payload.get("source") == "snapshot"
    assert "x = 1" in str(snap_payload.get("content", ""))
    status, payload = session.get_workspace_file(path_key)
    assert status == 200
    assert "x = 1" in str(payload.get("content", ""))


def test_run_artifact_dir_uses_nested_repo_layout(tmp_path: Path) -> None:
    run_id = "nested_layout_001"
    nested_repo = tmp_path / "agent_ontology"
    (tmp_path / "spec-viewer.html").write_text("<html></html>", encoding="utf-8")
    nested_live = (nested_repo / ".live_runs" / run_id)
    (tmp_path / "agent_ontology").mkdir(parents=True, exist_ok=True)
    nested_live.mkdir(parents=True, exist_ok=True)
    events_path = nested_live / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "event_index": 1,
                "seq": 1,
                "type": "started",
                "facet": "input",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _run_artifact_dir(tmp_path, run_id) == nested_live.resolve()

    session = _hydrate_persisted_run_session(tmp_path, run_id)
    assert session is not None
    assert session.run_dir == nested_live.resolve()


def test_run_session_parses_delegate_events() -> None:
    session = RunSession(
        run_id="test123",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="test",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=10,
        model_override="",
        max_budget_usd=None,
    )

    session._handle_line(
        '__AO_DELEGATE_TRACE__{"delegate_id":"d1","child_run_id":"child001","path":"/tmp/trace.json","total_calls":2,"total_duration_ms":55}'
    )
    session._handle_line(
        '__AO_DELEGATE_RUN__{"delegate_id":"d1","child_run_id":"child001","child_run_ids":["child001"],"agent_path":"/tmp/agent.py","returncode":0,"duration_ms":120}'
    )

    trace_events = [e for e in session.events if e.get("type") == "delegate_trace_file"]
    run_events = [e for e in session.events if e.get("type") == "delegate_run"]
    assert trace_events
    assert run_events
    trace_payload = trace_events[-1].get("payload")
    assert isinstance(trace_payload, dict)
    assert trace_payload.get("path") == "/tmp/trace.json"
    assert trace_events[-1].get("facet") == "delegate"
    assert trace_events[-1].get("related_run_id") == "child001"
    assert trace_events[-1].get("causal_link") == "delegate_parent_child_run"
    assert run_events[-1].get("facet") == "delegate"
    assert run_events[-1].get("related_run_id") == "child001"
    assert run_events[-1].get("causal_link") == "delegate_parent_child_run"


def test_run_session_emits_event_envelope_and_persists_jsonl(tmp_path: Path) -> None:
    session = RunSession(
        run_id="env001",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="envelope",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    ledger = tmp_path / "events.jsonl"
    session._events_log_path = ledger
    session._events_log_path.write_text("", encoding="utf-8")

    session.emit("started", input={"text": "build me a full app that handles todos"})
    session.emit("state", process="receive_request", iteration=1, process_metrics={})
    session.emit(
        "llm_call",
        process="receive_request",
        agent="Planner",
        model="openrouter/deepseek/deepseek-chat",
        duration_ms=12,
        process_metrics={},
    )

    assert len(session.events) == 3
    indices = [int(e.get("event_index", -1)) for e in session.events]
    assert indices == [1, 2, 3]
    assert [int(e.get("seq", -1)) for e in session.events] == indices
    for evt in session.events:
        assert evt.get("schema_version") == "1.0"
        assert evt.get("run_id") == "env001"
        assert isinstance(evt.get("ts"), str) and evt.get("ts")
        assert evt.get("event_id")
        assert evt.get("facet")
        assert "actor" in evt
        assert "span_id" in evt
        assert "parent_span_id" in evt

    started, state_evt, llm_evt = session.events
    assert started.get("facet") == "input"
    assert state_evt.get("facet") == "state"
    assert llm_evt.get("facet") == "llm"
    assert started.get("parent_span_id") is None
    assert isinstance(state_evt.get("parent_span_id"), str) and state_evt.get("parent_span_id")
    assert llm_evt.get("parent_span_id") == state_evt.get("span_id")

    lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [int(row["event_index"]) for row in parsed] == [1, 2, 3]
    assert [row["type"] for row in parsed] == ["started", "state", "llm_call"]


def test_run_session_final_data_summary_includes_quality_gate_fields() -> None:
    session = RunSession(
        run_id="quality1",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="quality",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    session.final_data = {
        "last_channel": "final",
        "quality_gate_failed": True,
        "quality_gate_reason": "all_candidate_files_empty",
        "quality_file_check": {
            "candidate_count": 3,
            "non_empty_count": 0,
            "empty_count": 3,
            "empty_files": ["README.md", "src/app.py"],
            "non_empty_files": [],
        },
    }

    summary = session._final_data_summary()
    assert summary.get("quality_gate_failed") is True
    assert summary.get("quality_gate_reason") == "all_candidate_files_empty"
    quality_check = summary.get("quality_file_check")
    assert isinstance(quality_check, dict)
    assert quality_check.get("candidate_count") == 3
    assert quality_check.get("non_empty_count") == 0
    assert quality_check.get("empty_count") == 3
    assert quality_check.get("empty_files") == ["README.md", "src/app.py"]


def test_run_session_golden_event_sequence_for_live_overlay_markers() -> None:
    session = RunSession(
        run_id="golden1",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="golden",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    session.run_dir = Path.cwd()

    lines = [
        "[Iteration 1] State: receive_request",
        '__AO_STEP_DIFF__{"process":"receive_request","added":["request"],"changed":[],"removed":[]}',
        '__AO_STORE_DIFF__{"process":"receive_request","added_stores":["memory"],"changed_stores":[],"removed_stores":[]}',
        '__AO_FILE_DIFF__{"process":"receive_request","added_files":["ws/todo.txt"],"changed_files":[],"removed_files":[]}',
        '__AO_DELEGATE_TRACE__{"delegate_id":"d2","path":"trace.json","total_calls":1,"total_duration_ms":12}',
        '__AO_DELEGATE_RUN__{"delegate_id":"d2","agent_path":"/tmp/child.py","returncode":0,"duration_ms":45}',
        "Trace written to trace.json (1 calls, 12ms total)",
        '__AO_FINAL_CHANNEL__{"source":"state.final_channel","value":{"answer":"ok"}}',
        '__AO_FINAL_DATA__{"answer":"ok","_done":true}',
    ]
    for line in lines:
        session._handle_line(line)

    event_types = [str(e.get("type")) for e in session.events if str(e.get("type")) != "log"]
    expected = [
        "state",
        "step_diff",
        "store_diff",
        "file_diff",
        "delegate_trace_file",
        "delegate_run",
        "trace_file",
        "final_channel",
        "final_data",
    ]
    assert event_types == expected


def test_run_session_golden_memory_vector_fixture_shape() -> None:
    session = RunSession(
        run_id="golden_memory_vector",
        root_dir=Path.cwd(),
        spec_path=Path.cwd() / "spec.yaml",
        agent_path=Path.cwd() / "agent.py",
        initial_data={},
        task="golden_memory_vector",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=5,
        model_override="",
        max_budget_usd=None,
    )
    session.run_dir = Path.cwd()
    payload = {
        "process": "write_memory",
        "added_stores": ["memory"],
        "changed_stores": [],
        "removed_stores": [],
        "value_preview": {
            "memory": {
                "before": "",
                "after": json.dumps(
                    {
                        "chunks": [
                            {"text": "alpha retrieval chunk"},
                            {"text": "beta retrieval chunk"},
                        ],
                        "vectors": [
                            [0.11, 0.22, 0.33, 0.44],
                            [0.15, 0.25, 0.35, 0.45],
                        ],
                    }
                ),
            }
        },
    }
    session._handle_line("__AO_STORE_DIFF__" + json.dumps(payload))

    store_events = [evt for evt in session.events if str(evt.get("type")) == "store_diff"]
    assert len(store_events) == 1
    event = store_events[0]
    payload_obj = event.get("payload")
    assert isinstance(payload_obj, dict)
    preview = payload_obj.get("value_preview")
    assert isinstance(preview, dict)
    memory_preview = preview.get("memory")
    assert isinstance(memory_preview, dict)
    encoded_after = memory_preview.get("after")
    assert isinstance(encoded_after, str) and encoded_after
    parsed_after = json.loads(encoded_after)
    assert isinstance(parsed_after.get("chunks"), list) and len(parsed_after["chunks"]) == 2
    assert isinstance(parsed_after.get("vectors"), list) and len(parsed_after["vectors"]) == 2
    assert len(parsed_after["vectors"][0]) == 4


def test_build_initial_data_seeds_babyagi_tasks_from_text() -> None:
    spec_path = Path("agent_ontology/specs/babyagi.yaml")
    payload = {
        "text": "write me code for a todo task api",
        "constraints": [],
        "requested_mode": "default",
    }
    initial = _build_initial_data_for_run(spec_path, payload)
    tasks = initial.get("tasks")
    assert isinstance(tasks, list) and tasks
    first = tasks[0]
    assert isinstance(first, dict)
    assert first.get("description") == "write me code for a todo task api"
    assert initial.get("max_tasks") == 5


def test_build_initial_data_includes_chat_conversation_aliases() -> None:
    spec_path = Path("agent_ontology/specs/autogpt.yaml")
    payload = {
        "text": "build todo api",
        "current_user_message": "build todo api",
        "chat_session_id": "chat_abc123",
        "conversation_history": [
            {"role": "user", "content": "build todo api"},
            {"role": "assistant", "content": "starting build"},
        ],
    }
    initial = _build_initial_data_for_run(spec_path, payload)
    assert initial.get("chat_session_id") == "chat_abc123"
    assert initial.get("current_user_message") == "build todo api"
    convo = initial.get("conversation_history")
    assert isinstance(convo, list) and len(convo) == 2
    assert initial.get("conversation_memory") == convo
    msgs = initial.get("messages")
    assert isinstance(msgs, list) and msgs == convo


def test_resolve_model_override_prefers_payload_value() -> None:
    assert _resolve_model_override("openai/gpt-5") == "openai/gpt-5"


def test_resolve_model_override_uses_default_when_blank(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_ONTOLOGY_DEFAULT_MODEL", raising=False)
    assert _resolve_model_override("") == "openrouter/deepseek/deepseek-chat"


def test_resolve_model_override_uses_env_default_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ONTOLOGY_DEFAULT_MODEL", "openrouter/deepseek/deepseek-r1")
    assert _resolve_model_override("") == "openrouter/deepseek/deepseek-r1"


def test_review_artifact_index_and_load_by_run_id(tmp_path: Path) -> None:
    review_dir = tmp_path / ".live_runs"
    review_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "deterministic_ok": True,
        "review_ok": True,
        "run_id": "abc123",
        "run_request": {"spec_path": "agent_ontology/specs/codex.yaml", "text": "build artifact"},
        "status": {"effective_model": "openrouter/deepseek/deepseek-chat", "returncode": 0},
        "trace_metrics": {"clean_exit": True},
        "event_summary": {"event_counts": {"file_diff": 1}},
        "workspace_files": [{"path": "artifact.txt", "size": 5, "preview": "done\n"}],
        "judge": {"report": {"verdict": "pass", "overall_score_normalized": 0.9}},
        "review_profile": {"profile": "babyagi_strict", "ok": True, "checks": []},
    }
    path = review_dir / "latest_e2e_review_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reviews = _list_review_artifacts(tmp_path, limit=10)
    assert reviews
    assert reviews[0].get("run_id") == "abc123"
    assert reviews[0].get("overall_score_normalized") == 0.9

    loaded = _load_review_artifact(tmp_path, "abc123")
    assert loaded is not None
    loaded_path, loaded_payload = loaded
    assert loaded_path == path
    assert loaded_payload.get("run_id") == "abc123"


def test_review_panel_summary_golden_shape() -> None:
    payload = {
        "ok": True,
        "deterministic_ok": True,
        "review_ok": False,
        "run_id": "golden-run",
        "run_request": {"spec_path": "agent_ontology/specs/codex.yaml", "text": "write artifact"},
        "status": {"effective_model": "openrouter/deepseek/deepseek-chat", "returncode": 0},
        "timed_out": False,
        "effective_min_overall_score": 0.75,
        "review_profile": {"profile": "babyagi_strict", "ok": True, "checks": []},
        "trace_metrics": {"total_llm_calls": 1, "clean_exit": True},
        "event_summary": {"event_counts": {"file_diff": 1, "llm_call": 1}},
        "judge": {"report": {"verdict": "pass", "overall_score_normalized": 1.0}},
        "workspace_files": [{"path": "artifact.txt", "size": 6, "preview": "done\\n"}],
    }
    summary = _review_panel_summary(payload)
    assert summary == {
        "ok": True,
        "deterministic_ok": True,
        "review_ok": False,
        "run_id": "golden-run",
        "spec_path": "agent_ontology/specs/codex.yaml",
        "objective": "write artifact",
        "effective_model": "openrouter/deepseek/deepseek-chat",
        "returncode": 0,
        "timed_out": False,
        "effective_min_overall_score": 0.75,
        "review_profile": {"profile": "babyagi_strict", "ok": True, "checks": []},
        "trace_metrics": {"total_llm_calls": 1, "clean_exit": True},
        "event_counts": {"file_diff": 1, "llm_call": 1},
        "judge_report": {"verdict": "pass", "overall_score_normalized": 1.0},
        "workspace_files": [{"path": "artifact.txt", "size": 6, "preview": "done\\n"}],
    }
