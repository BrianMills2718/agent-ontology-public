from __future__ import annotations

import contextlib
import io
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

from http.server import ThreadingHTTPServer

from agent_ontology.v2.api_server import (
    RUNS_V2,
    RUNS_V2_LOCK,
    LiveViewerV2Handler,
    V2RunSession,
    _append_session_turn,
    _attach_run_to_session,
    _ensure_session,
    _iter_run_dirs,
    _manifest_for_run,
    _manifest_from_events,
    _read_file_for_run,
    _read_snapshot_for_persisted_run,
)


def _evt(run_id: str, idx: int, event_type: str, facet: str, **extra):
    base = {
        "schema_version": "1.0",
        "run_id": run_id,
        "event_index": idx,
        "seq": idx,
        "event_id": f"{run_id}:{idx}",
        "ts": "2026-02-23T00:00:00+00:00",
        "type": event_type,
        "facet": facet,
        "entity_id": f"run:{run_id}",
        "span_id": f"span/{idx}",
        "parent_span_id": None,
        "actor": "runtime",
    }
    base.update(extra)
    return base


def _write_events(path: Path, events: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(evt) for evt in events) + "\n"
    path.write_text(payload, encoding="utf-8")


def _build_run(root: Path, run_id: str = "abc123def456") -> Path:
    run_dir = root / ".live_runs" / run_id
    workspace = run_dir / "autogpt_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "main.py").write_text("print('v2')\n", encoding="utf-8")
    (workspace / "README.md").write_text("# demo\n", encoding="utf-8")

    events = [
        _evt(
            run_id,
            1,
            "started",
            "input",
            input={
                "text": "build me todo api",
                "workspace_dir": str(workspace),
            },
        ),
        _evt(
            run_id,
            2,
            "final_data",
            "output",
            data={
                "workspace_dir": str(workspace),
                "artifact_files": ["main.py", "README.md"],
            },
        ),
        _evt(
            run_id,
            3,
            "completed",
            "done",
            returncode=0,
            error="",
            final_data_summary={"keys": ["workspace_dir", "artifact_files"]},
        ),
    ]
    _write_events(run_dir / "events.jsonl", events)
    return run_dir


@contextlib.contextmanager
def _serve_v2(root: Path):
    handler_cls = type("TestV2Handler", (LiveViewerV2Handler,), {"root_dir": root})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def _http_get_json(base: str, path: str, headers: dict[str, str] | None = None) -> dict:
    req = Request(base + path, method="GET", headers=headers or {})
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(base: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload or {}).encode("utf-8")
    req = Request(base + path, method="POST", data=body, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_sse_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for line in raw.splitlines():
        if line.startswith("id: "):
            ids.append(int(line.split(":", 1)[1].strip()))
    return ids


class _FakeStreamHandler:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self.wfile = io.BytesIO()
        self.status_code: int | None = None
        self.response_headers: list[tuple[str, str]] = []

    def send_response(self, code: int) -> None:
        self.status_code = code

    def send_header(self, key: str, value: str) -> None:
        self.response_headers.append((key, value))

    def end_headers(self) -> None:
        return


def test_iter_run_dirs_detects_persisted_runs(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    refs = _iter_run_dirs(tmp_path)
    assert refs
    assert refs[0].run_id == run_dir.name


def test_manifest_fallback_workspace_scan_without_file_diff_events(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    status, payload = _manifest_for_run(tmp_path, run_dir.name)
    assert status == 200
    assert payload["ok"] is True
    files = payload["files"]
    paths = {row["path"] for row in files}
    assert "autogpt_workspace/main.py" in paths
    assert "autogpt_workspace/README.md" in paths


def test_snapshot_read_falls_back_to_workspace_when_no_snapshots_exist(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    status, payload = _read_file_for_run(
        tmp_path,
        run_dir.name,
        "autogpt_workspace/main.py",
        "snapshot",
        at_event_index=None,
        start=0,
        limit=2048,
    )
    assert status == 200
    assert payload["ok"] is True
    assert payload.get("source_fallback") == "workspace"
    assert "print('v2')" in payload.get("content", "")


def test_session_binding_and_turn_persistence(tmp_path: Path):
    session = _ensure_session(tmp_path, "chat_alpha")
    assert session["session_id"] == "chat_alpha"
    assert session["run_ids"] == []
    assert session["turns"] == []

    _attach_run_to_session(tmp_path, "chat_alpha", "run_001")
    turn = _append_session_turn(
        tmp_path,
        "chat_alpha",
        role="user",
        text="build todo api",
        run_id="run_001",
        event_index=1,
    )
    assert turn is not None
    assert turn["role"] == "user"
    assert turn["run_id"] == "run_001"

    persisted = _ensure_session(tmp_path, "chat_alpha")
    assert persisted["run_ids"] == ["run_001"]
    assert len(persisted["turns"]) == 1
    assert persisted["turns"][0]["text"] == "build todo api"


def test_snapshot_at_cursor_reads_correct_version(tmp_path: Path):
    run_id = "snaprun001"
    run_dir = tmp_path / ".live_runs" / run_id
    snap_root = run_dir / "file_snapshots"
    snap_v2 = snap_root / "00000002" / "autogpt_workspace" / "main.py"
    snap_v5 = snap_root / "00000005" / "autogpt_workspace" / "main.py"
    snap_v2.parent.mkdir(parents=True, exist_ok=True)
    snap_v5.parent.mkdir(parents=True, exist_ok=True)
    snap_v2.write_text("print('v2')\n", encoding="utf-8")
    snap_v5.write_text("print('v5')\n", encoding="utf-8")

    events = [
        _evt(
            run_id,
            2,
            "file_diff",
            "files",
            path="autogpt_workspace/main.py",
            change_type="added",
            snapshot_rel="00000002/autogpt_workspace/main.py",
            is_text=True,
            size=12,
        ),
        _evt(
            run_id,
            5,
            "file_diff",
            "files",
            path="autogpt_workspace/main.py",
            change_type="modified",
            snapshot_rel="00000005/autogpt_workspace/main.py",
            is_text=True,
            size=12,
        ),
    ]

    manifest = _manifest_from_events(run_dir, events)
    assert len(manifest) == 1
    assert manifest[0]["path"] == "autogpt_workspace/main.py"
    assert manifest[0]["versions"] == 2

    s1, p1 = _read_snapshot_for_persisted_run(
        run_dir,
        events,
        run_id,
        "autogpt_workspace/main.py",
        3,
        0,
        4096,
    )
    assert s1 == 200
    assert p1["snapshot_event_index"] == 2
    assert "v2" in p1["content"]

    s2, p2 = _read_snapshot_for_persisted_run(
        run_dir,
        events,
        run_id,
        "autogpt_workspace/main.py",
        9,
        0,
        4096,
    )
    assert s2 == 200
    assert p2["snapshot_event_index"] == 5
    assert "v5" in p2["content"]


def test_session_endpoints_round_trip(tmp_path: Path):
    with _serve_v2(tmp_path) as base:
        created = _http_post_json(base, "/api/v2/session/new")
        sid = str(created.get("session_id") or "")
        assert sid

        turn = _http_post_json(
            base,
            f"/api/v2/session/{sid}/turn",
            {
                "role": "user",
                "text": "build me a todo api",
                "run_id": "run_http_001",
                "event_index": 1,
            },
        )
        assert turn.get("ok") is True
        assert isinstance(turn.get("turn"), dict)
        assert turn["turn"].get("text") == "build me a todo api"

        session = _http_get_json(base, f"/api/v2/session/{sid}")
        assert session.get("ok") is True
        assert session.get("session_id") == sid
        turns = session.get("turns")
        assert isinstance(turns, list)
        assert len(turns) == 1
        assert turns[0].get("run_id") == "run_http_001"

        listing = _http_get_json(base, "/api/v2/sessions?limit=20")
        assert listing.get("ok") is True
        sessions = listing.get("sessions")
        assert isinstance(sessions, list)
        assert any(str(row.get("session_id")) == sid for row in sessions)


def test_runs_endpoint_filters_by_session_id(tmp_path: Path):
    run_a = _build_run(tmp_path, run_id="runA001")
    run_b = _build_run(tmp_path, run_id="runB001")

    events_a = _load_events(run_a / "events.jsonl")
    events_b = _load_events(run_b / "events.jsonl")
    assert events_a and events_b
    events_a[0]["session_id"] = "sess_alpha"
    events_b[0]["session_id"] = "sess_beta"
    _write_events(run_a / "events.jsonl", events_a)
    _write_events(run_b / "events.jsonl", events_b)

    with _serve_v2(tmp_path) as base:
        rows_alpha = _http_get_json(base, "/api/v2/runs?session_id=sess_alpha&limit=20")
        assert rows_alpha.get("ok") is True
        runs_alpha = rows_alpha.get("runs")
        assert isinstance(runs_alpha, list)
        run_ids_alpha = {str(row.get("run_id")) for row in runs_alpha}
        assert "runA001" in run_ids_alpha
        assert "runB001" not in run_ids_alpha

        rows_beta = _http_get_json(base, "/api/v2/runs?session_id=sess_beta&limit=20")
        assert rows_beta.get("ok") is True
        runs_beta = rows_beta.get("runs")
        assert isinstance(runs_beta, list)
        run_ids_beta = {str(row.get("run_id")) for row in runs_beta}
        assert "runB001" in run_ids_beta
        assert "runA001" not in run_ids_beta


def _load_events(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        assert isinstance(obj, dict)
        rows.append(obj)
    return rows


class _FakeStreamSession:
    def __init__(self, events: list[dict]):
        self._events = list(events)

    def get_events_after(self, last_event_index: int, timeout: float = 1.0):
        rows = [dict(evt) for evt in self._events if int(evt.get("event_index", -1)) > int(last_event_index)]
        return rows, True


def _build_session_for_prune(tmp_path: Path, run_id: str = "prunerun001") -> V2RunSession:
    spec = tmp_path / "spec.yaml"
    agent = tmp_path / "agent.py"
    spec.write_text("name: spec\n", encoding="utf-8")
    agent.write_text("def run(initial=None):\n    return {}\n", encoding="utf-8")
    session = V2RunSession(
        run_id=run_id,
        session_id="sess_prune",
        root_dir=tmp_path,
        spec_path=spec,
        agent_path=agent,
        initial_data={"workspace_dir": str(tmp_path / ".live_runs" / run_id / "autogpt_workspace")},
        task="prune",
        requested_mode="default",
        mock_mode="deterministic",
        capability_profile="restricted",
        max_iter=10,
        model_override="",
        max_budget_usd=None,
    )
    session.run_dir.mkdir(parents=True, exist_ok=True)
    session._snapshot_dir.mkdir(parents=True, exist_ok=True)
    return session


def test_stream_resume_precedence_since_event_index_then_since_then_header():
    run_id = "streamv2resume"
    events = [
        _evt(run_id, 1, "started", "input"),
        _evt(run_id, 2, "state", "state", process="receive_request"),
        _evt(run_id, 3, "completed", "done", returncode=0, error=""),
    ]
    fake_session = _FakeStreamSession(events)
    with RUNS_V2_LOCK:
        RUNS_V2[run_id] = fake_session
    try:
        handler_1 = _FakeStreamHandler(headers={"Last-Event-ID": "0"})
        LiveViewerV2Handler._stream_events(handler_1, fake_session, "since_event_index=2")
        ids_1 = _parse_sse_ids(handler_1.wfile.getvalue().decode("utf-8"))
        assert handler_1.status_code == 200
        assert ids_1 == [3]

        handler_2 = _FakeStreamHandler(headers={"Last-Event-ID": "0"})
        LiveViewerV2Handler._stream_events(handler_2, fake_session, "since=1")
        ids_2 = _parse_sse_ids(handler_2.wfile.getvalue().decode("utf-8"))
        assert ids_2 == [2, 3]

        handler_3 = _FakeStreamHandler(headers={"Last-Event-ID": "1"})
        LiveViewerV2Handler._stream_events(handler_3, fake_session, "")
        ids_3 = _parse_sse_ids(handler_3.wfile.getvalue().decode("utf-8"))
        assert ids_3 == [2, 3]

        handler_4 = _FakeStreamHandler(headers={"Last-Event-ID": "bad"})
        LiveViewerV2Handler._stream_events(handler_4, fake_session, "")
        ids_4 = _parse_sse_ids(handler_4.wfile.getvalue().decode("utf-8"))
        assert ids_4 == [1, 2, 3]
    finally:
        with RUNS_V2_LOCK:
            RUNS_V2.pop(run_id, None)


def test_session_prune_snapshots_by_versions_per_file(tmp_path: Path):
    session = _build_session_for_prune(tmp_path, "prunevers001")
    session.snapshot_keep_events = 999
    session.snapshot_keep_versions_per_file = 2
    session.emit("started", facet="input")
    session.emit("state", facet="state")
    session.emit("state", facet="state")

    path_key = "autogpt_workspace/main.py"
    for idx, text in [(1, "v1\n"), (2, "v2\n"), (3, "v3\n")]:
        rel = f"{idx:08d}/{path_key}"
        snap = session._snapshot_dir / rel
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(text, encoding="utf-8")
        session._record_file_version(
            path_key,
            event_index=idx,
            sha1=f"sha{idx}",
            size=len(text),
            is_text=True,
            deleted=False,
            snapshot_rel=rel,
        )

    stats = session.prune_snapshots(force=False)
    assert stats["removed_versions"] == 1
    assert stats["removed_snapshot_files"] == 1
    assert not (session._snapshot_dir / f"{1:08d}" / path_key).exists()
    assert len(session._file_versions[path_key]) == 2

    status_old, payload_old = session.get_file_snapshot(path_key, at_event_index=1)
    assert status_old == 404
    assert "No snapshot at or before" in str(payload_old.get("error", ""))
    status_new, payload_new = session.get_file_snapshot(path_key, at_event_index=3)
    assert status_new == 200
    assert payload_new["snapshot_event_index"] == 3


def test_session_prune_snapshots_by_event_window(tmp_path: Path):
    session = _build_session_for_prune(tmp_path, "pruneevt001")
    session.snapshot_keep_events = 1
    session.snapshot_keep_versions_per_file = 10
    session.emit("started", facet="input")
    session.emit("state", facet="state")
    session.emit("state", facet="state")

    path_key = "autogpt_workspace/data.txt"
    for idx, text in [(1, "e1\n"), (2, "e2\n"), (3, "e3\n")]:
        rel = f"{idx:08d}/{path_key}"
        snap = session._snapshot_dir / rel
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(text, encoding="utf-8")
        session._record_file_version(
            path_key,
            event_index=idx,
            sha1=f"sha{idx}",
            size=len(text),
            is_text=True,
            deleted=False,
            snapshot_rel=rel,
        )

    stats = session.prune_snapshots(force=False)
    assert stats["cutoff_event_index"] >= 2
    assert stats["removed_versions"] == 1
    kept = [int(v["event_index"]) for v in session._file_versions[path_key]]
    assert kept == [2, 3]


class _FakePruneSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def prune_snapshots(self, *, force: bool = False):
        self.calls.append({"force": force})
        return {
            "removed_versions": 2,
            "removed_snapshot_files": 2,
            "removed_snapshot_bytes": 42,
            "cutoff_event_index": 10,
        }

    def emit(self, *args, **kwargs):
        self.calls.append({"emit": kwargs})
        return {}


def test_prune_endpoint_for_active_run(tmp_path: Path):
    run_id = "pruneapi001"
    fake = _FakePruneSession()
    with RUNS_V2_LOCK:
        RUNS_V2[run_id] = fake  # type: ignore[assignment]
    try:
        with _serve_v2(tmp_path) as base:
            payload = _http_post_json(base, f"/api/v2/prune/{run_id}", {"force": True})
            assert payload["ok"] is True
            assert payload["run_id"] == run_id
            assert payload["force"] is True
            assert payload["removed_versions"] == 2
            assert fake.calls and fake.calls[0]["force"] is True
    finally:
        with RUNS_V2_LOCK:
            RUNS_V2.pop(run_id, None)
