from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

EVENT_SCHEMA_VERSION = "1.0"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ITERATION_STATE_RE = re.compile(r"^\[Iteration\s+(?P<iter>\d+)\]\s+State:\s+(?P<state>.+)$")
LLM_CALL_RE = re.compile(r"^\s*\[(?P<agent>[^\]]+)\]\s+\((?P<model>[^,]+),\s*(?P<dur>\d+)ms\)")
TRACE_WRITTEN_RE = re.compile(r"^Trace written to\s+(?P<path>[^\s]+)\s+\((?P<calls>\d+)\s+calls,\s+(?P<dur>\d+)ms total\)")
FINAL_KEYS_RE = re.compile(r"^Final state\.data keys:\s*(?P<keys>.+)$")
STEP_ACTION_RE = re.compile(r"^\s*→\s+(?P<label>.+)$")
TOOL_OBSERVATION_RE = re.compile(r"^\s*Observation:\s*\[tool\s+(?P<tool>[A-Za-z0-9_.-]+)\s*(?P<detail>.*?)\]$")
OBSERVATION_RE = re.compile(r"^\s*Observation:\s*(?P<obs>.+)$")
SCHEMA_RE = re.compile(r"^\s*\[SCHEMA\]\s*(?P<schema>[^:]+):\s*(?P<detail>.+)$")
FAIL_RE = re.compile(r"^\s*\[FAIL\]\s*(?P<detail>.+)$")
LLM_SUMMARY_RE = re.compile(r"^\s*\[LLM\]\s*(?P<agent>.+?)\s*•\s*(?P<model>.+?)\s*•\s*(?P<dur>\d+)ms")
RUNTIME_RE = re.compile(r"^\s*\[(?P<tag>runtime)\]\s*(?P<msg>.+)$", flags=re.IGNORECASE)

MAX_EVENT_LIMIT = 20_000
MAX_FILE_BYTES = 240_000
MAX_MANIFEST_FILES = 5_000
MAX_SESSION_TURNS = 2_000
DEFAULT_SNAPSHOT_KEEP_EVENTS = 160
DEFAULT_SNAPSHOT_KEEP_VERSIONS_PER_FILE = 24


RUNS_V2: dict[str, "V2RunSession"] = {}
RUNS_V2_LOCK = threading.Lock()


def _is_text_blob(raw: bytes) -> bool:
    if not raw:
        return True
    return b"\x00" not in raw[:4096]


def _has_workspace_layout(root_dir: Path) -> bool:
    return (root_dir / "agent_ontology" / "spec-viewer.html").exists()


def _has_repo_layout(root_dir: Path) -> bool:
    return (root_dir / "spec-viewer.html").exists() and (root_dir / "agent_ontology").is_dir()


def _repo_dir(root_dir: Path) -> Path:
    root = root_dir.resolve()
    if _has_workspace_layout(root) and (root / "agent_ontology").is_dir():
        return (root / "agent_ontology").resolve()
    return root


def _find_default_root() -> Path:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[1],
    ]
    for candidate in candidates:
        if _has_workspace_layout(candidate) or _has_repo_layout(candidate):
            return candidate.resolve()
    return Path.cwd().resolve()


def _resolve_under_root(root_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = root_dir / path
    resolved = path.resolve()
    root = root_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path outside root: {raw_path}")
    return resolved


def _resolve_spec_path(root_dir: Path, raw_path: str) -> Path:
    direct = _resolve_under_root(root_dir, raw_path)
    if direct.exists():
        return direct

    viewer_relative = (root_dir / "agent_ontology" / raw_path).resolve()
    root = root_dir.resolve()
    if viewer_relative != root and root in viewer_relative.parents and viewer_relative.exists():
        return viewer_relative

    return direct


def _derive_agent_path(root_dir: Path, spec_path: Path) -> Path:
    stem = spec_path.stem.replace("-", "_")
    repo = _repo_dir(root_dir)
    return (repo / "agents" / f"{stem}_agent.py").resolve()


def _ensure_agent_generated(root_dir: Path, spec_path: Path, agent_path: Path) -> None:
    repo = _repo_dir(root_dir)
    instantiate_script = repo / "agent_ontology" / "instantiate.py"
    if agent_path.exists():
        try:
            agent_mtime = agent_path.stat().st_mtime
            if agent_mtime >= spec_path.stat().st_mtime and agent_mtime >= instantiate_script.stat().st_mtime:
                return
        except OSError:
            pass
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(instantiate_script), str(spec_path), "-o", str(agent_path)]
    result = subprocess.run(cmd, cwd=str(root_dir), capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "unknown instantiate error"
        raise RuntimeError(f"Failed to generate agent for {spec_path.name}: {detail}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _is_valid_run_id(run_id: str) -> bool:
    rid = str(run_id or "").strip()
    return bool(rid and RUN_ID_RE.match(rid))


def _run_root(root_dir: Path) -> Path:
    return (_repo_dir(root_dir) / ".live_runs").resolve()


def _run_artifact_dir(root_dir: Path, run_id: str) -> Path:
    base = _run_root(root_dir)
    candidate = (base / run_id).resolve()
    if candidate == base or base not in candidate.parents:
        raise ValueError(f"Invalid run id path: {run_id}")
    return candidate


def _session_dir(root_dir: Path) -> Path:
    return (_run_root(root_dir) / "_sessions").resolve()


def _is_valid_session_id(session_id: str) -> bool:
    sid = str(session_id or "").strip()
    return bool(sid and RUN_ID_RE.match(sid))


def _session_file(root_dir: Path, session_id: str) -> Path:
    if not _is_valid_session_id(session_id):
        raise ValueError(f"Invalid session id: {session_id}")
    base = _session_dir(root_dir)
    candidate = (base / f"{session_id}.json").resolve()
    if candidate == base or base not in candidate.parents:
        raise ValueError(f"Invalid session id path: {session_id}")
    return candidate


def _load_session(root_dir: Path, session_id: str) -> dict[str, Any] | None:
    try:
        path = _session_file(root_dir, session_id)
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _write_session(root_dir: Path, session: dict[str, Any]) -> None:
    session_id = str(session.get("session_id") or "").strip()
    path = _session_file(root_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _new_session(root_dir: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    session = {
        "session_id": uuid.uuid4().hex[:16],
        "created_at": now,
        "updated_at": now,
        "run_ids": [],
        "turns": [],
    }
    _write_session(root_dir, session)
    return session


def _ensure_session(root_dir: Path, session_id: str | None) -> dict[str, Any]:
    sid = str(session_id or "").strip()
    if sid:
        existing = _load_session(root_dir, sid)
        if existing is not None:
            return existing
        session = {
            "session_id": sid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_ids": [],
            "turns": [],
        }
        _write_session(root_dir, session)
        return session
    return _new_session(root_dir)


def _append_session_turn(
    root_dir: Path,
    session_id: str,
    *,
    role: str,
    text: str,
    run_id: str | None = None,
    event_index: int | None = None,
) -> dict[str, Any] | None:
    session = _load_session(root_dir, session_id)
    if session is None:
        return None
    turns = session.get("turns")
    if not isinstance(turns, list):
        turns = []
        session["turns"] = turns
    turn = {
        "time": datetime.now(timezone.utc).isoformat(),
        "role": str(role or "").strip() or "assistant",
        "text": str(text or ""),
        "run_id": str(run_id or ""),
        "event_index": int(event_index) if isinstance(event_index, int) else event_index,
    }
    turns.append(turn)
    if len(turns) > MAX_SESSION_TURNS:
        session["turns"] = turns[-MAX_SESSION_TURNS:]
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_session(root_dir, session)
    return turn


def _attach_run_to_session(root_dir: Path, session_id: str, run_id: str) -> None:
    session = _load_session(root_dir, session_id)
    if session is None:
        return
    runs = session.get("run_ids")
    if not isinstance(runs, list):
        runs = []
        session["run_ids"] = runs
    rid = str(run_id or "").strip()
    if rid and rid not in runs:
        runs.append(rid)
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_session(root_dir, session)


def _event_index_of(event: dict[str, Any], fallback: int = 0) -> int:
    try:
        return int(event.get("event_index", event.get("seq", fallback)) or fallback)
    except (TypeError, ValueError):
        return fallback


def _normalize_loaded_event(run_id: str, raw: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    event = dict(raw)
    idx = _event_index_of(event, fallback=fallback_index)
    event["schema_version"] = str(event.get("schema_version") or EVENT_SCHEMA_VERSION)
    event["run_id"] = str(event.get("run_id") or run_id)
    event["event_index"] = idx
    event["seq"] = idx
    event["event_id"] = str(event.get("event_id") or f"{run_id}:{idx}")
    event["ts"] = str(event.get("ts") or event.get("time") or datetime.now(timezone.utc).isoformat())
    event["type"] = str(event.get("type") or "event")
    event["facet"] = str(event.get("facet") or "control")
    if "entity_id" not in event:
        event["entity_id"] = None
    if "span_id" not in event:
        event["span_id"] = f"span/{idx}"
    if "parent_span_id" not in event:
        event["parent_span_id"] = None
    if "actor" not in event:
        event["actor"] = "runtime"
    return event


def _load_events_from_ledger(path: Path, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for i, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        rows.append(_normalize_loaded_event(run_id, raw, fallback_index=i))
    rows.sort(key=lambda e: _event_index_of(e))
    return rows


def _events_from_trace_json(path: Path, run_id: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    trace = data.get("trace")
    if not isinstance(trace, list):
        return []

    out: list[dict[str, Any]] = []
    idx = 0

    def append(event_type: str, facet: str, payload: dict[str, Any], *, ts: str | None = None) -> None:
        nonlocal idx
        idx += 1
        out.append(
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "run_id": run_id,
                "event_index": idx,
                "seq": idx,
                "event_id": f"{run_id}:{idx}",
                "ts": ts or datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "facet": facet,
                "entity_id": f"run:{run_id}",
                "span_id": f"span/{idx}",
                "parent_span_id": None,
                "actor": "runtime",
                **payload,
            }
        )

    append("started", "input", {"source": "trace_fallback", "trace_path": str(path)})
    for call in trace:
        if not isinstance(call, dict):
            continue
        append(
            "llm_call",
            "llm",
            {
                "agent": str(call.get("agent") or "agent"),
                "model": str(call.get("model") or ""),
                "duration_ms": _safe_int(call.get("duration_ms"), 0),
                "source": "trace_fallback",
                "payload": call,
            },
            ts=str(call.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        )
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    append(
        "completed",
        "done",
        {
            "returncode": 0 if bool(metrics.get("clean_exit", True)) else 1,
            "error": "" if bool(metrics.get("clean_exit", True)) else "trace indicates non-clean exit",
            "trace_path": str(path),
            "trace_metrics": metrics,
            "source": "trace_fallback",
        },
    )
    return out


def _load_persisted_run_events(root_dir: Path, run_id: str) -> tuple[list[dict[str, Any]], str]:
    run_dir = _run_artifact_dir(root_dir, run_id)
    ledger_path = (run_dir / "events.jsonl").resolve()
    if ledger_path.exists() and ledger_path.is_file():
        events = _load_events_from_ledger(ledger_path, run_id)
        if events:
            return events, "ledger"

    trace_path = (run_dir / "trace.json").resolve()
    if trace_path.exists() and trace_path.is_file():
        events = _events_from_trace_json(trace_path, run_id)
        if events:
            return events, "trace_fallback"

    return [], "none"


def _filter_events_window(events: list[dict[str, Any]], *, since_event_index: int, limit: int) -> list[dict[str, Any]]:
    filtered = [evt for evt in events if _event_index_of(evt) > since_event_index]
    if len(filtered) > limit:
        filtered = filtered[-limit:]
    return filtered


def _resolve_model_override(value: Any) -> str:
    raw = str(value or "").strip()
    if raw:
        return raw
    env_default = str(os.environ.get("AGENT_ONTOLOGY_DEFAULT_MODEL", "")).strip()
    if env_default:
        return env_default
    return "openrouter/deepseek/deepseek-chat"


def _build_initial_data_for_run(spec_path: Path, payload: dict[str, Any], *, root_dir: Path, run_id: str) -> dict[str, Any]:
    text = str(
        payload.get("text")
        or payload.get("objective")
        or payload.get("user")
        or payload.get("task")
        or ""
    ).strip()
    constraints = payload.get("constraints", [])
    if isinstance(constraints, str):
        constraints = [constraints] if constraints.strip() else []
    if not isinstance(constraints, list):
        constraints = []

    files = payload.get("files", "")
    if isinstance(files, list):
        files = ", ".join(str(x) for x in files if str(x).strip())
    files = str(files or "")

    run_dir = _run_artifact_dir(root_dir, run_id)
    workspace_dir_raw = str(payload.get("workspace_dir") or payload.get("workspace_path") or "").strip()
    if workspace_dir_raw:
        workspace_dir = _resolve_under_root(root_dir, workspace_dir_raw)
    else:
        workspace_dir = (run_dir / f"{spec_path.stem.lower()}_workspace").resolve()

    initial_data = {
        "text": text,
        "user": text,
        "objective": text,
        "task": str(payload.get("task") or text),
        "constraints": constraints,
        "files": files,
        "acceptance_criteria": payload.get("acceptance_criteria") if isinstance(payload.get("acceptance_criteria"), list) else [],
        "requested_mode": str(payload.get("requested_mode") or "default"),
        "workspace_dir": str(workspace_dir),
        "workspace_path": str(workspace_dir),
        "mock_mode": str(payload.get("mock_mode") or os.environ.get("AGENT_ONTOLOGY_MOCK_MODE", "live")),
        "capability_profile": str(payload.get("capability_profile") or os.environ.get("AGENT_ONTOLOGY_CAPABILITY_PROFILE", "external_enabled")),
        "model_override": _resolve_model_override(payload.get("model_override") or ""),
        "trace_id": f"agent_ontology/live_v2/{run_id}",
    }
    return initial_data


def _build_runner_code(agent_path: Path) -> str:
    return (
        "import importlib.util, json, os, sys, traceback\n"
        f"agent_path = {json.dumps(str(agent_path))}\n"
        "initial_data = {}\n"
        "try:\n"
        "    initial_data = json.loads(os.environ.get('AO_V2_INITIAL_DATA_JSON', '{}'))\n"
        "except Exception:\n"
        "    initial_data = {}\n"
        "final_path = os.environ.get('AO_V2_FINAL_DATA_PATH', '')\n"
        "state = None\n"
        "returncode = 0\n"
        "error = ''\n"
        "try:\n"
        "    spec = importlib.util.spec_from_file_location('ao_v2_agent', agent_path)\n"
        "    if spec is None or spec.loader is None:\n"
        "        raise RuntimeError(f'Unable to load agent module: {agent_path}')\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    if hasattr(module, 'run'):\n"
        "        state = module.run(initial_data)\n"
        "    elif hasattr(module, 'main'):\n"
        "        state = module.main()\n"
        "    else:\n"
        "        raise RuntimeError('Agent module missing run() or main()')\n"
        "except BaseException as exc:\n"
        "    returncode = 1\n"
        "    error = f'{type(exc).__name__}: {exc}'\n"
        "    traceback.print_exc()\n"
        "finally:\n"
        "    out = {'ok': returncode == 0, 'returncode': returncode, 'error': error, 'data': {}}\n"
        "    try:\n"
        "        if state is not None:\n"
        "            if hasattr(state, 'data') and isinstance(getattr(state, 'data'), dict):\n"
        "                out['data'] = getattr(state, 'data')\n"
        "                out['iteration'] = getattr(state, 'iteration', None)\n"
        "                out['schema_violations'] = getattr(state, 'schema_violations', None)\n"
        "            elif isinstance(state, dict):\n"
        "                out['data'] = state\n"
        "    except Exception:\n"
        "        pass\n"
        "    if final_path:\n"
        "        try:\n"
        "            with open(final_path, 'w', encoding='utf-8') as fh:\n"
        "                json.dump(out, fh, ensure_ascii=False, indent=2, default=str)\n"
        "        except Exception as exc:\n"
        "            print(f'[v2-final-data-write-error] {type(exc).__name__}: {exc}')\n"
        "sys.exit(returncode)\n"
    )


@dataclass(frozen=True)
class RunRef:
    run_id: str
    run_dir: Path
    mtime: float


def _run_roots(root_dir: Path) -> list[Path]:
    repo_root = _repo_dir(root_dir)
    candidates = [
        (repo_root / ".live_runs").resolve(),
        (root_dir / ".live_runs").resolve(),
        (root_dir / "agent_ontology" / ".live_runs").resolve(),
    ]
    out: list[Path] = []
    for path in candidates:
        if path in out:
            continue
        if path.exists() and path.is_dir():
            out.append(path)
    return out


def _iter_run_dirs(root_dir: Path) -> list[RunRef]:
    refs: list[RunRef] = []
    seen: set[str] = set()
    for base in _run_roots(root_dir):
        for child in base.iterdir():
            if not child.is_dir():
                continue
            rid = child.name.strip()
            if not _is_valid_run_id(rid):
                continue
            if rid in seen:
                continue
            events_path = child / "events.jsonl"
            trace_path = child / "trace.json"
            if not events_path.exists() and not trace_path.exists():
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            refs.append(RunRef(run_id=rid, run_dir=child.resolve(), mtime=mtime))
            seen.add(rid)
    refs.sort(key=lambda r: r.mtime, reverse=True)
    return refs


def _workspace_roots_from_events(run_dir: Path, events: list[dict[str, Any]]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(path_value: str) -> None:
        raw = str(path_value or "").strip()
        if not raw:
            return
        p = Path(raw)
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        else:
            p = p.resolve()
        if p in seen:
            return
        if p.exists() and p.is_dir():
            roots.append(p)
            seen.add(p)

    for evt in reversed(events):
        if evt.get("type") == "final_data":
            data = evt.get("data")
            if isinstance(data, dict):
                for key in ("workspace_dir", "workspace_path", "output_dir"):
                    add(str(data.get(key) or ""))
                break

    for evt in events:
        if evt.get("type") == "started":
            payload = evt.get("input")
            if isinstance(payload, dict):
                for key in ("workspace_dir", "workspace_path", "output_dir"):
                    add(str(payload.get(key) or ""))
            break

    if run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_dir() and "workspace" in child.name.lower():
                add(str(child))

    return roots


def _scan_workspace_manifest(run_dir: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_event_index = max((_event_index_of(e) for e in events), default=0)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in _workspace_roots_from_events(run_dir, events):
        prefix = root.name
        file_count = 0
        for file_path in sorted(root.rglob("*")):
            if file_count >= MAX_MANIFEST_FILES:
                break
            if not file_path.is_file():
                continue
            try:
                rel = file_path.relative_to(root).as_posix()
            except ValueError:
                continue
            key = f"{prefix}/{rel}" if rel else prefix
            if key in seen:
                continue
            seen.add(key)
            try:
                size = file_path.stat().st_size
            except OSError:
                size = 0
            rows.append(
                {
                    "path": key,
                    "versions": 1,
                    "latest_event_index": latest_event_index,
                    "deleted": False,
                    "is_text": True,
                    "size": size,
                    "content_available": True,
                    "event_indexes": [],
                    "source": "workspace_scan",
                }
            )
            file_count += 1
    return rows


def _manifest_from_events(run_dir: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    versions_by_path: dict[str, list[dict[str, Any]]] = {}
    for evt in events:
        if str(evt.get("type") or "") != "file_diff":
            continue
        path = str(evt.get("path") or "").strip().replace("\\", "/")
        if not path:
            continue
        versions = versions_by_path.setdefault(path, [])
        versions.append(
            {
                "event_index": _event_index_of(evt),
                "snapshot_rel": str(evt.get("snapshot_rel") or ""),
                "deleted": str(evt.get("change_type") or "").lower() == "deleted",
                "is_text": bool(evt.get("is_text", True)),
                "size": _safe_int(evt.get("size"), 0),
            }
        )

    rows: list[dict[str, Any]] = []
    for path, versions in versions_by_path.items():
        if not versions:
            continue
        versions.sort(key=lambda v: int(v.get("event_index", 0)))
        latest = versions[-1]
        rows.append(
            {
                "path": path,
                "versions": len(versions),
                "latest_event_index": int(latest.get("event_index", 0)),
                "deleted": bool(latest.get("deleted", False)),
                "is_text": bool(latest.get("is_text", True)),
                "size": int(latest.get("size", 0)),
                "content_available": bool(latest.get("snapshot_rel")),
                "event_indexes": [int(v.get("event_index", 0)) for v in versions][-60:],
                "source": "event_index",
            }
        )
    rows.sort(key=lambda r: str(r.get("path") or ""))
    return rows


def _read_snapshot_for_persisted_run(
    run_dir: Path,
    events: list[dict[str, Any]],
    run_id: str,
    rel_path: str,
    at_event_index: int | None,
    start: int,
    limit: int,
) -> tuple[int, dict[str, Any]]:
    versions: list[dict[str, Any]] = []
    for evt in events:
        if str(evt.get("type") or "") != "file_diff":
            continue
        path = str(evt.get("path") or "").strip().replace("\\", "/")
        if path != rel_path:
            continue
        versions.append(
            {
                "event_index": _event_index_of(evt),
                "snapshot_rel": str(evt.get("snapshot_rel") or ""),
                "deleted": str(evt.get("change_type") or "").lower() == "deleted",
            }
        )
    if not versions:
        return 404, {"ok": False, "error": f"No snapshot available for path: {rel_path}", "run_id": run_id, "path": rel_path}

    versions.sort(key=lambda v: int(v.get("event_index", 0)))
    target_idx = int(at_event_index) if isinstance(at_event_index, int) and at_event_index >= 0 else 10**12
    selected: dict[str, Any] | None = None
    for version in versions:
        if int(version.get("event_index", 0)) <= target_idx:
            selected = version
        else:
            break
    if not selected:
        return 404, {
            "ok": False,
            "error": f"No snapshot at or before event_index={target_idx} for path: {rel_path}",
            "run_id": run_id,
            "path": rel_path,
        }
    if bool(selected.get("deleted", False)):
        return 404, {
            "ok": False,
            "error": f"File deleted at selected cursor for path: {rel_path}",
            "run_id": run_id,
            "path": rel_path,
            "deleted": True,
        }

    snapshot_rel = str(selected.get("snapshot_rel") or "")
    if not snapshot_rel:
        return 404, {"ok": False, "error": f"Snapshot content missing for path: {rel_path}", "run_id": run_id, "path": rel_path}
    snapshot_root = (run_dir / "file_snapshots").resolve()
    snapshot_path = (snapshot_root / snapshot_rel).resolve()
    try:
        snapshot_path.relative_to(snapshot_root)
    except ValueError:
        return 404, {"ok": False, "error": f"Invalid snapshot path for {rel_path}", "run_id": run_id, "path": rel_path}

    try:
        raw = snapshot_path.read_bytes()
    except OSError as exc:
        return 404, {"ok": False, "error": f"Snapshot read failed: {exc}", "run_id": run_id, "path": rel_path}

    safe_start = max(0, int(start))
    safe_limit = max(256, min(MAX_FILE_BYTES, int(limit)))
    total = len(raw)
    chunk = raw[safe_start : safe_start + safe_limit]
    next_start = safe_start + len(chunk)
    truncated = next_start < total
    is_text = _is_text_blob(chunk)
    content = chunk.decode("utf-8", errors="replace") if is_text else ""
    return 200, {
        "ok": True,
        "run_id": run_id,
        "path": rel_path,
        "source": "snapshot",
        "snapshot_rel": snapshot_rel,
        "snapshot_event_index": int(selected.get("event_index", 0)),
        "snapshot_path": str(snapshot_path),
        "is_text": is_text,
        "size": total,
        "content": content,
        "content_total_chars": len(content),
        "range_start": safe_start,
        "range_limit": safe_limit,
        "truncated": truncated,
        "next_start": next_start if truncated else None,
    }


def _resolve_workspace_file_from_run(run_dir: Path, events: list[dict[str, Any]], rel_path: str) -> Path | None:
    key = str(rel_path or "").strip().replace("\\", "/")
    if not key:
        return None

    for root in _workspace_roots_from_events(run_dir, events):
        prefix = root.name
        if key == prefix:
            candidate = root
        elif key.startswith(prefix + "/"):
            candidate = (root / key[len(prefix) + 1 :]).resolve()
        else:
            continue
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def _read_workspace_file(path: Path, *, run_id: str, rel_path: str, start: int, limit: int) -> tuple[int, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return 500, {"ok": False, "error": f"read failed: {exc}", "run_id": run_id, "path": rel_path}

    total = len(raw)
    chunk = raw[start : start + limit]
    next_start = start + len(chunk)
    truncated = next_start < total
    if _is_text_blob(chunk):
        content = chunk.decode("utf-8", errors="replace")
        is_text = True
    else:
        content = ""
        is_text = False

    return 200, {
        "ok": True,
        "run_id": run_id,
        "path": rel_path,
        "source": "workspace",
        "workspace_path": str(path),
        "is_text": is_text,
        "size": total,
        "content": content,
        "content_total_chars": len(content),
        "range_start": start,
        "range_limit": limit,
        "truncated": truncated,
        "next_start": next_start if truncated else None,
    }


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65_536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class V2RunSession:
    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        root_dir: Path,
        spec_path: Path,
        agent_path: Path,
        initial_data: dict[str, Any],
        task: str,
        requested_mode: str,
        mock_mode: str,
        capability_profile: str,
        max_iter: int,
        model_override: str,
        max_budget_usd: float | None,
    ) -> None:
        self.run_id = run_id
        self.session_id = str(session_id or "")
        self.root_dir = root_dir.resolve()
        self.spec_path = spec_path.resolve()
        self.agent_path = agent_path.resolve()
        self.initial_data = dict(initial_data)
        self.task = str(task or "agent_ontology_live_viewer_v2")
        self.requested_mode = str(requested_mode or "default")
        self.mock_mode = str(mock_mode or "live")
        self.capability_profile = str(capability_profile or "external_enabled")
        self.max_iter = int(max_iter)
        self.model_override = str(model_override or "")
        self.max_budget_usd = max_budget_usd

        self.started_at = datetime.now(timezone.utc)
        self.done = False
        self.error = ""

        self.run_dir = _run_artifact_dir(self.root_dir, self.run_id)
        self._events_log_path = (self.run_dir / "events.jsonl").resolve()
        self._snapshot_dir = (self.run_dir / "file_snapshots").resolve()
        self._final_data_path = (self.run_dir / "final_data.json").resolve()

        self.events: list[dict[str, Any]] = []
        self.final_data: dict[str, Any] = {}
        self.final_channel: Any = None
        self.trace_path: str = ""
        self.current_process: str = ""
        self._current_process_span_id: str | None = None

        self._next_event_index = 1
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._process: subprocess.Popen[str] | None = None
        self._worker: threading.Thread | None = None
        self._stop_requested = False
        self._file_versions: dict[str, list[dict[str, Any]]] = {}
        self._latest_file_hash: dict[str, str] = {}
        self._known_file_paths: set[str] = set()
        self.snapshot_keep_events = _env_int(
            "AGENT_ONTOLOGY_V2_SNAPSHOT_KEEP_EVENTS",
            DEFAULT_SNAPSHOT_KEEP_EVENTS,
            minimum=1,
            maximum=5000,
        )
        self.snapshot_keep_versions_per_file = _env_int(
            "AGENT_ONTOLOGY_V2_SNAPSHOT_KEEP_VERSIONS_PER_FILE",
            DEFAULT_SNAPSHOT_KEEP_VERSIONS_PER_FILE,
            minimum=1,
            maximum=1000,
        )
        self.snapshot_prune_total = 0

    def _append_event_log(self, event: dict[str, Any]) -> None:
        try:
            self._events_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._events_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
        except OSError:
            # best-effort persistence only
            pass

    def emit(self, event_type: str, *, facet: str = "control", entity_id: str | None = None, **payload: Any) -> dict[str, Any]:
        with self._cond:
            idx = self._next_event_index
            self._next_event_index += 1
            event = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "event_index": idx,
                "seq": idx,
                "event_id": f"{self.run_id}:{idx}",
                "ts": datetime.now(timezone.utc).isoformat(),
                "time": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "facet": facet,
                "entity_id": entity_id,
                "span_id": payload.pop("span_id", f"span/{idx}"),
                "parent_span_id": payload.pop("parent_span_id", None),
                "actor": payload.pop("actor", "runtime"),
            }
            event.update(payload)
            self.events.append(event)
            self._append_event_log(event)
            self._cond.notify_all()
            return event

    def _parse_output_line(self, line: str) -> None:
        text = str(line or "").rstrip("\n")
        if not text.strip():
            return

        log_evt = self.emit(
            "log",
            facet="control",
            entity_id=f"run:{self.run_id}",
            parent_span_id=self._current_process_span_id,
            line=text,
        )

        stripped = text.strip()
        m_iter = ITERATION_STATE_RE.match(stripped)
        if m_iter:
            proc = str(m_iter.group("state") or "").strip()
            if proc.endswith("(fan-out)"):
                proc = proc[: -len("(fan-out)")].strip()
            self.current_process = proc
            state_evt = self.emit(
                "state",
                facet="state",
                entity_id=f"process:{proc}",
                process=proc,
                iteration=_safe_int(m_iter.group("iter"), 0),
                payload={"process": proc, "iteration": _safe_int(m_iter.group("iter"), 0)},
            )
            self._current_process_span_id = str(state_evt.get("span_id") or self._current_process_span_id or "")
            self._capture_workspace_snapshots(trigger_event_index=_event_index_of(state_evt), reason="state")
            return

        m_llm = LLM_CALL_RE.match(text)
        if m_llm:
            self.emit(
                "llm_call",
                facet="llm",
                entity_id=f"process:{self.current_process or 'unknown'}",
                parent_span_id=self._current_process_span_id,
                process=self.current_process,
                agent=str(m_llm.group("agent") or "").strip(),
                model=str(m_llm.group("model") or "").strip(),
                duration_ms=_safe_int(m_llm.group("dur"), 0),
            )
            return

        m_llm_summary = LLM_SUMMARY_RE.match(stripped)
        if m_llm_summary:
            self.emit(
                "llm_summary",
                facet="llm",
                entity_id=f"process:{self.current_process or 'unknown'}",
                parent_span_id=self._current_process_span_id,
                process=self.current_process,
                agent=str(m_llm_summary.group("agent") or "").strip(),
                model=str(m_llm_summary.group("model") or "").strip(),
                duration_ms=_safe_int(m_llm_summary.group("dur"), 0),
            )
            return

        m_step = STEP_ACTION_RE.match(stripped)
        if m_step:
            label = str(m_step.group("label") or "").strip()
            facet = "routing" if "?" in label or "gate" in label.lower() else "state"
            event_type = "routing_decision" if facet == "routing" else "step_action"
            if "plan" in label.lower():
                facet = "planning"
                event_type = "plan_step"
            self.emit(
                event_type,
                facet=facet,
                entity_id=f"process:{self.current_process or 'unknown'}",
                parent_span_id=self._current_process_span_id,
                process=self.current_process,
                label=label,
            )
            return

        m_tool_obs = TOOL_OBSERVATION_RE.match(stripped)
        if m_tool_obs:
            tool = str(m_tool_obs.group("tool") or "").strip()
            detail = str(m_tool_obs.group("detail") or "").strip()
            self.emit(
                "tool_result",
                facet="tools",
                entity_id=f"tool:{tool}",
                parent_span_id=self._current_process_span_id,
                process=self.current_process,
                tool=tool,
                detail=detail,
                status="not_implemented" if "not implemented" in detail.lower() else "observed",
            )
            return

        m_obs = OBSERVATION_RE.match(stripped)
        if m_obs:
            obs = str(m_obs.group("obs") or "").strip()
            self.emit(
                "observation",
                facet="observation",
                entity_id=f"process:{self.current_process or 'unknown'}",
                parent_span_id=self._current_process_span_id,
                process=self.current_process,
                observation=obs,
            )
            return

        m_schema = SCHEMA_RE.match(stripped)
        if m_schema:
            self.emit(
                "schema_violation",
                facet="error",
                entity_id=f"process:{self.current_process or 'unknown'}",
                parent_span_id=self._current_process_span_id,
                process=self.current_process,
                schema=str(m_schema.group("schema") or "").strip(),
                detail=str(m_schema.group("detail") or "").strip(),
            )
            return

        m_fail = FAIL_RE.match(stripped)
        if m_fail:
            self.emit(
                "failure",
                facet="error",
                entity_id=f"process:{self.current_process or 'unknown'}",
                parent_span_id=self._current_process_span_id,
                process=self.current_process,
                detail=str(m_fail.group("detail") or "").strip(),
            )
            return

        m_runtime = RUNTIME_RE.match(stripped)
        if m_runtime:
            msg = str(m_runtime.group("msg") or "").strip()
            self.emit(
                "runtime_info",
                facet="runtime",
                entity_id=f"run:{self.run_id}",
                parent_span_id=self._current_process_span_id,
                process=self.current_process,
                message=msg,
            )
            return

        m_trace = TRACE_WRITTEN_RE.match(stripped)
        if m_trace:
            trace_ref = str(m_trace.group("path") or "").strip()
            trace_path = Path(trace_ref)
            if not trace_path.is_absolute():
                trace_path = (self.run_dir / trace_path).resolve()
            self.trace_path = str(trace_path)
            self.emit(
                "trace_file",
                facet="control",
                entity_id=f"run:{self.run_id}",
                parent_span_id=self._current_process_span_id,
                path=self.trace_path,
                total_calls=_safe_int(m_trace.group("calls"), 0),
                total_duration_ms=_safe_int(m_trace.group("dur"), 0),
            )
            return

        m_keys = FINAL_KEYS_RE.match(stripped)
        if m_keys:
            raw_keys = m_keys.group("keys")
            keys: list[str] = []
            try:
                parsed = ast.literal_eval(raw_keys)
                if isinstance(parsed, list):
                    keys = [str(k) for k in parsed]
            except (SyntaxError, ValueError):
                keys = []
            self.emit(
                "final_keys",
                facet="output",
                entity_id=f"run:{self.run_id}",
                parent_span_id=self._current_process_span_id,
                keys=keys,
            )
            return

        if any(tok in stripped.lower() for tok in ("workspace", ".py", ".md", "artifacts")):
            self._capture_workspace_snapshots(trigger_event_index=_event_index_of(log_evt), reason="log")

    def _extract_final_channel(self) -> Any:
        if not isinstance(self.final_data, dict):
            return None
        for key in ("final_channel", "answer", "final_output", "draft_response", "text"):
            value = self.final_data.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    def _load_final_data(self) -> None:
        if not self._final_data_path.exists() or not self._final_data_path.is_file():
            return
        try:
            payload = json.loads(self._final_data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        data = payload.get("data")
        if isinstance(data, dict):
            self.final_data = data
        returncode = payload.get("returncode")
        if _safe_int(returncode, 0) != 0 and not self.error:
            self.error = str(payload.get("error") or "")

    def _final_summary(self) -> dict[str, Any]:
        if not isinstance(self.final_data, dict):
            return {"keys": [], "has_final_channel": False, "has_action_decision": False, "finalized": False, "answer": ""}
        return {
            "keys": sorted(self.final_data.keys()),
            "has_final_channel": "final_channel" in self.final_data,
            "has_action_decision": "action_decision" in self.final_data,
            "finalized": bool(self.final_data.get("finalized", False)),
            "answer": str(self.final_data.get("answer", ""))[:500],
        }

    def _workspace_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()

        def add(raw: str) -> None:
            path_raw = str(raw or "").strip()
            if not path_raw:
                return
            p = Path(path_raw)
            if not p.is_absolute():
                p = (self.run_dir / p).resolve()
            else:
                p = p.resolve()
            if p in seen:
                return
            if p.exists() and p.is_dir():
                roots.append(p)
                seen.add(p)

        for src in (self.initial_data, self.final_data):
            if not isinstance(src, dict):
                continue
            for key in ("workspace_dir", "workspace_path", "output_dir"):
                add(str(src.get(key) or ""))

        if self.run_dir.exists():
            for child in self.run_dir.iterdir():
                if child.is_dir() and "workspace" in child.name.lower():
                    add(str(child))
        return roots

    def _resolve_workspace_path(self, rel_path: str) -> Path | None:
        key = str(rel_path or "").strip().replace("\\", "/")
        if not key:
            return None
        for root in self._workspace_roots():
            prefix = root.name
            if key == prefix:
                cand = root
            elif key.startswith(prefix + "/"):
                cand = (root / key[len(prefix) + 1 :]).resolve()
            else:
                continue
            try:
                cand.relative_to(root.resolve())
            except ValueError:
                continue
            if cand.exists() and cand.is_file():
                return cand
        return None

    def _record_file_version(
        self,
        rel_path: str,
        *,
        event_index: int,
        sha1: str | None,
        size: int,
        is_text: bool,
        deleted: bool,
        snapshot_rel: str | None,
    ) -> None:
        versions = self._file_versions.setdefault(rel_path, [])
        version = {
            "event_index": int(event_index),
            "sha1": sha1,
            "size": int(size),
            "is_text": bool(is_text),
            "deleted": bool(deleted),
            "snapshot_rel": snapshot_rel,
        }
        versions.append(version)
        versions.sort(key=lambda v: int(v.get("event_index", 0)))
        if deleted:
            self._latest_file_hash.pop(rel_path, None)
        elif sha1:
            self._latest_file_hash[rel_path] = sha1
        self._known_file_paths.add(rel_path)

    def _prune_snapshots_unlocked(self, *, force: bool = False) -> dict[str, int]:
        latest_event = max((_event_index_of(evt) for evt in self.events), default=0)
        cutoff_event = max(0, int(latest_event) - int(self.snapshot_keep_events))
        keep_per_file = max(1, int(self.snapshot_keep_versions_per_file))

        removed_versions = 0
        removed_snapshot_files = 0
        removed_snapshot_bytes = 0
        removed_candidates: set[str] = set()
        rebuilt: dict[str, list[dict[str, Any]]] = {}

        for path, versions in self._file_versions.items():
            if not versions:
                rebuilt[path] = []
                continue
            ordered = sorted(versions, key=lambda v: int(v.get("event_index", 0)))
            keep = [v for v in ordered if int(v.get("event_index", 0)) >= cutoff_event]
            if not keep:
                keep = [ordered[-1]]
            if len(keep) > keep_per_file:
                keep = keep[-keep_per_file:]

            if len(keep) < len(ordered):
                removed_versions += (len(ordered) - len(keep))
                keep_ids = {id(v) for v in keep}
                for old in ordered:
                    if id(old) in keep_ids:
                        continue
                    rel = str(old.get("snapshot_rel") or "").strip()
                    if rel:
                        removed_candidates.add(rel)

            rebuilt[path] = keep

        if removed_versions == 0 and not force:
            return {
                "removed_versions": 0,
                "removed_snapshot_files": 0,
                "removed_snapshot_bytes": 0,
                "cutoff_event_index": cutoff_event,
            }

        self._file_versions = rebuilt
        self._latest_file_hash = {}
        self._known_file_paths = set(rebuilt.keys())
        live_refs: set[str] = set()
        for versions in rebuilt.values():
            if not versions:
                continue
            latest = versions[-1]
            if not bool(latest.get("deleted", False)):
                sha1 = str(latest.get("sha1") or "").strip()
                if sha1:
                    # path is reconstructed from map iteration below
                    pass
            for ver in versions:
                ref = str(ver.get("snapshot_rel") or "").strip()
                if ref:
                    live_refs.add(ref)

        for path, versions in rebuilt.items():
            if not versions:
                continue
            latest = versions[-1]
            if bool(latest.get("deleted", False)):
                continue
            sha1 = str(latest.get("sha1") or "").strip()
            if sha1:
                self._latest_file_hash[path] = sha1

        for rel in sorted(removed_candidates):
            if rel in live_refs:
                continue
            snap = (self._snapshot_dir / rel).resolve()
            try:
                snap.relative_to(self._snapshot_dir.resolve())
            except ValueError:
                continue
            try:
                if snap.exists() and snap.is_file():
                    size = int(snap.stat().st_size)
                    snap.unlink()
                    removed_snapshot_files += 1
                    removed_snapshot_bytes += size
            except OSError:
                continue

        if force and self._snapshot_dir.exists() and self._snapshot_dir.is_dir():
            for snap in sorted(self._snapshot_dir.rglob("*")):
                if not snap.is_file():
                    continue
                try:
                    rel = snap.relative_to(self._snapshot_dir).as_posix()
                except ValueError:
                    continue
                if rel in live_refs:
                    continue
                try:
                    size = int(snap.stat().st_size)
                    snap.unlink()
                    removed_snapshot_files += 1
                    removed_snapshot_bytes += size
                except OSError:
                    continue

        return {
            "removed_versions": removed_versions,
            "removed_snapshot_files": removed_snapshot_files,
            "removed_snapshot_bytes": removed_snapshot_bytes,
            "cutoff_event_index": cutoff_event,
        }

    def prune_snapshots(self, *, force: bool = False) -> dict[str, int]:
        with self._lock:
            stats = self._prune_snapshots_unlocked(force=force)
            removed = int(stats.get("removed_versions", 0))
            if removed > 0:
                self.snapshot_prune_total += removed
            return stats

    def _capture_workspace_snapshots(self, *, trigger_event_index: int, reason: str) -> None:
        if trigger_event_index <= 0:
            return
        if not self._workspace_roots():
            return

        seen_now: set[str] = set()
        changed_count = 0
        for root in self._workspace_roots():
            prefix = root.name
            file_count = 0
            for file_path in sorted(root.rglob("*")):
                if file_count >= MAX_MANIFEST_FILES:
                    break
                if not file_path.is_file():
                    continue
                try:
                    rel = file_path.relative_to(root).as_posix()
                except ValueError:
                    continue
                key = f"{prefix}/{rel}" if rel else prefix
                seen_now.add(key)
                file_count += 1

                try:
                    sha1 = _sha1_file(file_path)
                    size = file_path.stat().st_size
                except OSError:
                    continue

                last_hash = self._latest_file_hash.get(key)
                if last_hash == sha1:
                    continue

                try:
                    raw = file_path.read_bytes()
                except OSError:
                    continue
                is_text = _is_text_blob(raw)
                snapshot_rel = (Path(f"{int(trigger_event_index):08d}") / key).as_posix()
                snapshot_path = (self._snapshot_dir / snapshot_rel).resolve()
                try:
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    snapshot_path.write_bytes(raw)
                except OSError:
                    snapshot_rel = None

                self._record_file_version(
                    key,
                    event_index=trigger_event_index,
                    sha1=sha1,
                    size=size,
                    is_text=is_text,
                    deleted=False,
                    snapshot_rel=snapshot_rel,
                )
                changed_count += 1
                self.emit(
                    "file_diff",
                    facet="files",
                    entity_id=f"file:{key}",
                    parent_span_id=self._current_process_span_id,
                    path=key,
                    change_type="added" if last_hash is None else "modified",
                    snapshot_rel=snapshot_rel,
                    snapshot_event_index=trigger_event_index,
                    size=size,
                    is_text=is_text,
                    reason=reason,
                )

        removed = sorted(path for path in self._known_file_paths if path not in seen_now and path in self._latest_file_hash)
        for key in removed:
            self._record_file_version(
                key,
                event_index=trigger_event_index,
                sha1=None,
                size=0,
                is_text=True,
                deleted=True,
                snapshot_rel=None,
            )
            self.emit(
                "file_diff",
                facet="files",
                entity_id=f"file:{key}",
                parent_span_id=self._current_process_span_id,
                path=key,
                change_type="deleted",
                snapshot_rel=None,
                snapshot_event_index=trigger_event_index,
                size=0,
                is_text=True,
                reason=reason,
            )

        if changed_count > 0:
            self.emit(
                "file_snapshot_scan",
                facet="files",
                entity_id=f"run:{self.run_id}",
                parent_span_id=self._current_process_span_id,
                changed_files=changed_count,
                reason=reason,
                snapshot_event_index=trigger_event_index,
            )

        prune = self.prune_snapshots(force=False)
        if int(prune.get("removed_versions", 0)) > 0 or int(prune.get("removed_snapshot_files", 0)) > 0:
            self.emit(
                "snapshot_pruned",
                facet="files",
                entity_id=f"run:{self.run_id}",
                parent_span_id=self._current_process_span_id,
                trigger_event_index=trigger_event_index,
                reason=reason,
                **prune,
            )

    def file_manifest(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for path, versions in sorted(self._file_versions.items(), key=lambda item: item[0]):
            if not versions:
                continue
            latest = versions[-1]
            rows.append(
                {
                    "path": path,
                    "versions": len(versions),
                    "latest_event_index": int(latest.get("event_index", 0)),
                    "deleted": bool(latest.get("deleted", False)),
                    "is_text": bool(latest.get("is_text", True)),
                    "size": int(latest.get("size", 0)),
                    "content_available": bool(latest.get("snapshot_rel")),
                    "event_indexes": [int(v.get("event_index", 0)) for v in versions][-60:],
                    "source": "snapshot_index",
                }
            )

        if not rows:
            fallback = _scan_workspace_manifest(self.run_dir, self.events)
            return {"files": fallback, "count": len(fallback)}
        rows.sort(key=lambda r: str(r.get("path") or ""))
        return {"files": rows, "count": len(rows)}

    def get_workspace_file(self, rel_path: str, *, start: int = 0, limit: int = MAX_FILE_BYTES) -> tuple[int, dict[str, Any]]:
        candidate = self._resolve_workspace_path(rel_path)
        if candidate is None:
            return 404, {"ok": False, "error": f"File not found in workspace: {rel_path}", "run_id": self.run_id, "path": rel_path}
        safe_start = max(0, int(start))
        safe_limit = max(256, min(MAX_FILE_BYTES, int(limit)))
        return _read_workspace_file(candidate, run_id=self.run_id, rel_path=rel_path, start=safe_start, limit=safe_limit)

    def get_file_snapshot(
        self,
        rel_path: str,
        *,
        at_event_index: int | None = None,
        start: int = 0,
        limit: int = MAX_FILE_BYTES,
    ) -> tuple[int, dict[str, Any]]:
        versions = self._file_versions.get(rel_path, [])
        if not versions:
            return 404, {
                "ok": False,
                "error": f"No snapshot available for path: {rel_path}",
                "run_id": self.run_id,
                "path": rel_path,
                "at_event_index": at_event_index,
            }

        target_idx = int(at_event_index) if isinstance(at_event_index, int) and at_event_index >= 0 else 10**12
        selected: dict[str, Any] | None = None
        for version in versions:
            if int(version.get("event_index", 0)) <= target_idx:
                selected = version
            else:
                break

        if not selected:
            return 404, {
                "ok": False,
                "error": f"No snapshot at or before event_index={target_idx} for path: {rel_path}",
                "run_id": self.run_id,
                "path": rel_path,
                "at_event_index": at_event_index,
            }
        if bool(selected.get("deleted", False)):
            return 404, {
                "ok": False,
                "error": f"File deleted at selected cursor for path: {rel_path}",
                "run_id": self.run_id,
                "path": rel_path,
                "at_event_index": at_event_index,
                "deleted": True,
            }

        snapshot_rel = str(selected.get("snapshot_rel") or "")
        if not snapshot_rel:
            return 404, {
                "ok": False,
                "error": f"Snapshot content missing for path: {rel_path}",
                "run_id": self.run_id,
                "path": rel_path,
                "at_event_index": at_event_index,
            }

        snapshot_path = (self._snapshot_dir / snapshot_rel).resolve()
        try:
            snapshot_path.relative_to(self._snapshot_dir.resolve())
        except ValueError:
            return 404, {"ok": False, "error": f"Invalid snapshot path for {rel_path}", "run_id": self.run_id, "path": rel_path}

        try:
            raw = snapshot_path.read_bytes()
        except OSError as exc:
            return 404, {"ok": False, "error": f"Snapshot read failed: {exc}", "run_id": self.run_id, "path": rel_path}

        safe_start = max(0, int(start))
        safe_limit = max(256, min(MAX_FILE_BYTES, int(limit)))
        total = len(raw)
        chunk = raw[safe_start : safe_start + safe_limit]
        next_start = safe_start + len(chunk)
        truncated = next_start < total
        is_text = _is_text_blob(chunk)
        content = chunk.decode("utf-8", errors="replace") if is_text else ""
        return 200, {
            "ok": True,
            "run_id": self.run_id,
            "path": rel_path,
            "source": "snapshot",
            "snapshot_path": str(snapshot_path),
            "snapshot_event_index": int(selected.get("event_index", 0)),
            "snapshot_rel": snapshot_rel,
            "is_text": is_text,
            "size": total,
            "content": content,
            "content_total_chars": len(content),
            "range_start": safe_start,
            "range_limit": safe_limit,
            "truncated": truncated,
            "next_start": next_start if truncated else None,
        }

    def get_events_after(self, last_event_index: int, timeout: float = 1.0) -> tuple[list[dict[str, Any]], bool]:
        with self._cond:
            if not self.done and not any(_event_index_of(evt) > last_event_index for evt in self.events):
                self._cond.wait(timeout=max(0.01, float(timeout)))
            rows = [dict(evt) for evt in self.events if _event_index_of(evt) > last_event_index]
            return rows, self.done

    def snapshot(self) -> dict[str, Any]:
        latest = max((_event_index_of(e) for e in self.events), default=0)
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "spec_path": str(self.spec_path),
            "agent_path": str(self.agent_path),
            "started_at": self.started_at.isoformat(),
            "done": self.done,
            "error": self.error,
            "returncode": self._process.returncode if self._process is not None else None,
            "events": len(self.events),
            "latest_event_index": latest,
            "trace_path": self.trace_path,
            "run_dir": str(self.run_dir),
            "workspace_dir": str(self.initial_data.get("workspace_dir") or ""),
            "mock_mode": self.mock_mode,
            "capability_profile": self.capability_profile,
            "effective_model": self.model_override,
            "snapshot_keep_events": self.snapshot_keep_events,
            "snapshot_keep_versions_per_file": self.snapshot_keep_versions_per_file,
            "snapshot_prune_total": self.snapshot_prune_total,
        }

    def _run_worker(self) -> None:
        proc = self._process
        if proc is None:
            self.error = "Runner process not initialized"
            self.done = True
            return

        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    self._parse_output_line(line)
            returncode = proc.wait()
        except Exception as exc:
            returncode = 1
            if not self.error:
                self.error = f"{type(exc).__name__}: {exc}"

        self._load_final_data()

        snapshot_marker = max((_event_index_of(e) for e in self.events), default=0)
        if snapshot_marker > 0:
            self._capture_workspace_snapshots(trigger_event_index=snapshot_marker, reason="post-run")

        if self.final_data:
            final_evt = self.emit(
                "final_data",
                facet="output",
                entity_id=f"run:{self.run_id}",
                parent_span_id=self._current_process_span_id,
                data=self.final_data,
                summary={"keys": sorted(self.final_data.keys())},
            )
            self._capture_workspace_snapshots(trigger_event_index=_event_index_of(final_evt), reason="final_data")

        self.final_channel = self._extract_final_channel()
        if self.final_channel is not None:
            channel_evt = self.emit(
                "final_channel",
                facet="output",
                entity_id=f"run:{self.run_id}",
                parent_span_id=self._current_process_span_id,
                payload=self.final_channel,
            )
            if self.session_id:
                if isinstance(self.final_channel, str):
                    reply = self.final_channel
                else:
                    reply = json.dumps(self.final_channel, ensure_ascii=False, default=str)
                _append_session_turn(
                    self.root_dir,
                    self.session_id,
                    role="assistant",
                    text=reply,
                    run_id=self.run_id,
                    event_index=_event_index_of(channel_evt),
                )

        if returncode != 0 and not self.error:
            self.error = f"Process exited with return code {returncode}"

        completed_evt = self.emit(
            "completed",
            facet="done",
            entity_id=f"run:{self.run_id}",
            parent_span_id=self._current_process_span_id,
            returncode=returncode,
            error=self.error,
            trace_path=self.trace_path,
            final_channel=self.final_channel,
            final_data_summary=self._final_summary(),
            process_metrics={},
        )
        self._capture_workspace_snapshots(trigger_event_index=_event_index_of(completed_evt), reason="completed")
        if self.session_id and not self.final_channel:
            fallback_reply = ""
            if isinstance(self.final_data, dict):
                for key in ("final_output", "answer", "draft_response", "text"):
                    value = self.final_data.get(key)
                    if isinstance(value, str) and value.strip():
                        fallback_reply = value.strip()
                        break
            if self.error:
                fallback_reply = (fallback_reply + "\n\n" if fallback_reply else "") + f"[error] {self.error}"
            if fallback_reply:
                _append_session_turn(
                    self.root_dir,
                    self.session_id,
                    role="assistant",
                    text=fallback_reply,
                    run_id=self.run_id,
                    event_index=_event_index_of(completed_evt),
                )

        with self._cond:
            self.done = True
            self._cond.notify_all()

    def start(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir = Path(str(self.initial_data.get("workspace_dir") or "")).resolve()
            workspace_dir.mkdir(parents=True, exist_ok=True)

            started_evt = self.emit(
                "started",
                facet="input",
                entity_id=f"run:{self.run_id}",
                spec_path=str(self.spec_path),
                agent_path=str(self.agent_path),
                run_dir=str(self.run_dir),
                session_id=self.session_id,
                input=self.initial_data,
                mock_mode=self.mock_mode,
                capability_profile=self.capability_profile,
                model_override=self.model_override,
                effective_model=self.model_override,
            )
            self._capture_workspace_snapshots(trigger_event_index=_event_index_of(started_evt), reason="started")

            env = os.environ.copy()
            env["AO_V2_INITIAL_DATA_JSON"] = json.dumps(self.initial_data, ensure_ascii=False)
            env["AO_V2_FINAL_DATA_PATH"] = str(self._final_data_path)
            env["AGENT_ONTOLOGY_TASK"] = self.task
            env["AGENT_ONTOLOGY_MAX_ITER"] = str(self.max_iter)
            env["AGENT_ONTOLOGY_MOCK_MODE"] = self.mock_mode
            env["AGENT_ONTOLOGY_CAPABILITY_PROFILE"] = self.capability_profile
            env["AGENT_ONTOLOGY_TRACE_PREFIX"] = f"agent_ontology/live_v2/{self.run_id}"
            env["AGENT_ONTOLOGY_WORKSPACE_DIR"] = str(workspace_dir)
            if self.model_override:
                env["AGENT_ONTOLOGY_MODEL"] = self.model_override
            if self.max_budget_usd is not None:
                env["AGENT_ONTOLOGY_MAX_BUDGET_USD"] = str(self.max_budget_usd)

            cmd = [sys.executable, "-u", "-c", _build_runner_code(self.agent_path)]
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self.root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            self._worker = threading.Thread(target=self._run_worker, name=f"v2-run-{self.run_id}", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            proc = self._process
        if proc is None:
            return
        if proc.poll() is not None:
            return
        self.emit("stop_requested", facet="control", entity_id=f"run:{self.run_id}")
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        except OSError:
            pass


def _session_for_run(run_id: str) -> V2RunSession | None:
    with RUNS_V2_LOCK:
        return RUNS_V2.get(run_id)


def _load_events_for_run(root_dir: Path, run_id: str) -> tuple[list[dict[str, Any]], str]:
    session = _session_for_run(run_id)
    if session is not None:
        return [dict(evt) for evt in session.events], "session"
    return _load_persisted_run_events(root_dir, run_id)


def _extract_objective(events: list[dict[str, Any]]) -> str:
    for evt in events:
        if evt.get("type") == "started":
            payload = evt.get("input")
            if isinstance(payload, dict):
                for key in ("text", "objective", "task", "user"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
    for evt in reversed(events):
        if evt.get("type") == "final_data":
            data = evt.get("data")
            if isinstance(data, dict):
                for key in ("objective", "text", "task", "user"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
    return ""


def _extract_workspace_dir(events: list[dict[str, Any]], run_dir: Path) -> str:
    for evt in reversed(events):
        if evt.get("type") == "final_data":
            data = evt.get("data")
            if isinstance(data, dict):
                for key in ("workspace_dir", "workspace_path", "output_dir"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
    for evt in events:
        if evt.get("type") == "started":
            payload = evt.get("input")
            if isinstance(payload, dict):
                for key in ("workspace_dir", "workspace_path", "output_dir"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
    for candidate in sorted(run_dir.iterdir() if run_dir.exists() else [], key=lambda p: p.name):
        if candidate.is_dir() and "workspace" in candidate.name.lower():
            return str(candidate)
    return ""


def _run_summary(root_dir: Path, ref: RunRef) -> dict[str, Any]:
    events, source = _load_events_for_run(root_dir, ref.run_id)
    count = len(events)
    latest_idx = max((_event_index_of(e) for e in events), default=0)
    latest_ts = str(events[-1].get("ts") or events[-1].get("time") or "") if events else ""

    started = next((e for e in events if e.get("type") == "started"), None)
    completed = next((e for e in reversed(events) if e.get("type") == "completed"), None)

    status = "unknown"
    returncode = None
    error = ""
    if isinstance(completed, dict):
        returncode = completed.get("returncode")
        error = str(completed.get("error") or "").strip()
        status = "completed" if _safe_int(returncode, 1) == 0 and not error else "error"

    session = _session_for_run(ref.run_id)
    if session is not None and not session.done:
        status = "running"

    spec_path = str(started.get("spec_path") or "") if isinstance(started, dict) else ""
    model = ""
    session_id = ""
    if isinstance(started, dict):
        model = str(started.get("effective_model") or started.get("model_override") or "")
        session_id = str(started.get("session_id") or "")

    return {
        "run_id": ref.run_id,
        "run_dir": str(ref.run_dir),
        "source": source,
        "event_count": count,
        "latest_event_index": latest_idx,
        "started_at": str(started.get("ts") or started.get("time") or "") if isinstance(started, dict) else "",
        "updated_at": latest_ts,
        "status": status,
        "returncode": returncode,
        "error": error,
        "objective": _extract_objective(events),
        "workspace_dir": _extract_workspace_dir(events, ref.run_dir),
        "spec_path": spec_path,
        "model": model,
        "session_id": session_id,
        "mtime": datetime.fromtimestamp(ref.mtime, tz=timezone.utc).isoformat(),
    }


def _manifest_for_run(root_dir: Path, run_id: str) -> tuple[int, dict[str, Any]]:
    session = _session_for_run(run_id)
    if session is not None:
        try:
            manifest = session.file_manifest()
            rows = manifest.get("files") if isinstance(manifest, dict) else []
        except Exception:
            rows = []
        rows = [r for r in rows if isinstance(r, dict) and str(r.get("path") or "").strip()]
        rows.sort(key=lambda r: str(r.get("path") or ""))
        return 200, {"ok": True, "run_id": run_id, "count": len(rows), "files": rows}

    try:
        events, _ = _load_persisted_run_events(root_dir, run_id)
        run_dir = _run_artifact_dir(root_dir, run_id)
    except ValueError:
        return 404, {"ok": False, "run_id": run_id, "error": f"Unknown run: {run_id}"}

    if not events and not run_dir.exists():
        return 404, {"ok": False, "run_id": run_id, "error": f"Unknown run: {run_id}"}

    rows = _manifest_from_events(run_dir, events)
    if not rows:
        rows = _scan_workspace_manifest(run_dir, events)
    rows.sort(key=lambda r: str(r.get("path") or ""))
    return 200, {"ok": True, "run_id": run_id, "count": len(rows), "files": rows}


def _read_file_for_run(
    root_dir: Path,
    run_id: str,
    rel_path: str,
    source: str,
    at_event_index: int | None,
    start: int,
    limit: int,
) -> tuple[int, dict[str, Any]]:
    session = _session_for_run(run_id)
    safe_start = max(0, start)
    safe_limit = max(256, min(MAX_FILE_BYTES, limit if limit > 0 else MAX_FILE_BYTES))
    if session is not None:
        if source == "workspace":
            return session.get_workspace_file(rel_path, start=safe_start, limit=safe_limit)
        status, payload = session.get_file_snapshot(rel_path, at_event_index=at_event_index, start=safe_start, limit=safe_limit)
        if status == 404:
            status_ws, payload_ws = session.get_workspace_file(rel_path, start=safe_start, limit=safe_limit)
            payload_ws = dict(payload_ws)
            payload_ws["source_fallback"] = "workspace"
            return status_ws, payload_ws
        return status, payload

    try:
        run_dir = _run_artifact_dir(root_dir, run_id)
    except ValueError:
        return 404, {"ok": False, "run_id": run_id, "path": rel_path, "error": f"Unknown run: {run_id}"}

    events, _ = _load_persisted_run_events(root_dir, run_id)
    if source == "snapshot":
        status_snap, payload_snap = _read_snapshot_for_persisted_run(
            run_dir,
            events,
            run_id,
            rel_path,
            at_event_index,
            safe_start,
            safe_limit,
        )
        if status_snap == 200:
            return status_snap, payload_snap

    candidate = _resolve_workspace_file_from_run(run_dir, events, rel_path)
    if candidate is None:
        return 404, {"ok": False, "run_id": run_id, "path": rel_path, "error": f"File not found in workspace: {rel_path}"}
    status, payload = _read_workspace_file(candidate, run_id=run_id, rel_path=rel_path, start=safe_start, limit=safe_limit)
    if source == "snapshot":
        payload["source_fallback"] = "workspace"
    return status, payload


def _spec_paths(root_dir: Path) -> list[str]:
    roots = [
        (root_dir / "agent_ontology" / "specs").resolve(),
        (root_dir / "specs").resolve(),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for base in roots:
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.glob("*.yaml")):
            rel = path.resolve().relative_to(root_dir.resolve()).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            out.append(rel)
    return out


class LiveViewerV2Handler(SimpleHTTPRequestHandler):
    root_dir: Path = Path.cwd()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.root_dir), **kwargs)

    def _send_json(self, payload: object, status: int = 200) -> None:
        blob = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def _read_json_body(self) -> dict[str, object]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ValueError(f"Invalid Content-Length: {raw_len}") from exc
        payload = self.rfile.read(length) if length > 0 else b"{}"
        if not payload:
            return {}
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/v2/health":
            self._send_json({"ok": True, "root": str(self.root_dir.resolve()), "version": "v2"})
            return

        if path == "/api/v2/specs":
            specs = _spec_paths(self.root_dir)
            default = next((s for s in specs if s.endswith("autogpt.yaml")), specs[0] if specs else "")
            self._send_json({"ok": True, "specs": specs, "default_spec": default})
            return

        if path == "/api/v2/sessions":
            params = parse_qs(parsed.query or "")
            limit = max(1, min(200, _safe_int((params.get("limit") or ["80"])[0], 80)))
            session_root = _session_dir(self.root_dir)
            rows: list[dict[str, Any]] = []
            if session_root.exists() and session_root.is_dir():
                for session_file in sorted(session_root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if len(rows) >= limit:
                        break
                    try:
                        payload = json.loads(session_file.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    rows.append(
                        {
                            "session_id": str(payload.get("session_id") or session_file.stem),
                            "created_at": str(payload.get("created_at") or ""),
                            "updated_at": str(payload.get("updated_at") or ""),
                            "turn_count": len(payload.get("turns") or []),
                            "run_count": len(payload.get("run_ids") or []),
                        }
                    )
            self._send_json({"ok": True, "count": len(rows), "sessions": rows})
            return

        if path == "/api/v2/runs":
            params = parse_qs(parsed.query or "")
            limit = max(1, min(300, _safe_int((params.get("limit") or ["80"])[0], 80)))
            session_filter = str((params.get("session_id") or [""])[0]).strip()
            refs = _iter_run_dirs(self.root_dir)[:limit]
            rows = [_run_summary(self.root_dir, ref) for ref in refs]
            if session_filter:
                rows = [row for row in rows if str(row.get("session_id") or "") == session_filter]
            self._send_json({"ok": True, "count": len(rows), "runs": rows})
            return

        if path.startswith("/api/v2/events/"):
            run_id = path.rsplit("/", 1)[-1]
            if not _is_valid_run_id(run_id):
                self._send_json({"ok": False, "error": f"Invalid run id: {run_id}"}, status=400)
                return
            params = parse_qs(parsed.query or "")
            since = _safe_int((params.get("since_event_index") or params.get("since") or ["-1"])[0], -1)
            limit = max(1, min(MAX_EVENT_LIMIT, _safe_int((params.get("limit") or ["5000"])[0], 5000)))
            events, source = _load_events_for_run(self.root_dir, run_id)
            if not events:
                self._send_json({"ok": False, "error": f"Unknown run: {run_id}"}, status=404)
                return
            window = _filter_events_window(events, since_event_index=since, limit=limit)
            latest = max((_event_index_of(e) for e in events), default=0)
            self._send_json(
                {
                    "ok": True,
                    "run_id": run_id,
                    "source": source,
                    "count": len(window),
                    "latest_event_index": latest,
                    "events": window,
                }
            )
            return

        if path.startswith("/api/v2/files/"):
            run_id = path.rsplit("/", 1)[-1]
            if not _is_valid_run_id(run_id):
                self._send_json({"ok": False, "error": f"Invalid run id: {run_id}"}, status=400)
                return
            status, payload = _manifest_for_run(self.root_dir, run_id)
            self._send_json(payload, status=status)
            return

        if path.startswith("/api/v2/file/"):
            run_id = path.rsplit("/", 1)[-1]
            if not _is_valid_run_id(run_id):
                self._send_json({"ok": False, "error": f"Invalid run id: {run_id}"}, status=400)
                return
            params = parse_qs(parsed.query or "")
            rel_path = str((params.get("path") or [""])[0]).strip()
            if not rel_path:
                self._send_json({"ok": False, "error": "Missing required query: path"}, status=400)
                return
            source = str((params.get("source") or ["snapshot"])[0]).strip().lower()
            if source not in {"snapshot", "workspace"}:
                source = "snapshot"
            at_event_index = _safe_int((params.get("at_event_index") or [""])[0], -1)
            start = _safe_int((params.get("start") or ["0"])[0], 0)
            limit = _safe_int((params.get("limit") or [str(MAX_FILE_BYTES)])[0], MAX_FILE_BYTES)
            status, payload = _read_file_for_run(
                self.root_dir,
                run_id,
                rel_path,
                source,
                at_event_index if at_event_index >= 0 else None,
                start,
                limit,
            )
            payload = dict(payload)
            payload["ok"] = status == 200
            self._send_json(payload, status=status)
            return

        if path.startswith("/api/v2/status/"):
            run_id = path.rsplit("/", 1)[-1]
            session = _session_for_run(run_id)
            if session is None:
                events, _ = _load_events_for_run(self.root_dir, run_id)
                if not events:
                    self._send_json({"ok": False, "error": f"Unknown run: {run_id}"}, status=404)
                    return
                latest = max((_event_index_of(e) for e in events), default=0)
                self._send_json({"ok": True, "run_id": run_id, "done": True, "events": len(events), "latest_event_index": latest})
                return
            self._send_json({"ok": True, **session.snapshot()})
            return

        if path.startswith("/api/v2/session/"):
            session_id = path.rsplit("/", 1)[-1]
            if not _is_valid_session_id(session_id):
                self._send_json({"ok": False, "error": f"Invalid session id: {session_id}"}, status=400)
                return
            session = _load_session(self.root_dir, session_id)
            if session is None:
                self._send_json({"ok": False, "error": f"Unknown session: {session_id}"}, status=404)
                return
            runs = session.get("run_ids")
            if not isinstance(runs, list):
                runs = []
            run_set = {str(r) for r in runs}
            refs = _iter_run_dirs(self.root_dir)
            run_rows = [_run_summary(self.root_dir, ref) for ref in refs if ref.run_id in run_set]
            out = dict(session)
            out["ok"] = True
            out["runs"] = run_rows
            out["turn_count"] = len(out.get("turns") or [])
            self._send_json(out)
            return

        if path.startswith("/api/v2/stream/"):
            run_id = path.rsplit("/", 1)[-1]
            session = _session_for_run(run_id)
            if session is None:
                self._send_json({"ok": False, "error": f"Unknown active run: {run_id}"}, status=404)
                return
            self._stream_events(session, parsed.query)
            return

        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/v2/session/new":
            session = _new_session(self.root_dir)
            self._send_json({"ok": True, **session})
            return

        if path.startswith("/api/v2/session/") and path.endswith("/turn"):
            session_id = path.split("/")[4] if len(path.split("/")) > 4 else ""
            if not _is_valid_session_id(session_id):
                self._send_json({"ok": False, "error": f"Invalid session id: {session_id}"}, status=400)
                return
            session = _load_session(self.root_dir, session_id)
            if session is None:
                self._send_json({"ok": False, "error": f"Unknown session: {session_id}"}, status=404)
                return
            try:
                payload = self._read_json_body()
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            role = str(payload.get("role") or "user").strip().lower()
            if role not in {"user", "assistant", "system"}:
                role = "user"
            text = str(payload.get("text") or "").strip()
            if not text:
                self._send_json({"ok": False, "error": "Missing turn text"}, status=400)
                return
            run_id = str(payload.get("run_id") or "").strip() or None
            event_index = payload.get("event_index")
            event_idx = int(event_index) if isinstance(event_index, int) else None
            turn = _append_session_turn(
                self.root_dir,
                session_id,
                role=role,
                text=text,
                run_id=run_id,
                event_index=event_idx,
            )
            self._send_json({"ok": True, "session_id": session_id, "turn": turn})
            return

        if path == "/api/v2/run":
            self._handle_start_run()
            return

        if path.startswith("/api/v2/stop/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_V2_LOCK:
                session = RUNS_V2.get(run_id)
            if not session:
                self._send_json({"ok": False, "error": f"Unknown run: {run_id}"}, status=404)
                return
            session.stop()
            self._send_json({"ok": True, "run_id": run_id})
            return

        if path.startswith("/api/v2/prune/"):
            run_id = path.rsplit("/", 1)[-1]
            if not _is_valid_run_id(run_id):
                self._send_json({"ok": False, "error": f"Invalid run id: {run_id}"}, status=400)
                return
            with RUNS_V2_LOCK:
                session = RUNS_V2.get(run_id)
            if session is None:
                self._send_json({"ok": False, "error": f"Unknown active run: {run_id}"}, status=404)
                return
            force = False
            try:
                payload = self._read_json_body()
            except ValueError:
                payload = {}
            if isinstance(payload, dict):
                force = bool(payload.get("force", False))
            stats = session.prune_snapshots(force=force)
            session.emit("snapshot_pruned_manual", facet="files", entity_id=f"run:{run_id}", reason="manual", force=force, **stats)
            self._send_json({"ok": True, "run_id": run_id, "force": force, **stats})
            return

        self._send_json({"ok": False, "error": f"Unknown endpoint: {path}"}, status=404)

    def _stream_events(self, session: V2RunSession, query: str) -> None:
        params = parse_qs(query or "")
        last_seq = _safe_int((params.get("since_event_index") or params.get("since") or ["-1"])[0], -1)
        last_header = self.headers.get("Last-Event-ID")
        if last_seq < 0 and last_header is not None:
            last_seq = _safe_int(last_header, -1)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            while True:
                events, done = session.get_events_after(last_seq, timeout=1.0)
                if not events:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    if done:
                        break
                    continue

                for event in events:
                    idx = _event_index_of(event, last_seq)
                    blob = json.dumps(event, default=str)
                    frame = f"id: {idx}\ndata: {blob}\n\n".encode("utf-8")
                    self.wfile.write(frame)
                    last_seq = idx

                self.wfile.flush()
                if done:
                    break
        except (BrokenPipeError, ConnectionResetError):
            return

    def _handle_start_run(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        spec_path_raw = str(payload.get("spec_path", "")).strip()
        if not spec_path_raw:
            self._send_json({"ok": False, "error": "Missing required field: spec_path"}, status=400)
            return

        try:
            spec_path = _resolve_spec_path(self.root_dir, spec_path_raw)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        if not spec_path.exists():
            self._send_json({"ok": False, "error": f"Spec not found: {spec_path}"}, status=404)
            return

        agent_path_raw = str(payload.get("agent_path", "")).strip()
        if agent_path_raw:
            try:
                agent_path = _resolve_under_root(self.root_dir, agent_path_raw)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
        else:
            agent_path = _derive_agent_path(self.root_dir, spec_path)

        try:
            _ensure_agent_generated(self.root_dir, spec_path, agent_path)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        spec_stem = spec_path.stem.lower()
        default_max_iter = 220 if spec_stem.startswith("autogpt") else 80
        max_iter = _safe_int(payload.get("max_iter", default_max_iter), default_max_iter)
        if max_iter < 1:
            max_iter = default_max_iter

        max_budget_usd = _safe_float(payload.get("max_budget_usd"), None)

        run_id = uuid.uuid4().hex[:12]
        session_id_raw = str(payload.get("session_id") or "").strip()
        if session_id_raw and not _is_valid_session_id(session_id_raw):
            self._send_json({"ok": False, "error": f"Invalid session_id: {session_id_raw}"}, status=400)
            return
        session = _ensure_session(self.root_dir, session_id_raw or None)
        session_id = str(session.get("session_id") or "")
        initial_data = _build_initial_data_for_run(spec_path, payload, root_dir=self.root_dir, run_id=run_id)
        initial_data["session_id"] = session_id

        requested_mode = str(payload.get("requested_mode", "default") or "default")
        mock_mode = str(payload.get("mock_mode", os.environ.get("AGENT_ONTOLOGY_MOCK_MODE", "live")) or "live").strip().lower()
        if mock_mode in {"off", "none", "disabled"}:
            mock_mode = "live"
        if mock_mode not in {"deterministic", "live", "llm_mock"}:
            self._send_json({"ok": False, "error": f"Invalid mock_mode: {mock_mode}"}, status=400)
            return

        capability_profile = str(
            payload.get("capability_profile", os.environ.get("AGENT_ONTOLOGY_CAPABILITY_PROFILE", "external_enabled"))
            or "external_enabled"
        ).strip().lower()
        if capability_profile in {"external", "unshackled", "full"}:
            capability_profile = "external_enabled"
        if capability_profile not in {"restricted", "external_enabled"}:
            self._send_json({"ok": False, "error": f"Invalid capability_profile: {capability_profile}"}, status=400)
            return

        task = str(payload.get("task", "agent_ontology_live_viewer_v2"))
        model_override = _resolve_model_override(payload.get("model_override", ""))

        session = V2RunSession(
            run_id=run_id,
            session_id=session_id,
            root_dir=self.root_dir,
            spec_path=spec_path,
            agent_path=agent_path,
            initial_data=initial_data,
            task=task,
            requested_mode=requested_mode,
            mock_mode=mock_mode,
            capability_profile=capability_profile,
            max_iter=max_iter,
            model_override=model_override,
            max_budget_usd=max_budget_usd,
        )

        with RUNS_V2_LOCK:
            RUNS_V2[run_id] = session

        _attach_run_to_session(self.root_dir, session_id, run_id)
        user_text = str(payload.get("text") or payload.get("objective") or payload.get("user") or "").strip()
        if user_text:
            _append_session_turn(self.root_dir, session_id, role="user", text=user_text, run_id=run_id, event_index=None)

        try:
            session.start()
        except Exception as exc:
            session.error = f"{type(exc).__name__}: {exc}"
            session.done = True
            session.emit("error", facet="error", entity_id=f"run:{run_id}", message=session.error)
            self._send_json({"ok": False, "error": session.error}, status=500)
            return

        self._send_json(
            {
                "ok": True,
                "run_id": run_id,
                "stream_url": f"/api/v2/stream/{run_id}",
                "status_url": f"/api/v2/status/{run_id}",
                "stop_url": f"/api/v2/stop/{run_id}",
                "spec_path": str(spec_path),
                "agent_path": str(agent_path),
                "session_id": session_id,
                "mock_mode": mock_mode,
                "capability_profile": capability_profile,
                "effective_model": model_override,
            }
        )


def _cleanup_finished_sessions() -> None:
    with RUNS_V2_LOCK:
        sessions = list(RUNS_V2.values())
    now = datetime.now(timezone.utc)
    for session in sessions:
        if not session.done:
            continue
        age = (now - session.started_at).total_seconds()
        if age <= 1200:
            continue
        with RUNS_V2_LOCK:
            RUNS_V2.pop(session.run_id, None)


def run_server(*, host: str, port: int, root: Path) -> None:
    root = root.resolve()
    handler_cls = type("LiveViewerV2BoundHandler", (LiveViewerV2Handler,), {"root_dir": root})
    server = ThreadingHTTPServer((host, port), handler_cls)

    stop_event = threading.Event()

    def janitor() -> None:
        while not stop_event.wait(30.0):
            _cleanup_finished_sessions()

    janitor_thread = threading.Thread(target=janitor, name="live-v2-janitor", daemon=True)
    janitor_thread.start()

    print(f"[live-v2] Serving on http://{host}:{port}")
    print(f"[live-v2] Root: {root}")
    print(f"[live-v2] Viewer: http://{host}:{port}/agent_ontology/v2/spec-viewer-v2.html")

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run clean-room v2 live viewer server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--root", default=str(_find_default_root()))
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"error: root path does not exist or is not a directory: {root}")
        return 2

    run_server(host=args.host, port=args.port, root=root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
