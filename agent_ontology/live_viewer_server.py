#!/usr/bin/env python3
"""Live server for spec-viewer with real-time agent execution streaming."""

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
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ITERATION_RE = re.compile(r"^\[Iteration\s+(?P<iter>\d+)\]\s+State:\s+(?P<state>[A-Za-z0-9_]+)")
LLM_CALL_RE = re.compile(r"^\s*\[(?P<label>.+?)\]\s+\((?P<model>[^,]+),\s*(?P<ms>\d+)ms\)")
TRACE_WRITTEN_RE = re.compile(
    r"^Trace written to (?P<path>.+?) \((?P<calls>\d+) calls, (?P<dur>\d+)ms total\)"
)
FALLBACK_MODEL_OVERRIDE = "openrouter/deepseek/deepseek-chat"
EVENT_SCHEMA_VERSION = "1.0"
FILE_SNAPSHOT_MAX_BYTES = int(os.environ.get("AGENT_ONTOLOGY_FILE_SNAPSHOT_MAX_BYTES", "1000000") or "1000000")
FILE_SLICE_DEFAULT_CHARS = int(os.environ.get("AGENT_ONTOLOGY_FILE_SLICE_DEFAULT_CHARS", "120000") or "120000")
FILE_SLICE_MAX_CHARS = int(os.environ.get("AGENT_ONTOLOGY_FILE_SLICE_MAX_CHARS", "300000") or "300000")
EVENT_QUERY_DEFAULT_LIMIT = int(os.environ.get("AGENT_ONTOLOGY_EVENTS_QUERY_DEFAULT_LIMIT", "5000") or "5000")
EVENT_QUERY_MAX_LIMIT = int(os.environ.get("AGENT_ONTOLOGY_EVENTS_QUERY_MAX_LIMIT", "20000") or "20000")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

RUNS: dict[str, "RunSession"] = {}
RUNS_LOCK = threading.Lock()


def _default_model_override() -> str:
    configured = str(os.environ.get("AGENT_ONTOLOGY_DEFAULT_MODEL", FALLBACK_MODEL_OVERRIDE) or "").strip()
    return configured or FALLBACK_MODEL_OVERRIDE


def _resolve_model_override(raw_value: object) -> str:
    raw = str(raw_value or "").strip()
    return raw or _default_model_override()


def _parse_cursor_value(raw: object) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return max(value, 0)


def _parse_slice_limit(raw: object) -> int:
    text = str(raw or "").strip()
    if not text:
        return max(1, min(FILE_SLICE_DEFAULT_CHARS, FILE_SLICE_MAX_CHARS))
    try:
        value = int(text)
    except ValueError:
        return max(1, min(FILE_SLICE_DEFAULT_CHARS, FILE_SLICE_MAX_CHARS))
    return max(1, min(value, FILE_SLICE_MAX_CHARS))


def _slice_text_content(content: str, start: int, limit: int) -> tuple[str, int, bool, int | None]:
    total = len(content)
    begin = max(0, int(start))
    end = min(total, begin + max(1, int(limit)))
    chunk = content[begin:end]
    truncated = end < total
    next_start = end if truncated else None
    return chunk, total, truncated, next_start


def _parse_event_limit(raw: object) -> int:
    text = str(raw or "").strip()
    if not text:
        return max(1, min(EVENT_QUERY_DEFAULT_LIMIT, EVENT_QUERY_MAX_LIMIT))
    try:
        value = int(text)
    except ValueError:
        return max(1, min(EVENT_QUERY_DEFAULT_LIMIT, EVENT_QUERY_MAX_LIMIT))
    return max(1, min(value, EVENT_QUERY_MAX_LIMIT))


def _resolve_stream_since(params: dict[str, list[str]], headers: object) -> int:
    for key in ("since_event_index", "since"):
        values = params.get(key)
        if not values:
            continue
        parsed = _parse_cursor_value(values[0])
        if parsed is not None:
            return parsed

    header_value = None
    if hasattr(headers, "get"):
        try:
            header_value = headers.get("Last-Event-ID")
        except Exception:
            header_value = None
    parsed_header = _parse_cursor_value(header_value)
    if parsed_header is not None:
        return parsed_header
    return 0


def _is_path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


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
    if _has_workspace_layout(root) or _has_repo_layout(root):
        return (root / "agent_ontology").resolve()
    return root


def _default_viewer_path(root_dir: Path) -> str:
    if _has_workspace_layout(root_dir):
        return "/agent_ontology/spec-viewer.html"
    return "/spec-viewer.html"


def _find_default_root() -> Path:
    """Find workspace/repo root containing spec-viewer.html."""
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


def _derive_agent_path(root_dir: Path, spec_path: Path) -> Path:
    stem = spec_path.stem.replace("-", "_")
    repo = _repo_dir(root_dir)
    return (repo / "agents" / f"{stem}_agent.py").resolve()


def _resolve_spec_path(root_dir: Path, raw_path: str) -> Path:
    """Resolve spec path from either workspace-root or viewer-relative forms."""
    direct = _resolve_under_root(root_dir, raw_path)
    if direct.exists():
        return direct

    # spec-viewer.html lives at /agent_ontology/spec-viewer.html and references
    # "agent_ontology/specs/*.yaml", which resolves to /agent_ontology/agent_ontology/specs/*
    viewer_relative = (root_dir / "agent_ontology" / raw_path).resolve()
    root = root_dir.resolve()
    if viewer_relative != root and root in viewer_relative.parents and viewer_relative.exists():
        return viewer_relative

    return direct


def _review_dir(root_dir: Path) -> Path:
    return (_repo_dir(root_dir) / ".live_runs").resolve()


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _is_e2e_review_payload(data: dict[str, Any]) -> bool:
    return isinstance(data, dict) and isinstance(data.get("run_id"), str) and (
        "deterministic_ok" in data or "review_ok" in data or "event_summary" in data
    )


def _preview_text(value: Any, limit: int = 320) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _review_panel_summary(data: dict[str, Any]) -> dict[str, Any]:
    status = data.get("status", {})
    trace_metrics = data.get("trace_metrics", {})
    event_summary = data.get("event_summary", {})
    judge = data.get("judge", {})
    judge_report = judge.get("report", {}) if isinstance(judge, dict) else {}
    review_profile = data.get("review_profile", {})
    workspace_files = data.get("workspace_files", [])
    if not isinstance(status, dict):
        status = {}
    if not isinstance(trace_metrics, dict):
        trace_metrics = {}
    if not isinstance(event_summary, dict):
        event_summary = {}
    if not isinstance(judge_report, dict):
        judge_report = {}
    if not isinstance(review_profile, dict):
        review_profile = {}
    if not isinstance(workspace_files, list):
        workspace_files = []
    return {
        "ok": bool(data.get("ok")),
        "deterministic_ok": bool(data.get("deterministic_ok")),
        "review_ok": bool(data.get("review_ok")),
        "run_id": str(data.get("run_id", "")),
        "spec_path": str((data.get("run_request") or {}).get("spec_path", "")),
        "objective": str((data.get("run_request") or {}).get("text", "")),
        "effective_model": str(status.get("effective_model", "")),
        "returncode": status.get("returncode"),
        "timed_out": bool(data.get("timed_out")),
        "effective_min_overall_score": data.get("effective_min_overall_score"),
        "review_profile": review_profile,
        "trace_metrics": trace_metrics,
        "event_counts": event_summary.get("event_counts", {}),
        "judge_report": judge_report,
        "workspace_files": [
            {
                "path": str((row or {}).get("path", "")),
                "size": int((row or {}).get("size", 0) or 0),
                "preview": _preview_text((row or {}).get("preview", ""), 320),
            }
            for row in workspace_files
            if isinstance(row, dict)
        ],
    }


def _review_artifact_summary(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    mtime = 0.0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        pass
    run_request = data.get("run_request", {}) if isinstance(data.get("run_request"), dict) else {}
    judge_report = (
        (data.get("judge") or {}).get("report", {})
        if isinstance(data.get("judge"), dict)
        else {}
    )
    if not isinstance(judge_report, dict):
        judge_report = {}
    return {
        "run_id": str(data.get("run_id", "")),
        "file": str(path),
        "file_name": path.name,
        "mtime": mtime,
        "ok": bool(data.get("ok")),
        "deterministic_ok": bool(data.get("deterministic_ok")),
        "review_ok": bool(data.get("review_ok")),
        "spec_path": str(run_request.get("spec_path", "")),
        "objective_preview": _preview_text(run_request.get("text", ""), 140),
        "judge_verdict": str(judge_report.get("verdict", "")),
        "overall_score_normalized": judge_report.get("overall_score_normalized"),
    }


def _list_review_artifacts(root_dir: Path, limit: int = 30) -> list[dict[str, Any]]:
    review_dir = _review_dir(root_dir)
    if not review_dir.exists() or not review_dir.is_dir():
        return []
    files = sorted(review_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []
    for path in files:
        payload = _read_json_file(path)
        if payload is None or not _is_e2e_review_payload(payload):
            continue
        items.append(_review_artifact_summary(path, payload))
        if len(items) >= max(1, int(limit)):
            break
    return items


def _load_review_artifact(root_dir: Path, run_id: str | None = None) -> tuple[Path, dict[str, Any]] | None:
    review_dir = _review_dir(root_dir)
    if not review_dir.exists() or not review_dir.is_dir():
        return None
    files = sorted(review_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_id is None:
        for path in files:
            payload = _read_json_file(path)
            if payload is not None and _is_e2e_review_payload(payload):
                return path, payload
        return None
    want = str(run_id).strip()
    for path in files:
        payload = _read_json_file(path)
        if payload is None or not _is_e2e_review_payload(payload):
            continue
        payload_run_id = str(payload.get("run_id", ""))
        if payload_run_id == want:
            return path, payload
    return None


def _is_valid_run_id(run_id: str) -> bool:
    rid = str(run_id or "").strip()
    return bool(rid and RUN_ID_RE.match(rid))


def _run_artifact_dir(root_dir: Path, run_id: str) -> Path:
    base = _review_dir(root_dir)
    candidate = (base / run_id).resolve()
    if candidate == base or base not in candidate.parents:
        raise ValueError(f"Invalid run id path: {run_id}")
    return candidate


def _event_index_of(event: dict[str, Any], fallback: int = 0) -> int:
    try:
        return int(event.get("event_index", event.get("seq", fallback)) or fallback)
    except (TypeError, ValueError):
        return fallback


def _normalize_loaded_event(run_id: str, raw: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    event = dict(raw)
    event_index = _event_index_of(event, fallback=fallback_index)
    event["schema_version"] = str(event.get("schema_version") or EVENT_SCHEMA_VERSION)
    event["run_id"] = str(event.get("run_id") or run_id)
    event["event_index"] = event_index
    event["seq"] = event_index
    event["event_id"] = str(event.get("event_id") or f"{run_id}:{event_index}")
    event["ts"] = str(event.get("ts") or datetime.now(timezone.utc).isoformat())
    event["type"] = str(event.get("type") or "event")
    event["facet"] = str(event.get("facet") or "control")
    if "entity_id" not in event:
        event["entity_id"] = None
    event["actor"] = str(event.get("actor") or "replay")
    event["span_id"] = str(event.get("span_id") or f"span.replay.{run_id}.{event_index}")
    if "parent_span_id" not in event:
        event["parent_span_id"] = f"span.run.{run_id}"
    return event


def _load_events_from_ledger(ledger_path: Path, run_id: str) -> list[dict[str, Any]]:
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    fallback_index = 0
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        fallback_index += 1
        out.append(_normalize_loaded_event(run_id, parsed, fallback_index))
    out.sort(key=lambda item: _event_index_of(item))
    return out


def _events_from_trace_json(trace_path: Path, run_id: str) -> list[dict[str, Any]]:
    payload = _read_json_file(trace_path)
    if not isinstance(payload, dict):
        return []
    calls = payload.get("trace")
    if not isinstance(calls, list):
        calls = []
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    events: list[dict[str, Any]] = []
    seq = 0

    def _append(event_type: str, facet: str, body: dict[str, Any], ts: str | None = None) -> None:
        nonlocal seq
        seq += 1
        events.append(
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "run_id": run_id,
                "event_id": f"{run_id}:{seq}",
                "event_index": seq,
                "seq": seq,
                "ts": ts or datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "facet": facet,
                "entity_id": f"run:{run_id}" if event_type in {"started", "completed"} else None,
                "actor": "replay",
                "span_id": f"span.replay.{run_id}.{seq}",
                "parent_span_id": f"span.run.{run_id}",
                **body,
            }
        )

    _append(
        "started",
        "input",
        {
            "run_id": run_id,
            "trace_path": str(trace_path),
            "source": "trace_fallback",
        },
    )
    for call in calls:
        if not isinstance(call, dict):
            continue
        _append(
            "llm_call",
            "llm",
            {
                "process": str(call.get("process") or "replay"),
                "agent": str(call.get("agent") or "agent"),
                "model": str(call.get("model") or ""),
                "duration_ms": int(call.get("duration_ms", 0) or 0),
                "source": "trace_fallback",
                "trace_call": call,
            },
            ts=str(call.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        )
    _append(
        "completed",
        "done",
        {
            "run_id": run_id,
            "returncode": 0 if bool(metrics.get("clean_exit", True)) else 1,
            "error": None if bool(metrics.get("clean_exit", True)) else "trace indicates non-clean exit",
            "trace_path": str(trace_path),
            "process_metrics": {},
            "trace_metrics": metrics,
            "source": "trace_fallback",
        },
    )
    return events


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


def _ensure_agent_generated(root_dir: Path, spec_path: Path, agent_path: Path) -> None:
    repo = _repo_dir(root_dir)
    instantiate_script = repo / "agent_ontology" / "instantiate.py"
    if agent_path.exists():
        try:
            agent_mtime = agent_path.stat().st_mtime
            if (
                agent_mtime >= spec_path.stat().st_mtime
                and agent_mtime >= instantiate_script.stat().st_mtime
            ):
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


def _build_runner_code(agent_path: Path, initial_data: dict[str, object]) -> str:
    initial_json = json.dumps(initial_data)
    escaped_initial_json = json.dumps(initial_json)
    return (
        "import difflib as _difflib, hashlib as _hashlib, importlib, importlib.util, json, os, pathlib as _pathlib\n"
        f"agent_path = {json.dumps(str(agent_path))}\n"
        "def _runtime_info():\n"
        "    info = {}\n"
        "    try:\n"
        "        llm_mod = importlib.import_module('llm_client')\n"
        "        info['llm_client_module'] = getattr(llm_mod, '__file__', None)\n"
        "        info['llm_client_version_attr'] = getattr(llm_mod, '__version__', None)\n"
        "        info['has_call_llm'] = callable(getattr(llm_mod, 'call_llm', None))\n"
        "        info['has_call_llm_structured'] = callable(getattr(llm_mod, 'call_llm_structured', None))\n"
        "    except Exception as exc:\n"
        "        info['llm_client_import_error'] = f'{type(exc).__name__}: {exc}'\n"
        "    try:\n"
        "        import importlib.metadata as _md\n"
        "        info['llm_client_version'] = _md.version('llm_client')\n"
        "    except Exception:\n"
        "        pass\n"
        "    return info\n"
        "\n"
        "_runtime = _runtime_info()\n"
        "print('__AO_RUNTIME_INFO__' + json.dumps(_runtime, default=str), flush=True)\n"
        "if not _runtime.get('has_call_llm'):\n"
        "    raise RuntimeError('llm_client.call_llm unavailable; refusing non-llm_client execution path')\n"
        "\n"
        "spec = importlib.util.spec_from_file_location('ao_live_agent', agent_path)\n"
        "if spec is None or spec.loader is None:\n"
        "    raise RuntimeError(f'Unable to load agent module: {agent_path}')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "\n"
        "def _preview(value, limit=500):\n"
        "    try:\n"
        "        text = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)\n"
        "    except Exception:\n"
        "        text = repr(value)\n"
        "    if len(text) > limit:\n"
        "        return text[:limit] + '...'\n"
        "    return text\n"
        "\n"
        "def _emit_step_diff(process_id, before_data, after_data):\n"
        "    before_keys = set(before_data.keys())\n"
        "    after_keys = set(after_data.keys())\n"
        "    added = sorted(after_keys - before_keys)\n"
        "    removed = sorted(before_keys - after_keys)\n"
        "    changed = []\n"
        "    for key in sorted(before_keys & after_keys):\n"
        "        if _preview(before_data.get(key), limit=2000) != _preview(after_data.get(key), limit=2000):\n"
        "            changed.append(key)\n"
        "    value_preview = {}\n"
        "    for key in (added + changed)[:10]:\n"
        "        value_preview[key] = _preview(after_data.get(key), limit=500)\n"
        "    payload = {\n"
        "        'process': process_id,\n"
        "        'added': added,\n"
        "        'removed': removed,\n"
        "        'changed': changed,\n"
        "        'value_preview': value_preview,\n"
        "    }\n"
        "    print('__AO_STEP_DIFF__' + json.dumps(payload, default=str, ensure_ascii=False), flush=True)\n"
        "\n"
        "def _is_store_like(name, obj):\n"
        "    if name in {'data', 'iteration', 'schema_violations'}:\n"
        "        return False\n"
        "    if name.startswith('_'):\n"
        "        return False\n"
        "    return callable(getattr(obj, 'read', None)) and callable(getattr(obj, 'write', None))\n"
        "\n"
        "def _snapshot_stores(state):\n"
        "    snap = {}\n"
        "    for _name, _obj in getattr(state, '__dict__', {}).items():\n"
        "        if not _is_store_like(_name, _obj):\n"
        "            continue\n"
        "        try:\n"
        "            snap[_name] = _obj.read()\n"
        "        except Exception as exc:\n"
        "            snap[_name] = {'__error__': f'{type(exc).__name__}: {exc}'}\n"
        "    return snap\n"
        "\n"
        "def _emit_store_diff(process_id, before_store, after_store):\n"
        "    before_keys = set(before_store.keys())\n"
        "    after_keys = set(after_store.keys())\n"
        "    added = sorted(after_keys - before_keys)\n"
        "    removed = sorted(before_keys - after_keys)\n"
        "    changed = []\n"
        "    for key in sorted(before_keys & after_keys):\n"
        "        if _preview(before_store.get(key), limit=2000) != _preview(after_store.get(key), limit=2000):\n"
        "            changed.append(key)\n"
        "    value_preview = {}\n"
        "    for key in (added + changed)[:20]:\n"
        "        value_preview[key] = {\n"
        "            'before': _preview(before_store.get(key), limit=700),\n"
        "            'after': _preview(after_store.get(key), limit=700),\n"
        "        }\n"
        "    payload = {\n"
        "        'process': process_id,\n"
        "        'added_stores': added,\n"
        "        'removed_stores': removed,\n"
        "        'changed_stores': changed,\n"
        "        'value_preview': value_preview,\n"
        "    }\n"
        "    print('__AO_STORE_DIFF__' + json.dumps(payload, default=str, ensure_ascii=False), flush=True)\n"
        "\n"
        "class _FallbackChannel:\n"
        "    def __init__(self):\n"
        "        self.messages = []\n"
        "    def read(self, key=None):\n"
        "        if not self.messages:\n"
        "            return None\n"
        "        return self.messages[-1]\n"
        "    def write(self, value, key=None):\n"
        "        self.messages = [value]\n"
        "\n"
        "def _ensure_runtime_channels(state):\n"
        "    for _name in ('final_channel', 'commentary_channel', 'stream_channel', 'rpc_channel'):\n"
        "        if not hasattr(state, _name):\n"
        "            setattr(state, _name, _FallbackChannel())\n"
        "\n"
        "_FILE_DIFF_MAX_TRACKED = int(os.environ.get('AGENT_ONTOLOGY_FILE_MAX_TRACKED', '400') or '400')\n"
        "_FILE_DIFF_MAX_BYTES = int(os.environ.get('AGENT_ONTOLOGY_FILE_MAX_BYTES', '120000') or '120000')\n"
        "_FILE_DIFF_MAX_PATCH = int(os.environ.get('AGENT_ONTOLOGY_FILE_MAX_PATCH', '14000') or '14000')\n"
        "_FILE_DIFF_IGN_DIRS = {\n"
        "    '.git', 'node_modules', '.venv', 'venv', '__pycache__', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.live_runs'\n"
        "}\n"
        "\n"
        "def _state_data(state):\n"
        "    if isinstance(state, dict):\n"
        "        return state\n"
        "    return getattr(state, 'data', {})\n"
        "\n"
        "def _is_text_blob(blob):\n"
        "    if not blob:\n"
        "        return True\n"
        "    if b'\\x00' in blob[:4096]:\n"
        "        return False\n"
        "    return True\n"
        "\n"
        "def _candidate_file_roots(state):\n"
        "    roots = []\n"
        "    data = _state_data(state)\n"
        "    for _k in ('workspace_dir', 'workspace_path', 'output_dir'):\n"
        "        _v = data.get(_k) if isinstance(data, dict) else None\n"
        "        if isinstance(_v, str) and _v.strip():\n"
        "            _p = _pathlib.Path(_v).expanduser()\n"
        "            if not _p.is_absolute():\n"
        "                _p = (_pathlib.Path.cwd() / _p).resolve()\n"
        "            if _p.exists() and _p.is_dir():\n"
        "                roots.append(str(_p))\n"
        "    for _name, _obj in getattr(state, '__dict__', {}).items():\n"
        "        _path_attr = getattr(_obj, 'path', None)\n"
        "        if isinstance(_path_attr, str) and _path_attr.strip():\n"
        "            _p = _pathlib.Path(_path_attr).expanduser()\n"
        "            if not _p.is_absolute():\n"
        "                _p = (_pathlib.Path.cwd() / _p).resolve()\n"
        "            if _p.exists():\n"
        "                if _p.is_dir():\n"
        "                    roots.append(str(_p))\n"
        "                elif _p.is_file():\n"
        "                    roots.append(str(_p.parent))\n"
        "    out = []\n"
        "    seen = set()\n"
        "    for _r in roots:\n"
        "        if _r in seen:\n"
        "            continue\n"
        "        seen.add(_r)\n"
        "        out.append(_r)\n"
        "    if not out:\n"
        "        out.append(str(_pathlib.Path.cwd().resolve()))\n"
        "    return out\n"
        "\n"
        "def _snapshot_files(state):\n"
        "    snap = {}\n"
        "    roots = _candidate_file_roots(state)\n"
        "    if not roots:\n"
        "        return snap\n"
        "    for _root_idx, _root in enumerate(roots):\n"
        "        _rp = _pathlib.Path(_root)\n"
        "        _label = _rp.name or f'root{_root_idx}'\n"
        "        if not _rp.exists() or not _rp.is_dir():\n"
        "            continue\n"
        "        for _p in sorted(_rp.rglob('*')):\n"
        "            if len(snap) >= _FILE_DIFF_MAX_TRACKED:\n"
        "                return snap\n"
        "            if _p.is_dir():\n"
        "                continue\n"
        "            _rel_parts = _p.relative_to(_rp).parts\n"
        "            if any(_part in _FILE_DIFF_IGN_DIRS for _part in _rel_parts):\n"
        "                continue\n"
        "            try:\n"
        "                _size = int(_p.stat().st_size)\n"
        "            except Exception:\n"
        "                continue\n"
        "            if _size > (_FILE_DIFF_MAX_BYTES * 4):\n"
        "                continue\n"
        "            try:\n"
        "                _raw = _p.read_bytes()\n"
        "            except Exception:\n"
        "                continue\n"
        "            _digest = _hashlib.sha1(_raw).hexdigest()\n"
        "            _is_text = _is_text_blob(_raw)\n"
        "            _text = None\n"
        "            if _is_text and _size <= _FILE_DIFF_MAX_BYTES:\n"
        "                try:\n"
        "                    _text = _raw.decode('utf-8')\n"
        "                except Exception:\n"
        "                    try:\n"
        "                        _text = _raw.decode('utf-8', errors='replace')\n"
        "                    except Exception:\n"
        "                        _text = None\n"
        "            _rel = _p.relative_to(_rp).as_posix()\n"
        "            _key = f'{_label}/{_rel}'\n"
        "            snap[_key] = {\n"
        "                'root': str(_rp),\n"
        "                'path': str(_p),\n"
        "                'rel': _rel,\n"
        "                'size': _size,\n"
        "                'hash': _digest,\n"
        "                'is_text': bool(_text is not None),\n"
        "                'text': _text,\n"
        "            }\n"
        "    return snap\n"
        "\n"
        "def _emit_file_diff(process_id, before_files, after_files):\n"
        "    before_keys = set(before_files.keys())\n"
        "    after_keys = set(after_files.keys())\n"
        "    added = sorted(after_keys - before_keys)\n"
        "    removed = sorted(before_keys - after_keys)\n"
        "    changed = []\n"
        "    for _k in sorted(before_keys & after_keys):\n"
        "        if before_files.get(_k, {}).get('hash') != after_files.get(_k, {}).get('hash'):\n"
        "            changed.append(_k)\n"
        "    if not added and not removed and not changed:\n"
        "        return\n"
        "    patches = {}\n"
        "    previews = {}\n"
        "    _focus = (added + changed + removed)[:30]\n"
        "    for _k in _focus:\n"
        "        _b = before_files.get(_k)\n"
        "        _a = after_files.get(_k)\n"
        "        _status = 'changed'\n"
        "        if _b is None:\n"
        "            _status = 'added'\n"
        "        elif _a is None:\n"
        "            _status = 'removed'\n"
        "        previews[_k] = {\n"
        "            'status': _status,\n"
        "            'size_before': int(_b.get('size', 0)) if isinstance(_b, dict) else 0,\n"
        "            'size_after': int(_a.get('size', 0)) if isinstance(_a, dict) else 0,\n"
        "            'path': (_a or _b or {}).get('path', ''),\n"
        "        }\n"
        "        _bt = _b.get('text') if isinstance(_b, dict) else None\n"
        "        _at = _a.get('text') if isinstance(_a, dict) else None\n"
        "        if isinstance(_bt, str) or isinstance(_at, str):\n"
        "            _from = f\"a/{_k}\" if _b is not None else '/dev/null'\n"
        "            _to = f\"b/{_k}\" if _a is not None else '/dev/null'\n"
        "            _diff_lines = list(_difflib.unified_diff(\n"
        "                (_bt or '').splitlines(),\n"
        "                (_at or '').splitlines(),\n"
        "                fromfile=_from,\n"
        "                tofile=_to,\n"
        "                lineterm='',\n"
        "            ))\n"
        "            _patch = '\\n'.join(_diff_lines)\n"
        "            if len(_patch) > _FILE_DIFF_MAX_PATCH:\n"
        "                _patch = _patch[:_FILE_DIFF_MAX_PATCH] + '\\n...<truncated>'\n"
        "            patches[_k] = _patch or f'{_status}: {_k}'\n"
        "            if isinstance(_at, str) and _at:\n"
        "                previews[_k]['content_preview'] = _preview(_at, limit=1400)\n"
        "        else:\n"
        "            patches[_k] = f'{_status}: {_k} (binary/non-text)'\n"
        "    payload = {\n"
        "        'process': process_id,\n"
        "        'added_files': added,\n"
        "        'removed_files': removed,\n"
        "        'changed_files': changed,\n"
        "        'patches': patches,\n"
        "        'previews': previews,\n"
        "    }\n"
        "    print('__AO_FILE_DIFF__' + json.dumps(payload, default=str, ensure_ascii=False), flush=True)\n"
        "\n"
        "for _pid, _fn in list(getattr(module, 'PROCESSES', {}).items()):\n"
        "    def _make_wrapper(pid, fn):\n"
        "        def _wrapped(state):\n"
        "            _ensure_runtime_channels(state)\n"
        "            before_data = dict(getattr(state, 'data', {}))\n"
        "            before_store = _snapshot_stores(state)\n"
        "            before_files = _snapshot_files(state)\n"
        "            try:\n"
        "                result = fn(state)\n"
        "            except Exception as exc:\n"
        "                err_payload = {'process': pid, 'error': f'{type(exc).__name__}: {exc}', 'raised': True}\n"
        "                print('__AO_STEP_DIFF__' + json.dumps(err_payload, default=str, ensure_ascii=False), flush=True)\n"
        "                raise\n"
        "            after_data = dict(getattr(state, 'data', {}))\n"
        "            after_store = _snapshot_stores(state)\n"
        "            after_files = _snapshot_files(state)\n"
        "            _emit_step_diff(pid, before_data, after_data)\n"
        "            _emit_store_diff(pid, before_store, after_store)\n"
        "            _emit_file_diff(pid, before_files, after_files)\n"
        "            single_turn = os.environ.get('AGENT_ONTOLOGY_SINGLE_TURN', '1').lower() not in ('0', 'false', 'no')\n"
        "            pid_str = str(pid)\n"
        "            if single_turn and (\n"
        "                pid_str == 'emit_final'\n"
        "                or pid_str.startswith('emit_final')\n"
        "                or pid_str == 'persist'\n"
        "                or pid_str == 'persist_turn'\n"
        "                or pid_str.startswith('persist')\n"
        "            ):\n"
        "                try:\n"
        "                    state.data['_done'] = True\n"
        "                except Exception:\n"
        "                    pass\n"
        "            return result\n"
        "        return _wrapped\n"
        "    module.PROCESSES[_pid] = _make_wrapper(_pid, _fn)\n"
        "\n"
        "def _non_empty(value):\n"
        "    return value not in (None, '', [], {})\n"
        "\n"
        "def _read_channel_attr(state, name):\n"
        "    obj = getattr(state, name, None)\n"
        "    if obj is None:\n"
        "        return None\n"
        "    if callable(getattr(obj, 'read', None)):\n"
        "        try:\n"
        "            return obj.read()\n"
        "        except Exception as exc:\n"
        "            return {'__error__': f'{type(exc).__name__}: {exc}'}\n"
        "    return None\n"
        "\n"
        "def _infer_final_output(state):\n"
        "    if isinstance(state, dict):\n"
        "        data = state\n"
        "    else:\n"
        "        data = getattr(state, 'data', {})\n"
        "\n"
        "    direct = _read_channel_attr(state, 'final_channel')\n"
        "    if _non_empty(direct):\n"
        "        return {'source': 'state.final_channel', 'value': direct}\n"
        "\n"
        "    for _name, _obj in getattr(state, '__dict__', {}).items():\n"
        "        _lname = str(_name).lower()\n"
        "        if _name == 'final_channel':\n"
        "            continue\n"
        "        if not any(tok in _lname for tok in ('channel', 'out', 'reply', 'response')):\n"
        "            continue\n"
        "        if callable(getattr(_obj, 'read', None)):\n"
        "            try:\n"
        "                _val = _obj.read()\n"
        "            except Exception as exc:\n"
        "                _val = {'__error__': f'{type(exc).__name__}: {exc}'}\n"
        "            if _non_empty(_val):\n"
        "                return {'source': f'state.{_name}', 'value': _val}\n"
        "\n"
        "    if isinstance(data, dict):\n"
        "        for _k in (\n"
            "            'answer', 'final', 'final_output', 'draft_response', 'response',\n"
            "            'content', 'action_decision', 'api_response',\n"
            "            'llm_call_result', 'error', 'text'\n"
        "        ):\n"
        "            if _k in data and _non_empty(data.get(_k)):\n"
        "                return {'source': f'state.data.{_k}', 'value': data.get(_k)}\n"
        "        _nz = {k: v for k, v in data.items() if _non_empty(v)}\n"
        "        if _nz:\n"
        "            _preview_map = {}\n"
        "            for _k, _v in list(_nz.items())[:12]:\n"
        "                _preview_map[_k] = _preview(_v, limit=500)\n"
        "            return {'source': 'state.data', 'value_preview': _preview_map}\n"
        "\n"
        "    return {'source': 'none', 'value': None}\n"
        "\n"
        f"initial = json.loads({escaped_initial_json})\n"
        "state = module.run(initial)\n"
        "final_channel = _infer_final_output(state)\n"
        "print('__AO_FINAL_CHANNEL__' + json.dumps(final_channel, default=str), flush=True)\n"
        "final_data = state if isinstance(state, dict) else getattr(state, 'data', {})\n"
        "print('__AO_FINAL_DATA__' + json.dumps(final_data, default=str), flush=True)\n"
    )


def _build_initial_data_for_run(
    spec_path: Path,
    payload: dict[str, object],
    *,
    root_dir: Path | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    def _normalize_conversation(raw: object, *, limit: int = 80) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            role = "user"
            content = ""
            if isinstance(item, dict):
                role_raw = str(item.get("role", "") or "").strip().lower()
                if role_raw in {"user", "assistant", "system", "tool"}:
                    role = role_raw
                content = str(item.get("content", item.get("text", "")) or "").strip()
            else:
                content = str(item or "").strip()
            if not content:
                continue
            out.append({"role": role, "content": content})
            if len(out) >= max(1, int(limit)):
                break
        return out

    text_value = str(payload.get("text", ""))
    files_value = payload.get("files", "")
    constraints_value = payload.get("constraints", "")
    chat_session_id = str(payload.get("chat_session_id", "") or "").strip()
    current_user_message = str(payload.get("current_user_message", text_value) or text_value)
    conversation_history = _normalize_conversation(payload.get("conversation_history"))
    messages = _normalize_conversation(payload.get("messages"))
    if not messages:
        messages = list(conversation_history)

    initial_data: dict[str, object] = {
        "text": text_value,
        "files": files_value,
        "constraints": constraints_value,
        # Generic aliases so specs that key off these fields get useful input.
        "task": text_value,
        "user": text_value,
        "objective": text_value,
        "acceptance_criteria": [str(c) for c in constraints_value] if isinstance(constraints_value, list) else [],
        "requested_mode": str(payload.get("requested_mode", "default") or "default"),
        "current_user_message": current_user_message,
    }
    if chat_session_id:
        initial_data["chat_session_id"] = chat_session_id
    if conversation_history:
        initial_data["conversation_history"] = conversation_history
        initial_data["conversation_memory"] = list(conversation_history)
    if messages:
        initial_data["messages"] = messages

    # BabyAGI-style specs expect a pre-populated task queue; without it they stop
    # immediately with "No tasks remaining!" and never call the LLM.
    spec_stem = spec_path.stem.lower()
    if spec_stem.startswith("babyagi") and text_value.strip():
        max_tasks_raw = payload.get("max_tasks", 5)
        try:
            max_tasks = int(max_tasks_raw)
        except (TypeError, ValueError):
            max_tasks = 5
        if max_tasks < 1:
            max_tasks = 1
        initial_data.setdefault(
            "tasks",
            [{"id": 1, "description": text_value.strip(), "status": "pending", "priority": 1}],
        )
        initial_data.setdefault("completed_count", 0)
        initial_data.setdefault("max_tasks", max_tasks)

    workspace_dir_raw = str(payload.get("workspace_dir", "") or "").strip()
    if workspace_dir_raw:
        initial_data["workspace_dir"] = workspace_dir_raw
    elif root_dir is not None and run_id:
        # Keep runtime workspace outside .live_runs so file diffs/artifacts stay visible.
        workspace_root = (_repo_dir(root_dir) / "workspaces").resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        initial_data["workspace_dir"] = str((workspace_root / run_id).resolve())

    return initial_data


@dataclass
class RunSession:
    run_id: str
    root_dir: Path
    spec_path: Path
    agent_path: Path
    initial_data: dict[str, object]
    task: str
    requested_mode: str
    mock_mode: str
    capability_profile: str
    max_iter: int
    model_override: str
    max_budget_usd: float | None

    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    done: bool = False
    returncode: int | None = None
    error: str | None = None
    current_process: str | None = None
    trace_path: str | None = None
    final_channel: object = None
    final_data: object = None
    effective_model: str = ""

    process: subprocess.Popen[str] | None = None
    run_dir: Path | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    process_metrics: dict[str, dict[str, int]] = field(default_factory=dict)
    _next_seq: int = 1
    _cv: threading.Condition = field(default_factory=threading.Condition)
    _reader_thread: threading.Thread | None = None
    _events_log_path: Path | None = None
    _run_span_id: str = ""
    _active_transition_span_id: str | None = None
    _snapshot_dir: Path | None = None
    _file_history: dict[str, list[dict[str, object]]] = field(default_factory=dict)

    def _event_facet(self, event_type: str) -> str:
        if event_type in {"started", "runtime_info"}:
            return "input"
        if event_type in {"delegate_run", "delegate_trace_file"}:
            return "delegate"
        if event_type == "state":
            return "state"
        if event_type == "llm_call":
            return "llm"
        if event_type == "step_diff":
            return "state"
        if event_type == "store_diff":
            return "stores"
        if event_type == "file_diff":
            return "files"
        if event_type in {"final_channel", "final_data", "final_keys"}:
            return "output"
        if event_type in {"completed", "stopped"}:
            return "done"
        if event_type == "error":
            return "error"
        return "control"

    def _event_entity_id(self, event_type: str, payload: dict[str, object], seq: int) -> str | None:
        if event_type in {"started", "runtime_info", "completed", "stopped", "error"}:
            return f"run:{self.run_id}"
        if event_type == "state":
            process = str(payload.get("process") or "").strip()
            return f"process:{process}" if process else None
        if event_type == "llm_call":
            process = str(payload.get("process") or "").strip()
            agent = str(payload.get("agent") or "").strip()
            if process and agent:
                return f"llm:{process}:{agent}:{seq}"
            if process:
                return f"llm:{process}:{seq}"
            return f"llm:{seq}"
        if event_type == "step_diff":
            body = payload.get("payload")
            if isinstance(body, dict):
                process = str(body.get("process") or "").strip()
                if process:
                    return f"state:{process}"
            return "state:unknown"
        if event_type == "store_diff":
            body = payload.get("payload")
            if isinstance(body, dict):
                names: list[str] = []
                for key in ("changed_stores", "added_stores", "removed_stores"):
                    values = body.get(key)
                    if isinstance(values, list):
                        names.extend(str(v).strip() for v in values if str(v).strip())
                if names:
                    return f"store:{names[0]}"
                process = str(body.get("process") or "").strip()
                if process:
                    return f"store-process:{process}"
            return "store:unknown"
        if event_type == "file_diff":
            body = payload.get("payload")
            if isinstance(body, dict):
                names: list[str] = []
                for key in ("changed_files", "added_files", "removed_files"):
                    values = body.get(key)
                    if isinstance(values, list):
                        names.extend(str(v).strip() for v in values if str(v).strip())
                if names:
                    return f"file:{names[0]}"
                process = str(body.get("process") or "").strip()
                if process:
                    return f"file-process:{process}"
            return "file:unknown"
        if event_type == "trace_file":
            path = str(payload.get("path") or "").strip()
            return f"trace:{path}" if path else "trace:unknown"
        if event_type == "delegate_trace_file":
            body = payload.get("payload")
            if isinstance(body, dict):
                delegate_id = str(body.get("delegate_id") or "").strip()
                child_run_id = str(body.get("child_run_id") or "").strip()
                path = str(body.get("path") or "").strip()
                if delegate_id and child_run_id and path:
                    return f"delegate-trace:{delegate_id}:{child_run_id}:{path}"
                if delegate_id and child_run_id:
                    return f"delegate-trace:{delegate_id}:{child_run_id}"
                if delegate_id and path:
                    return f"delegate-trace:{delegate_id}:{path}"
                if delegate_id:
                    return f"delegate-trace:{delegate_id}"
            return "delegate-trace:unknown"
        if event_type == "delegate_run":
            body = payload.get("payload")
            if isinstance(body, dict):
                delegate_id = str(body.get("delegate_id") or "").strip()
                child_run_id = str(body.get("child_run_id") or "").strip()
                if not child_run_id:
                    child_run_ids = body.get("child_run_ids")
                    if isinstance(child_run_ids, list):
                        for raw in child_run_ids:
                            candidate = str(raw or "").strip()
                            if candidate:
                                child_run_id = candidate
                                break
                if delegate_id and child_run_id:
                    return f"delegate-run:{delegate_id}:{child_run_id}"
                if delegate_id:
                    return f"delegate-run:{delegate_id}"
            return "delegate-run:unknown"
        if event_type in {"final_channel", "final_data", "final_keys"}:
            return f"output:{self.run_id}"
        return None

    def _event_actor(self, event_type: str, payload: dict[str, object]) -> str:
        if event_type == "llm_call":
            agent = str(payload.get("agent") or "").strip()
            return f"agent:{agent}" if agent else "agent:unknown"
        if event_type in {"delegate_trace_file", "delegate_run"}:
            body = payload.get("payload")
            if isinstance(body, dict):
                delegate_id = str(body.get("delegate_id") or "").strip()
                if delegate_id:
                    return f"delegate:{delegate_id}"
            return "delegate:unknown"
        return "runtime"

    def _event_related_run_id(self, event_type: str, payload: dict[str, object]) -> str | None:
        if event_type not in {"delegate_trace_file", "delegate_run"}:
            return None
        body = payload.get("payload")
        if not isinstance(body, dict):
            return None
        child_run_id = str(body.get("child_run_id") or "").strip()
        if child_run_id:
            return child_run_id
        child_run_ids = body.get("child_run_ids")
        if isinstance(child_run_ids, list):
            for raw in child_run_ids:
                candidate = str(raw or "").strip()
                if candidate:
                    return candidate
        return None

    def _make_span_ids(self, event_type: str, seq: int) -> tuple[str | None, str | None]:
        if not self._run_span_id:
            self._run_span_id = f"span.run.{self.run_id}"

        if event_type == "started":
            self._active_transition_span_id = None
            return self._run_span_id, None
        if event_type == "state":
            span_id = f"span.state.{self.run_id}.{seq}"
            self._active_transition_span_id = span_id
            return span_id, self._run_span_id
        if event_type in {"completed", "stopped"}:
            return f"span.done.{self.run_id}.{seq}", self._run_span_id
        if event_type == "error":
            parent = self._active_transition_span_id or self._run_span_id
            return f"span.error.{self.run_id}.{seq}", parent

        parent = self._active_transition_span_id or self._run_span_id
        return f"span.{event_type}.{self.run_id}.{seq}", parent

    def _append_event_to_ledger(self, event: dict[str, object]) -> None:
        if self._events_log_path is None:
            return
        try:
            with self._events_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str, ensure_ascii=False))
                fh.write("\n")
        except OSError:
            # Best-effort persistence; live run should continue even if local disk write fails.
            return

    def _infer_workspace_root(self, path_key: str, abs_path: Path) -> Path:
        rel_part = path_key.split("/", 1)[1] if "/" in path_key else Path(path_key).name
        rel_parts = Path(rel_part).parts
        abs_parts = abs_path.parts
        if rel_parts and len(abs_parts) >= len(rel_parts):
            if tuple(abs_parts[-len(rel_parts) :]) == rel_parts:
                root_parts = abs_parts[: -len(rel_parts)]
                if root_parts:
                    return Path(*root_parts)
        return abs_path.parent

    def _workspace_root_label(self, root: Path) -> str:
        root_name = root.name or "workspace"
        if root_name == self.run_id:
            return "workspace"
        return root_name

    def _normalize_file_path_key(self, path_key: str) -> str:
        cleaned = str(path_key or "").strip().replace("\\", "/")
        if not cleaned:
            return cleaned

        if self.run_id and cleaned.startswith(f"{self.run_id}/"):
            cleaned = cleaned.split("/", 1)[1]

        try:
            path_obj = Path(cleaned).expanduser()
            if path_obj.is_absolute():
                for root in self._workspace_roots_unlocked():
                    if _is_path_within_root(path_obj, root):
                        try:
                            rel = path_obj.relative_to(root).as_posix()
                            return f"{self._workspace_root_label(root)}/{rel}" if rel else self._workspace_root_label(root)
                        except ValueError:
                            continue
        except OSError:
            return cleaned

        return cleaned

    def _find_file_history_key(self, requested_key: str) -> str | None:
        clean = str(requested_key or "").strip().replace("\\", "/")
        if not clean:
            return None
        normalized = self._normalize_file_path_key(clean)
        if normalized in self._file_history:
            return normalized
        if clean in self._file_history:
            return clean

        if self.run_id:
            run_prefixed = f"{self.run_id}/{clean}"
            if run_prefixed in self._file_history:
                return run_prefixed
            if normalized != clean:
                run_prefixed_normalized = f"{self.run_id}/{normalized}"
                if run_prefixed_normalized in self._file_history:
                    return run_prefixed_normalized

        candidates: list[str] = []
        suffix = f"/{clean}"
        for key in self._file_history.keys():
            if key == clean or key == normalized:
                return key
            if key.endswith(suffix):
                candidates.append(key)

        if not candidates:
            return None
        candidates.sort()
        return candidates[0]

    def _record_file_history(self, path_key: str, entry: dict[str, object]) -> None:
        with self._cv:
            path_key = self._normalize_file_path_key(path_key)
            history = self._file_history.setdefault(path_key, [])
            history.append(entry)

    def _latest_event_index_unlocked(self) -> int:
        if self.events:
            try:
                return max(0, int(self.events[-1].get("event_index", self.events[-1].get("seq", 0)) or 0))
            except (TypeError, ValueError, AttributeError):
                return max(0, int(self._next_seq) - 1)
        return max(0, int(self._next_seq) - 1)

    def _workspace_roots_unlocked(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[str] = set()

        def _add(raw: object) -> None:
            text = str(raw or "").strip()
            if not text:
                return
            try:
                path = Path(text).expanduser().resolve()
            except OSError:
                return
            if not path.exists() or not path.is_dir():
                return
            key = str(path)
            if key in seen:
                return
            seen.add(key)
            roots.append(path)

        for source in (self.initial_data, self.final_data):
            if not isinstance(source, dict):
                continue
            for key in ("workspace_dir", "workspace_path", "output_dir"):
                _add(source.get(key))

        if self.run_dir and self.run_dir.exists():
            try:
                for child in self.run_dir.iterdir():
                    if not child.is_dir():
                        continue
                    if child.name.endswith("_workspace"):
                        _add(child)
            except OSError:
                pass

        return roots

    def _is_within_workspace_root(self, path: Path) -> bool:
        roots = self._workspace_roots_unlocked()
        if not roots:
            return True
        for root in self._workspace_roots_unlocked():
            if _is_path_within_root(path, root):
                return True
        return False

    def _write_text_snapshot_unlocked(
        self,
        *,
        path_key: str,
        abs_path: Path,
        workspace_root: Path,
        event_index: int,
        size: int,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "snapshot_path": "",
            "content_available": False,
            "content_size": int(size),
            "is_text": False,
        }
        if self.run_dir is None:
            return result
        if size < 0 or size > FILE_SNAPSHOT_MAX_BYTES:
            return result

        try:
            raw = abs_path.read_bytes()
        except OSError:
            return result

        is_text = _is_text_blob(raw)
        result["is_text"] = bool(is_text)
        if not is_text:
            return result

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")

        if self._snapshot_dir is None:
            self._snapshot_dir = (self.run_dir / "file_snapshots").resolve()
        try:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return result

        snapshot_batch_dir = (self._snapshot_dir / f"{int(event_index):08d}").resolve()
        try:
            snapshot_batch_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return result

        snapshot_name = hashlib.sha1(f"fallback:{path_key}:{event_index}".encode("utf-8")).hexdigest() + ".json"
        snapshot_path = (snapshot_batch_dir / snapshot_name).resolve()
        snapshot_payload = {
            "path": path_key,
            "event_index": int(event_index),
            "absolute_path": str(abs_path),
            "workspace_root": str(workspace_root),
            "size": int(size),
            "is_text": True,
            "content": text,
            "fallback": True,
        }
        try:
            snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            return result

        try:
            rel = snapshot_path.relative_to(self.run_dir).as_posix()
        except ValueError:
            rel = ""

        if rel:
            result["snapshot_path"] = rel
            result["content_available"] = True
            result["content_size"] = len(text.encode("utf-8"))
        return result

    def _artifact_file_names_unlocked(self) -> list[str]:
        if not isinstance(self.final_data, dict):
            return []
        raw = self.final_data.get("artifact_files")
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for row in raw:
            text = str(row or "").strip()
            if text:
                out.append(text)
        return out

    def _ensure_workspace_file_history_fallback_unlocked(self, max_files: int = 500) -> None:
        if max_files <= 0:
            return
        roots = self._workspace_roots_unlocked()
        if not roots:
            return

        artifact_names = self._artifact_file_names_unlocked()
        latest_event_index = max(1, self._latest_event_index_unlocked())
        added_count = 0

        for root in roots:
            if added_count >= max_files:
                break
            root_label = self._workspace_root_label(root)
            candidates: list[Path] = []
            candidate_from_artifacts = False

            if artifact_names:
                for raw_name in artifact_names:
                    if added_count + len(candidates) >= max_files:
                        break
                    name_path = Path(str(raw_name))
                    trial_paths: list[Path] = []
                    if name_path.is_absolute():
                        trial_paths.append(name_path)
                    else:
                        trial_paths.append(root / name_path)
                        parts = name_path.parts
                        if parts and parts[0] == root_label and len(parts) > 1:
                            trial_paths.append(root / Path(*parts[1:]))

                    resolved: Path | None = None
                    for trial in trial_paths:
                        try:
                            candidate = trial.expanduser().resolve()
                        except OSError:
                            continue
                        if not _is_path_within_root(candidate, root):
                            continue
                        if candidate.exists() and candidate.is_file():
                            resolved = candidate
                            break
                    if resolved is not None:
                        candidates.append(resolved)
                if candidates:
                    candidate_from_artifacts = True
            else:
                try:
                    for candidate in sorted(root.rglob("*")):
                        if added_count + len(candidates) >= max_files:
                            break
                        if candidate.is_file():
                            if not _is_path_within_root(candidate, root):
                                continue
                            try:
                                candidates.append(candidate.resolve())
                            except OSError:
                                continue
                except OSError:
                    pass
            if artifact_names and not candidate_from_artifacts:
                # Artifact list was provided but did not resolve to valid workspace files.
                # Fall back to a full workspace scan so files still appear in the explorer.
                try:
                    for candidate in sorted(root.rglob("*")):
                        if added_count + len(candidates) >= max_files:
                            break
                        if candidate.is_file():
                            if not _is_path_within_root(candidate, root):
                                continue
                            try:
                                candidates.append(candidate.resolve())
                            except OSError:
                                continue
                except OSError:
                    pass

            seen_rel: set[str] = set()
            for abs_path in candidates:
                if added_count >= max_files:
                    break
                try:
                    rel = abs_path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if not rel or rel in seen_rel:
                    continue
                seen_rel.add(rel)
                path_key = self._normalize_file_path_key(f"{root_label}/{rel}")

                history = self._file_history.get(path_key) or []
                latest_known = history[-1] if history else {}
                if history:
                    try:
                        known_event = int(latest_known.get("event_index", 0) or 0)
                    except (TypeError, ValueError):
                        known_event = 0
                    if (
                        known_event >= latest_event_index
                        and bool(latest_known.get("content_available", False))
                        and not bool(latest_known.get("deleted", False))
                    ):
                        continue

                try:
                    size = int(abs_path.stat().st_size)
                except OSError:
                    size = 0

                snapshot_meta = self._write_text_snapshot_unlocked(
                    path_key=path_key,
                    abs_path=abs_path,
                    workspace_root=root,
                    event_index=latest_event_index,
                    size=size,
                )
                is_text = bool(snapshot_meta.get("is_text", False))

                entry = {
                    "event_index": latest_event_index,
                    "deleted": False,
                    "absolute_path": str(abs_path),
                    "workspace_root": str(root),
                    "size": size,
                    "is_text": bool(is_text),
                    "snapshot_path": str(snapshot_meta.get("snapshot_path") or ""),
                    "content_available": bool(snapshot_meta.get("content_available", False)),
                    "content_size": int(snapshot_meta.get("content_size", size) or size),
                    "fallback": True,
                }
                if history:
                    self._file_history[path_key].append(entry)
                else:
                    self._file_history[path_key] = [entry]
                added_count += 1

    def _capture_file_snapshots(self, payload: dict[str, object], event_index: int) -> None:
        if self.run_dir is None or self._snapshot_dir is None:
            return
        if not isinstance(payload, dict):
            return

        previews = payload.get("previews")
        preview_map = previews if isinstance(previews, dict) else {}
        changed_keys = payload.get("changed_files")
        added_keys = payload.get("added_files")
        removed_keys = payload.get("removed_files")
        changed = [str(k) for k in changed_keys] if isinstance(changed_keys, list) else []
        added = [str(k) for k in added_keys] if isinstance(added_keys, list) else []
        removed = [str(k) for k in removed_keys] if isinstance(removed_keys, list) else []

        snapshot_batch_dir = (self._snapshot_dir / f"{int(event_index):08d}").resolve()
        snapshot_batch_dir.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        for path_key in added + changed:
            if not path_key or path_key in seen:
                continue
            raw_key = str(path_key)
            normalized_key = self._normalize_file_path_key(raw_key)
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            path_key = normalized_key
            preview = preview_map.get(raw_key) or preview_map.get(path_key)
            abs_raw = ""
            if isinstance(preview, dict):
                abs_raw = str(preview.get("path") or "").strip()
            if not abs_raw:
                self._record_file_history(
                    path_key,
                    {"event_index": event_index, "deleted": True, "reason": "missing_path"},
                )
                continue

            file_path = Path(abs_raw).expanduser()
            if not file_path.is_absolute():
                file_path = (self.run_dir / file_path).resolve()
            else:
                file_path = file_path.resolve()

            if not self._is_within_workspace_root(file_path):
                continue

            if not file_path.exists() or not file_path.is_file():
                self._record_file_history(
            path_key,
            {
                        "event_index": event_index,
                        "deleted": True,
                        "absolute_path": str(file_path),
                        "reason": "missing_file",
                    },
                )
                continue

            try:
                size = int(file_path.stat().st_size)
            except OSError:
                size = 0
            workspace_root = self._infer_workspace_root(path_key, file_path).resolve()
            snapshot_rel_path = ""
            is_text = False
            content_available = False
            content_size = size

            if size <= FILE_SNAPSHOT_MAX_BYTES:
                try:
                    raw = file_path.read_bytes()
                except OSError:
                    raw = b""
                is_text = _is_text_blob(raw)
                if is_text:
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw.decode("utf-8", errors="replace")
                    snapshot_name = hashlib.sha1(
                        f"{path_key}:{event_index}".encode("utf-8")
                    ).hexdigest() + ".json"
                    snapshot_path = (snapshot_batch_dir / snapshot_name).resolve()
                    snapshot_payload = {
                        "path": path_key,
                        "event_index": event_index,
                        "absolute_path": str(file_path),
                        "workspace_root": str(workspace_root),
                        "size": size,
                        "is_text": True,
                        "content": text,
                    }
                    snapshot_path.write_text(
                        json.dumps(snapshot_payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    snapshot_rel_path = snapshot_path.relative_to(self.run_dir).as_posix()
                    content_available = True
                    content_size = len(text.encode("utf-8"))

            self._record_file_history(
                path_key,
                {
                    "event_index": event_index,
                    "deleted": False,
                    "absolute_path": str(file_path),
                    "workspace_root": str(workspace_root),
                    "size": size,
                    "is_text": is_text,
                    "snapshot_path": snapshot_rel_path,
                    "content_available": content_available,
                    "content_size": content_size,
                },
            )

        for path_key in removed:
            if not path_key:
                continue
            raw_key = str(path_key)
            path_key = self._normalize_file_path_key(raw_key)
            preview = preview_map.get(raw_key) or preview_map.get(path_key)
            abs_raw = ""
            if isinstance(preview, dict):
                abs_raw = str(preview.get("path") or "").strip()
            entry: dict[str, object] = {"event_index": event_index, "deleted": True}
            if abs_raw:
                file_path = Path(abs_raw).expanduser()
                if not file_path.is_absolute():
                    file_path = (self.run_dir / file_path).resolve()
                else:
                    file_path = file_path.resolve()
                if not self._is_within_workspace_root(file_path):
                    continue
                entry["absolute_path"] = str(file_path)
                entry["workspace_root"] = str(self._infer_workspace_root(path_key, file_path).resolve())
            self._record_file_history(path_key, entry)

    def _file_entry_for_event_unlocked(
        self, path_key: str, at_event_index: int | None
    ) -> dict[str, object] | None:
        history = self._file_history.get(path_key)
        if not history:
            return None
        limit = int(at_event_index) if at_event_index is not None else 2**63 - 1
        selected = None
        for entry in history:
            try:
                idx = int(entry.get("event_index", 0))
            except (TypeError, ValueError):
                idx = 0
            if idx <= limit:
                selected = entry
            else:
                break
        return dict(selected) if isinstance(selected, dict) else None

    def file_manifest(self) -> dict[str, object]:
        with self._cv:
            self._ensure_workspace_file_history_fallback_unlocked()
            files: list[dict[str, object]] = []
            for path_key in sorted(self._file_history.keys()):
                history = self._file_history.get(path_key, [])
                if not history:
                    continue
                latest = history[-1]
                event_indexes: list[int] = []
                for entry in history:
                    try:
                        idx = int(entry.get("event_index", 0) or 0)
                    except (TypeError, ValueError):
                        idx = 0
                    if idx <= 0:
                        continue
                    if bool(entry.get("deleted", False)):
                        continue
                    if not str(entry.get("snapshot_path") or "").strip():
                        continue
                    event_indexes.append(idx)
                files.append(
                    {
                        "path": path_key,
                        "versions": len(history),
                        "latest_event_index": int(latest.get("event_index", 0) or 0),
                        "deleted": bool(latest.get("deleted", False)),
                        "is_text": bool(latest.get("is_text", False)),
                        "size": int(latest.get("size", 0) or 0),
                        "content_available": bool(latest.get("content_available", False)),
                        "event_indexes": event_indexes[-200:],
                    }
                )
        return {"files": files}

    def get_file_snapshot(
        self, path_key: str, at_event_index: int | None, start: int = 0, limit: int | None = None
    ) -> tuple[int, dict[str, object]]:
        clean_path = str(path_key or "").strip()
        if not clean_path:
            return 400, {"error": "Missing required query parameter: path"}

        with self._cv:
            self._ensure_workspace_file_history_fallback_unlocked()
            resolved_path_key = self._find_file_history_key(clean_path)
            if resolved_path_key is None:
                return 404, {"error": f"Unknown file path: {clean_path}"}
            entry = self._file_entry_for_event_unlocked(resolved_path_key, at_event_index)
            run_dir = self.run_dir
        if entry is None:
            return 404, {"error": f"Unknown file path: {clean_path}"}
        if bool(entry.get("deleted", False)):
            return 404, {
                "error": "File is deleted at requested event index",
                "path": clean_path,
                "deleted_event_index": int(entry.get("event_index", 0) or 0),
            }
        snapshot_rel = str(entry.get("snapshot_path") or "").strip()
        if not snapshot_rel:
            return 404, {
                "error": "No text snapshot available for requested file/event",
                "path": clean_path,
                "event_index": int(entry.get("event_index", 0) or 0),
            }
        if run_dir is None:
            return 500, {"error": "Run directory unavailable"}
        try:
            snapshot_path = _resolve_under_root(run_dir, snapshot_rel)
        except ValueError:
            return 500, {"error": "Invalid snapshot path"}
        if not snapshot_path.exists():
            return 404, {"error": "Snapshot artifact not found", "path": clean_path}
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 500, {"error": "Failed to read snapshot artifact", "path": clean_path}
        content = str(payload.get("content", ""))
        effective_limit = _parse_slice_limit(limit)
        chunk, total_chars, truncated, next_start = _slice_text_content(content, int(start), effective_limit)
        return 200, {
            "ok": True,
            "run_id": self.run_id,
            "path": resolved_path_key,
            "snapshot_event_index": int(entry.get("event_index", 0) or 0),
            "source": "snapshot",
            "snapshot_path": snapshot_rel,
            "is_text": bool(payload.get("is_text", True)),
            "size": int(payload.get("size", entry.get("size", 0)) or 0),
            "content": chunk,
            "content_total_chars": total_chars,
            "range_start": max(0, int(start)),
            "range_limit": effective_limit,
            "truncated": truncated,
            "next_start": next_start,
        }

    def get_workspace_file(
        self, path_key: str, start: int = 0, limit: int | None = None
    ) -> tuple[int, dict[str, object]]:
        clean_path = str(path_key or "").strip()
        if not clean_path:
            return 400, {"error": "Missing required query parameter: path"}
        with self._cv:
            self._ensure_workspace_file_history_fallback_unlocked()
            resolved_path_key = self._find_file_history_key(clean_path)
            if resolved_path_key is None:
                return 404, {"error": f"Unknown file path: {clean_path}"}
            entry = self._file_entry_for_event_unlocked(resolved_path_key, None)
        if entry is None:
            return 404, {"error": f"Unknown file path: {clean_path}"}
        if bool(entry.get("deleted", False)):
            return 404, {"error": "File is deleted", "path": clean_path}

        abs_raw = str(entry.get("absolute_path") or "").strip()
        root_raw = str(entry.get("workspace_root") or "").strip()
        if not abs_raw:
            return 404, {"error": "No workspace file path available", "path": clean_path}
        file_path = Path(abs_raw).expanduser().resolve()
        if root_raw:
            workspace_root = Path(root_raw).expanduser().resolve()
            if not _is_path_within_root(file_path, workspace_root):
                return 403, {"error": "Workspace path outside allowed root", "path": clean_path}
        if not file_path.exists() or not file_path.is_file():
            return 404, {"error": "Workspace file not found", "path": clean_path}
        try:
            size = int(file_path.stat().st_size)
        except OSError:
            size = 0
        if size > FILE_SNAPSHOT_MAX_BYTES:
            return 413, {
                "error": "Workspace file too large for inline response",
                "path": clean_path,
                "size": size,
                "max_bytes": FILE_SNAPSHOT_MAX_BYTES,
            }
        try:
            raw = file_path.read_bytes()
        except OSError:
            return 500, {"error": "Failed to read workspace file", "path": clean_path}
        if not _is_text_blob(raw):
            return 415, {"error": "Workspace file is binary", "path": clean_path, "size": size}
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")
        effective_limit = _parse_slice_limit(limit)
        chunk, total_chars, truncated, next_start = _slice_text_content(content, int(start), effective_limit)
        return 200, {
            "ok": True,
            "run_id": self.run_id,
            "path": clean_path,
            "source": "workspace",
            "workspace_path": str(file_path),
            "size": size,
            "content": chunk,
            "content_total_chars": total_chars,
            "range_start": max(0, int(start)),
            "range_limit": effective_limit,
            "truncated": truncated,
            "next_start": next_start,
        }

    def emit(self, event_type: str, **payload: object) -> None:
        with self._cv:
            seq = self._next_seq
            now = time.time()
            ts_iso = datetime.fromtimestamp(now, timezone.utc).isoformat()
            payload_obj = dict(payload)
            related_run_id = self._event_related_run_id(event_type, payload_obj)
            related_span_id = f"span.run.{related_run_id}" if related_run_id else None
            span_id, parent_span_id = self._make_span_ids(event_type, seq)
            event = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "event_id": f"{self.run_id}:{seq}",
                "event_index": seq,
                "seq": seq,
                "time": now,
                "ts": ts_iso,
                "type": event_type,
                "facet": self._event_facet(event_type),
                "entity_id": self._event_entity_id(event_type, payload_obj, seq),
                "actor": self._event_actor(event_type, payload_obj),
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "related_run_id": related_run_id,
                "related_span_id": related_span_id,
                "causal_link": "delegate_parent_child_run" if related_run_id else None,
                **payload_obj,
            }
            self._next_seq += 1
            self.events.append(event)
            self._append_event_to_ledger(event)
            self._cv.notify_all()

    def get_events_after(self, last_seq: int, timeout: float = 1.0) -> tuple[list[dict[str, object]], bool]:
        with self._cv:
            end = time.time() + timeout
            while not self.done and (not self.events or self.events[-1]["seq"] <= last_seq):
                remaining = end - time.time()
                if remaining <= 0:
                    break
                self._cv.wait(remaining)
            items = [e for e in self.events if e["seq"] > last_seq]
            return items, self.done

    def snapshot(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "spec_path": str(self.spec_path),
            "agent_path": str(self.agent_path),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "done": self.done,
            "returncode": self.returncode,
            "error": self.error,
            "current_process": self.current_process,
            "trace_path": self.trace_path,
            "event_count": len(self.events),
            "event_ledger_path": str(self._events_log_path) if self._events_log_path else None,
            "file_snapshot_dir": str(self._snapshot_dir) if self._snapshot_dir else None,
            "tracked_files_count": len(self._file_history),
            "process_metrics": self.process_metrics,
            "mock_mode": self.mock_mode,
            "capability_profile": self.capability_profile,
            "effective_model": self.effective_model,
        }

    def start(self) -> None:
        self.run_dir = (_repo_dir(self.root_dir) / ".live_runs" / self.run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_log_path = (self.run_dir / "events.jsonl").resolve()
        self._events_log_path.write_text("", encoding="utf-8")
        self._snapshot_dir = (self.run_dir / "file_snapshots").resolve()
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._file_history = {}
        self._run_span_id = f"span.run.{self.run_id}"
        self._active_transition_span_id = None

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["AGENT_ONTOLOGY_TRACE_PREFIX"] = f"agent_ontology/live/{self.run_id}"
        env["AGENT_ONTOLOGY_TASK"] = self.task
        env["AGENT_ONTOLOGY_MAX_ITER"] = str(self.max_iter)
        env["AGENT_ONTOLOGY_MOCK_MODE"] = self.mock_mode
        env["AGENT_ONTOLOGY_CAPABILITY_PROFILE"] = self.capability_profile
        env["AGENT_ONTOLOGY_SINGLE_TURN"] = os.environ.get("AGENT_ONTOLOGY_SINGLE_TURN", "1")
        env["AGENT_ONTOLOGY_EMIT_DELEGATE_EVENTS"] = "1"
        env["AGENT_ONTOLOGY_MODEL"] = _resolve_model_override(self.model_override)
        self.effective_model = env["AGENT_ONTOLOGY_MODEL"]
        if self.max_budget_usd is not None:
            env["AGENT_ONTOLOGY_MAX_BUDGET_USD"] = str(self.max_budget_usd)

        pythonpath_parts = [str(self.root_dir), str(self.root_dir / "agent_ontology")]
        llm_client_repo = self.root_dir / "llm_client"
        if llm_client_repo.exists():
            pythonpath_parts.append(str(llm_client_repo))
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        initial = dict(self.initial_data)
        initial["requested_mode"] = self.requested_mode
        runner_code = _build_runner_code(self.agent_path, initial)

        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", runner_code],
            cwd=str(self.run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self.emit(
            "started",
            run_id=self.run_id,
            spec_path=str(self.spec_path),
            agent_path=str(self.agent_path),
            input=initial,
            run_dir=str(self.run_dir),
            mock_mode=self.mock_mode,
            capability_profile=self.capability_profile,
            model_override=self.effective_model,
            effective_model=self.effective_model,
        )

        self._reader_thread = threading.Thread(target=self._consume_output, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.emit("stopped", run_id=self.run_id)

    def _consume_output(self) -> None:
        try:
            if self.process and self.process.stdout:
                for raw_line in self.process.stdout:
                    self._handle_line(raw_line.rstrip("\n"))
            if self.process:
                self.returncode = self.process.wait()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.emit("error", message=self.error)
        finally:
            self.done = True
            self.ended_at = time.time()
            self.emit(
                "completed",
                run_id=self.run_id,
                returncode=self.returncode,
                error=self.error,
                trace_path=self.trace_path,
                process_metrics=self.process_metrics,
                final_channel=self.final_channel,
                final_data_summary=self._final_data_summary(),
            )

    def _final_data_summary(self) -> dict[str, object]:
        if not isinstance(self.final_data, dict):
            return {}
        summary: dict[str, object] = {
            "keys": sorted(self.final_data.keys()),
            "last_channel": self.final_data.get("last_channel"),
            "finalized": self.final_data.get("finalized"),
            "has_action_decision": "action_decision" in self.final_data,
            "has_final_channel": "final_channel" in self.final_data,
        }
        draft_response = self.final_data.get("draft_response")
        if isinstance(draft_response, str) and draft_response:
            summary["draft_response"] = (
                draft_response if len(draft_response) <= 2000 else draft_response[:2000] + "..."
            )
        final_answer = self.final_data.get("answer")
        if isinstance(final_answer, str) and final_answer:
            summary["answer"] = final_answer if len(final_answer) <= 2000 else final_answer[:2000] + "..."
        action_decision = self.final_data.get("action_decision")
        if isinstance(action_decision, dict) and action_decision:
            try:
                action_preview = json.dumps(action_decision, default=str, ensure_ascii=False)
            except Exception:
                action_preview = str(action_decision)
            summary["action_decision"] = (
                action_preview if len(action_preview) <= 3000 else action_preview[:3000] + "..."
            )
        quality_gate_failed = self.final_data.get("quality_gate_failed")
        if isinstance(quality_gate_failed, bool):
            summary["quality_gate_failed"] = quality_gate_failed
        quality_gate_reason = self.final_data.get("quality_gate_reason")
        if isinstance(quality_gate_reason, str) and quality_gate_reason:
            summary["quality_gate_reason"] = (
                quality_gate_reason
                if len(quality_gate_reason) <= 220
                else quality_gate_reason[:220] + "..."
            )
        quality_file_check = self.final_data.get("quality_file_check")
        if isinstance(quality_file_check, dict) and quality_file_check:
            compact_quality: dict[str, object] = {}
            for key in ("candidate_count", "non_empty_count", "empty_count"):
                raw_value = quality_file_check.get(key)
                try:
                    compact_quality[key] = int(raw_value)
                except (TypeError, ValueError):
                    pass
            for key in ("non_empty_files", "empty_files"):
                raw_value = quality_file_check.get(key)
                if not isinstance(raw_value, list):
                    continue
                compact_quality[key] = [str(item)[:260] for item in raw_value[:20]]
            if compact_quality:
                summary["quality_file_check"] = compact_quality
        return summary

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()

        if stripped.startswith("__AO_RUNTIME_INFO__"):
            raw = stripped[len("__AO_RUNTIME_INFO__") :]
            try:
                info = json.loads(raw)
            except json.JSONDecodeError:
                info = {"raw": raw}
            self.emit("runtime_info", info=info)
            return

        if stripped.startswith("__AO_FINAL_CHANNEL__"):
            raw = stripped[len("__AO_FINAL_CHANNEL__") :]
            try:
                self.final_channel = json.loads(raw)
            except json.JSONDecodeError:
                self.final_channel = raw
            self.emit("final_channel", payload=self.final_channel)
            return

        if stripped.startswith("__AO_FINAL_DATA__"):
            raw = stripped[len("__AO_FINAL_DATA__") :]
            try:
                self.final_data = json.loads(raw)
            except json.JSONDecodeError:
                self.final_data = raw
            self.emit("final_data", summary=self._final_data_summary(), data=self.final_data)
            return

        if stripped.startswith("__AO_DELEGATE_TRACE__"):
            raw = stripped[len("__AO_DELEGATE_TRACE__") :]
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            self.emit("delegate_trace_file", payload=payload)
            return

        if stripped.startswith("__AO_DELEGATE_RUN__"):
            raw = stripped[len("__AO_DELEGATE_RUN__") :]
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            self.emit("delegate_run", payload=payload)
            return

        if stripped.startswith("__AO_STEP_DIFF__"):
            raw = stripped[len("__AO_STEP_DIFF__") :]
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            self.emit("step_diff", payload=payload)
            return

        if stripped.startswith("__AO_STORE_DIFF__"):
            raw = stripped[len("__AO_STORE_DIFF__") :]
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            self.emit("store_diff", payload=payload)
            return

        if stripped.startswith("__AO_FILE_DIFF__"):
            raw = stripped[len("__AO_FILE_DIFF__") :]
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            self.emit("file_diff", payload=payload)
            event_index = int(self.events[-1].get("event_index", 0)) if self.events else 0
            if event_index <= 0:
                event_index = 1
            self._capture_file_snapshots(payload if isinstance(payload, dict) else {}, event_index)
            return

        self.emit("log", line=line)
        if not stripped:
            return

        m_state = ITERATION_RE.match(stripped)
        if m_state:
            process_id = m_state.group("state")
            iteration = int(m_state.group("iter"))
            self.current_process = process_id
            bucket = self.process_metrics.setdefault(
                process_id, {"count": 0, "duration_ms": 0, "llm_calls": 0}
            )
            bucket["count"] += 1
            self.emit(
                "state",
                iteration=iteration,
                process=process_id,
                process_metrics=self.process_metrics,
            )
            return

        m_call = LLM_CALL_RE.match(line)
        if m_call:
            process_id = self.current_process or "_unknown_"
            duration_ms = int(m_call.group("ms"))
            bucket = self.process_metrics.setdefault(
                process_id, {"count": 0, "duration_ms": 0, "llm_calls": 0}
            )
            bucket["llm_calls"] += 1
            bucket["duration_ms"] += duration_ms
            self.emit(
                "llm_call",
                process=process_id,
                agent=m_call.group("label"),
                model=m_call.group("model").strip(),
                duration_ms=duration_ms,
                process_metrics=self.process_metrics,
            )
            return

        m_trace = TRACE_WRITTEN_RE.match(stripped)
        if m_trace:
            trace_ref = m_trace.group("path").strip()
            trace_path = Path(trace_ref)
            if not trace_path.is_absolute() and self.run_dir is not None:
                trace_path = (self.run_dir / trace_path).resolve()
            self.trace_path = str(trace_path)
            self.emit(
                "trace_file",
                path=self.trace_path,
                total_calls=int(m_trace.group("calls")),
                total_duration_ms=int(m_trace.group("dur")),
            )
            return

        if stripped.startswith("Final state.data keys:"):
            raw_keys = stripped.split(":", 1)[1].strip()
            try:
                keys = ast.literal_eval(raw_keys)
                if not isinstance(keys, list):
                    keys = []
            except (SyntaxError, ValueError):
                keys = []
            self.emit("final_keys", keys=keys)


def _hydrate_persisted_run_session(root_dir: Path, run_id: str) -> RunSession | None:
    if not _is_valid_run_id(run_id):
        return None
    try:
        events, _ = _load_persisted_run_events(root_dir, run_id)
    except ValueError:
        return None
    if not events:
        return None
    try:
        run_dir = _run_artifact_dir(root_dir, run_id)
    except ValueError:
        return None

    session = RunSession(
        run_id=run_id,
        root_dir=root_dir,
        spec_path=Path(""),
        agent_path=Path(""),
        initial_data={},
        task="",
        requested_mode="default",
        mock_mode="live",
        capability_profile="external_enabled",
        max_iter=0,
        model_override="",
        max_budget_usd=None,
    )
    session.run_dir = run_dir
    session._events_log_path = (run_dir / "events.jsonl").resolve()
    snapshot_dir = (run_dir / "file_snapshots").resolve()
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    session._snapshot_dir = snapshot_dir if snapshot_dir.exists() and snapshot_dir.is_dir() else None
    session.events = [dict(evt) for evt in events]
    if session.events:
        session._next_seq = max(_event_index_of(evt) for evt in session.events) + 1
    session.done = True

    for event in session.events:
        event_type = str(event.get("type") or "")
        if event_type == "started":
            raw_input = event.get("input")
            if isinstance(raw_input, dict):
                for key in ("workspace_dir", "workspace_path", "output_dir"):
                    value = raw_input.get(key)
                    if isinstance(value, str) and value.strip():
                        session.initial_data[key] = value
        elif event_type == "final_data":
            raw_data = event.get("data")
            if isinstance(raw_data, dict):
                session.final_data = raw_data
        elif event_type == "final_channel":
            payload = event.get("payload")
            if payload is not None:
                session.final_channel = payload

    return session


class LiveViewerHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving static viewer files + live execution API."""

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
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/live/health":
            self._send_json({"ok": True, "root": str(self.root_dir)})
            return

        if path == "/api/live/runs":
            with RUNS_LOCK:
                runs = [session.snapshot() for session in RUNS.values()]
            runs.sort(key=lambda r: r["started_at"], reverse=True)
            self._send_json({"runs": runs})
            return

        if path == "/api/live/reviews":
            params = parse_qs(parsed.query or "")
            try:
                limit = int((params.get("limit") or ["30"])[0])
            except (TypeError, ValueError):
                limit = 30
            reviews = _list_review_artifacts(self.root_dir, limit=max(1, min(limit, 200)))
            self._send_json({"reviews": reviews})
            return

        if path == "/api/live/review/latest":
            loaded = _load_review_artifact(self.root_dir, None)
            if loaded is None:
                self._send_json({"error": "No review artifacts found"}, status=404)
                return
            file_path, payload = loaded
            self._send_json(
                {
                    "ok": True,
                    "file": str(file_path),
                    "summary": _review_artifact_summary(file_path, payload),
                    "panel_summary": _review_panel_summary(payload),
                    "data": payload,
                }
            )
            return

        if path.startswith("/api/live/review/"):
            run_id = path.rsplit("/", 1)[-1]
            loaded = _load_review_artifact(self.root_dir, run_id)
            if loaded is None:
                self._send_json({"error": f"No review artifact found for run_id={run_id}"}, status=404)
                return
            file_path, payload = loaded
            self._send_json(
                {
                    "ok": True,
                    "file": str(file_path),
                    "summary": _review_artifact_summary(file_path, payload),
                    "panel_summary": _review_panel_summary(payload),
                    "data": payload,
                }
            )
            return

        if path.startswith("/api/live/files/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                session = RUNS.get(run_id)
            if not session:
                restored = _hydrate_persisted_run_session(self.root_dir, run_id)
                if restored is not None:
                    with RUNS_LOCK:
                        session = RUNS.setdefault(run_id, restored)
                else:
                    self._send_json({"error": f"Unknown run: {run_id}"}, status=404)
                    return
            manifest = session.file_manifest()
            self._send_json({"ok": True, "run_id": run_id, **manifest})
            return

        if path.startswith("/api/live/file_snapshot/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                session = RUNS.get(run_id)
            if not session:
                restored = _hydrate_persisted_run_session(self.root_dir, run_id)
                if restored is not None:
                    with RUNS_LOCK:
                        session = RUNS.setdefault(run_id, restored)
                else:
                    self._send_json({"error": f"Unknown run: {run_id}"}, status=404)
                    return
            params = parse_qs(parsed.query or "")
            path_key = str((params.get("path") or [""])[0] or "")
            at_event_raw = (params.get("at_event_index") or [""])[0]
            at_event_index = _parse_cursor_value(at_event_raw)
            start_raw = (params.get("start") or ["0"])[0]
            start = _parse_cursor_value(start_raw) or 0
            limit_raw = (params.get("limit") or [""])[0]
            status, payload = session.get_file_snapshot(
                path_key,
                at_event_index,
                start=start,
                limit=_parse_slice_limit(limit_raw),
            )
            self._send_json(payload, status=status)
            return

        if path.startswith("/api/live/file_workspace/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                session = RUNS.get(run_id)
            if not session:
                restored = _hydrate_persisted_run_session(self.root_dir, run_id)
                if restored is not None:
                    with RUNS_LOCK:
                        session = RUNS.setdefault(run_id, restored)
                else:
                    self._send_json({"error": f"Unknown run: {run_id}"}, status=404)
                    return
            params = parse_qs(parsed.query or "")
            path_key = str((params.get("path") or [""])[0] or "")
            start_raw = (params.get("start") or ["0"])[0]
            start = _parse_cursor_value(start_raw) or 0
            limit_raw = (params.get("limit") or [""])[0]
            status, payload = session.get_workspace_file(
                path_key,
                start=start,
                limit=_parse_slice_limit(limit_raw),
            )
            self._send_json(payload, status=status)
            return

        if path.startswith("/api/live/status/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                session = RUNS.get(run_id)
            if not session:
                self._send_json({"error": f"Unknown run: {run_id}"}, status=404)
                return
            self._send_json(session.snapshot())
            return

        if path.startswith("/api/live/events/"):
            run_id = path.rsplit("/", 1)[-1]
            if not _is_valid_run_id(run_id):
                self._send_json({"error": f"Invalid run id: {run_id}"}, status=400)
                return
            params = parse_qs(parsed.query or "")
            since_event_index = _resolve_stream_since(params, self.headers)
            limit = _parse_event_limit((params.get("limit") or [""])[0])

            with RUNS_LOCK:
                session = RUNS.get(run_id)

            source = "memory"
            done = True
            status_payload: dict[str, object] | None = None
            if session is not None:
                with session._cv:
                    all_events = [dict(evt) for evt in session.events]
                    done = bool(session.done)
                    status_payload = session.snapshot()
            else:
                try:
                    all_events, source = _load_persisted_run_events(self.root_dir, run_id)
                except ValueError:
                    self._send_json({"error": f"Invalid run id path: {run_id}"}, status=400)
                    return
                if not all_events:
                    self._send_json({"error": f"No events found for run: {run_id}"}, status=404)
                    return
                source = str(source or "none")

            events = _filter_events_window(
                all_events,
                since_event_index=since_event_index,
                limit=limit,
            )
            latest_event_index = 0
            if all_events:
                latest_event_index = max(_event_index_of(evt) for evt in all_events)
            self._send_json(
                {
                    "ok": True,
                    "run_id": run_id,
                    "source": source,
                    "done": done,
                    "latest_event_index": latest_event_index,
                    "event_count": len(events),
                    "events": events,
                    "status": status_payload,
                }
            )
            return

        if path.startswith("/api/live/stream/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                session = RUNS.get(run_id)
            if not session:
                self._send_json({"error": f"Unknown run: {run_id}"}, status=404)
                return
            self._stream_events(session, parsed.query)
            return

        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/live/run":
            self._handle_start_run()
            return

        if path.startswith("/api/live/stop/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                session = RUNS.get(run_id)
            if not session:
                self._send_json({"error": f"Unknown run: {run_id}"}, status=404)
                return
            session.stop()
            self._send_json({"ok": True, "run_id": run_id})
            return

        self._send_json({"error": f"Unknown endpoint: {path}"}, status=404)

    def _handle_start_run(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        spec_path_raw = str(payload.get("spec_path", "")).strip()
        if not spec_path_raw:
            self._send_json({"error": "Missing required field: spec_path"}, status=400)
            return

        try:
            spec_path = _resolve_spec_path(self.root_dir, spec_path_raw)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        if not spec_path.exists():
            self._send_json({"error": f"Spec not found: {spec_path}"}, status=404)
            return

        agent_path_raw = str(payload.get("agent_path", "")).strip()
        if agent_path_raw:
            try:
                agent_path = _resolve_under_root(self.root_dir, agent_path_raw)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
        else:
            agent_path = _derive_agent_path(self.root_dir, spec_path)

        try:
            _ensure_agent_generated(self.root_dir, spec_path, agent_path)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._send_json({"error": str(exc)}, status=500)
            return

        spec_stem = spec_path.stem.lower()
        default_max_iter = 220 if spec_stem.startswith("autogpt") else 80
        max_iter_raw = payload.get("max_iter", default_max_iter)
        try:
            max_iter = int(max_iter_raw)
            if max_iter < 1:
                raise ValueError("max_iter must be >= 1")
        except ValueError as exc:
            self._send_json({"error": f"Invalid max_iter: {exc}"}, status=400)
            return

        max_budget_usd: float | None
        max_budget_raw = str(payload.get("max_budget_usd", "")).strip()
        if max_budget_raw:
            try:
                max_budget_usd = float(max_budget_raw)
            except ValueError:
                self._send_json({"error": f"Invalid max_budget_usd: {max_budget_raw}"}, status=400)
                return
        else:
            max_budget_usd = None

        run_id = uuid.uuid4().hex[:12]
        initial_data = _build_initial_data_for_run(
            spec_path,
            payload,
            root_dir=self.root_dir,
            run_id=run_id,
        )

        requested_mode = str(payload.get("requested_mode", "default") or "default")
        mock_mode = str(payload.get("mock_mode", os.environ.get("AGENT_ONTOLOGY_MOCK_MODE", "live")) or "live").strip().lower()
        if mock_mode in {"off", "none", "disabled"}:
            mock_mode = "live"
        if mock_mode not in {"deterministic", "live", "llm_mock"}:
            self._send_json({"error": f"Invalid mock_mode: {mock_mode}"}, status=400)
            return
        capability_profile = str(
            payload.get(
                "capability_profile",
                os.environ.get("AGENT_ONTOLOGY_CAPABILITY_PROFILE", "external_enabled"),
            )
            or "external_enabled"
        ).strip().lower()
        if capability_profile in {"external", "unshackled", "full"}:
            capability_profile = "external_enabled"
        if capability_profile not in {"restricted", "external_enabled"}:
            self._send_json({"error": f"Invalid capability_profile: {capability_profile}"}, status=400)
            return
        initial_data["capability_profile"] = capability_profile
        task = str(payload.get("task", "agent_ontology_live_viewer"))
        model_override = _resolve_model_override(payload.get("model_override", ""))

        session = RunSession(
            run_id=run_id,
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

        with RUNS_LOCK:
            RUNS[run_id] = session

        try:
            session.start()
        except Exception as exc:
            session.error = f"{type(exc).__name__}: {exc}"
            session.done = True
            session.emit("error", message=session.error)
            self._send_json({"error": session.error}, status=500)
            return

        self._send_json(
            {
                "ok": True,
                "run_id": run_id,
                "stream_url": f"/api/live/stream/{run_id}",
                "status_url": f"/api/live/status/{run_id}",
                "stop_url": f"/api/live/stop/{run_id}",
                "spec_path": str(spec_path),
                "agent_path": str(agent_path),
                "mock_mode": mock_mode,
                "capability_profile": capability_profile,
            }
        )

    def _stream_events(self, session: RunSession, query: str) -> None:
        params = parse_qs(query or "")
        last_seq = _resolve_stream_since(params, self.headers)

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
                    event_index = int(event.get("event_index", event.get("seq", last_seq)))
                    blob = json.dumps(event, default=str).encode("utf-8")
                    self.wfile.write(f"id: {event_index}\n".encode("utf-8"))
                    self.wfile.write(b"data: " + blob + b"\n\n")
                    self.wfile.flush()
                    last_seq = event_index

                if done:
                    break
        except (BrokenPipeError, ConnectionResetError):
            return


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve spec-viewer with live agent execution and real-time trace streaming."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--root",
        default=str(_find_default_root()),
        help=(
            "Root to serve. Supported layouts: workspace root containing "
            "agent_ontology/spec-viewer.html, or repo root containing spec-viewer.html."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    root_dir = Path(args.root).resolve()
    if not (_has_workspace_layout(root_dir) or _has_repo_layout(root_dir)):
        raise SystemExit(
            f"Invalid root: {root_dir} (missing spec-viewer.html). "
            "Use workspace root (contains agent_ontology/spec-viewer.html) "
            "or repo root (contains spec-viewer.html)."
        )

    LiveViewerHandler.root_dir = root_dir

    server = ThreadingHTTPServer((args.host, args.port), LiveViewerHandler)
    print(f"Live viewer server running on http://{args.host}:{args.port}")
    print(f"Open: http://{args.host}:{args.port}{_default_viewer_path(root_dir)}")
    print(f"Serving root: {root_dir}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down live viewer server...")
    finally:
        with RUNS_LOCK:
            sessions = list(RUNS.values())
        for session in sessions:
            session.stop()
        server.server_close()


if __name__ == "__main__":
    main()
