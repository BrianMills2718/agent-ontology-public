#!/usr/bin/env python3
"""Run a live backend E2E and (optionally) judge outputs via llm_client."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_MODEL = "openrouter/deepseek/deepseek-chat"
DEFAULT_JUDGE_MODEL = os.environ.get("AGENT_ONTOLOGY_JUDGE_MODEL", DEFAULT_MODEL)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _http_get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_health(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            return _http_get_json(f"{base_url}/api/live/health", timeout=5.0)
        except Exception as exc:  # pragma: no cover - defensive runtime behavior
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for /api/live/health ({last_error})")


def _stream_events(stream_url: str, timeout_seconds: float) -> tuple[list[dict[str, Any]], bool]:
    events: list[dict[str, Any]] = []
    timed_out = False
    started = time.monotonic()
    with request.urlopen(stream_url, timeout=timeout_seconds + 20) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                if time.monotonic() - started > timeout_seconds:
                    timed_out = True
                    break
                continue
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if isinstance(evt, dict):
                events.append(evt)
            if evt.get("type") == "completed":
                break
            if time.monotonic() - started > timeout_seconds:
                timed_out = True
                break
    return events, timed_out


def summarize_live_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    started_event: dict[str, Any] | None = None
    completed_event: dict[str, Any] | None = None
    final_data_event: dict[str, Any] | None = None
    runtime_info_event: dict[str, Any] | None = None
    log_lines: list[str] = []
    llm_calls: list[dict[str, Any]] = []
    file_diff_samples: list[dict[str, Any]] = []
    step_diff_samples: list[dict[str, Any]] = []

    def _compact_started(evt: dict[str, Any]) -> dict[str, Any]:
        payload = evt.get("input", {})
        if not isinstance(payload, dict):
            payload = {}
        objective = payload.get("objective") or payload.get("text") or ""
        return {
            "run_id": evt.get("run_id"),
            "spec_path": evt.get("spec_path"),
            "mock_mode": evt.get("mock_mode"),
            "capability_profile": evt.get("capability_profile"),
            "model_override": evt.get("model_override"),
            "effective_model": evt.get("effective_model"),
            "run_dir": evt.get("run_dir"),
            "objective_preview": _trim(str(objective), 320),
        }

    def _compact_completed(evt: dict[str, Any]) -> dict[str, Any]:
        final_channel = evt.get("final_channel", {})
        source = ""
        preview = ""
        if isinstance(final_channel, dict):
            source = str(final_channel.get("source", ""))
            preview = _trim(as_pretty_text(final_channel.get("value", "")), 420)
        return {
            "run_id": evt.get("run_id"),
            "returncode": evt.get("returncode"),
            "error": evt.get("error"),
            "trace_path": evt.get("trace_path"),
            "process_metrics": evt.get("process_metrics", {}),
            "final_channel_source": source,
            "final_channel_preview": preview,
        }

    for evt in events:
        event_type = str(evt.get("type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
        if event_type == "started":
            started_event = _compact_started(evt)
        elif event_type == "completed":
            completed_event = _compact_completed(evt)
        elif event_type == "final_data":
            final_data_event = evt
        elif event_type == "runtime_info":
            runtime_info_event = evt
        elif event_type == "log":
            line = str(evt.get("line", ""))
            if line:
                log_lines.append(_trim(line, 320))
        elif event_type == "llm_call":
            if len(llm_calls) < 20:
                llm_calls.append(
                    {
                        "process": evt.get("process"),
                        "agent": evt.get("agent"),
                        "model": evt.get("model"),
                        "duration_ms": evt.get("duration_ms"),
                    }
                )
        elif event_type == "file_diff":
            payload = evt.get("payload", {})
            if isinstance(payload, dict) and len(file_diff_samples) < 4:
                file_diff_samples.append(
                    {
                        "process": payload.get("process"),
                        "added_files": payload.get("added_files", []),
                        "changed_files": payload.get("changed_files", []),
                        "removed_files": payload.get("removed_files", []),
                        "patches": {
                            str(k): _trim(str(v), 420)
                            for k, v in (payload.get("patches", {}) or {}).items()
                        },
                    }
                )
        elif event_type == "step_diff":
            payload = evt.get("payload", {})
            if isinstance(payload, dict) and len(step_diff_samples) < 8:
                step_diff_samples.append(
                    {
                        "process": payload.get("process"),
                        "added": payload.get("added", []),
                        "changed": payload.get("changed", []),
                        "removed": payload.get("removed", []),
                    }
                )

    summary = (final_data_event or {}).get("summary", {})
    if isinstance(summary, dict):
        compact_summary = dict(summary)
        for key in ("draft_response", "answer", "action_decision"):
            if key in compact_summary:
                compact_summary[key] = _trim(str(compact_summary.get(key, "")), 700)
    else:
        compact_summary = {}

    return {
        "event_counts": counts,
        "has_completed_event": completed_event is not None,
        "started_event": started_event or {},
        "completed_event": completed_event or {},
        "final_data_summary": compact_summary,
        "runtime_info": (runtime_info_event or {}).get("info", {}),
        "log_tail": log_lines[-40:],
        "llm_call_samples": llm_calls,
        "file_diff_samples": file_diff_samples,
        "step_diff_samples": step_diff_samples,
    }


def as_pretty_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _is_text_file(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    return True


def _snapshot_workspace_files(
    run_dir: Path | None,
    *,
    max_files: int = 6,
    max_bytes: int = 120_000,
    max_preview: int = 800,
) -> list[dict[str, Any]]:
    if run_dir is None or not run_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for file_path in sorted(run_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name == "trace.json":
            continue
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if size > 250_000:
            continue
        if not _is_text_file(file_path):
            continue
        if total_bytes + size > max_bytes:
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows.append(
            {
                "path": str(file_path.relative_to(run_dir)),
                "size": size,
                "preview": _trim(content, max_preview),
            }
        )
        total_bytes += size
        if len(rows) >= max_files:
            break
    return rows


def _load_trace_metrics(trace_path: str | None) -> dict[str, Any]:
    if not trace_path:
        return {}
    path = Path(trace_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    metrics = data.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    if raw.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", raw)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        if cleaned:
            raw = cleaned

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _canonical_action_type(value: Any) -> str:
    action_type = str(value or "").strip().lower()
    if action_type in {"external", "externalcall"}:
        return "external_call"
    if action_type in {"none", "skip", "no_action"}:
        return "none"
    if action_type in {"command", "external_call", "none"}:
        return action_type
    return ""


def _canonical_interface_name(value: Any) -> str:
    interface_name = str(value or "").strip().lower()
    if interface_name in {"search", "lookup", "wikipedia_lookup", "web_fetch"}:
        return "semantic_search"
    if interface_name == "mcp_tool_call":
        return "mcp_call"
    if interface_name == "rpc_event":
        return "rpc_emit"
    return interface_name


def _load_trace_observations(trace_path: str | None) -> dict[str, Any]:
    payload = {
        "executor_calls": 0,
        "executor_external_call_count": 0,
        "executor_action_types": [],
        "executor_interfaces": [],
        "executor_interfaces_canonical": [],
    }
    if not trace_path:
        return payload
    path = Path(trace_path)
    if not path.exists():
        return payload

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload

    trace_rows = data.get("trace", [])
    if not isinstance(trace_rows, list):
        return payload

    action_types: list[str] = []
    interfaces: list[str] = []
    interfaces_canonical: list[str] = []
    executor_calls = 0
    executor_external_calls = 0

    for row in trace_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("agent", "")).strip().lower() != "executor":
            continue
        executor_calls += 1
        parsed = _extract_json_object(str(row.get("response", "")))
        if not isinstance(parsed, dict):
            continue

        action_type = _canonical_action_type(parsed.get("action_type"))
        if action_type and action_type not in action_types:
            action_types.append(action_type)
        if action_type == "external_call":
            executor_external_calls += 1

        interface_raw = str(parsed.get("interface_name", "") or "").strip()
        if interface_raw and interface_raw not in interfaces:
            interfaces.append(interface_raw)

        interface_canonical = _canonical_interface_name(interface_raw)
        if interface_canonical and interface_canonical not in interfaces_canonical:
            interfaces_canonical.append(interface_canonical)

    payload["executor_calls"] = executor_calls
    payload["executor_external_call_count"] = executor_external_calls
    payload["executor_action_types"] = action_types
    payload["executor_interfaces"] = interfaces
    payload["executor_interfaces_canonical"] = interfaces_canonical
    return payload


def _build_judge_payload(
    *,
    run_request: dict[str, Any],
    run_response: dict[str, Any],
    status: dict[str, Any],
    event_summary: dict[str, Any],
    trace_metrics: dict[str, Any],
    workspace_files: list[dict[str, Any]],
) -> dict[str, Any]:
    status_compact = {
        "run_id": status.get("run_id"),
        "done": status.get("done"),
        "returncode": status.get("returncode"),
        "error": status.get("error"),
        "current_process": status.get("current_process"),
        "trace_path": status.get("trace_path"),
        "event_count": status.get("event_count"),
        "effective_model": status.get("effective_model"),
        "process_metrics": status.get("process_metrics", {}),
    }
    request_compact = {
        "spec_path": run_request.get("spec_path"),
        "text": _trim(str(run_request.get("text", "")), 400),
        "requested_mode": run_request.get("requested_mode"),
        "max_iter": run_request.get("max_iter"),
        "mock_mode": run_request.get("mock_mode"),
        "capability_profile": run_request.get("capability_profile"),
    }
    response_compact = {
        "run_id": run_response.get("run_id"),
        "spec_path": run_response.get("spec_path"),
        "agent_path": run_response.get("agent_path"),
        "mock_mode": run_response.get("mock_mode"),
        "capability_profile": run_response.get("capability_profile"),
    }
    return {
        "objective": _trim(str(run_request.get("text", "")), 500),
        "runtime_request": request_compact,
        "runtime_response": response_compact,
        "runtime_status": status_compact,
        "event_summary": event_summary,
        "trace_metrics": trace_metrics,
        "workspace_files": workspace_files,
    }


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _normalize_scale_0_to_1(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    normalized = raw / 5.0 if raw > 1.0 else raw
    return _clamp01(normalized)


def _criteria_score_normalized(criteria: Any) -> float | None:
    if not isinstance(criteria, list) or not criteria:
        return None
    raw_scores: list[float] = []
    for item in criteria:
        if not isinstance(item, dict):
            continue
        try:
            raw_scores.append(float(item.get("score", 0)))
        except (TypeError, ValueError):
            continue
    if not raw_scores:
        return None
    return _clamp01(sum(raw_scores) / (len(raw_scores) * 5.0))


def evaluate_review_profile(
    *,
    profile: str,
    no_judge: bool,
    trace_metrics: dict[str, Any],
    event_summary: dict[str, Any],
    workspace_files: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if profile != "babyagi_strict":
        return {"profile": profile, "ok": True, "checks": checks}

    if no_judge:
        _check("judge_enabled", False, "babyagi_strict requires judge step (omit --no-judge)")
    else:
        _check("judge_enabled", True, "judge step enabled")

    clean_exit = bool(trace_metrics.get("clean_exit"))
    _check("trace_clean_exit", clean_exit, f"clean_exit={clean_exit}")

    total_llm_calls = _as_int(trace_metrics.get("total_llm_calls"), 0)
    _check("llm_calls_present", total_llm_calls >= 1, f"total_llm_calls={total_llm_calls}")

    event_counts = event_summary.get("event_counts", {})
    if not isinstance(event_counts, dict):
        event_counts = {}
    file_diff_events = _as_int(event_counts.get("file_diff"), 0)
    workspace_count = len(workspace_files)
    artifact_signal = file_diff_events >= 1 or workspace_count >= 1
    _check(
        "artifact_signal_present",
        artifact_signal,
        f"file_diff_events={file_diff_events}, workspace_files={workspace_count}",
    )

    final_summary = event_summary.get("final_data_summary", {})
    has_final_summary = isinstance(final_summary, dict) and bool(final_summary)
    _check("final_data_present", has_final_summary, f"final_data_keys={sorted((final_summary or {}).keys())}")

    quality_gate_failed = False
    quality_gate_reason = ""
    candidate_count = 0
    non_empty_count = 0
    if isinstance(final_summary, dict):
        quality_gate_failed = bool(final_summary.get("quality_gate_failed", False))
        quality_gate_reason = str(final_summary.get("quality_gate_reason", "") or "").strip()
        quality_file_check = final_summary.get("quality_file_check", {})
        if isinstance(quality_file_check, dict):
            candidate_count = _as_int(quality_file_check.get("candidate_count"), 0)
            non_empty_count = _as_int(quality_file_check.get("non_empty_count"), 0)
    empty_candidate_failure = candidate_count > 0 and non_empty_count == 0
    quality_ok = not quality_gate_failed and not empty_candidate_failure
    _check(
        "quality_gate_passed",
        quality_ok,
        (
            f"quality_gate_failed={quality_gate_failed}, "
            f"quality_gate_reason={quality_gate_reason or '(none)'}, "
            f"candidate_count={candidate_count}, non_empty_count={non_empty_count}"
        ),
    )

    ok = all(bool(item.get("ok")) for item in checks)
    return {"profile": profile, "ok": ok, "checks": checks}


def run_llm_judge(
    payload: dict[str, Any],
    *,
    judge_model: str,
    run_id: str,
    max_budget: float,
) -> dict[str, Any]:
    try:
        from pydantic import BaseModel, Field
        from typing import Literal

        from llm_client import call_llm_structured
    except Exception as exc:  # pragma: no cover - runtime dependency issues
        return {"error": f"Judge unavailable: {type(exc).__name__}: {exc}"}

    class CriterionScore(BaseModel):
        criterion: str
        score: int = Field(ge=1, le=5)
        reasoning: str

    class JudgeReport(BaseModel):
        verdict: Literal["pass", "borderline", "fail"]
        overall_score: float = Field(ge=0.0, le=5.0)
        criteria: list[CriterionScore]
        strengths: list[str]
        issues: list[str]
        recommendations: list[str]
        confidence: float = Field(ge=0.0, le=5.0)
        summary: str

    system = (
        "You are a strict QA judge for agent runtime E2E runs.\n"
        "Score against this rubric (1-5 each):\n"
        "- objective_completion: does output materially satisfy the user objective?\n"
        "- artifact_quality: are emitted files coherent, structured, and plausibly runnable?\n"
        "- runtime_reliability: did execution complete cleanly with healthy trace/events?\n"
        "- observability_quality: are logs/diffs/trace sufficient to debug behavior?\n"
        "- risk_posture: identify regressions, contradictions, or unsafe patterns.\n"
        "Return conservative scores. Use `fail` if objective is clearly not met or runtime is unstable.\n"
        "Output strict JSON only (no markdown). Be concise:\n"
        "- criterion reasoning: <= 25 words each\n"
        "- strengths/issues/recommendations: <= 4 bullets each\n"
        "- summary: <= 80 words"
    )
    user = json.dumps(payload, ensure_ascii=False)

    try:
        parsed, meta = call_llm_structured(
            judge_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=JudgeReport,
            task="agent_ontology_live_e2e_review",
            trace_id=f"agent_ontology/live_e2e_review/{run_id}",
            max_budget=max_budget,
            max_tokens=2200,
        )
    except Exception as exc:
        return {"error": f"Judge call failed: {type(exc).__name__}: {exc}"}

    report_payload = parsed.model_dump()
    criteria = report_payload.get("criteria", [])
    criteria_norm = _criteria_score_normalized(criteria)
    declared_raw = float(report_payload.get("overall_score", 0.0) or 0.0)
    declared_norm = _normalize_scale_0_to_1(declared_raw)
    overall_norm = criteria_norm if criteria_norm is not None else declared_norm

    report_payload["overall_score_raw"] = round(declared_raw, 4)
    report_payload["overall_score_normalized"] = round(overall_norm, 4)
    if criteria_norm is not None:
        report_payload["criteria_overall_score_normalized"] = round(criteria_norm, 4)
        # Compatibility for existing consumers.
        report_payload["computed_overall_score"] = round(criteria_norm, 4)
        if abs(declared_norm - criteria_norm) > 0.2:
            report_payload["score_consistency_warning"] = (
                "declared overall score materially disagrees with criteria-derived score "
                f"({declared_norm:.3f} vs {criteria_norm:.3f})"
            )
    report_payload["declared_overall_score_normalized"] = round(declared_norm, 4)

    confidence_raw = float(report_payload.get("confidence", 0.0) or 0.0)
    report_payload["confidence_raw"] = round(confidence_raw, 4)
    report_payload["confidence_normalized"] = round(_normalize_scale_0_to_1(confidence_raw), 4)

    meta_payload = {
        "model": getattr(meta, "model", judge_model),
        "cost": getattr(meta, "cost", None),
        "usage": getattr(meta, "usage", None),
        "billing_mode": getattr(meta, "billing_mode", None),
    }
    return {"report": report_payload, "meta": meta_payload}


def run_live_e2e_review(args: argparse.Namespace) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    root_dir = Path(args.root).resolve()
    server_script = Path(__file__).resolve().parent / "live_viewer_server.py"

    server_cmd = [
        sys.executable,
        "-u",
        str(server_script),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--root",
        str(root_dir),
    ]
    server_proc = subprocess.Popen(  # noqa: S603
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    server_lines: list[str] = []
    try:
        _wait_for_health(base_url, timeout_seconds=20.0)

        run_request = {
            "spec_path": args.spec_path,
            "text": args.text,
            "requested_mode": args.requested_mode,
            "max_iter": args.max_iter,
            "model_override": args.model_override,
            "mock_mode": args.mock_mode,
            "capability_profile": args.capability_profile,
        }

        run_response = _http_post_json(f"{base_url}/api/live/run", run_request, timeout=30.0)
        run_id = str(run_response.get("run_id", ""))
        if not run_id:
            raise RuntimeError(f"Run response missing run_id: {run_response}")

        stream_url = base_url + str(run_response.get("stream_url", f"/api/live/stream/{run_id}"))
        events, timed_out = _stream_events(stream_url, timeout_seconds=args.timeout_seconds)
        status = _http_get_json(f"{base_url}/api/live/status/{run_id}", timeout=15.0)
        event_summary = summarize_live_events(events)
        trace_metrics = _load_trace_metrics(status.get("trace_path"))
        trace_observations = _load_trace_observations(status.get("trace_path"))

        run_dir_raw = event_summary.get("started_event", {}).get("run_dir")
        run_dir = Path(run_dir_raw) if isinstance(run_dir_raw, str) and run_dir_raw else None
        workspace_files = _snapshot_workspace_files(run_dir)

        judge_payload = _build_judge_payload(
            run_request=run_request,
            run_response=run_response,
            status=status,
            event_summary=event_summary,
            trace_metrics=trace_metrics,
            workspace_files=workspace_files,
        )

        judge_result: dict[str, Any] = {}
        if not args.no_judge:
            judge_result = run_llm_judge(
                judge_payload,
                judge_model=args.judge_model,
                run_id=run_id,
                max_budget=args.judge_max_budget,
            )

        deterministic_ok = bool(
            status.get("done")
            and status.get("returncode") == 0
            and event_summary.get("has_completed_event")
            and not timed_out
        )
        review_ok = True
        effective_min_overall_score = args.min_overall_score
        if args.review_profile == "babyagi_strict":
            effective_min_overall_score = max(effective_min_overall_score, 0.75)
        if not args.no_judge:
            report = judge_result.get("report", {})
            if isinstance(report, dict):
                verdict = str(report.get("verdict", "fail")).lower()
                overall = float(
                    report.get(
                        "overall_score_normalized",
                        report.get("computed_overall_score", _normalize_scale_0_to_1(report.get("overall_score", 0.0))),
                    )
                    or 0.0
                )
                review_ok = verdict != "fail" and overall >= effective_min_overall_score
            else:
                review_ok = False
        profile_result = evaluate_review_profile(
            profile=args.review_profile,
            no_judge=bool(args.no_judge),
            trace_metrics=trace_metrics,
            event_summary=event_summary,
            workspace_files=workspace_files,
        )

        report_payload = {
            "ok": bool(deterministic_ok and review_ok and profile_result.get("ok", True)),
            "deterministic_ok": deterministic_ok,
            "review_ok": review_ok,
            "timed_out": timed_out,
            "run_id": run_id,
            "run_request": run_request,
            "run_response": run_response,
            "status": status,
            "trace_metrics": trace_metrics,
            "trace_observations": trace_observations,
            "event_summary": event_summary,
            "workspace_files": workspace_files,
            "review_profile": profile_result,
            "effective_min_overall_score": effective_min_overall_score,
            "judge": judge_result,
        }
        return report_payload
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=2)
        try:
            remaining, _ = server_proc.communicate(timeout=2)
            if remaining:
                server_lines = remaining.splitlines()
        except Exception:
            server_lines = []
        if args.include_server_log and server_lines:
            sys.stderr.write("\n".join(server_lines[-60:]) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live backend E2E + optional LLM output review.")
    parser.add_argument("--host", default="127.0.0.1", help="Live server host")
    parser.add_argument("--port", type=int, default=19100, help="Live server port")
    parser.add_argument("--root", default=str(Path.cwd()), help="Workspace root containing /agent_ontology")

    parser.add_argument("--spec-path", default="agent_ontology/specs/babyagi.yaml", help="Spec path for /api/live/run")
    parser.add_argument("--text", required=True, help="Input text/objective")
    parser.add_argument("--requested-mode", default="default", choices=["default", "plan"])
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--model-override", default="", help="Explicit AGENT_ONTOLOGY_MODEL override")
    parser.add_argument("--mock-mode", default="live", choices=["deterministic", "live", "llm_mock"])
    parser.add_argument(
        "--capability-profile",
        default="restricted",
        choices=["restricted", "external_enabled"],
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)

    parser.add_argument("--no-judge", action="store_true", help="Skip llm_client judging step")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model for llm_client")
    parser.add_argument(
        "--judge-max-budget",
        type=float,
        default=0.0,
        help="max_budget passed to llm_client for judge call (0 = unlimited)",
    )
    parser.add_argument(
        "--min-overall-score",
        type=float,
        default=0.65,
        help="Minimum normalized judge score in [0,1]",
    )
    parser.add_argument(
        "--review-profile",
        default="default",
        choices=["default", "babyagi_strict"],
        help="Additional pass/fail gating profile applied after deterministic + judge checks",
    )
    parser.add_argument("--json-out", default="", help="Optional output file path for full JSON report")
    parser.add_argument(
        "--include-server-log",
        action="store_true",
        help="Emit recent server logs to stderr",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        report = run_live_e2e_review(args)
    except error.HTTPError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"HTTPError: {exc}",
                    "status": getattr(exc, "status", None),
                },
                indent=2,
            )
        )
        raise SystemExit(2) from exc
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        raise SystemExit(2) from exc

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report.get("ok") else 2)


if __name__ == "__main__":
    main()
