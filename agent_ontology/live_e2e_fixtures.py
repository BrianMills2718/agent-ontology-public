#!/usr/bin/env python3
"""Run a fixture pack of live backend E2E reviews."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from agent_ontology.live_e2e_review import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    run_live_e2e_review,
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slug(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return token.strip("._-") or "fixture"


def _load_fixture_pack(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[Any]
    if isinstance(data, dict):
        rows = data.get("fixtures", [])
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Fixture file must be object or list: {path}")

    fixtures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        fixture = dict(row)
        fixture_id = str(fixture.get("id") or f"fixture_{idx + 1}").strip()
        fixture["id"] = fixture_id
        tags = fixture.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        fixture["tags"] = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
        fixtures.append(fixture)
    return fixtures


def _select_fixtures(
    fixtures: list[dict[str, Any]],
    *,
    only_ids: set[str],
    required_tags: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_id = str(fixture.get("id", "")).strip()
        tags = set(str(tag).strip().lower() for tag in fixture.get("tags", []))
        if only_ids and fixture_id not in only_ids:
            continue
        if required_tags and required_tags.isdisjoint(tags):
            continue
        selected.append(fixture)
    return selected


def _assertions_failures(report: dict[str, Any], assertions: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    if "expect_ok" in assertions:
        expect_ok = bool(assertions.get("expect_ok"))
        actual_ok = bool(report.get("ok"))
        if actual_ok != expect_ok:
            failures.append(f"expected ok={expect_ok} but got {actual_ok}")

    event_counts = (report.get("event_summary") or {}).get("event_counts", {})
    if not isinstance(event_counts, dict):
        event_counts = {}
    min_event_counts = assertions.get("min_event_counts", {})
    if isinstance(min_event_counts, dict):
        for key, value in min_event_counts.items():
            need = _as_int(value, 0)
            got = _as_int(event_counts.get(str(key), 0), 0)
            if got < need:
                failures.append(f"event_counts[{key}] expected >= {need}, got {got}")

    process_metrics = (report.get("status") or {}).get("process_metrics", {})
    if not isinstance(process_metrics, dict):
        process_metrics = {}
    min_process_counts = assertions.get("min_process_counts", {})
    if isinstance(min_process_counts, dict):
        for process_id, value in min_process_counts.items():
            need = _as_int(value, 0)
            bucket = process_metrics.get(str(process_id), {})
            got = _as_int((bucket or {}).get("count", 0), 0) if isinstance(bucket, dict) else 0
            if got < need:
                failures.append(f"process_metrics[{process_id}].count expected >= {need}, got {got}")

    trace_metrics = report.get("trace_metrics", {})
    if not isinstance(trace_metrics, dict):
        trace_metrics = {}
    min_trace_metrics = assertions.get("min_trace_metrics", {})
    if isinstance(min_trace_metrics, dict):
        for key, value in min_trace_metrics.items():
            need = _as_int(value, 0)
            got = _as_int(trace_metrics.get(str(key), 0), 0)
            if got < need:
                failures.append(f"trace_metrics[{key}] expected >= {need}, got {got}")

    trace_obs = report.get("trace_observations", {})
    if not isinstance(trace_obs, dict):
        trace_obs = {}
    if "min_executor_external_calls" in assertions:
        need = _as_int(assertions.get("min_executor_external_calls"), 0)
        got = _as_int(trace_obs.get("executor_external_call_count"), 0)
        if got < need:
            failures.append(f"trace_observations.executor_external_call_count expected >= {need}, got {got}")

    required_interfaces = assertions.get("require_executor_interfaces", [])
    if isinstance(required_interfaces, list) and required_interfaces:
        got_interfaces = {
            str(item).strip().lower()
            for item in trace_obs.get("executor_interfaces_canonical", [])
            if str(item).strip()
        }
        missing = [str(item) for item in required_interfaces if str(item).strip().lower() not in got_interfaces]
        if missing:
            failures.append(f"missing executor interface(s): {', '.join(missing)}")

    min_workspace_files = assertions.get("min_workspace_files")
    if min_workspace_files is not None:
        need = _as_int(min_workspace_files, 0)
        workspace_files = report.get("workspace_files", [])
        got = len(workspace_files) if isinstance(workspace_files, list) else 0
        if got < need:
            failures.append(f"workspace_files expected >= {need}, got {got}")

    return failures


def _build_live_args(
    *,
    base_args: argparse.Namespace,
    fixture: dict[str, Any],
    port: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        host=base_args.host,
        port=port,
        root=base_args.root,
        spec_path=str(fixture.get("spec_path") or ""),
        text=str(fixture.get("text") or ""),
        requested_mode=str(fixture.get("requested_mode") or base_args.requested_mode),
        max_iter=_as_int(fixture.get("max_iter"), _as_int(base_args.max_iter, 20)),
        model_override=str(fixture.get("model_override") or base_args.model_override),
        mock_mode=str(fixture.get("mock_mode") or base_args.mock_mode),
        capability_profile=str(fixture.get("capability_profile") or base_args.capability_profile),
        timeout_seconds=float(fixture.get("timeout_seconds") or base_args.timeout_seconds),
        no_judge=bool(fixture.get("no_judge", base_args.no_judge_default)),
        judge_model=str(fixture.get("judge_model") or base_args.judge_model),
        judge_max_budget=float(fixture.get("judge_max_budget") or base_args.judge_max_budget),
        min_overall_score=float(fixture.get("min_overall_score") or base_args.min_overall_score),
        review_profile=str(fixture.get("review_profile") or base_args.review_profile),
        json_out="",
        include_server_log=bool(base_args.include_server_log),
    )


def run_fixture_pack(args: argparse.Namespace) -> dict[str, Any]:
    fixture_path = Path(args.fixtures).resolve()
    fixtures = _load_fixture_pack(fixture_path)
    selected = _select_fixtures(
        fixtures,
        only_ids={str(item).strip() for item in args.only_id if str(item).strip()},
        required_tags={str(item).strip().lower() for item in args.tag if str(item).strip()},
    )

    reports_dir = Path(args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    started_at = time.time()

    for idx, fixture in enumerate(selected):
        fixture_id = str(fixture.get("id", f"fixture_{idx + 1}"))
        run_port = _as_int(args.base_port, 19100) + idx
        run_args = _build_live_args(base_args=args, fixture=fixture, port=run_port)

        fixture_started = time.monotonic()
        fixture_errors: list[str] = []
        report: dict[str, Any] = {}

        if not run_args.spec_path:
            fixture_errors.append("missing spec_path")
        if not run_args.text:
            fixture_errors.append("missing text")

        if not fixture_errors:
            try:
                report = run_live_e2e_review(run_args)
            except Exception as exc:  # pragma: no cover - runtime integration path
                fixture_errors.append(f"runtime error: {type(exc).__name__}: {exc}")

        assertions = fixture.get("assertions", {})
        if isinstance(assertions, dict) and report:
            fixture_errors.extend(_assertions_failures(report, assertions))

        fixture_ok = not fixture_errors
        duration_seconds = round(time.monotonic() - fixture_started, 3)
        run_id = str(report.get("run_id", "")) if isinstance(report, dict) else ""
        slug = _slug(fixture_id)
        report_path = reports_dir / f"{slug}.json"
        report_payload = {
            "fixture": fixture,
            "ok": fixture_ok,
            "errors": fixture_errors,
            "report": report,
            "duration_seconds": duration_seconds,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        row = {
            "id": fixture_id,
            "tags": fixture.get("tags", []),
            "ok": fixture_ok,
            "errors": fixture_errors,
            "duration_seconds": duration_seconds,
            "run_id": run_id,
            "report_path": str(report_path),
            "deterministic_ok": bool(report.get("deterministic_ok")) if isinstance(report, dict) else False,
            "review_ok": bool(report.get("review_ok")) if isinstance(report, dict) else False,
            "event_counts": ((report.get("event_summary") or {}).get("event_counts", {})) if isinstance(report, dict) else {},
            "trace_observations": report.get("trace_observations", {}) if isinstance(report, dict) else {},
        }
        results.append(row)

        if fixture_ok:
            passed += 1
        else:
            failed += 1
            if args.fail_fast:
                break

    summary = {
        "ok": failed == 0 and len(selected) > 0,
        "fixture_pack": str(fixture_path),
        "selected_fixture_count": len(selected),
        "passed": passed,
        "failed": failed,
        "started_at_unix": started_at,
        "duration_seconds": round(time.time() - started_at, 3),
        "results": results,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixture packs for live backend E2E review.")
    parser.add_argument(
        "--fixtures",
        default="agent_ontology/benchmarks/live_e2e_fixtures.json",
        help="Fixture JSON path",
    )
    parser.add_argument("--root", default=str(Path.cwd()), help="Workspace root containing /agent_ontology")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=19100, help="Base port (fixture index is added)")
    parser.add_argument("--requested-mode", default="default", choices=["default", "plan"])
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--model-override", default=DEFAULT_MODEL)
    parser.add_argument("--mock-mode", default="deterministic", choices=["deterministic", "live", "llm_mock"])
    parser.add_argument(
        "--capability-profile",
        default="restricted",
        choices=["restricted", "external_enabled"],
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)

    parser.add_argument("--no-judge-default", action="store_true", help="Default fixtures to no-judge when unset")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-max-budget", type=float, default=0.0)
    parser.add_argument("--min-overall-score", type=float, default=0.65)
    parser.add_argument("--review-profile", default="default", choices=["default", "babyagi_strict"])

    parser.add_argument("--only-id", action="append", default=[], help="Run only exact fixture id (repeatable)")
    parser.add_argument("--tag", action="append", default=[], help="Run fixtures matching any tag (repeatable)")
    parser.add_argument("--reports-dir", default=".live_runs/e2e_fixture_reports")
    parser.add_argument("--json-out", default="", help="Optional aggregate output JSON path")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--include-server-log", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_fixture_pack(args)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report.get("ok") else 2)


if __name__ == "__main__":
    main()
