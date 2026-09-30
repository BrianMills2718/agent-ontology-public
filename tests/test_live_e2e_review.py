from __future__ import annotations

from pathlib import Path
import json

from agent_ontology.live_e2e_review import (
    _criteria_score_normalized,
    _load_trace_observations,
    _normalize_scale_0_to_1,
    _snapshot_workspace_files,
    evaluate_review_profile,
    summarize_live_events,
)


def test_summarize_live_events_collects_expected_fields() -> None:
    events = [
        {"type": "started", "run_id": "abc123", "model_override": "openrouter/deepseek/deepseek-chat"},
        {"type": "runtime_info", "info": {"has_call_llm": True}},
        {"type": "log", "line": "line 1"},
        {"type": "llm_call", "process": "decide", "agent": "Planner", "model": "deepseek", "duration_ms": 1234},
        {
            "type": "file_diff",
            "payload": {
                "process": "pull_task",
                "added_files": ["autogpt_workspace/app.py"],
                "changed_files": [],
                "removed_files": [],
                "patches": {"autogpt_workspace/app.py": "--- /dev/null\n+++ b/autogpt_workspace/app.py\n+print('x')"},
            },
        },
        {"type": "step_diff", "payload": {"process": "pull_task", "added": ["task"], "changed": [], "removed": []}},
        {"type": "final_data", "summary": {"keys": ["answer"], "finalized": True}},
        {"type": "completed", "returncode": 0},
    ]

    summary = summarize_live_events(events)
    assert summary["has_completed_event"] is True
    assert summary["event_counts"]["started"] == 1
    assert summary["event_counts"]["file_diff"] == 1
    assert summary["started_event"]["run_id"] == "abc123"
    assert summary["runtime_info"]["has_call_llm"] is True
    assert summary["final_data_summary"]["keys"] == ["answer"]
    assert summary["llm_call_samples"][0]["model"] == "deepseek"
    assert summary["file_diff_samples"][0]["added_files"] == ["autogpt_workspace/app.py"]


def test_snapshot_workspace_files_skips_trace_and_binary(tmp_path: Path) -> None:
    run_dir = tmp_path
    (run_dir / "autogpt_workspace").mkdir()
    (run_dir / "autogpt_workspace" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (run_dir / "trace.json").write_text("{}", encoding="utf-8")
    (run_dir / "autogpt_workspace" / "blob.bin").write_bytes(b"\x00\x01\x02")

    rows = _snapshot_workspace_files(run_dir, max_files=10, max_bytes=50_000, max_preview=120)
    paths = {row["path"] for row in rows}
    assert "autogpt_workspace/app.py" in paths
    assert "trace.json" not in paths
    assert "autogpt_workspace/blob.bin" not in paths


def test_babyagi_strict_profile_requires_artifacts_and_judge() -> None:
    trace_metrics = {"clean_exit": True, "total_llm_calls": 2}
    event_summary = {"event_counts": {"file_diff": 0}, "final_data_summary": {"keys": ["answer"]}}
    workspace_files = []

    strict_no_judge = evaluate_review_profile(
        profile="babyagi_strict",
        no_judge=True,
        trace_metrics=trace_metrics,
        event_summary=event_summary,
        workspace_files=workspace_files,
    )
    assert strict_no_judge["ok"] is False

    strict_with_artifact = evaluate_review_profile(
        profile="babyagi_strict",
        no_judge=False,
        trace_metrics=trace_metrics,
        event_summary=event_summary,
        workspace_files=[{"path": "todo_api.py", "size": 123, "preview": "print('ok')"}],
    )
    assert strict_with_artifact["ok"] is True


def test_babyagi_strict_profile_fails_when_quality_gate_failed() -> None:
    trace_metrics = {"clean_exit": True, "total_llm_calls": 2}
    event_summary = {
        "event_counts": {"file_diff": 1},
        "final_data_summary": {
            "keys": ["quality_gate_failed", "quality_gate_reason", "quality_file_check"],
            "quality_gate_failed": True,
            "quality_gate_reason": "all_candidate_files_empty",
            "quality_file_check": {"candidate_count": 3, "non_empty_count": 0, "empty_count": 3},
        },
    }
    workspace_files = [{"path": "README.md", "size": 32, "preview": "todo app"}]

    strict = evaluate_review_profile(
        profile="babyagi_strict",
        no_judge=False,
        trace_metrics=trace_metrics,
        event_summary=event_summary,
        workspace_files=workspace_files,
    )
    assert strict["ok"] is False
    checks = {str(row.get("name")): bool(row.get("ok")) for row in strict.get("checks", [])}
    assert checks.get("quality_gate_passed") is False


def test_babyagi_strict_profile_fails_when_candidate_files_are_all_empty() -> None:
    trace_metrics = {"clean_exit": True, "total_llm_calls": 2}
    event_summary = {
        "event_counts": {"file_diff": 1},
        "final_data_summary": {
            "keys": ["quality_file_check"],
            "quality_file_check": {"candidate_count": 2, "non_empty_count": 0, "empty_count": 2},
        },
    }
    workspace_files = [{"path": "artifact.txt", "size": 1, "preview": "x"}]

    strict = evaluate_review_profile(
        profile="babyagi_strict",
        no_judge=False,
        trace_metrics=trace_metrics,
        event_summary=event_summary,
        workspace_files=workspace_files,
    )
    assert strict["ok"] is False
    checks = {str(row.get("name")): bool(row.get("ok")) for row in strict.get("checks", [])}
    assert checks.get("quality_gate_passed") is False


def test_score_normalization_helpers_use_canonical_zero_to_one_scale() -> None:
    assert _normalize_scale_0_to_1(4.0) == 0.8
    assert _normalize_scale_0_to_1(0.8) == 0.8
    assert _normalize_scale_0_to_1("bad") == 0.0

    criteria = [
        {"criterion": "objective_completion", "score": 5},
        {"criterion": "artifact_quality", "score": 3},
    ]
    assert _criteria_score_normalized(criteria) == 0.8


def test_load_trace_observations_extracts_executor_interfaces(tmp_path: Path) -> None:
    trace = {
        "metrics": {"total_llm_calls": 2},
        "trace": [
            {
                "agent": "Executor",
                "response": "```json\n{\"action_type\":\"external_call\",\"interface_name\":\"web_fetch\"}\n```",
            },
            {
                "agent": "Executor",
                "response": "{\"action_type\":\"external_call\",\"interface_name\":\"mcp_tool_call\"}",
            },
        ],
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    obs = _load_trace_observations(str(trace_path))
    assert obs["executor_calls"] == 2
    assert obs["executor_external_call_count"] == 2
    assert "external_call" in obs["executor_action_types"]
    assert "web_fetch" in obs["executor_interfaces"]
    assert "mcp_tool_call" in obs["executor_interfaces"]
    assert "semantic_search" in obs["executor_interfaces_canonical"]
    assert "mcp_call" in obs["executor_interfaces_canonical"]
