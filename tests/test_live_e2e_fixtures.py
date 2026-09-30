from __future__ import annotations

from pathlib import Path
import json

from agent_ontology.live_e2e_fixtures import (
    _assertions_failures,
    _load_fixture_pack,
    _select_fixtures,
)


def test_load_fixture_pack_supports_object_shape(tmp_path: Path) -> None:
    fixture_file = tmp_path / "fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {"id": "a", "tags": ["x"]},
                    {"id": "b", "tags": "y"},
                ]
            }
        ),
        encoding="utf-8",
    )
    fixtures = _load_fixture_pack(fixture_file)
    assert [row["id"] for row in fixtures] == ["a", "b"]
    assert fixtures[0]["tags"] == ["x"]
    assert fixtures[1]["tags"] == ["y"]


def test_select_fixtures_filters_by_id_and_tag() -> None:
    fixtures = [
        {"id": "one", "tags": ["ci", "external"]},
        {"id": "two", "tags": ["ci"]},
    ]
    selected = _select_fixtures(fixtures, only_ids={"one"}, required_tags={"external"})
    assert len(selected) == 1
    assert selected[0]["id"] == "one"


def test_assertions_failures_validates_external_observations() -> None:
    report = {
        "ok": True,
        "event_summary": {"event_counts": {"completed": 1}},
        "status": {"process_metrics": {"execute_external_call": {"count": 2}}},
        "trace_metrics": {"total_llm_calls": 3},
        "trace_observations": {
            "executor_external_call_count": 2,
            "executor_interfaces_canonical": ["semantic_search", "mcp_call"],
        },
        "workspace_files": [{"path": "x.py"}],
    }
    failures = _assertions_failures(
        report,
        {
            "expect_ok": True,
            "min_event_counts": {"completed": 1},
            "min_process_counts": {"execute_external_call": 1},
            "min_trace_metrics": {"total_llm_calls": 1},
            "min_executor_external_calls": 1,
            "require_executor_interfaces": ["mcp_call"],
            "min_workspace_files": 1,
        },
    )
    assert failures == []

    failures_missing = _assertions_failures(
        report,
        {"require_executor_interfaces": ["rpc_emit"]},
    )
    assert failures_missing
