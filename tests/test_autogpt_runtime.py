from __future__ import annotations

from pathlib import Path
import json

from agents import autogpt_agent


def _new_state() -> autogpt_agent.AgentState:
    return autogpt_agent.AgentState()


def test_set_goal_normalizes_objective_and_workspace(tmp_path: Path) -> None:
    state = _new_state()
    state.data = {
        "text": "build me a full app that handles todos",
        "workspace_dir": str(tmp_path / "ws"),
    }
    autogpt_agent.process_set_goal(state)
    assert state.data.get("objective") == "build me a full app that handles todos"
    workspace_dir = state.data.get("workspace_dir")
    assert isinstance(workspace_dir, str)
    assert Path(workspace_dir).exists()
    assert state.data.get("capability_profile") == "restricted"


def test_think_overrides_env_setup_plan_for_todo_full_app(monkeypatch) -> None:
    state = _new_state()
    state.data = {
        "objective": "build me a full app that handles todos",
        "plan": [
            "Create a new directory and initialize git",
            "Create and activate a virtual environment",
            "Implement app code",
        ],
        "current_step_idx": 0,
        "results": [],
    }

    def _fake_thinker(_msg: str, output_schema: str | None = None) -> str:
        assert output_schema == "Thought"
        return json.dumps(
            {
                "reasoning": "Use file-first project setup.",
                "proposed_action": "create initial app files",
            }
        )

    monkeypatch.setattr(autogpt_agent, "invoke_thinker", _fake_thinker)
    autogpt_agent.process_think(state)
    plan = state.data.get("plan")
    assert isinstance(plan, list)
    assert len(plan) >= 6
    assert "project structure" in str(plan[0]).lower()
    assert "dependency-free" in " ".join(str(x).lower() for x in plan)
    assert state.data.get("current_step") == plan[0]


def test_execute_command_runs_simple_shell_command(tmp_path: Path) -> None:
    state = _new_state()
    workspace = tmp_path / "ws"
    state.data = {
        "workspace_dir": str(workspace),
        "command": "echo hello > hello.txt",
        "current_step_idx": 0,
    }
    autogpt_agent.process_execute_command(state)
    assert state.data.get("success") is True
    assert state.data.get("executed_command") == "echo hello > hello.txt"
    assert (workspace / "hello.txt").exists()
    assert state.data.get("key") == "step_0"
    assert isinstance(state.data.get("value"), str)


def test_execute_command_blocks_heavy_or_risky_commands(tmp_path: Path) -> None:
    state = _new_state()
    state.data = {
        "workspace_dir": str(tmp_path / "ws"),
        "command": "npm install express",
        "current_step_idx": 1,
        "capability_profile": "restricted",
    }
    autogpt_agent.process_execute_command(state)
    assert state.data.get("success") is False
    assert "Blocked" in str(state.data.get("action_result", ""))


def test_execute_command_allows_dependency_commands_when_external_enabled(tmp_path: Path) -> None:
    state = _new_state()
    state.data = {
        "workspace_dir": str(tmp_path / "ws"),
        "command": "pip install",
        "current_step_idx": 2,
        "capability_profile": "external_enabled",
    }
    autogpt_agent.process_execute_command(state)
    action_result = str(state.data.get("action_result", ""))
    assert "Blocked by capability profile" not in action_result
    assert "Blocked potentially destructive or long-running command" not in action_result


def test_execute_command_autofallback_generates_files_for_build_intent(tmp_path: Path) -> None:
    state = _new_state()
    workspace = tmp_path / "ws"
    state.data = {
        "workspace_dir": str(workspace),
        "command": "",
        "action_type": "command",
        "objective": "build a todo api app",
        "current_step": "create initial project files",
        "proposed_action": "write starter code files",
        "current_step_idx": 0,
    }
    autogpt_agent.process_execute_command(state)
    assert state.data.get("success") is True
    assert state.data.get("autofallback_generated_command") is True
    assert (workspace / "README.md").exists()
    assert (workspace / "src" / "main.py").exists()
    assert any(
        (workspace / "tests" / name).exists()
        for name in ("test_smoke.py", "test_todo_app.py")
    )


def test_execute_command_autofallback_replaces_privileged_install_for_build_intent(tmp_path: Path) -> None:
    state = _new_state()
    workspace = tmp_path / "ws"
    state.data = {
        "workspace_dir": str(workspace),
        "command": "sudo apt-get install sqlite3",
        "action_type": "command",
        "objective": "build a todo api app",
        "current_step": "implement local persistence and app files",
        "proposed_action": "install sqlite then continue",
        "current_step_idx": 0,
        "capability_profile": "external_enabled",
    }
    autogpt_agent.process_execute_command(state)
    assert state.data.get("success") is True
    assert state.data.get("autofallback_generated_command") is True
    assert state.data.get("autofallback_reason") == "blocked_privileged_or_package_manager_command"
    assert str(state.data.get("autofallback_original_command", "")).startswith("sudo apt-get")
    assert (workspace / "README.md").exists()
    assert (workspace / "src" / "main.py").exists()
    assert any(
        (workspace / "tests" / name).exists()
        for name in ("test_smoke.py", "test_todo_app.py")
    )


def test_execute_command_autofallback_when_missing_tool_for_build_intent(tmp_path: Path) -> None:
    state = _new_state()
    workspace = tmp_path / "ws"
    state.data = {
        "workspace_dir": str(workspace),
        "command": "definitely_missing_command_xyz123 --version",
        "action_type": "command",
        "objective": "build a todo api app",
        "current_step": "create project files",
        "proposed_action": "run missing tool then continue",
        "current_step_idx": 0,
    }
    autogpt_agent.process_execute_command(state)
    assert state.data.get("success") is True
    assert state.data.get("autofallback_generated_command") is True
    assert state.data.get("autofallback_reason") == "missing_tool_for_build_intent"
    assert "Original command failed due missing tooling" in str(state.data.get("action_result", ""))
    assert (workspace / "README.md").exists()
    assert (workspace / "src" / "main.py").exists()


def test_execute_command_quality_gate_rejects_all_empty_candidate_files_for_build_intent(tmp_path: Path) -> None:
    state = _new_state()
    workspace = tmp_path / "ws"
    state.data = {
        "workspace_dir": str(workspace),
        "command": "mkdir -p todo_app && : > todo_app/README.md && : > todo_app/main.py",
        "action_type": "command",
        "objective": "build a full app that handles todos",
        "current_step": "create app files",
        "proposed_action": "write initial files",
        "current_step_idx": 0,
    }
    autogpt_agent.process_execute_command(state)
    assert state.data.get("success") is False
    assert state.data.get("quality_gate_failed") is True
    assert state.data.get("quality_gate_reason") == "all_candidate_files_empty"
    quality = state.data.get("quality_file_check")
    assert isinstance(quality, dict)
    assert int(quality.get("candidate_count", 0) or 0) >= 2
    assert int(quality.get("non_empty_count", 0) or 0) == 0
    assert "Quality gate failed" in str(state.data.get("action_result", ""))


def test_execute_command_quality_gate_passes_with_non_empty_file_for_build_intent(tmp_path: Path) -> None:
    state = _new_state()
    workspace = tmp_path / "ws"
    state.data = {
        "workspace_dir": str(workspace),
        "command": "mkdir -p todo_app && printf 'print(1)\\n' > todo_app/main.py",
        "action_type": "command",
        "objective": "build a full app that handles todos",
        "current_step": "create app files",
        "proposed_action": "write initial files",
        "current_step_idx": 0,
    }
    autogpt_agent.process_execute_command(state)
    assert state.data.get("success") is True
    assert state.data.get("quality_gate_failed") is False
    quality = state.data.get("quality_file_check")
    assert isinstance(quality, dict)
    assert int(quality.get("candidate_count", 0) or 0) >= 1
    assert int(quality.get("non_empty_count", 0) or 0) >= 1


def test_execute_external_call_dispatches_in_external_enabled_mode() -> None:
    state = _new_state()
    state.data = {
        "capability_profile": "external_enabled",
        "action_type": "external_call",
        "interface_name": "semantic_search",
        "payload": {"query": "todo app architecture"},
        "current_step_idx": 0,
    }
    autogpt_agent.process_execute_external_call(state)
    assert state.data.get("success") is True
    external_result = state.data.get("external_result")
    assert isinstance(external_result, dict)
    assert external_result.get("handled") is True
    assert state.data.get("executed_interface") == "semantic_search"
    assert isinstance(state.data.get("action_result"), str)


def test_execute_external_call_blocked_when_restricted() -> None:
    state = _new_state()
    state.data = {
        "capability_profile": "restricted",
        "action_type": "external_call",
        "interface_name": "semantic_search",
        "payload": {"query": "todo app architecture"},
        "current_step_idx": 0,
    }
    autogpt_agent.process_execute_external_call(state)
    assert state.data.get("success") is False
    assert "Blocked by capability profile" in str(state.data.get("action_result", ""))


def test_execution_mode_gate_routes_external_call() -> None:
    state = _new_state()
    state.data = {"action_type": "external_call"}
    assert autogpt_agent.process_execution_mode_gate(state) == "execute_external_call"


def test_post_act_routing_override_prefers_external_call_for_retrieval_intent(monkeypatch) -> None:
    state = _new_state()
    state.data = {
        "objective": "research livekit session lifecycle and find references",
        "current_step": "search for sources about livekit session lifecycle",
        "proposed_action": "look up livekit session lifecycle architecture",
    }

    def _fake_executor(_msg: str, output_schema: str | None = None) -> str:
        assert output_schema == "ActionResult"
        return json.dumps(
            {
                "action_type": "command",
                "command": "",
                "interface_name": "",
                "payload": {},
                "routing_policy": {},
                "action_result": "planned",
                "success": True,
            }
        )

    monkeypatch.setattr(autogpt_agent, "invoke_executor", _fake_executor)
    autogpt_agent.process_act(state)
    autogpt_agent.process_normalize_action_routing(state)
    assert state.data.get("action_type") == "external_call"
    assert state.data.get("interface_name") == "semantic_search"
    payload = state.data.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("query")
    assert state.data.get("routing_override_reason") == "retrieval_intent_prefers_external_call"


def test_post_act_routing_override_prefers_mcp_call_when_intent_mentions_mcp(monkeypatch) -> None:
    state = _new_state()
    state.data = {
        "objective": "research via mcp and summarize findings",
        "current_step": "search for sources using mcp",
        "proposed_action": "look up sources with mcp search",
    }

    def _fake_executor(_msg: str, output_schema: str | None = None) -> str:
        assert output_schema == "ActionResult"
        return json.dumps(
            {
                "action_type": "command",
                "command": "",
                "interface_name": "",
                "payload": {},
                "routing_policy": {},
                "action_result": "planned",
                "success": True,
            }
        )

    monkeypatch.setattr(autogpt_agent, "invoke_executor", _fake_executor)
    autogpt_agent.process_act(state)
    autogpt_agent.process_normalize_action_routing(state)
    assert state.data.get("action_type") == "external_call"
    assert state.data.get("interface_name") == "mcp_call"
    payload = state.data.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("tool") == "search"
    args = payload.get("args")
    assert isinstance(args, dict)
    assert args.get("query")


def test_post_act_routing_override_prefers_external_when_build_objective_has_retrieval_step(monkeypatch) -> None:
    state = _new_state()
    state.data = {
        "objective": "build a full todo app end-to-end",
        "current_step": "search for reliable references about todo API pagination best practices",
        "proposed_action": "look up examples for todo API pagination and response schemas",
    }

    def _fake_executor(_msg: str, output_schema: str | None = None) -> str:
        assert output_schema == "ActionResult"
        return json.dumps(
            {
                "action_type": "command",
                "command": "curl https://example.com/todo-api-pagination",
                "interface_name": "",
                "payload": {},
                "routing_policy": {},
                "action_result": "planned",
                "success": True,
            }
        )

    monkeypatch.setattr(autogpt_agent, "invoke_executor", _fake_executor)
    autogpt_agent.process_act(state)
    autogpt_agent.process_normalize_action_routing(state)
    assert state.data.get("action_type") == "external_call"
    assert state.data.get("interface_name") == "semantic_search"
    payload = state.data.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("query")
    assert state.data.get("routing_override_reason") == "retrieval_intent_prefers_external_call"


def test_post_act_routing_override_does_not_force_external_for_build_step(monkeypatch) -> None:
    state = _new_state()
    state.data = {
        "objective": "build a full todo app end-to-end",
        "current_step": "implement sqlite-backed todo models and storage layer",
        "proposed_action": "write code for the todo models in the project workspace",
    }

    def _fake_executor(_msg: str, output_schema: str | None = None) -> str:
        assert output_schema == "ActionResult"
        return json.dumps(
            {
                "action_type": "command",
                "command": "cat > models.py",
                "interface_name": "",
                "payload": {},
                "routing_policy": {},
                "action_result": "planned",
                "success": True,
            }
        )

    monkeypatch.setattr(autogpt_agent, "invoke_executor", _fake_executor)
    autogpt_agent.process_act(state)
    autogpt_agent.process_normalize_action_routing(state)
    assert state.data.get("action_type") == "command"
    assert state.data.get("interface_name") == ""
    assert state.data.get("routing_override_reason") != "retrieval_intent_prefers_external_call"


def test_routing_policy_from_executor_forces_external_call(monkeypatch) -> None:
    state = _new_state()
    state.data = {
        "objective": "build a todo app",
        "current_step": "collect references for api pagination",
        "proposed_action": "research api pagination patterns",
    }

    def _fake_executor(_msg: str, output_schema: str | None = None) -> str:
        assert output_schema == "ActionResult"
        return json.dumps(
            {
                "action_type": "command",
                "command": "echo placeholder",
                "interface_name": "",
                "payload": {},
                "routing_policy": {
                    "prefer_external_call": True,
                    "preferred_interface": "semantic_search",
                    "query": "todo api pagination best practices",
                    "reason": "retrieval step requires references",
                },
                "action_result": "planned",
                "success": True,
            }
        )

    monkeypatch.setattr(autogpt_agent, "invoke_executor", _fake_executor)
    autogpt_agent.process_act(state)
    autogpt_agent.process_normalize_action_routing(state)
    assert state.data.get("action_type") == "external_call"
    assert state.data.get("interface_name") == "semantic_search"
    assert state.data.get("routing_override_reason") == "routing_policy_prefers_external_call"
    assert state.data.get("routing_override_policy_reason") == "retrieval step requires references"


def test_normalize_action_routing_repairs_external_call_missing_interface() -> None:
    state = _new_state()
    state.data = {
        "objective": "research todo api pagination references",
        "current_step": "look up best practices",
        "proposed_action": "find sources for todo api pagination",
        "action_type": "external_call",
        "interface_name": "",
        "payload": {},
        "routing_policy": {},
    }
    autogpt_agent.process_normalize_action_routing(state)
    assert state.data.get("action_type") == "external_call"
    assert state.data.get("interface_name") == "semantic_search"
    payload = state.data.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("query")
    assert state.data.get("routing_override_reason") == "external_call_interface_repaired"
    telemetry = state.data.get("routing_telemetry")
    assert isinstance(telemetry, dict)
    assert int(telemetry.get("external_interface_repaired", 0) or 0) >= 1


def test_execute_external_call_uses_semantic_fallback_when_interface_unhandled(monkeypatch) -> None:
    state = _new_state()
    state.data = {
        "capability_profile": "external_enabled",
        "action_type": "external_call",
        "interface_name": "mcp_call",
        "payload": {"query": "todo api design"},
        "current_step_idx": 0,
    }

    def _fake_dispatch(interface_name: str, payload: dict[str, object]) -> dict[str, object]:
        if interface_name == "mcp_call":
            return {"handled": False, "ok": True, "provider": "stub", "mode": "deterministic"}
        if interface_name == "semantic_search":
            return {
                "handled": True,
                "ok": True,
                "text": "Found semantic references",
                "provider": "stub",
                "mode": "deterministic",
            }
        raise AssertionError(f"unexpected interface {interface_name}")

    monkeypatch.setattr(autogpt_agent, "_external_tool_dispatch", _fake_dispatch)
    autogpt_agent.process_execute_external_call(state)
    assert state.data.get("success") is True
    assert state.data.get("executed_interface") == "semantic_search"
    dispatch = state.data.get("semantic_fallback_dispatch")
    assert isinstance(dispatch, dict)
    assert dispatch.get("from_interface") == "mcp_call"
    telemetry = state.data.get("routing_telemetry")
    assert isinstance(telemetry, dict)
    assert int(telemetry.get("semantic_fallback_dispatches", 0) or 0) >= 1


def test_resume_checkpoint_restores_selected_fields() -> None:
    state = _new_state()
    state.data = {
        "resume_requested": True,
        "checkpoint": {
            "plan": ["step a", "step b"],
            "current_step_idx": 1,
            "results": ["ok"],
            "completed_steps": ["step a"],
            "verify_retry_count": 1,
            "verification_fail_count": 2,
        },
    }
    autogpt_agent.process_resume_checkpoint(state)
    assert state.data.get("resume_applied") is True
    assert state.data.get("resume_requested") is False
    assert state.data.get("current_step_idx") == 1
    assert state.data.get("plan") == ["step a", "step b"]
    assert state.data.get("verify_retry_count") == 1


def test_checkpoint_state_persists_snapshot_history() -> None:
    state = _new_state()
    state.data = {
        "current_step_idx": 2,
        "total_steps": 5,
        "current_step": "implement api",
        "action_type": "command",
        "success": True,
        "results": ["ok1", "ok2"],
        "plan": ["a", "b", "c"],
        "completed_steps": ["a", "b"],
        "checkpoints": [],
    }
    autogpt_agent.process_checkpoint_state(state)
    checkpoint = state.data.get("checkpoint")
    assert isinstance(checkpoint, dict)
    assert checkpoint.get("current_step_idx") == 2
    assert checkpoint.get("results_count") == 2
    checkpoints = state.data.get("checkpoints")
    assert isinstance(checkpoints, list) and len(checkpoints) == 1


def test_verify_progress_requests_retry_when_failed_and_budget_available() -> None:
    state = _new_state()
    state.data = {
        "constraints": ["must include api endpoint"],
        "action_result": "implemented storage layer only",
        "success": False,
        "current_step_idx": 3,
        "verify_retry_count": 0,
        "max_verify_retries": 2,
        "verification_fail_count": 0,
    }
    autogpt_agent.process_verify_progress(state)
    assert state.data.get("verification_status") == "retry"
    assert state.data.get("verify_retry_count") == 1
    assert state.data.get("current_step_idx") == 2
    assert state.data.get("verification_fail_count") == 1


def test_verify_progress_pass_resets_retry_counter() -> None:
    state = _new_state()
    state.data = {
        "constraints": [],
        "action_result": "done",
        "success": True,
        "current_step_idx": 2,
        "verify_retry_count": 2,
    }
    autogpt_agent.process_verify_progress(state)
    assert state.data.get("verification_status") == "pass"
    assert state.data.get("verify_retry_count") == 0


def test_handle_rejection_reaches_act_at_retry_limit() -> None:
    state = _new_state()
    state.data = {
        "criticism_count": 2,
        "max_criticism_retries": 3,
        "improvements": "needs changes",
    }
    autogpt_agent.process_handle_rejection(state)
    assert state.data.get("criticism_count") == 3
    assert autogpt_agent.process_retry_or_proceed(state) == "act"
