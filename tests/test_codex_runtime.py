from __future__ import annotations

from agents import codex_agent


def _new_state() -> codex_agent.AgentState:
    return codex_agent.AgentState()


def test_action_gate_uses_nested_action_type_fallback() -> None:
    state = _new_state()
    state.data = {
        "action_decision": {
            "action_type": "tool_calls",
        }
    }
    assert codex_agent.process_action_gate(state) == "run_tools"


def test_action_gate_response_branch_when_no_tool_calls() -> None:
    state = _new_state()
    state.data = {
        "action_type": "respond",
        "action_decision": {"action_type": "respond"},
    }
    assert codex_agent.process_action_gate(state) == "finalize_response"


def test_run_tools_supports_batch_calls_and_aliases() -> None:
    state = _new_state()
    state.data = {
        "tool_calls": [
            {
                "recipient_name": "functions.exec_command",
                "parameters": {"cmd": "echo hello"},
            },
            {
                "tool_name": "unknown_custom_tool",
                "arguments": "demo",
            },
        ]
    }

    result_state = codex_agent.process_run_tools(state)
    assert result_state is state
    tool_results = state.data.get("tool_results")
    assert isinstance(tool_results, list)
    assert len(tool_results) == 2
    first_output = str(tool_results[0].get("output", "")).lower()
    assert "unknown tool" not in first_output
    assert tool_results[0].get("tool_name") == "exec_command"
    assert "unknown tool" in str(tool_results[1].get("output", "")).lower()
    assert "observation" in state.data
