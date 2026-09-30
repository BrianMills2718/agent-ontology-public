from __future__ import annotations

import json
from pathlib import Path

from agents import babyagi_agent


NO_QUESTIONS_PROMPT = (
    "write me code for a todo task api. full code dont ask any questions just work until its done"
)


def _new_state() -> babyagi_agent.AgentState:
    return babyagi_agent.AgentState()


def _seed_state(tmp_path: Path, prompt: str) -> babyagi_agent.AgentState:
    state = _new_state()
    workspace = tmp_path / "ws"
    state.data = {
        "text": prompt,
        "objective": prompt,
        "constraints": [],
        "workspace_dir": str(workspace),
        "max_tasks": 1,
        "tasks": [
            {
                "id": 1,
                "description": prompt,
                "status": "pending",
                "priority": 1,
            }
        ],
    }
    return state


def test_pull_task_executes_filesystem_tool_calls_and_writes_files(tmp_path: Path, monkeypatch) -> None:
    state = _seed_state(tmp_path, NO_QUESTIONS_PROMPT)

    def _fake_execution_agent(_msg: str, output_schema: str | None = None) -> str:
        assert output_schema == "ExecutionOutput"
        return json.dumps(
            {
                "result": "Planned todo API scaffold.",
                "task_id": 1,
                "tool_calls": [
                    {
                        "tool_name": "filesystem_tool",
                        "input": {
                            "operation": "write_file",
                            "path": "todo_api/app.py",
                            "content": "print('todo api')\n",
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(babyagi_agent, "invoke_execution_agent", _fake_execution_agent)

    babyagi_agent.process_pull_task(state)

    out_file = Path(state.data["workspace_dir"]) / "todo_api" / "app.py"
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == "print('todo api')\n"

    tool_outputs = state.data.get("tool_outputs")
    assert isinstance(tool_outputs, list)
    assert tool_outputs
    assert bool(tool_outputs[0].get("ok")) is True
    assert "Executed 1 tool call(s): 1 ok, 0 failed." in str(state.data.get("result", ""))


def test_pull_task_no_questions_mode_falls_back_to_default_todo_scaffold(tmp_path: Path, monkeypatch) -> None:
    state = _seed_state(tmp_path, NO_QUESTIONS_PROMPT)

    def _fake_execution_agent(_msg: str, output_schema: str | None = None) -> str:
        assert output_schema == "ExecutionOutput"
        return json.dumps(
            {
                "result": "To build the calculator, I need a few preferences so I can set up the repo correctly: which language?",
                "task_id": 1,
                "tool_calls": [],
            }
        )

    monkeypatch.setattr(babyagi_agent, "invoke_execution_agent", _fake_execution_agent)

    babyagi_agent.process_pull_task(state)

    workspace = Path(state.data["workspace_dir"])
    app_file = workspace / "todo_api" / "app.py"
    readme_file = workspace / "todo_api" / "README.md"

    assert state.data.get("no_questions_mode") is True
    assert state.data.get("used_default_scaffold") is True
    assert app_file.exists()
    assert readme_file.exists()
    assert "No-questions mode fallback" in str(state.data.get("result", ""))
